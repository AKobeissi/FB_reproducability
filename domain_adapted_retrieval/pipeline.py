"""
Financial RAG retrieval pipeline.

Full method: Fine-tuned bi-encoder  +  Doc-filtered retrieval  +  Multi-HyDE
             +  Cross-encoder reranking

Two retrieval strategies are supported:

  STANDARD (page-level index + doc-filter):
    1. Filter ChromaDB to only the target document's pages via `where` clause.
    2. Embed original query + HyDE hypotheticals.
    3. Query page-level ChromaDB (doc-filtered) with each embedding.
    4. RRF fusion of multiple ranked lists.
    5. Cross-encoder reranking → top-k pages.

  HIERARCHICAL (chunk-level index + doc-filter):
    1. Filter ChromaDB chunk collection to the target document.
    2. Retrieve top-N chunks via dense search + RRF.
    3. Aggregate chunks to their parent pages (rank by best chunk similarity).
    4. Cross-encoder reranking of the top page candidates → top-k pages.

Doc-filter is the key improvement: since FinanceBench provides doc_name per
question and DocRec@20 ≈ 0.97 with vanilla BGE-M3, document routing is
already solved.  Restricting search to the ~143 pages of the target document
converts the task from global search to within-doc page ranking.
"""

import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text chunking utility (for generation context, not for indexing)
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """Split text into overlapping character chunks."""
    if not text or not text.strip():
        return []

    chunks: List[str] = []
    start = 0
    text = text.strip()

    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunk = text[start:]
        else:
            ws = text.rfind(" ", start, end)
            end = ws if ws > start else end
            chunk = text[start:end]

        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap
        if start <= 0 or start >= len(text):
            break

    return chunks


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    ranked_lists: List[List[str]],
    k: int = 60,
) -> List[str]:
    """Merge ranked lists via Reciprocal Rank Fusion (Cormack 2009)."""
    scores: Dict[str, float] = defaultdict(float)
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores, key=lambda x: -scores[x])


# ---------------------------------------------------------------------------
# HyDE generator
# ---------------------------------------------------------------------------

class HyDEGenerator:
    """
    Generates hypothetical financial passages using a Qwen model.
    Each passage is embedded and fused with the raw query embedding via RRF.
    """

    def __init__(self, config):
        hc = config.hyde
        self.enabled = hc.enabled
        self.num_hypotheticals = hc.num_hypotheticals
        self.max_new_tokens = hc.max_new_tokens
        self.temperature = hc.temperature
        self.do_sample = hc.do_sample
        self.prompt_template = hc.prompt_template
        self.model = None
        self.tokenizer = None
        # Cache: question -> list of hypothetical strings.
        # Populated by pre_generate_all(); generate() returns from cache if present
        # so Qwen does not need to be in GPU memory during retrieval.
        self._cache: Dict[str, List[str]] = {}

        if self.enabled:
            self._load_model(hc)

    def _load_model(self, hc) -> None:
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            import torch
        except ImportError:
            raise ImportError("transformers + bitsandbytes required for HyDE")

        logger.info(f"Loading HyDE model: {hc.qwen_model_name}")

        bnb_config = None
        if hc.load_in_4bit:
            try:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            except Exception:
                bnb_config = None

        self.tokenizer = AutoTokenizer.from_pretrained(
            hc.qwen_model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            hc.qwen_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        logger.info("HyDE model loaded.")

    def pre_generate_all(self, questions: List[str]) -> None:
        """
        Pre-generate hypotheticals for all questions, then free GPU memory.

        Call this BEFORE loading embedding models / reranker for eval so that
        Qwen is the only model in VRAM during generation (~4–5 GB), rather than
        sharing the GPU with two BGE-M3 models and the reranker (~14 GB total).
        After this returns, generate() serves cached results without any GPU.
        """
        from tqdm import tqdm

        if not self.enabled or self.model is None:
            logger.warning("HyDE pre-generation skipped: model not loaded.")
            return

        missing = [q for q in questions if q not in self._cache]
        logger.info(
            f"Pre-generating hypotheticals for {len(missing)} questions "
            f"({len(questions) - len(missing)} already cached)…"
        )

        for q in tqdm(missing, desc="HyDE pre-generation"):
            self._cache[q] = self._generate_impl(q)

        self.free_memory()
        logger.info("HyDE pre-generation complete — Qwen freed from GPU.")

    def generate(self, question: str) -> List[str]:
        """
        Return hypotheticals for question.

        If pre_generate_all() was called earlier the result comes from the
        in-memory cache (no GPU required).  Falls back to live generation if
        the model is still loaded and the question is not cached.
        """
        if question in self._cache:
            return self._cache[question]

        if not self.enabled or self.model is None:
            return []

        result = self._generate_impl(question)
        self._cache[question] = result
        return result

    def _generate_impl(self, question: str) -> List[str]:
        """Run Qwen inference to generate hypothetical passages."""
        import torch

        prompt = self.prompt_template.format(question=question)
        hypotheticals: List[str] = []

        try:
            if hasattr(self.tokenizer, "apply_chat_template"):
                messages = [{"role": "user", "content": prompt}]
                formatted = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
            else:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature if self.do_sample else 1.0,
                    do_sample=self.do_sample,
                    num_return_sequences=self.num_hypotheticals,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            prompt_len = inputs["input_ids"].shape[-1]
            for out in outputs:
                generated = self.tokenizer.decode(
                    out[prompt_len:], skip_special_tokens=True
                ).strip()
                if generated:
                    hypotheticals.append(generated)

        except Exception as e:
            logger.warning(f"HyDE generation failed: {e}. Using raw query only.")

        return hypotheticals

    def free_memory(self) -> None:
        if self.model is not None:
            import torch
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("HyDE model freed from GPU memory.")


# ---------------------------------------------------------------------------
# Cross-encoder reranker
# ---------------------------------------------------------------------------

class CrossEncoderReranker:
    """Reranks (query, page_text) pairs using a cross-encoder."""

    def __init__(self, model_name: str):
        logger.info(f"Loading cross-encoder reranker: {model_name}")
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name, max_length=512)
        except Exception as e:
            logger.warning(f"Could not load reranker ({e}). Reranking skipped.")
            self.model = None

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 5,
    ) -> List[Dict]:
        if self.model is None or not candidates:
            return candidates[:top_k]

        pairs = [(query, c["text"][:1024]) for c in candidates]
        try:
            scores = self.model.predict(pairs)
            ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
            return [c for _, c in ranked[:top_k]]
        except Exception as e:
            logger.warning(f"Reranking failed: {e}. Returning unranked top-k.")
            return candidates[:top_k]


# ---------------------------------------------------------------------------
# Main retrieval pipeline
# ---------------------------------------------------------------------------

class FinancialRetrievalPipeline:
    """
    End-to-end retrieval pipeline for financial Q&A.

    Supports six operating modes (ablation variants):
      1. global          : raw query, BGE-M3 page index, global search (baseline)
      2. docfilter       : raw query, page index, doc-filtered search
      3. docfilter_hyde  : Multi-HyDE + doc-filtered page index
      4. docfilter_rerank: doc-filtered + cross-encoder reranking
      5. docfilter_hier  : hierarchical chunk search + doc-filtered + reranking
      6. docfilter_hyde_rerank (FULL): all components enabled

    Args:
        config          : ExperimentConfig
        embed_model     : SentenceTransformer for encoding queries/passages
        page_collection : ChromaDB page-level collection
        chunk_collection: ChromaDB chunk-level collection (for hierarchical mode)
        hyde_generator  : HyDEGenerator (or None to skip HyDE)
        reranker        : CrossEncoderReranker (or None to skip reranking)
        use_doc_filter  : restrict ChromaDB queries to the target document
        use_hierarchical: use chunk-level collection for retrieval
    """

    def __init__(
        self,
        config,
        embed_model,
        page_collection,
        chunk_collection=None,
        hyde_generator: Optional[HyDEGenerator] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        use_doc_filter: bool = True,
        use_hierarchical: bool = False,
    ):
        self.config = config
        self.embed_model = embed_model
        self.page_collection = page_collection
        self.chunk_collection = chunk_collection
        self.hyde = hyde_generator
        self.reranker = reranker
        self.use_doc_filter = use_doc_filter
        self.use_hierarchical = use_hierarchical

        rc = config.retrieval
        self.candidate_pages = rc.candidate_pages
        self.final_k = rc.final_k
        self.chunk_size = rc.chunk_size
        self.chunk_overlap = rc.chunk_overlap
        self.rrf_k = rc.rrf_k
        self.hier_n_chunks = rc.hier_n_chunks

    def _embed(self, texts: List[str]) -> np.ndarray:
        return self.embed_model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )

    def _query_collection(
        self,
        collection,
        embedding: np.ndarray,
        n_results: int,
        doc_name: Optional[str] = None,
    ) -> List[Dict]:
        """
        Query a ChromaDB collection.

        When doc_name is provided and use_doc_filter=True, the query is
        restricted to pages/chunks from that document via a `where` clause.
        This is the key efficiency and accuracy improvement over global search.
        """
        where = None
        if self.use_doc_filter and doc_name:
            where = {"doc_name": doc_name}

        count = collection.count()
        # When doc-filtering, we can't know the per-doc count without querying,
        # so we cap n_results at total collection count.
        n_results = min(n_results, count)

        try:
            kwargs = dict(
                query_embeddings=[embedding.tolist()],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
            if where is not None:
                kwargs["where"] = where

            results = collection.query(**kwargs)
        except Exception as e:
            # Fallback: if filtering returns too few results, retry without filter
            logger.warning(f"Filtered query failed ({e}), retrying without filter.")
            results = collection.query(
                query_embeddings=[embedding.tolist()],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )

        candidates = []
        for doc_text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            candidates.append({
                "text": doc_text,
                "metadata": {
                    "doc_name": meta.get("doc_name", ""),
                    "page": meta.get("page", -1),
                    "chunk_idx": meta.get("chunk_idx", 0),
                    "pdf_path": meta.get("pdf_path", ""),
                    "source": meta.get("doc_name", ""),
                },
                "_score": float(1.0 - dist),
                "_id": f"{meta.get('doc_name','')}_{meta.get('page','')}",
            })
        return candidates

    # -----------------------------------------------------------------------
    # Standard page-level retrieval
    # -----------------------------------------------------------------------

    def retrieve_pages(
        self,
        question: str,
        doc_name: Optional[str] = None,
        enable_hyde: bool = True,
        enable_rerank: bool = True,
        k: int = None,
    ) -> List[Dict]:
        """
        Retrieve top-k pages for a question.

        Args:
            question    : Natural language question.
            doc_name    : Target document (used for doc-filtering when enabled).
            enable_hyde : If True, generate and fuse hypothetical passages.
            enable_rerank: If True, rerank candidates with the cross-encoder.
            k           : Number of pages to return.
        """
        k = k or self.final_k

        # Step 1: Gather query texts
        query_texts = [question]
        if enable_hyde and self.hyde is not None:
            hyps = self.hyde.generate(question)
            query_texts.extend(hyps)
            logger.debug(f"HyDE generated {len(hyps)} hypotheticals")

        # Step 2: Embed all query texts
        embeddings = self._embed(query_texts)

        # Step 3: Search page collection per query embedding
        n_fetch = min(self.candidate_pages * 2, self.page_collection.count())
        all_result_lists = []
        id_to_candidate: Dict[str, Dict] = {}

        for emb in embeddings:
            results = self._query_collection(
                self.page_collection, emb, n_fetch, doc_name=doc_name
            )
            ranked_ids = []
            for r in results:
                rid = r["_id"]
                ranked_ids.append(rid)
                if rid not in id_to_candidate:
                    id_to_candidate[rid] = r
            all_result_lists.append(ranked_ids)

        # Step 4: RRF fusion
        if len(all_result_lists) > 1:
            fused_ids = reciprocal_rank_fusion(all_result_lists, k=self.rrf_k)
        else:
            fused_ids = all_result_lists[0] if all_result_lists else []

        top_ids = fused_ids[: self.candidate_pages]
        candidates = [id_to_candidate[cid] for cid in top_ids if cid in id_to_candidate]

        # Step 5: Cross-encoder reranking
        if enable_rerank and self.reranker is not None:
            return self.reranker.rerank(question, candidates, top_k=k)
        else:
            return candidates[:k]

    # -----------------------------------------------------------------------
    # Hierarchical chunk-level retrieval
    # -----------------------------------------------------------------------

    def retrieve_pages_hierarchical(
        self,
        question: str,
        doc_name: Optional[str] = None,
        enable_hyde: bool = True,
        enable_rerank: bool = True,
        k: int = None,
    ) -> List[Dict]:
        """
        Retrieve top-k pages via a chunk-level index.

        1. Dense retrieval in the chunk-level index (with doc-filter).
        2. Aggregate chunks → unique parent pages, ranked by best chunk score.
        3. Cross-encoder reranking of page candidates.

        This is more precise than page-level search because 400-char chunks
        embed closer to specific tables/paragraphs than 2000-char pages.
        """
        if self.chunk_collection is None:
            logger.warning("Hierarchical retrieval requested but no chunk_collection provided. "
                           "Falling back to page-level retrieval.")
            return self.retrieve_pages(question, doc_name, enable_hyde, enable_rerank, k)

        k = k or self.final_k

        # Step 1: Gather query texts
        query_texts = [question]
        if enable_hyde and self.hyde is not None:
            hyps = self.hyde.generate(question)
            query_texts.extend(hyps)

        # Step 2: Embed
        embeddings = self._embed(query_texts)

        # Step 3: Search chunk collection per query
        n_fetch_chunks = min(self.hier_n_chunks * len(query_texts),
                             self.chunk_collection.count())
        all_chunk_lists = []
        chunk_id_to_chunk: Dict[str, Dict] = {}

        for emb in embeddings:
            results = self._query_collection(
                self.chunk_collection, emb, self.hier_n_chunks, doc_name=doc_name
            )
            ranked_ids = []
            for r in results:
                # Unique chunk ID includes chunk_idx
                cid = (f"{r['metadata']['doc_name']}_"
                       f"page_{r['metadata']['page']}_"
                       f"chunk_{r['metadata'].get('chunk_idx', 0)}")
                ranked_ids.append(cid)
                if cid not in chunk_id_to_chunk:
                    chunk_id_to_chunk[cid] = r
            all_chunk_lists.append(ranked_ids)

        # Step 4: RRF over chunks
        if len(all_chunk_lists) > 1:
            fused_chunk_ids = reciprocal_rank_fusion(all_chunk_lists, k=self.rrf_k)
        else:
            fused_chunk_ids = all_chunk_lists[0] if all_chunk_lists else []

        # Step 5: Aggregate chunks → pages (keep best score per page)
        # Page ID = "{doc_name}_page_{page_num}"
        page_best_score: Dict[str, float] = {}
        page_best_chunk: Dict[str, Dict] = {}

        for rank, chunk_id in enumerate(fused_chunk_ids):
            chunk = chunk_id_to_chunk.get(chunk_id)
            if chunk is None:
                continue
            page_id = (f"{chunk['metadata']['doc_name']}_"
                       f"page_{chunk['metadata']['page']}")
            # RRF score: higher rank = higher score (1/(k+rank))
            rrf_score = 1.0 / (self.rrf_k + rank + 1)
            if page_id not in page_best_score or rrf_score > page_best_score[page_id]:
                page_best_score[page_id] = rrf_score
                page_best_chunk[page_id] = chunk

        # Sort pages by their best-chunk score
        sorted_page_ids = sorted(page_best_score, key=lambda p: -page_best_score[p])

        # Build page candidates (use the full page text from the page-level index
        # so the reranker has complete context, not just one chunk)
        page_candidates = []
        seen_pages = set()
        for page_id in sorted_page_ids[: self.candidate_pages]:
            chunk = page_best_chunk[page_id]
            page_key = (chunk["metadata"]["doc_name"], chunk["metadata"]["page"])
            if page_key in seen_pages:
                continue
            seen_pages.add(page_key)

            # Try to get full page text from the page-level index
            full_text = self._fetch_page_text(
                chunk["metadata"]["doc_name"], chunk["metadata"]["page"]
            )
            page_candidates.append({
                "text": full_text or chunk["text"],
                "metadata": {
                    "doc_name": chunk["metadata"]["doc_name"],
                    "page": chunk["metadata"]["page"],
                    "pdf_path": chunk["metadata"].get("pdf_path", ""),
                    "source": chunk["metadata"]["doc_name"],
                },
                "_score": page_best_score[page_id],
                "_id": page_id,
            })

        # Step 6: Cross-encoder reranking
        if enable_rerank and self.reranker is not None:
            return self.reranker.rerank(question, page_candidates, top_k=k)
        else:
            return page_candidates[:k]

    def _fetch_page_text(self, doc_name: str, page_num: int) -> Optional[str]:
        """Retrieve full page text from the page-level collection."""
        try:
            result = self.page_collection.get(
                ids=[f"{doc_name}_page_{page_num}"],
                include=["documents"],
            )
            if result["documents"]:
                return result["documents"][0]
        except Exception:
            pass
        return None

    # -----------------------------------------------------------------------
    # Public interface: retrieve_chunks (for generation)
    # -----------------------------------------------------------------------

    def retrieve_chunks(
        self,
        question: str,
        doc_name: Optional[str] = None,
        enable_hyde: bool = True,
        enable_rerank: bool = True,
        k: int = None,
    ) -> List[Dict]:
        """
        Retrieve top-k pages and split them into chunks for the LLM context.

        Returns chunks with page metadata compatible with RetrievalEvaluator.
        """
        k = k or self.final_k

        if self.use_hierarchical and self.chunk_collection is not None:
            pages = self.retrieve_pages_hierarchical(
                question, doc_name, enable_hyde, enable_rerank, k
            )
        else:
            pages = self.retrieve_pages(
                question, doc_name, enable_hyde, enable_rerank, k
            )

        chunks: List[Dict] = []
        for page in pages:
            for ct in chunk_text(page["text"], self.chunk_size, self.chunk_overlap):
                chunks.append({
                    "text": ct,
                    "metadata": {
                        "doc_name": page["metadata"]["doc_name"],
                        "page": page["metadata"]["page"],
                        "source": page["metadata"]["doc_name"],
                    },
                })

        return chunks


# ---------------------------------------------------------------------------
# Generator (kept for completeness)
# ---------------------------------------------------------------------------

class QwenGenerator:
    """Generates answers to financial questions given retrieved context."""

    _PROMPT = (
        "You are a financial analyst answering questions based on SEC filings. "
        "Use ONLY the provided context. If the context does not contain the answer, "
        "say 'I cannot determine this from the provided information.' "
        "Be concise and precise, especially for numerical answers.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )

    def __init__(self, model_name: str, load_in_4bit: bool = True):
        logger.info(f"Loading generator model: {model_name}")
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            import torch

            bnb = None
            if load_in_4bit:
                try:
                    bnb = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                except Exception:
                    bnb = None

            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=bnb, device_map="auto",
                trust_remote_code=True,
            )
            self.model.eval()
            self.model_name = model_name
        except Exception as e:
            logger.error(f"Generator failed to load: {e}")
            self.model = None
            self.tokenizer = None
            self.model_name = model_name

    def generate(self, question: str, context_chunks: List[Dict]) -> str:
        if self.model is None:
            return ""

        import torch

        context_text = "\n\n".join(c["text"] for c in context_chunks[:5])[:3000]
        prompt = self._PROMPT.format(context=context_text, question=question)

        try:
            if hasattr(self.tokenizer, "apply_chat_template"):
                msgs = [{"role": "user", "content": prompt}]
                formatted = self.tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
                inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
            else:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            prompt_len = inputs["input_ids"].shape[-1]
            return self.tokenizer.decode(
                out[0][prompt_len:], skip_special_tokens=True
            ).strip()

        except Exception as e:
            logger.warning(f"Generation failed: {e}")
            return ""

    def free_memory(self) -> None:
        if self.model is not None:
            import torch
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
