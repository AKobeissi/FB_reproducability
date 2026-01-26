"""
Hybrid Retrieval Runner (Dense + Sparse).

Supports:
  - Dense: Any embedding model configured in RAGExperiment (e.g., BGE-M3, MPNet).
  - Sparse: BM25 or SPLADE (configurable).
  - Fusion: Weighted Reciprocal Rank Fusion (wRRF).
"""

import hashlib
import logging
import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from tqdm import tqdm

# --- Core Dependencies ---
from src.retrieval.vectorstore import build_chroma_store
from src.core.rag_dependencies import BM25Retriever

# --- Sparse (BM25) helpers ---
from src.retrieval.bm25 import (
    _compute_corpus_fingerprint,
    _get_chunk_cache_path,
    finance_preprocess_func,
)

# Try importing SPLADE components
try:
    from src.experiments.splade import (
        SpladeEncoder,
        _index_cache_path,
        _build_splade_index,
        _score_query,
        SpladeIndex,
    )
    SPLADE_AVAILABLE = True
except ImportError:
    SPLADE_AVAILABLE = False

logger = logging.getLogger(__name__)

# PDF loader
try:
    from langchain_community.document_loaders import PyMuPDFLoader
except ImportError:
    try:
        from langchain.document_loaders import PyMuPDFLoader
    except ImportError:
        PyMuPDFLoader = None


# ---------------------------------------------------------------------------
# 1) Robust alignment keying
# ---------------------------------------------------------------------------

def _stable_sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

_WS_RE = re.compile(r"\s+")

def _normalize_text_for_hash(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\u00a0", " ")
    text = _WS_RE.sub(" ", text)
    return text.strip().lower()

def _extract_text_and_meta(item: Union[Any, Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    if isinstance(item, dict):
        content = item.get("page_content") or item.get("text") or ""
        meta = item.get("metadata") or {}
        return content, meta
    content = getattr(item, "page_content", "") or getattr(item, "text", "") or ""
    meta = getattr(item, "metadata", {}) or {}
    return content, meta

def _make_alignment_key(item: Union[Any, Dict[str, Any], str]) -> str:
    if isinstance(item, str):
        text = item
        meta = {}
    else:
        text, meta = _extract_text_and_meta(item)

    norm = _normalize_text_for_hash(text)
    h = _stable_sha1(norm)

    src = meta.get("source") or meta.get("doc_name") or meta.get("file_name")
    page = meta.get("page") or meta.get("page_number") or meta.get("page_index")
    if src is not None and page is not None:
        return f"sp:{str(src)}::p{str(page)}::h:{h}"

    return f"h:{h}"


# ---------------------------------------------------------------------------
# 2) Helpers (Chunk Loading & Retrieval)
# ---------------------------------------------------------------------------

def _load_chunks_from_pdfs(experiment) -> List[Any]:
    """Load and split PDFs from the experiment directory."""
    if PyMuPDFLoader is None:
        raise ImportError("PyMuPDFLoader required for PDF loading.")

    pdf_dir = Path(experiment.pdf_local_dir)
    logger.info(f"[Hybrid] Loading PDFs from {pdf_dir}...")
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning(f"[Hybrid] No PDFs found in {pdf_dir}")
        return []

    raw_docs = []
    for p in tqdm(pdf_files, desc="Loading PDFs"):
        try:
            docs = PyMuPDFLoader(str(p)).load()
            for d in docs:
                d.metadata.setdefault("doc_name", p.stem)
                d.metadata.setdefault("source", p.stem)
            raw_docs.extend(docs)
        except Exception as e:
            logger.warning(f"[Hybrid] Failed to load {p}: {e}")

    logger.info(f"[Hybrid] Splitting {len(raw_docs)} documents...")
    chunks = experiment.text_splitter.split_documents(raw_docs)
    logger.info(f"[Hybrid] Generated {len(chunks)} chunks.")
    return chunks

def _get_docs(retriever: Any, query: str) -> List[Any]:
    if retriever is None:
        return []
    try:
        if hasattr(retriever, "invoke"):
            return retriever.invoke(query)
        if hasattr(retriever, "get_relevant_documents"):
            return retriever.get_relevant_documents(query)
        if hasattr(retriever, "similarity_search"):
            return retriever.similarity_search(query)
    except Exception as e:
        logger.error(f"[Hybrid] Error during retrieval: {e}")
        return []
    
    raise AttributeError(f"Provided object {type(retriever)} is not a valid retriever.")


# ---------------------------------------------------------------------------
# 3) BM25 Cache & Builder
# ---------------------------------------------------------------------------

def _bm25_cache_path(experiment, fingerprint: str) -> str:
    base = _get_chunk_cache_path(experiment, fingerprint)
    cs = getattr(experiment, "chunk_size", None)
    ov = getattr(experiment, "chunk_overlap", None)
    unit = getattr(experiment, "chunking_unit", "chars") 
    
    if cs is None and ov is None:
        return base
        
    stem, ext = os.path.splitext(base)
    # Append unit to ensure uniqueness
    return f"{stem}_cs{cs}_ov{ov}_{unit}{ext or '.pkl'}"

def _load_or_build_bm25_chunks(experiment, fingerprint: str):
    cache_path = _bm25_cache_path(experiment, fingerprint)

    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                logger.info(f"[Hybrid] Loading BM25 chunks from {cache_path}")
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"[Hybrid] Failed to load BM25 chunk cache: {e}")

    # Fallback: Load fresh
    chunks = _load_chunks_from_pdfs(experiment)
    
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(chunks, f)

    return chunks


# ---------------------------------------------------------------------------
# 4) SPLADE Wrapper
# ---------------------------------------------------------------------------

class HybridSpladeRetriever:
    def __init__(self, index: "SpladeIndex", encoder: "SpladeEncoder", top_k: int = 50):
        self.index = index
        self.encoder = encoder
        self.top_k = top_k

    def get_relevant_documents(self, query: str) -> List[Dict[str, Any]]:
        q_terms, q_weights = self.encoder.encode_topk(
            [query], self.index.top_n_terms, batch_size=1
        )[0]
        scores, ids = _score_query(
            q_terms, q_weights, self.index, self.top_k
        )
        out = []
        for s, idx in zip(scores, ids):
            out.append({
                "page_content": self.index.texts[idx],
                "metadata": self.index.metadatas[idx],
                "score": float(s),
            })
        return out


# ---------------------------------------------------------------------------
# 5) Dense Retriever Init (Robust to Missing Store)
# ---------------------------------------------------------------------------

def _get_dense_retriever(experiment, k: int):
    logger.info(f"[Hybrid] Initializing Dense Retriever (Embeddings: {experiment.embedding_model})...")
    
    retriever = None
    vectordb = None
    should_build = False

    # 1. Attempt Lazy Load (Probe)
    try:
        # We try to load. If the DB doesn't exist, Chroma might create an empty shell.
        # We must verify if it actually contains data.
        retriever, vectordb, is_empty_flag = build_chroma_store(experiment, docs="all", lazy_load=True)
        
        if is_empty_flag:
            logger.info("[Hybrid] Store flagged as empty by builder.")
            should_build = True
        elif vectordb is not None:
            # SANITY CHECK: Access the collection to ensure tables exist
            # This catches 'no such table: collections'
            count = vectordb._collection.count()
            if count == 0:
                logger.info("[Hybrid] Vector store exists but has 0 documents. Triggering rebuild.")
                should_build = True
            else:
                logger.info(f"[Hybrid] Successfully loaded existing VectorStore with {count} docs.")
                
    except Exception as e:
        logger.warning(f"[Hybrid] Failed to probe VectorStore (will rebuild): {e}")
        should_build = True

    # 2. Build from Scratch if needed
    if should_build:
        logger.info("[Hybrid] Building VectorStore from scratch (this may take time)...")
        chunks = _load_chunks_from_pdfs(experiment)
        
        # Call again with actual documents to populate
        # Note: lazy_load=False ensures it persists
        store_result = build_chroma_store(experiment, docs="all", documents=chunks, lazy_load=False)
        
        if isinstance(store_result, tuple):
             retriever = store_result[0]
        else:
             retriever = store_result
    
    # 3. Configure k
    if retriever:
        if hasattr(retriever, "search_kwargs"):
            retriever.search_kwargs["k"] = k
        elif hasattr(retriever, "k"):
            retriever.k = k
        
    return retriever


# ---------------------------------------------------------------------------
# 6) Weighted RRF
# ---------------------------------------------------------------------------

def _perform_rrf(
    dense_results: List[Any],
    sparse_results: List[Any],
    rrf_k: int = 60,
    top_k: int = 5,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> List[Dict[str, Any]]:
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict[str, Any]] = {}

    def _accumulate(items: List[Any], weight: float):
        if not items or weight <= 0:
            return
        seen = set()
        for rank, item in enumerate(items):
            key = _make_alignment_key(item)
            if key in seen:
                continue
            seen.add(key)

            text, meta = _extract_text_and_meta(item)
            if key not in doc_map:
                doc_map[key] = {"text": text, "metadata": meta}
                scores[key] = 0.0

            scores[key] += weight * (1.0 / (rrf_k + rank + 1))

    _accumulate(dense_results, dense_weight)
    _accumulate(sparse_results, sparse_weight)

    if not scores:
        return []

    best_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)[:top_k]
    return [{"text": doc_map[k]["text"], "metadata": doc_map[k]["metadata"], "score": scores[k]} for k in best_keys]


# ---------------------------------------------------------------------------
# 7) Runners
# ---------------------------------------------------------------------------

def run_hybrid_search(
    experiment,
    data: List[Dict[str, Any]],
    *,
    dense_weight: Optional[float] = None,
    sparse_weight: Optional[float] = None,
    rrf_k: Optional[int] = None,
    retrieval_only: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING HYBRID EXPERIMENT (wRRF)")
    logger.info("=" * 80)

    candidate_k = int(getattr(experiment, "hybrid_candidate_k", 50))
    sparse_model = str(getattr(experiment, "hybrid_sparse_model", "bm25")).lower()

    alpha = getattr(experiment, "hybrid_alpha", None)
    if dense_weight is None:
        dense_weight = float(alpha) if alpha is not None else float(getattr(experiment, "hybrid_dense_weight", 1.0))
    if sparse_weight is None:
        sparse_weight = float(1.0 - dense_weight) if alpha is not None else float(getattr(experiment, "hybrid_sparse_weight", 1.0))
    if rrf_k is None:
        rrf_k = int(getattr(experiment, "hybrid_rrf_k", 60))
    if retrieval_only is None:
        retrieval_only = bool(getattr(experiment, "retrieval_only", False))

    logger.info(f"Config: candidate_k={candidate_k}, sparse_model={sparse_model}, top_k={experiment.top_k}")
    logger.info(f"Fusion: rrf_k={rrf_k}, dense_weight={dense_weight:.3f}, sparse_weight={sparse_weight:.3f}")

    # Initialize Dense (with robust auto-build)
    dense_retriever = _get_dense_retriever(experiment, k=candidate_k)

    # Initialize Sparse
    sparse_retriever = None
    if sparse_model == "splade":
        if not SPLADE_AVAILABLE:
            raise ImportError("SPLADE dependencies not found. Check src/experiments/splade.py")
        logger.info("[Hybrid] Initializing SPLADE...")
        encoder = SpladeEncoder(device=experiment.device)
        fingerprint = _compute_corpus_fingerprint(Path(experiment.pdf_local_dir))
        top_n = int(getattr(experiment, "splade_top_n_terms", 256))
        cache_path = _index_cache_path(experiment, fingerprint, top_n)

        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                index = pickle.load(f)
        else:
            index = _build_splade_index(experiment, encoder, Path(experiment.pdf_local_dir), fingerprint, top_n)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(index, f)

        sparse_retriever = HybridSpladeRetriever(index, encoder, top_k=candidate_k)
    else:
        logger.info("[Hybrid] Initializing BM25...")
        fingerprint = _compute_corpus_fingerprint(Path(experiment.pdf_local_dir))
        chunks = _load_or_build_bm25_chunks(experiment, fingerprint)
        if chunks:
            sparse_retriever = BM25Retriever.from_documents(chunks, preprocess_func=finance_preprocess_func)
            sparse_retriever.k = candidate_k
        else:
            logger.warning("[Hybrid] No chunks for BM25; running Dense-only.")

    results = experiment._create_skipped_results(data, "hybrid", "hybrid", "pdf", "hybrid", start_id=0)

    for i, sample in enumerate(tqdm(data, desc="Hybrid Inference")):
        q = sample.get("question")
        results[i]["doc_name"] = sample.get("doc_name")
        results[i]["doc_link"] = sample.get("doc_link")
        if not q:
            continue

        dense_docs = _get_docs(dense_retriever, q) or []
        sparse_docs = _get_docs(sparse_retriever, q) or []

        fused = _perform_rrf(
            dense_docs, sparse_docs,
            rrf_k=rrf_k,
            top_k=experiment.top_k,
            dense_weight=float(dense_weight),
            sparse_weight=float(sparse_weight),
        )

        results[i]["retrieved_chunks"] = fused
        results[i]["num_retrieved"] = len(fused)

        gold_segments, gold_text = experiment._prepare_gold_evidence(sample.get("evidence"))
        results[i]["gold_evidence"] = gold_text
        results[i]["gold_evidence_segments"] = gold_segments

        if not retrieval_only:
            context = "\n\n".join([c["text"] for c in fused]) if fused else ""
            answer, prompt = experiment._generate_answer(q, context, return_prompt=True)
            results[i]["generated_answer"] = answer
            results[i]["final_prompt"] = prompt

        experiment.notify_sample_complete(1)

    return results


def run_hybrid_rrf_sweep(
    experiment,
    data: List[Dict[str, Any]],
    *,
    alphas: Optional[List[float]] = None,
    rrf_ks: Optional[List[int]] = None,
    retrieval_only: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:

    if alphas is None:
        alphas = list(getattr(experiment, "hybrid_sweep_alphas", [0.6, 0.75, 0.85, 0.9, 0.95]))
    if rrf_ks is None:
        rrf_ks = list(getattr(experiment, "hybrid_sweep_rrf_ks", [10, 30, 60]))

    candidate_k = int(getattr(experiment, "hybrid_candidate_k", 50))
    sparse_model = str(getattr(experiment, "hybrid_sparse_model", "bm25")).lower()

    logger.info("\n" + "=" * 80)
    logger.info("RUNNING HYBRID SWEEP (wRRF)")
    logger.info("=" * 80)
    logger.info(f"Sweep alphas={alphas}")
    logger.info(f"Sweep rrf_ks={rrf_ks}")

    # Initialize Dense (with robust auto-build)
    dense_retriever = _get_dense_retriever(experiment, k=candidate_k)

    # Initialize Sparse
    sparse_retriever = None
    if sparse_model == "splade":
        if not SPLADE_AVAILABLE:
            raise ImportError("SPLADE dependencies not found.")
        encoder = SpladeEncoder(device=experiment.device)
        fingerprint = _compute_corpus_fingerprint(Path(experiment.pdf_local_dir))
        top_n = int(getattr(experiment, "splade_top_n_terms", 256))
        cache_path = _index_cache_path(experiment, fingerprint, top_n)
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                index = pickle.load(f)
        else:
            index = _build_splade_index(experiment, encoder, Path(experiment.pdf_local_dir), fingerprint, top_n)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(index, f)
        sparse_retriever = HybridSpladeRetriever(index, encoder, top_k=candidate_k)
    else:
        fingerprint = _compute_corpus_fingerprint(Path(experiment.pdf_local_dir))
        chunks = _load_or_build_bm25_chunks(experiment, fingerprint)
        if chunks:
            sparse_retriever = BM25Retriever.from_documents(chunks, preprocess_func=finance_preprocess_func)
            sparse_retriever.k = candidate_k

    out: Dict[str, List[Dict[str, Any]]] = {}
    configs: List[Tuple[str, float, float, int]] = []
    for a in alphas:
        a = float(a)
        a = max(0.0, min(1.0, a))
        for k in rrf_ks:
            cfg_name = f"hybrid_rrf_alpha{a:.2f}_k{k}"
            out[cfg_name] = experiment._create_skipped_results(data, cfg_name, cfg_name, "pdf", "hybrid", start_id=0)
            configs.append((cfg_name, a, 1.0 - a, int(k)))

    for i, sample in enumerate(tqdm(data, desc="Hybrid Sweep Inference")):
        q = sample.get("question")
        if not q:
            continue

        dense_docs = _get_docs(dense_retriever, q) or []
        sparse_docs = _get_docs(sparse_retriever, q) or []
        
        gold_segments, gold_text = experiment._prepare_gold_evidence(sample.get("evidence"))

        for (cfg_name, dw, sw, rk) in configs:
            res = out[cfg_name]
            res[i]["doc_name"] = sample.get("doc_name")
            res[i]["doc_link"] = sample.get("doc_link")
            res[i]["gold_evidence"] = gold_text
            res[i]["gold_evidence_segments"] = gold_segments
            res[i]["hybrid_alpha"] = dw
            res[i]["hybrid_rrf_k"] = rk

            fused = _perform_rrf(
                dense_docs, sparse_docs,
                rrf_k=rk,
                top_k=experiment.top_k,
                dense_weight=dw,
                sparse_weight=sw,
            )
            res[i]["retrieved_chunks"] = fused
            res[i]["num_retrieved"] = len(fused)

            if not retrieval_only:
                context = "\n\n".join([c["text"] for c in fused]) if fused else ""
                answer, prompt = experiment._generate_answer(q, context, return_prompt=True)
                res[i]["generated_answer"] = answer
                res[i]["final_prompt"] = prompt

        experiment.notify_sample_complete(1)

    return out