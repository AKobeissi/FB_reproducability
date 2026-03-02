"""
RAG Experiment with Cross-Encoder Re-ranking.
Literature Standard: Retrieve Top-N (e.g. 20) -> Rerank -> Select Top-K (e.g. 5).
"""
from typing import List, Dict, Any, Tuple
import logging
import torch
import os
import json
from pathlib import Path
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

from src.retrieval.vectorstore import (
    build_chroma_store, 
    populate_chroma_store, 
    save_store_config, 
    get_chroma_db_path
)
from src.ingestion.pdf_utils import load_pdf_with_fallback
from src.experiments.rag_shared_vector import (
    _get_doc_text,
    _create_skipped_result,
    _log_pdf_sources
)

logger = logging.getLogger(__name__)

# --- Configuration ---
RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"
DEFAULT_LATE_INTERACTION_MODEL = "colbert-ir/colbertv2.0"
INITIAL_TOP_N = 20  # Pool size before re-ranking


class LateInteractionReranker:
    """ColBERT-style late interaction reranker using token-level embeddings."""

    def __init__(
        self,
        model_id: str,
        device: str,
        query_max_len: int = 64,
        doc_max_len: int = 256,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.query_max_len = query_max_len
        self.doc_max_len = doc_max_len

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        self.model.to(device)
        self.model.eval()

    def _encode_query(self, query: str) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            max_length=self.query_max_len,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = self.model(**encoded)
        embeddings = outputs.last_hidden_state.squeeze(0)
        mask = encoded["attention_mask"].squeeze(0).bool()
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings, mask

    def _encode_docs(self, docs: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            docs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.doc_max_len,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = self.model(**encoded)
        embeddings = outputs.last_hidden_state
        mask = encoded["attention_mask"].bool()
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings, mask

    def score(self, query: str, docs: List[str], batch_size: int = 8) -> List[float]:
        if not docs:
            return []

        query_embeddings, query_mask = self._encode_query(query)
        scores: List[float] = []

        for start in range(0, len(docs), batch_size):
            batch = docs[start : start + batch_size]
            doc_embeddings, doc_mask = self._encode_docs(batch)

            sim = torch.einsum("qh,bdh->bqd", query_embeddings, doc_embeddings)
            sim = sim.masked_fill(~doc_mask.unsqueeze(1), -1e4)
            max_sim = sim.max(dim=-1).values
            max_sim = max_sim * query_mask.unsqueeze(0)
            batch_scores = max_sim.sum(dim=-1)
            scores.extend(batch_scores.detach().cpu().tolist())

        return scores

def run_reranking(experiment, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Run experiment: Shared Vector Store + Cross-Encoder Reranking.
    """
    logger.info("\n" + "=" * 80)
    reranker_style = getattr(experiment, "reranker_style", "cross_encoder")
    candidate_k = getattr(experiment, "k_cand", None) or INITIAL_TOP_N
    reranker_model_id = getattr(experiment, "reranker_model", RERANKER_MODEL_ID)
    late_interaction_model = getattr(experiment, "late_interaction_model", DEFAULT_LATE_INTERACTION_MODEL)
    reranker_batch_size = getattr(experiment, "reranker_batch_size", 8)

    logger.info(f"RUNNING RERANKING EXPERIMENT")
    logger.info(
        "Style: %s | Initial-K: %s -> Top-K: %s",
        reranker_style,
        candidate_k,
        experiment.top_k,
    )
    if reranker_style == "late_interaction":
        logger.info("Late-interaction model: %s", late_interaction_model)
    else:
        logger.info("Cross-encoder model: %s", reranker_model_id)
    logger.info("=" * 80)

    # --- PART 0: AUTO-DETECT PDF DIRECTORY ---
    # Fix for when running from src.core causing wrong default path
    current_pdf_dir = getattr(experiment, "pdf_local_dir", None)
    
    # If path is None, doesn't exist, or is empty, try to find the real 'pdfs' folder
    if not current_pdf_dir or not os.path.exists(current_pdf_dir) or not os.listdir(current_pdf_dir):
        logger.info(f"Default PDF dir '{current_pdf_dir}' seems invalid. Searching for 'pdfs' folder...")
        
        # Look in likely locations relative to this script
        potential_paths = [
            os.path.join(os.getcwd(), "pdfs"),                            # ./pdfs
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../../pdfs")), # ../../pdfs (Project Root)
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../pdfs")), # ../../../pdfs
        ]
        
        for p in potential_paths:
            if os.path.exists(p) and os.path.isdir(p) and os.listdir(p):
                logger.info(f"✓ Auto-detected PDF directory at: {p}")
                experiment.pdf_local_dir = Path(p)
                break
        else:
            logger.warning("Could not auto-locate 'pdfs' directory. Ingestion may fail.")

    # --- PART 1: ENSURE VECTOR STORE IS POPULATED ---
    unique_docs = {}
    
    # 1. Collect Docs from Data
    for sample in data:
        doc_name = sample.get('doc_name', 'unknown')
        if doc_name not in unique_docs:
            unique_docs[doc_name] = sample.get('doc_link', '')

    # 2. Collect Docs from Local Directory (using the fixed path)
    if getattr(experiment, "use_all_pdfs", False) and getattr(experiment, "pdf_local_dir", None):
        pdf_dir = str(experiment.pdf_local_dir)
        if os.path.exists(pdf_dir):
            import glob
            pdf_files = glob.glob(os.path.join(pdf_dir, "**", "*.pdf"), recursive=True)
            pdf_files += glob.glob(os.path.join(pdf_dir, "**", "*.PDF"), recursive=True)
            for pdf_path in pdf_files:
                filename = os.path.basename(pdf_path)
                doc_name = os.path.splitext(filename)[0]
                unique_docs[doc_name] = ""

    logger.info(f"Collected {len(unique_docs)} unique documents")

    # 3. Initialize Chroma (Lazy Load)
    try:
        base_retriever, vectordb, is_new = build_chroma_store(
            experiment,
            "all",
            lazy_load=True,
        )
    except Exception as e:
        logger.error("Chroma build failed: %s", e)
        return []

    # 4. Check Metadata
    db_name, db_path = get_chroma_db_path(experiment, "all")
    meta_path = os.path.join(db_path, "shared_meta.json")
    available_docs = set()
    pdf_source_map = {}

    if not is_new and os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                available_docs = set(meta.get("available_docs", []))
                pdf_source_map = meta.get("pdf_source_map", {})
        except Exception:
            pass

    # 5. Ingest Missing Documents
    docs_to_process = {k: v for k, v in unique_docs.items() if k not in available_docs}
    
    if docs_to_process:
        logger.info(f"Ingesting {len(docs_to_process)} missing documents...")
        for i, (doc_name, doc_link) in enumerate(docs_to_process.items()):
            
            pdf_docs, pdf_source = load_pdf_with_fallback(
                doc_name=doc_name,
                doc_link=doc_link,
                local_dir=getattr(experiment, 'pdf_local_dir', None),
            )
            pdf_source_map[doc_name] = pdf_source

            if not pdf_docs:
                logger.warning(f"No PDF pages for '{doc_name}'. Skipping.")
                continue

            chunks = experiment._chunk_text_langchain(
                pdf_docs,
                metadata={'doc_name': doc_name, 'source': 'pdf', 'doc_link': doc_link}
            )
            
            if chunks:
                populate_chroma_store(experiment, vectordb, chunks, db_name)
                available_docs.add(doc_name)
        
        # Save updated metadata
        save_store_config(experiment, db_path)
        with open(meta_path, 'w') as f:
            json.dump({"available_docs": list(available_docs), "pdf_source_map": pdf_source_map}, f)

    _log_pdf_sources(pdf_source_map)


    # --- PART 2: INITIALIZE RERANKER ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reranker = None
    if reranker_style == "late_interaction":
        logger.info(f"Loading Late-Interaction Reranker on {device}...")
        try:
            reranker = LateInteractionReranker(
                late_interaction_model,
                device=device,
                query_max_len=getattr(experiment, "late_interaction_query_max_len", 64),
                doc_max_len=getattr(experiment, "late_interaction_doc_max_len", 256),
            )
        except Exception as e:
            logger.error(f"Failed to load late-interaction reranker: {e}")
            raise
    else:
        logger.info(f"Loading Cross-Encoder on {device}...")
        try:
            reranker = CrossEncoder(
                reranker_model_id,
                model_kwargs={"torch_dtype": torch.float16 if device=="cuda" else torch.float32},
                device=device,
                trust_remote_code=True,
            )
        except Exception as e:
            logger.error(f"Failed to load reranker: {e}")
            raise


    # --- PART 3: RUN EXPERIMENT LOOP ---
    base_retriever.search_kwargs["k"] = candidate_k

    results = []
    
    for i, sample in enumerate(data):
        logger.info(f"\n--- Sample {i+1}/{len(data)} ---")
        
        doc_name = sample.get('doc_name', 'unknown')
        question = sample.get('question', '')
        
        # A. Initial Retrieval
        try:
            initial_docs = base_retriever.invoke(question)
        except Exception as e:
            logger.warning(f"Retrieval failed: {e}")
            initial_docs = []

        reranked_chunks = []
        if not initial_docs:
            logger.warning("No documents retrieved. Skipping reranking.")
        else:
            # B. Re-ranking
            pairs = []
            valid_docs = []
            for doc in initial_docs:
                text = _get_doc_text(doc)
                if text.strip():
                    pairs.append([question, text])
                    valid_docs.append(doc)
            
            if pairs:
                if reranker_style == "late_interaction":
                    doc_texts = [p[1] for p in pairs]
                    scores = reranker.score(question, doc_texts, batch_size=reranker_batch_size)
                else:
                    scores = reranker.predict(pairs, batch_size=reranker_batch_size, show_progress_bar=False)

                scored_docs = []
                for doc, score in zip(valid_docs, scores):
                    doc.metadata["rerank_score"] = float(score)
                    scored_docs.append((doc, score))

                scored_docs.sort(key=lambda x: x[1], reverse=True)
                top_k_docs = [d[0] for d in scored_docs[:experiment.top_k]]

                reranked_chunks = _docs_to_chunks(top_k_docs)
            
        # C. Generation
        context_text = "\n\n".join([c['text'] for c in reranked_chunks])
        generated_answer, final_prompt = experiment._generate_answer(
            question, context_text, return_prompt=True
        )

        # D. Build Result
        gold_segments, gold_str = experiment._prepare_gold_evidence(sample.get('evidence', ''))
        
        result = {
            'sample_id': i,
            'doc_name': doc_name,
            'question': question,
            'reference_answer': sample.get('answer', ''),
            'gold_evidence': gold_str,
            'gold_evidence_segments': gold_segments,
            'retrieved_chunks': reranked_chunks,
            'generated_answer': generated_answer,
            'experiment_type': "reranking",
            'final_prompt': final_prompt,
            'context_length': len(context_text),
            'generation_length': len(generated_answer)
        }
        
        results.append(result)
        logger.info(f"Completed sample {i+1}")

    del reranker
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results

def _docs_to_chunks(documents):
    chunks = []
    for rank, doc in enumerate(documents):
        text = _get_doc_text(doc)
        chunks.append({
            'rank': rank + 1,
            'text': text,
            'score': doc.metadata.get("rerank_score"),
            'metadata': doc.metadata
        })
    return chunks