"""
Hybrid Retrieval Runner (BM25 + Dense Vector Store).
"""

import hashlib
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

from src.retrieval.vectorstore import build_chroma_store
from src.retrieval.bm25 import _compute_corpus_fingerprint, _get_chunk_cache_path, finance_preprocess_func
from src.core.rag_dependencies import BM25Retriever

logger = logging.getLogger(__name__)

# Helpers omitted for brevity (same as before), just ensuring imports match
try:
    from langchain_community.document_loaders import PyMuPDFLoader
except Exception:
    try:
        from langchain.document_loaders import PyMuPDFLoader
    except Exception:
        PyMuPDFLoader = None

def _get_docs(retriever, query: str):
    if hasattr(retriever, "invoke"): return retriever.invoke(query)
    return retriever.get_relevant_documents(query)

def _stable_sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _make_stable_doc_key(doc) -> str:
    md = getattr(doc, "metadata", {}) or {}
    text = getattr(doc, "page_content", "") or ""
    for k in ("chunk_id", "id", "doc_id"):
        if k in md and md[k]: return f"{k}:{md[k]}"
    return f"h:{_stable_sha1(text)}"

def _ensure_chunk_key_in_metadata(doc):
    key = _make_stable_doc_key(doc)
    doc.metadata.setdefault("chunk_id", key)
    return key

def _load_or_build_chunks(experiment, fingerprint):
    cache_path = _get_chunk_cache_path(experiment, fingerprint)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f: return pickle.load(f)
        except Exception: pass
    
    # Rebuild
    if PyMuPDFLoader is None: raise ImportError("PyMuPDFLoader required")
    pdf_files = sorted(Path(experiment.pdf_local_dir).glob("*.pdf"))
    raw_docs = []
    for p in tqdm(pdf_files, desc="Loading PDFs"):
        raw_docs.extend(PyMuPDFLoader(str(p)).load())
    chunks = experiment.text_splitter.split_documents(raw_docs)
    for c in chunks: _ensure_chunk_key_in_metadata(c)
    
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f: pickle.dump(chunks, f)
    return chunks

def _perform_rrf(dense_docs, sparse_docs, rrf_k, top_k, dense_weight=1.0, sparse_weight=1.0):
    scores = {}
    doc_map = {}
    
    def _process(docs, weight):
        for rank, doc in enumerate(docs):
            key = _ensure_chunk_key_in_metadata(doc)
            scores.setdefault(key, 0.0)
            doc_map[key] = doc
            scores[key] += weight * (1.0 / (rrf_k + rank + 1))
            
    _process(dense_docs, dense_weight)
    _process(sparse_docs, sparse_weight)
    
    sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)[:top_k]
    return [{"text": doc_map[k].page_content, "metadata": doc_map[k].metadata, "score": scores[k]} for k in sorted_keys]

def run_hybrid_search(experiment, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    logger.info("\n" + "=" * 80)
    logger.info(f"RUNNING HYBRID EXPERIMENT")
    logger.info("=" * 80)

    # 1. Initialize Dense
    dense_retriever, _ = build_chroma_store(experiment, docs="all", lazy_load=False)
    candidate_k = int(getattr(experiment, "hybrid_candidate_k", 50))
    try: dense_retriever.search_kwargs["k"] = candidate_k
    except: dense_retriever.k = candidate_k

    # 2. Initialize Sparse
    fingerprint = _compute_corpus_fingerprint(Path(experiment.pdf_local_dir))
    chunks = _load_or_build_chunks(experiment, fingerprint)
    bm25_retriever = BM25Retriever.from_documents(chunks, preprocess_func=finance_preprocess_func)
    bm25_retriever.k = candidate_k

    # 3. Results Container
    results = experiment._create_skipped_results(
        data, "hybrid_placeholder", "hybrid_placeholder", "pdf", "hybrid", start_id=0
    )

    logger.info(f"Processing {len(data)} samples with Hybrid RRF...")
    
    for i, sample in enumerate(tqdm(data, desc="Hybrid Inference")):
        question = sample.get("question")
        
        results[i]['doc_name'] = sample.get('doc_name')
        results[i]['doc_link'] = sample.get('doc_link')
        
        if not question: continue

        dense_docs = _get_docs(dense_retriever, question)
        sparse_docs = _get_docs(bm25_retriever, question)
        
        fused = _perform_rrf(dense_docs or [], sparse_docs or [], rrf_k=60, top_k=experiment.top_k)
        context = "\n\n".join([c["text"] for c in fused])
        
        answer, prompt = experiment._generate_answer(question, context, return_prompt=True)
        gold_segments, gold_text = experiment._prepare_gold_evidence(sample.get("evidence"))

        results[i]["gold_evidence"] = gold_text
        results[i]["gold_evidence_segments"] = gold_segments
        results[i]["retrieved_chunks"] = fused
        results[i]["num_retrieved"] = len(fused)
        results[i]["generated_answer"] = answer
        results[i]["final_prompt"] = prompt
        
        experiment.notify_sample_complete(1)

    return results