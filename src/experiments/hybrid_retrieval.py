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

# Helpers for PDF loading
try:
    from langchain_community.document_loaders import PyMuPDFLoader
except Exception:
    try:
        from langchain.document_loaders import PyMuPDFLoader
    except Exception:
        PyMuPDFLoader = None

def _get_docs(retriever, query: str):
    """
    Safely invoke a retriever, handling both new (invoke) and old (get_relevant_documents) LangChain APIs.
    """
    if hasattr(retriever, "invoke"):
        return retriever.invoke(query)
    elif hasattr(retriever, "get_relevant_documents"):
        return retriever.get_relevant_documents(query)
    else:
        # If it's a VectorStore trying to act as a retriever, fallback to similarity_search
        if hasattr(retriever, "similarity_search"):
            return retriever.similarity_search(query)
        raise AttributeError(f"Provided object {type(retriever)} is not a valid retriever.")

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
    for p in tqdm(pdf_files, desc="Loading PDFs for Sparse"):
        try:
            raw_docs.extend(PyMuPDFLoader(str(p)).load())
        except Exception as e:
            logger.warning(f"Failed to load {p}: {e}")
            
    chunks = experiment.text_splitter.split_documents(raw_docs)
    for c in chunks: _ensure_chunk_key_in_metadata(c)
    
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f: pickle.dump(chunks, f)
    return chunks

def _perform_rrf(dense_docs, sparse_docs, rrf_k, top_k, dense_weight=1.0, sparse_weight=1.0):
    scores = {}
    doc_map = {}
    
    def _process(docs, weight):
        if not docs: return
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

    candidate_k = int(getattr(experiment, "hybrid_candidate_k", 50))

    # 1. Initialize Dense
    # FIX: Handle return type of build_chroma_store safely
    logger.info("Initializing Dense Retriever...")
    store_result = build_chroma_store(experiment, docs="all", lazy_load=False)
    
    dense_retriever = None
    if isinstance(store_result, tuple):
        # Likely (vectorstore, retriever) or (retriever, vectorstore)
        # We assume standard LangChain convention often used in these repos: (store, retriever)
        # But we verify capabilities.
        obj1, obj2 = store_result
        if hasattr(obj2, "invoke") or hasattr(obj2, "get_relevant_documents"):
            dense_retriever = obj2
        elif hasattr(obj1, "invoke") or hasattr(obj1, "get_relevant_documents"):
            dense_retriever = obj1
        else:
            # Neither is a retriever? Try converting the one that looks like a store.
            if hasattr(obj1, "as_retriever"):
                dense_retriever = obj1.as_retriever()
            elif hasattr(obj2, "as_retriever"):
                dense_retriever = obj2.as_retriever()
    else:
        # Single return value
        if hasattr(store_result, "as_retriever"):
            dense_retriever = store_result.as_retriever()
        else:
            dense_retriever = store_result

    if dense_retriever is None:
        raise ValueError("Could not initialize a valid Dense Retriever from build_chroma_store.")

    # Set K for dense retriever
    if hasattr(dense_retriever, "search_kwargs"):
        dense_retriever.search_kwargs["k"] = candidate_k
    else:
        dense_retriever.k = candidate_k

    # 2. Initialize Sparse
    logger.info("Initializing Sparse (BM25) Retriever...")
    fingerprint = _compute_corpus_fingerprint(Path(experiment.pdf_local_dir))
    chunks = _load_or_build_chunks(experiment, fingerprint)
    if not chunks:
        logger.warning("No chunks found for BM25! Hybrid search will rely solely on Dense.")
    
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
        
        if not fused:
            logger.warning(f"No documents retrieved for Q: {question[:30]}...")
            context = ""
        else:
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