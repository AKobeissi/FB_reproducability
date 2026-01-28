"""
Unified RAG Pipeline.
Allows composable usage of:
  - Query Expansion: HyDE (Single or Multi)
  - Retrieval: Dense, Sparse (BM25/SPLADE), or Hybrid (RRF)
  - Post-Processing: Cross-Encoder Reranking
"""
import logging
import os
import json
import shutil
import numpy as np
import torch
from typing import List, Dict, Any
from tqdm import tqdm
from sentence_transformers import CrossEncoder
from pathlib import Path

# Import helpers
from src.experiments.rag_hyde_shared import _generate_hypotheticals
from src.experiments.hybrid_retrieval import (
    _get_dense_retriever,
    _get_docs,
    _perform_rrf,
    _load_or_build_bm25_chunks,
    finance_preprocess_func
)
from src.retrieval.bm25 import _compute_corpus_fingerprint
from src.core.rag_dependencies import BM25Retriever
from src.retrieval.vectorstore import (
    build_chroma_store, 
    populate_chroma_store, 
    save_store_config, 
    get_chroma_db_path
)
from src.ingestion.pdf_utils import load_pdf_with_fallback

logger = logging.getLogger(__name__)

# Attempt to import logging helper
try:
    from src.experiments.rag_shared_vector import _log_pdf_sources
except ImportError:
    def _log_pdf_sources(mapping):
        logger.info(f"PDF Source Map: {len(mapping)} documents tracked.")

def run_unified_pipeline(experiment, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING UNIFIED PIPELINE")
    logger.info("=" * 80)

    # --- Configuration Extraction ---
    use_hyde = getattr(experiment, "unified_use_hyde", False)
    hyde_k = getattr(experiment, "unified_hyde_k", 1)
    retrieval_mode = getattr(experiment, "unified_retrieval", "dense") # dense, sparse, hybrid
    use_rerank = getattr(experiment, "unified_use_rerank", False)
    
    # Heuristic: If reranking, fetch more candidates
    candidate_k = experiment.top_k * 4 if use_rerank else experiment.top_k
    
    logger.info(f"Pipeline Configuration:")
    logger.info(f"  [1] HyDE Enabled: {use_hyde} (Generations: {hyde_k})")
    logger.info(f"  [2] Retrieval Mode: {retrieval_mode.upper()} (Candidates: {candidate_k})")
    logger.info(f"  [3] Reranking Enabled: {use_rerank}")

    # =========================================================================
    # PART 1: INGESTION & VECTOR STORE SETUP
    # =========================================================================
    
    # A. Identify which documents are needed for this run
    unique_docs = {}
    
    # 1. From Config (All PDFs)
    if getattr(experiment, "use_all_pdfs", False) and getattr(experiment, "pdf_local_dir", None):
        if os.path.exists(str(experiment.pdf_local_dir)):
            import glob
            p = str(experiment.pdf_local_dir)
            # Find all PDFs recursively
            for f in glob.glob(os.path.join(p, "**", "*.pdf"), recursive=True):
                unique_docs[os.path.splitext(os.path.basename(f))[0]] = ""
                
    # 2. From Dataset (Ensure at least these are present)
    for sample in data:
        unique_docs.setdefault(sample.get('doc_name', 'unknown'), sample.get('doc_link', ''))

    vectordb = None
    # We need the dense store if mode is dense/hybrid OR if HyDE is enabled
    need_dense_store = (retrieval_mode in ["dense", "hybrid"]) or use_hyde

    if need_dense_store:
        try:
            # Initialize Store (Lazy Load)
            _, vectordb, is_new = build_chroma_store(experiment, "all", lazy_load=True)
            
            # --- VERIFICATION & SELF-HEALING ---
            if not is_new and vectordb:
                try:
                    # Run a dummy query to check if vectors exist and dimensions match
                    _ = vectordb.similarity_search("sanity_check_query", k=1)
                    logger.info(f"✓ Vector store sanity check passed.")
                except Exception as e:
                    logger.error(f"⚠️ Vector store is corrupt or incompatible (Error: {e}).")
                    logger.info("   -> Deleting and rebuilding from scratch...")
                    
                    _, db_path = get_chroma_db_path(experiment, "all")
                    if os.path.exists(db_path):
                        shutil.rmtree(db_path)
                    
                    # Re-initialize as new
                    _, vectordb, is_new = build_chroma_store(experiment, "all", lazy_load=True)
                    is_new = True 
            # ----------------------------------------------
            
            # --- Ingestion Loop ---
            _, db_path = get_chroma_db_path(experiment, "all")
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

            docs_to_process = {k: v for k, v in unique_docs.items() if k not in available_docs}
            
            if docs_to_process:
                logger.info(f"Ingesting {len(docs_to_process)} missing documents into Shared Store...")
                
                for doc_name, doc_link in tqdm(docs_to_process.items(), desc="Ingesting PDFs"):
                    pdf_docs, src = load_pdf_with_fallback(doc_name, doc_link, getattr(experiment, 'pdf_local_dir', None))
                    pdf_source_map[doc_name] = src
                    
                    if pdf_docs:
                        chunks = experiment._chunk_text_langchain(pdf_docs, metadata={'doc_name': doc_name})
                        if chunks:
                            populate_chroma_store(experiment, vectordb, chunks, "all")
                            available_docs.add(doc_name)
                
                save_store_config(experiment, db_path)
                with open(meta_path, 'w') as f:
                    json.dump({"available_docs": list(available_docs), "pdf_source_map": pdf_source_map}, f)
            
            _log_pdf_sources(pdf_source_map)

        except Exception as e:
            logger.error(f"Chroma initialization/ingestion failed: {e}")
            if retrieval_mode == "dense":
                return []

    # =========================================================================
    # PART 2: SPARSE RETRIEVER SETUP (BM25)
    # =========================================================================
    sparse_retriever = None
    if retrieval_mode in ["sparse", "hybrid"]:
        logger.info("Initializing Sparse Retriever (BM25)...")
        fingerprint = _compute_corpus_fingerprint(Path(experiment.pdf_local_dir))
        chunks = _load_or_build_bm25_chunks(experiment, fingerprint)
        if chunks:
            sparse_retriever = BM25Retriever.from_documents(chunks, preprocess_func=finance_preprocess_func)
            sparse_retriever.k = candidate_k
        else:
            logger.warning("BM25 initialization failed (no chunks). Falling back to Dense only.")
            retrieval_mode = "dense"

    # =========================================================================
    # PART 3: RERANKER SETUP
    # =========================================================================
    reranker = None
    if use_rerank:
        model_id = "BAAI/bge-reranker-v2-m3"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing Reranker ({model_id}) on {device}...")
        try:
            reranker = CrossEncoder(
                model_id, 
                model_kwargs={"torch_dtype": torch.float16 if device=="cuda" else torch.float32},
                device=device,
                trust_remote_code=True
            )
        except Exception as e:
            logger.warning(f"Failed to load reranker: {e}. Continuing without reranking.")
            use_rerank = False

    # =========================================================================
    # PART 4: EXECUTION LOOP
    # =========================================================================
    results = experiment._create_skipped_results(data, "unified", "unified", "pdf", "unified", start_id=0)

    consecutive_empty_count = 0
    MAX_EMPTY_STRIKES = 5

    for i, sample in enumerate(tqdm(data, desc="Unified Pipeline")):
        question = sample.get("question")
        if not question: 
            continue

        results[i]["doc_name"] = sample.get("doc_name")
        results[i]["doc_link"] = sample.get("doc_link")
        
        # --- Stage 1: HyDE (Query Expansion) ---
        search_query_vector = None
        hypotheticals = []
        
        if use_hyde:
            hypotheticals = _generate_hypotheticals(experiment, question, n=hyde_k)
            if experiment.embeddings:
                embeddings = experiment.embeddings.embed_documents(hypotheticals)
                if len(embeddings) > 1:
                    search_query_vector = np.mean(embeddings, axis=0).tolist()
                elif embeddings:
                    search_query_vector = embeddings[0]
            results[i]["hypotheticals"] = hypotheticals

        # --- Stage 2: Retrieval ---
        dense_docs = []
        sparse_docs = []
        
        # A. Dense Search
        if need_dense_store and vectordb:
            try:
                if search_query_vector:
                    # FIX: Robust Method Calls to avoid AttributeError
                    docs_scores = []
                    
                    # 1. Try modern LangChain method (returns docs + scores)
                    if hasattr(vectordb, "similarity_search_by_vector_with_relevance_scores"):
                        docs_scores = vectordb.similarity_search_by_vector_with_relevance_scores(search_query_vector, k=candidate_k)
                    
                    # 2. Try standard method (returns docs only) and fake the scores
                    elif hasattr(vectordb, "similarity_search_by_vector"):
                        docs = vectordb.similarity_search_by_vector(search_query_vector, k=candidate_k)
                        docs_scores = [(d, 0.0) for d in docs]
                        
                    # 3. Fallback: If HyDE vector exists but vector search is missing on object
                    else:
                        logger.warning(f"Sample {i}: Vector search method missing on store. Falling back to text search.")
                        docs = vectordb.similarity_search(question, k=candidate_k)
                        docs_scores = [(d, 0.0) for d in docs]

                    dense_docs = []
                    for doc, score in docs_scores:
                        doc.metadata['score'] = float(score)
                        dense_docs.append(doc)
                else:
                    # Standard Dense (Query text -> Vector handled by store)
                    # Try to get scores if possible
                    if hasattr(vectordb, "similarity_search_with_score"):
                        docs_scores = vectordb.similarity_search_with_score(question, k=candidate_k)
                        dense_docs = []
                        for doc, score in docs_scores:
                            doc.metadata['score'] = float(score)
                            dense_docs.append(doc)
                    else:
                        dense_docs = vectordb.similarity_search(question, k=candidate_k)
                        for d in dense_docs:
                            d.metadata['score'] = 0.0
                            
            except Exception as e:
                logger.warning(f"Dense retrieval failed for sample {i}: {e}")

        # B. Sparse Search (BM25)
        if sparse_retriever:
            sparse_docs = _get_docs(sparse_retriever, question)

        # C. Fusion or Selection
        final_retrieved_docs = []
        
        if retrieval_mode == "hybrid":
            fused = _perform_rrf(
                dense_docs, sparse_docs, 
                rrf_k=60, 
                top_k=candidate_k 
            )
            from langchain.schema import Document
            for item in fused:
                d = Document(page_content=item['text'], metadata=item['metadata'])
                d.metadata['score'] = item['score']
                final_retrieved_docs.append(d)
        elif retrieval_mode == "sparse":
            final_retrieved_docs = sparse_docs
        else:
            final_retrieved_docs = dense_docs

        # --- Stage 3: Reranking ---
        if use_rerank and final_retrieved_docs:
            pairs = []
            valid_docs = []
            for d in final_retrieved_docs:
                text = d.page_content
                if not text.strip(): continue
                pairs.append([question, text])
                valid_docs.append(d)
            
            if pairs:
                try:
                    scores = reranker.predict(pairs, batch_size=8, show_progress_bar=False)
                    scored = []
                    for d, s in zip(valid_docs, scores):
                        d.metadata['rerank_score'] = float(s)
                        scored.append(d)
                    
                    # Sort by new score
                    scored.sort(key=lambda x: x.metadata['rerank_score'], reverse=True)
                    final_retrieved_docs = scored[:experiment.top_k]
                except Exception as e:
                    logger.warning(f"Reranking failed sample {i}: {e}")
                    final_retrieved_docs = final_retrieved_docs[:experiment.top_k]
            else:
                 final_retrieved_docs = final_retrieved_docs[:experiment.top_k]
        else:
             final_retrieved_docs = final_retrieved_docs[:experiment.top_k]

        # Format for output
        formatted_chunks = []
        for rank, d in enumerate(final_retrieved_docs):
            formatted_chunks.append({
                "rank": rank + 1,
                "text": d.page_content,
                "score": d.metadata.get('rerank_score') or d.metadata.get('score', 0),
                "metadata": d.metadata
            })
            
        results[i]["retrieved_chunks"] = formatted_chunks
        results[i]["num_retrieved"] = len(formatted_chunks)
        
        # --- EARLY STOPPING CHECK ---
        if len(formatted_chunks) == 0:
            consecutive_empty_count += 1
            logger.warning(f"Sample {i}: No chunks retrieved. (Strike {consecutive_empty_count}/{MAX_EMPTY_STRIKES})")
            
            if consecutive_empty_count >= MAX_EMPTY_STRIKES:
                logger.error("🛑 STOPPING EXPERIMENT: No data retrieved for 5 consecutive samples.")
                break 
        else:
            consecutive_empty_count = 0 
        
        # --- Stage 4: Generation ---
        context = "\n\n".join([c["text"] for c in formatted_chunks])
        ans, prompt = experiment._generate_answer(question, context, return_prompt=True)
        
        results[i]["generated_answer"] = ans
        results[i]["final_prompt"] = prompt
        
        gold_segs, gold_str = experiment._prepare_gold_evidence(sample.get('evidence', ''))
        results[i]["gold_evidence"] = gold_str
        results[i]["gold_evidence_segments"] = gold_segs

        experiment.notify_sample_complete(1)

    return results