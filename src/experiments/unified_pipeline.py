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
from src.retrieval.ot_reranker import OTReranker
from src.retrieval.late_chunking import load_late_index_for_scope

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
    create_faiss_store,
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
    retrieval_mode = getattr(experiment, "unified_retrieval", "dense") # dense, sparse, hybrid, bert
    use_rerank = getattr(experiment, "unified_use_rerank", False)
    reranker_style = getattr(experiment, "unified_reranker_style", "cross_encoder")
    ot_model = getattr(experiment, "unified_ot_model", experiment.embedding_model)
    ot_query_sentences = getattr(experiment, "unified_ot_query_sentences", 8)
    ot_doc_sentences = getattr(experiment, "unified_ot_doc_sentences", 24)
    ot_reg = getattr(experiment, "unified_ot_reg", 0.05)
    ot_iters = getattr(experiment, "unified_ot_iters", 40)
    ot_prune_k = getattr(experiment, "unified_ot_prune_k", 20)
    
    # Heuristic: If reranking, fetch more candidates
    candidate_k = experiment.top_k * 4 if use_rerank else experiment.top_k
    
    logger.info(f"Pipeline Configuration:")
    logger.info(f"  [1] HyDE Enabled: {use_hyde} (Generations: {hyde_k})")
    logger.info(f"  [2] Retrieval Mode: {retrieval_mode.upper()} (Candidates: {candidate_k})")
    if retrieval_mode == "bert":
        logger.info(f"      BERT Embedding Model: {getattr(experiment, 'embedding_model', 'unknown')}")
    logger.info(f"  [3] Reranking Enabled: {use_rerank}")
    if use_rerank:
        logger.info(f"  [4] Reranker Style: {reranker_style}")

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
    late_index = None
    # We need the dense store if mode is dense/hybrid OR if HyDE is enabled
    need_dense_store = (retrieval_mode in ["dense", "hybrid", "bert"]) or use_hyde
    use_faiss = getattr(experiment, "use_faiss_chunking", False)

    if need_dense_store:
        if getattr(experiment, "chunking_strategy", "recursive") == "late":
            logger.info("Initializing late chunking dense index...")
            all_pages: List[Any] = []
            for doc_name, doc_link in unique_docs.items():
                pdf_docs, src = load_pdf_with_fallback(doc_name, doc_link, getattr(experiment, 'pdf_local_dir', None))
                for d in pdf_docs or []:
                    d.metadata.setdefault("doc_name", doc_name)
                    d.metadata.setdefault("source", src or "pdf")
                all_pages.extend(pdf_docs or [])
            if all_pages:
                late_index = load_late_index_for_scope(experiment, all_pages, scope="all", pdf_dir=Path(experiment.pdf_local_dir))
            else:
                logger.warning("Late chunking: no PDF pages loaded. Dense retrieval disabled.")
                need_dense_store = False
        elif use_faiss:
            logger.info("Using FAISS for unified dense retrieval (in-memory).")
            all_chunks = []
            pdf_source_map = {}
            for doc_name, doc_link in unique_docs.items():
                pdf_docs, src = load_pdf_with_fallback(doc_name, doc_link, getattr(experiment, 'pdf_local_dir', None))
                pdf_source_map[doc_name] = src
                if pdf_docs:
                    chunks = experiment._chunk_text_langchain(pdf_docs, metadata={'doc_name': doc_name})
                    if chunks:
                        all_chunks.extend(chunks)
            _log_pdf_sources(pdf_source_map)

            if all_chunks:
                try:
                    vectordb = create_faiss_store(experiment, all_chunks, index_name="unified")
                except Exception as e:
                    logger.error(f"FAISS initialization failed: {e}")
                    if retrieval_mode == "dense":
                        return []
            else:
                logger.error("FAISS dense setup failed: no chunks built.")
                if retrieval_mode == "dense":
                    return []
        else:
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
    ce_reranker = None
    ot_reranker = None
    if use_rerank:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            if reranker_style in ["ot", "ot_then_cross_encoder"]:
                logger.info(f"Initializing OT Reranker ({ot_model}) on {device}...")
                ot_reranker = OTReranker(
                    model_name=ot_model,
                    device=device,
                    query_max_sentences=ot_query_sentences,
                    doc_max_sentences=ot_doc_sentences,
                    sinkhorn_reg=ot_reg,
                    sinkhorn_iters=ot_iters,
                )
            if reranker_style in ["cross_encoder", "ot_then_cross_encoder"]:
                model_id = "BAAI/bge-reranker-v2-m3"
                logger.info(f"Initializing Reranker ({model_id}) on {device}...")
                ce_reranker = CrossEncoder(
                    model_id,
                    model_kwargs={"torch_dtype": torch.float16 if device=="cuda" else torch.float32},
                    device=device,
                    trust_remote_code=True
                )
            reranker = ot_reranker or ce_reranker
        except Exception as e:
            logger.warning(f"Failed to load reranker: {e}. Continuing without reranking.")
            use_rerank = False

    # =========================================================================
    # PART 4: EXECUTION LOOP
    # =========================================================================
    results = experiment._create_skipped_results(data, "unified", "unified", "pdf", "unified", start_id=0)

    consecutive_empty_count = 0
    MAX_EMPTY_STRIKES = len(data)  # never kill experiment early; log instead

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
            if late_index is not None and hypotheticals:
                late_vecs = [late_index.embed_query(h) for h in hypotheticals]
                if late_vecs:
                    search_query_vector = np.mean(late_vecs, axis=0)
            elif experiment.embeddings:
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
        if need_dense_store and late_index is not None:
            try:
                if search_query_vector:
                    dense_docs = late_index.search_by_vector(np.array(search_query_vector), k=candidate_k)
                else:
                    dense_docs = late_index.search(question, k=candidate_k)
                logger.debug(f"Sample {i}: late_index returned {len(dense_docs)} docs")
            except Exception as e:
                logger.error(f"Late chunking retrieval FAILED for sample {i}: {e}", exc_info=True)
        elif need_dense_store and vectordb:
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
                    if reranker_style == "ot":
                        doc_texts = [doc.page_content for doc in valid_docs]
                        scores = ot_reranker.score(question, doc_texts)
                    elif reranker_style == "ot_then_cross_encoder":
                        doc_texts = [doc.page_content for doc in valid_docs]
                        ot_scores = ot_reranker.score(question, doc_texts)
                        ot_ranked = sorted(
                            zip(valid_docs, ot_scores),
                            key=lambda x: x[1],
                            reverse=True,
                        )
                        prune_k = min(max(1, ot_prune_k), len(ot_ranked))
                        pruned_docs = [doc for doc, _ in ot_ranked[:prune_k]]
                        pruned_pairs = [[question, doc.page_content] for doc in pruned_docs]
                        ce_scores = ce_reranker.predict(pruned_pairs, batch_size=8, show_progress_bar=False)
                        scored = []
                        for d, s in zip(pruned_docs, ce_scores):
                            d.metadata['ot_score'] = float(next(score for doc, score in ot_ranked if doc is d))
                            d.metadata['rerank_score'] = float(s)
                            scored.append(d)
                        scored.sort(key=lambda x: x.metadata['rerank_score'], reverse=True)
                        final_retrieved_docs = scored[:experiment.top_k]
                        scores = None
                    else:
                        scores = ce_reranker.predict(pairs, batch_size=8, show_progress_bar=False)

                    if scores is not None:
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