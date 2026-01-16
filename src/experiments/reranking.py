"""
RAG Experiment with Cross-Encoder Re-ranking.
Literature Standard: Retrieve Top-N (e.g. 20) -> Rerank -> Select Top-K (e.g. 5).
"""
from typing import List, Dict, Any
import logging
import torch
import os
import json
from pathlib import Path
from sentence_transformers import CrossEncoder

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
INITIAL_TOP_N = 20  # Pool size before re-ranking

def run_reranking(experiment, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Run experiment: Shared Vector Store + Cross-Encoder Reranking.
    """
    logger.info("\n" + "=" * 80)
    logger.info(f"RUNNING RERANKING EXPERIMENT")
    logger.info(f"Model: {RERANKER_MODEL_ID} | Top-N: {INITIAL_TOP_N} -> Top-K: {experiment.top_k}")
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
    logger.info(f"Loading Cross-Encoder on {device}...")
    try:
        reranker = CrossEncoder(
            RERANKER_MODEL_ID, 
            model_kwargs={"torch_dtype": torch.float16 if device=="cuda" else torch.float32},
            device=device,
            trust_remote_code=True
        )
    except Exception as e:
        logger.error(f"Failed to load reranker: {e}")
        raise


    # --- PART 3: RUN EXPERIMENT LOOP ---
    base_retriever.search_kwargs["k"] = INITIAL_TOP_N

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
                scores = reranker.predict(pairs, batch_size=8, show_progress_bar=False)
                
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