"""
Expanded Shared Vector Experiment.
Search over ALL documents (Shared Index) using Query Expansion.
"""
from typing import List, Dict, Any
import logging
import os
import json

try:
    from .pdf_utils import load_pdf_with_fallback
    from .vectorstore import build_chroma_store, populate_chroma_store, save_store_config, get_chroma_db_path
    from .query_expansion import process_query_for_experiment
    from .rag_shared_vector import (
        _get_or_create_retrieval_prompt, 
        _get_or_create_documents_chain, 
        _build_chunks_from_docs, 
        _fallback_retrieval_qa,
        _supports_modern_retrieval,
        _log_pdf_sources,
        _create_skipped_result,
        _create_all_skipped_results
    )
except ImportError:
    from pdf_utils import load_pdf_with_fallback
    from vectorstore import build_chroma_store, populate_chroma_store, save_store_config, get_chroma_db_path
    from query_expansion import process_query_for_experiment
    from rag_shared_vector import (
        _get_or_create_retrieval_prompt, 
        _get_or_create_documents_chain, 
        _build_chunks_from_docs, 
        _fallback_retrieval_qa,
        _supports_modern_retrieval,
        _log_pdf_sources,
        _create_skipped_result,
        _create_all_skipped_results
    )

logger = logging.getLogger(__name__)

def run_expanded_shared(experiment, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Run SHARED vector store experiment with Query Expansion.
    """
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING EXPANDED SHARED EXPERIMENT (Expansion + Global Search)")
    logger.info("=" * 80)

    # --- Step 1: Initialize the Shared "All" Index (Same as rag_shared_vector) ---
    unique_docs = {}
    
    # Scan local dir if configured
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

    # Collect docs from data
    for sample in data:
        doc_name = sample.get('doc_name', 'unknown')
        if doc_name not in unique_docs:
            unique_docs[doc_name] = sample.get('doc_link', '')

    # Initialize Chroma
    try:
        retriever, vectordb, is_new = build_chroma_store(experiment, "all", lazy_load=True)
    except Exception as e:
        logger.error("Chroma build failed: %s", e)
        return _create_all_skipped_results(data, "expanded_shared", experiment=experiment)

    # Metadata and Ingestion checks
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

    # Ingest missing docs
    if docs_to_process:
        logger.info(f"Ingesting {len(docs_to_process)} new documents...")
        for doc_name, doc_link in docs_to_process.items():
            pdf_docs, pdf_source = load_pdf_with_fallback(
                doc_name=doc_name,
                doc_link=doc_link,
                local_dir=getattr(experiment, 'pdf_local_dir', None),
            )
            pdf_source_map[doc_name] = pdf_source
            if not pdf_docs: continue

            chunks = experiment._chunk_text_langchain(pdf_docs, metadata={'doc_name': doc_name, 'source': 'pdf'})
            if chunks:
                populate_chroma_store(experiment, vectordb, chunks, "all")
                available_docs.add(doc_name)
        
        save_store_config(experiment, db_path)
        with open(meta_path, 'w') as f:
            json.dump({"available_docs": list(available_docs), "pdf_source_map": pdf_source_map}, f)

    _log_pdf_sources(pdf_source_map)

    # --- Step 2: Process Samples with Expansion ---
    results = []
    for i, sample in enumerate(data):
        doc_name = sample.get('doc_name', 'unknown')
        
        # We can't evaluate retrieval if the doc isn't in the index
        if doc_name not in available_docs:
            skipped = _create_skipped_result(sample, i, doc_name, "expanded_shared", experiment=experiment)
            results.append(skipped)
            continue

        result = _process_expanded_shared_sample(
            experiment=experiment,
            sample=sample,
            sample_id=i,
            doc_name=doc_name,
            pdf_source=pdf_source_map.get(doc_name, 'none'),
            retriever=retriever
        )
        results.append(result)
        logger.info(f"Completed sample {i+1}")

    return results

def _process_expanded_shared_sample(
    experiment, sample, sample_id, doc_name, pdf_source, retriever
):
    raw_question = sample.get('question', '')
    
    # 1. APPLY EXPANSION
    retrieval_query, generation_query = process_query_for_experiment(raw_question)
    
    logger.info(f"Original:  {raw_question}")
    logger.info(f"Retrieval: {retrieval_query} (Searching GLOBAL index)")
    
    # 2. Run Retrieval using retrieval_query (with anchors)
    generated_answer, retrieved_chunks, final_prompt = _run_expanded_retrieval(
        experiment=experiment,
        retrieval_query=retrieval_query,
        generation_query=generation_query,
        retriever=retriever
    )

    # Calculate statistics for Mixin compatibility
    context_text = "\n\n".join([chunk['text'] for chunk in retrieved_chunks])
    context_length = len(context_text)
    generation_length = len(generated_answer)
    
    # Helper for Gold Evidence (to prevent other errors down the line)
    gold_segments, gold_evidence_str = experiment._prepare_gold_evidence(sample.get('evidence', ''))
    
    # Log where we found the data
    source_docs = [
        {'doc_name': c['metadata'].get('doc_name'), 'score': c.get('score')} 
        for c in retrieved_chunks
    ]
    
    return {
        'sample_id': sample_id,
        'doc_name': doc_name,
        'question': raw_question,
        'expanded_retrieval_query': retrieval_query,
        'expanded_generation_query': generation_query,
        'reference_answer': sample.get('answer', ''),
        
        # --- Standard Fields Required by Mixins ---
        'question_type': sample.get('question_type'),
        'question_reasoning': sample.get('question_reasoning'),
        'gold_evidence': gold_evidence_str,
        'gold_evidence_segments': gold_segments,
        'num_retrieved': len(retrieved_chunks),
        'context_length': context_length,
        'generation_length': generation_length,
        # ----------------------------------------

        'retrieved_chunks': retrieved_chunks,
        'retrieved_from_docs': source_docs,
        'generated_answer': generated_answer,
        'experiment_type': "expanded_shared",
        'final_prompt': final_prompt,
        'pdf_source': pdf_source
    }

def _run_expanded_retrieval(experiment, retrieval_query, generation_query, retriever):
    """
    Decoupled retrieval and generation for the shared index.
    """
    experiment.ensure_langchain_llm()
    
    # A. Retrieve (using anchored query)
    try:
        # Modern
        docs = retriever.invoke(retrieval_query)
    except:
        # Legacy
        docs = retriever.get_relevant_documents(retrieval_query)
        
    chunks = _build_chunks_from_docs(docs)
    context_text = "\n\n".join(chunk['text'] for chunk in chunks)
    
    # B. Generate (using clean query + context)
    prompt = experiment._build_financebench_prompt(
        question=generation_query, 
        context=context_text, 
        mode="expanded_shared"
    )
    
    if experiment.use_api:
        answer = experiment._generate_answer(generation_query, context_text)
    else:
        outputs = experiment.llm_pipeline(prompt, max_new_tokens=experiment.max_new_tokens)
        answer = outputs[0]["generated_text"]

    return answer.strip(), chunks, prompt