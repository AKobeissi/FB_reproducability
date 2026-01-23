import logging
import os
import json
import gc
import numpy as np
from collections import defaultdict
from typing import List, Dict, Any

from langchain_core.documents import Document

from src.ingestion.pdf_utils import load_pdf_with_fallback
from src.retrieval.vectorstore import build_chroma_store, get_chroma_db_path, build_index_config

logger = logging.getLogger(__name__)

def run_oracle_document(experiment, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Oracle Document Retrieval:
    Restricts search to ONLY the specific vector store of the gold document.
    """
    return _run_oracle_using_chroma(experiment, data, mode='document')


def run_oracle_page(experiment, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Oracle Page Retrieval:
    Restricts search to ONLY the gold pages within the gold document's vector store.
    """
    return _run_oracle_using_chroma(experiment, data, mode='page')


def _run_oracle_using_chroma(experiment, data: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    """
    Main execution loop for Oracle experiments (Document or Page level).
    Handles vector store loading/building per document to ensure isolation.
    """
    logger.info("\n" + "=" * 80)
    logger.info(f"RUNNING ORACLE EXPERIMENT (CHROMA): {mode.upper()}")
    logger.info("=" * 80)

    results = []
    
    # 1. Group samples by document to minimize vector store switching
    doc_groups = defaultdict(list)
    for sample in data:
        doc_name = sample.get('doc_name', 'unknown')
        doc_groups[doc_name].append(sample)

    logger.info(f"Processing {len(doc_groups)} unique documents")

    # Iterate through each document group
    for i, (doc_name, samples) in enumerate(doc_groups.items()):
        logger.info(f"\n[{i+1}/{len(doc_groups)}] Processing Document Context: {doc_name} ({len(samples)} samples)")
        
        vectordb = None
        retriever = None
        
        try:
            # --- 2. Smart Vector Store Loading ---
            # We determine if we can reuse an existing store or need to build one.
            db_name, db_path = get_chroma_db_path(experiment, doc_name)
            config_path = os.path.join(db_path, "config.json")
            
            needs_building = True
            
            # Check if a valid, non-stale store exists WITHOUT opening Chroma yet
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        stored_config = json.load(f)
                    
                    current_config = build_index_config(experiment)
                    
                    # If configs match (chunk size, etc.) and directory looks populated
                    if stored_config == current_config:
                        if len(os.listdir(db_path)) > 1: # Basic check for content
                            needs_building = False
                except Exception as e:
                    logger.warning(f"Error checking config for {db_name}, will rebuild: {e}")

            if not needs_building:
                logger.info(f"Loading existing vector store: {db_name}")
                # We simply load it. The library will handle the rest.
                retriever, vectordb = build_chroma_store(experiment, docs=doc_name)
            else:
                logger.info(f"Building vector store for: {db_name}")
                
                # Load PDF using the fallback logic (checks local dir then URL)
                doc_link = samples[0].get('doc_link', '')
                pdf_docs, pdf_source = load_pdf_with_fallback(
                    doc_name=doc_name,
                    doc_link=doc_link,
                    local_dir=getattr(experiment, 'pdf_local_dir', None),
                )

                if not pdf_docs:
                    logger.warning(f"No PDF found for {doc_name}. Skipping {len(samples)} samples.")
                    continue

                # Create Chunks
                # We attach doc_name and source metadata here
                chunks = experiment._chunk_text_langchain(
                    pdf_docs,
                    metadata={'doc_name': doc_name, 'source': 'pdf', 'pdf_source': pdf_source}
                )
                
                if not chunks:
                    logger.warning(f"No chunks generated for {doc_name}.")
                    continue

                # Build and Persist
                retriever, vectordb = build_chroma_store(
                    experiment,
                    docs=doc_name, 
                    documents=chunks
                )
                
        except Exception as e:
            logger.error(f"Failed to initialize vector store for {doc_name}: {e}")
            # Force cleanup
            gc.collect()
            continue

        # --- 3. Process Samples for this Document ---
        if vectordb:
            for sample in samples:
                try:
                    result = _process_sample_chroma(
                        experiment=experiment,
                        sample=sample,
                        vectordb=vectordb,
                        mode=mode,
                        sample_id=len(results)
                    )
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing sample {sample.get('question', '')[:30]}: {e}", exc_info=True)

                if (len(results) % 10) == 0:
                    logger.info(f"Processed {len(results)} samples...")
        
        # --- 4. Cleanup ---
        # Crucial: Delete references and GC to release file locks on the ChromaDB folder
        # before the next iteration tries to open a different DB (or the same one).
        if vectordb:
            del vectordb
        if retriever:
            del retriever
        
        gc.collect()

    return results


def _process_sample_chroma(
    experiment, 
    sample: Dict[str, Any], 
    vectordb, 
    mode: str, 
    sample_id: int
) -> Dict[str, Any]:
    
    question = sample['question']
    k = experiment.top_k
    
    # --- Step A: Setup Page Oracle Filter ---
    search_kwargs = {}
    gold_pages = set()
    
    if mode == 'page':
        # Extract gold pages from evidence using the specific field names
        gold_pages = _extract_gold_pages(sample.get('evidence', []))
        
        if gold_pages:
            # Construct Chroma Metadata Filter
            # This forces the vector store to ONLY search chunks from these pages.
            if len(gold_pages) == 1:
                search_filter = {'page': list(gold_pages)[0]}
            else:
                search_filter = {'page': {'$in': list(gold_pages)}}
            
            search_kwargs['filter'] = search_filter
            logger.debug(f"Applied page filter for sample {sample_id}: {search_filter}")
        else:
            logger.warning(f"Oracle Page requested but no 'evidence_page_num' found for sample {sample_id}. Falling back to full Doc Oracle.")

    # --- Step B: Retrieval ---
    # Retrieve chunks (restricted by filter if in page mode)
    docs_and_scores = vectordb.similarity_search_with_score(
        question, 
        k=k,
        **search_kwargs
    )
    
    # Format Retrieved Chunks
    retrieved_chunks = []
    for rank, (doc, score) in enumerate(docs_and_scores):
        retrieved_chunks.append({
            'text': doc.page_content,
            'score': float(score),
            'metadata': doc.metadata,
            'rank': rank + 1
        })

    # --- Step C: Verification (Optional logging) ---
    if mode == "page" and gold_pages:
        got_pages = {c["metadata"].get("page") for c in retrieved_chunks}
        # Normalize to ints for comparison
        got_pages_int = {int(p) for p in got_pages if p is not None and str(p).isdigit()}
        
        # If we got results, they MUST be from the gold pages.
        if retrieved_chunks and not got_pages_int.issubset(gold_pages):
             logger.warning(f"Page-oracle leak detected! Retrieved pages {got_pages_int} not in gold {gold_pages}")

    # --- Step D: Generation ---
    context_text = "\n\n".join([c['text'] for c in retrieved_chunks])
    
    answer, prompt_snapshot = experiment._generate_answer(
        question, 
        context_text, 
        return_prompt=True
    )
    
    if not prompt_snapshot:
        prompt_snapshot = experiment._build_financebench_prompt(
            question, context_text, mode=f"oracle_{mode}"
        )

    # --- Step E: Result Formatting ---
    gold_segments, gold_evidence_str = experiment._prepare_gold_evidence(sample.get('evidence', ''))
    
    return {
        'sample_id': sample_id,
        'doc_name': sample.get('doc_name'),
        'doc_link': sample.get('doc_link', ''),
        'question': question,
        'reference_answer': sample['answer'],
        'question_type': sample.get('question_type'),
        'question_reasoning': sample.get('question_reasoning'),
        'gold_evidence': gold_evidence_str,
        'gold_evidence_segments': gold_segments,
        'retrieved_chunks': retrieved_chunks,
        'num_retrieved': len(retrieved_chunks),
        'context_length': len(context_text),
        'generated_answer': answer,
        'generation_length': len(answer),
        'experiment_type': f"oracle_{mode}",
        'vector_store_type': 'chroma',
        'final_prompt': prompt_snapshot,
    }


def _extract_gold_pages(evidence_list) -> set:
    """
    Extract unique (0-indexed) page indices from FinanceBench-style evidence.
    Returns a set[int].
    """
    pages = set()
    
    # 1. Handle None
    if evidence_list is None:
        return pages

    # 2. Fix: Handle NumPy arrays explicitly to avoid "truth value ambiguous" errors
    if isinstance(evidence_list, np.ndarray):
        # Check if empty array
        if evidence_list.size == 0:
            return pages
        # Convert to standard python list for iteration
        evidence_list = evidence_list.tolist()

    # 3. Handle standard empty lists (now safe)
    if not evidence_list:
        return pages
    
    for ev in evidence_list:
        if not isinstance(ev, dict):
            continue

        found_page = False
        
        # Priority 1: The specific field for FinanceBench gold pages
        if "evidence_page_num" in ev and ev["evidence_page_num"] is not None:
            try:
                pages.add(int(ev["evidence_page_num"]))
                found_page = True
            except (ValueError, TypeError):
                pass
        
        if found_page:
            continue

        # Priority 2: Fallback keys
        for key in ["page_ix", "page_idx", "page", "page_number"]:
            if key in ev and ev[key] is not None:
                try:
                    pages.add(int(ev[key]))
                    break
                except (ValueError, TypeError):
                    pass

    return pages