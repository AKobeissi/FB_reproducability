import logging
import numpy as np
from collections import defaultdict
from typing import List, Dict, Any

from src.ingestion.pdf_utils import load_pdf_with_fallback

logger = logging.getLogger(__name__)

def run_oracle_document(experiment, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Oracle Document Retrieval:
    Restricts search to ONLY the gold document associated with the question.
    Uses fast in-memory embeddings (numpy) instead of Chroma.
    """
    return _run_oracle_base(experiment, data, mode='document')


def run_oracle_page(experiment, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Oracle Page Retrieval:
    Restricts search to ONLY the gold pages within the gold document.
    """
    return _run_oracle_base(experiment, data, mode='page')


def _run_oracle_base(experiment, data: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    logger.info("\n" + "=" * 80)
    logger.info(f"RUNNING ORACLE EXPERIMENT: {mode.upper()}")
    logger.info("=" * 80)

    results = []
    
    # 1. Group samples by document to minimize PDF loading/embedding
    doc_groups = defaultdict(list)
    for sample in data:
        doc_name = sample.get('doc_name', 'unknown')
        doc_groups[doc_name].append(sample)

    logger.info(f"Processing {len(doc_groups)} unique documents")

    for doc_name, samples in doc_groups.items():
        logger.info(f"\nProcessing Document: {doc_name} ({len(samples)} samples)")
        
        # 2. Load and Chunk PDF
        doc_link = samples[0].get('doc_link', '')
        pdf_docs, pdf_source = load_pdf_with_fallback(
            doc_name=doc_name,
            doc_link=doc_link,
            local_dir=getattr(experiment, 'pdf_local_dir', None),
        )

        if not pdf_docs:
            logger.warning(f"No PDF found for {doc_name}. Skipping {len(samples)} samples.")
            continue

        chunks = experiment._chunk_text_langchain(
            pdf_docs,
            metadata={'doc_name': doc_name, 'source': 'pdf', 'pdf_source': pdf_source}
        )
        
        if not chunks:
            logger.warning(f"No chunks generated for {doc_name}.")
            continue

        # 3. Embed all chunks for this document once (Fast In-Memory)
        texts = [c.page_content for c in chunks]
        try:
            # Check if using HF or OpenAI embeddings
            if hasattr(experiment.embeddings, 'embed_documents'):
                doc_vectors = experiment.embeddings.embed_documents(texts)
            else:
                # Fallback for simpler interfaces
                doc_vectors = [experiment.embeddings.embed_query(t) for t in texts]
            
            doc_vectors = np.array(doc_vectors)
            # Normalize for cosine similarity
            norm = np.linalg.norm(doc_vectors, axis=1, keepdims=True)
            doc_vectors = doc_vectors / (norm + 1e-9)
        except Exception as e:
            logger.error(f"Embedding failed for {doc_name}: {e}")
            continue

        # 4. Process Samples
        for i, sample in enumerate(samples):
            try:
                result = _process_sample_oracle(
                    experiment=experiment,
                    sample=sample,
                    chunks=chunks,
                    doc_vectors=doc_vectors,
                    mode=mode,
                    sample_id=len(results),
                    doc_name=doc_name,
                    pdf_source=pdf_source
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing sample {sample.get('question', '')[:30]}: {e}")

            if (len(results) % 10) == 0:
                logger.info(f"Processed {len(results)} samples...")

    return results


def _process_sample_oracle(
    experiment, 
    sample: Dict[str, Any], 
    chunks: List[Any], 
    doc_vectors: np.ndarray, 
    mode: str, 
    sample_id: int,
    doc_name: str,
    pdf_source: str
) -> Dict[str, Any]:
    
    question = sample['question']
    
    # --- Step A: Identify Candidate Chunks ---
    candidate_indices = []
    
    if mode == 'document':
        # Use all chunks in the document
        candidate_indices = list(range(len(chunks)))
    
    elif mode == 'page':
        # Filter chunks that belong to gold pages
        gold_pages = _extract_gold_pages(sample.get('evidence', []))
        
        if not gold_pages:
            # Fallback: if no page info in evidence, use whole doc (or return empty?)
            # Usually better to be strict for "Oracle Page", but we'll log warning
            # and fallback to document search to avoid crashing.
            candidate_indices = list(range(len(chunks)))
        else:
            for idx, chunk in enumerate(chunks):
                # LangChain PDF loaders usually put page number in metadata['page']
                # pypdf is 0-indexed.
                chunk_page = chunk.metadata.get('page')
                if chunk_page is not None and chunk_page in gold_pages:
                    candidate_indices.append(idx)
                    
            if not candidate_indices:
                # No chunks matched the gold pages (parsing mismatch?)
                candidate_indices = list(range(len(chunks)))

    # --- Step B: Retrieval (Cosine Similarity) ---
    if not candidate_indices:
        retrieved_chunks = []
    else:
        # Get query vector
        q_vec = np.array(experiment.embeddings.embed_query(question))
        q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-9)
        
        # Select candidate vectors
        cand_vectors = doc_vectors[candidate_indices]
        
        # Dot product (Cosine Sim since normalized)
        scores = np.dot(cand_vectors, q_vec)
        
        # Get Top K
        k = min(experiment.top_k, len(scores))
        top_k_local_indices = np.argsort(scores)[::-1][:k]
        
        retrieved_chunks = []
        for local_idx in top_k_local_indices:
            global_idx = candidate_indices[local_idx]
            chunk = chunks[global_idx]
            retrieved_chunks.append({
                'text': chunk.page_content,
                'score': float(scores[local_idx]),
                'metadata': chunk.metadata,
                'rank': len(retrieved_chunks) + 1
            })

    # --- Step C: Generation ---
    context_text = "\n\n".join([c['text'] for c in retrieved_chunks])
    
    # Use the standard generation pipeline from the experiment class
    answer, prompt_snapshot = experiment._generate_answer(
        question, 
        context_text, 
        return_prompt=True
    )
    
    if not prompt_snapshot:
        prompt_snapshot = experiment._build_financebench_prompt(
            question, context_text, mode=f"oracle_{mode}"
        )

    # --- Step D: Format Output (Same metrics as other experiments) ---
    gold_segments, gold_evidence_str = experiment._prepare_gold_evidence(sample.get('evidence', ''))
    
    return {
        'sample_id': sample_id,
        'doc_name': doc_name,
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
        'vector_store_type': 'numpy',
        'pdf_source': pdf_source,
        'final_prompt': prompt_snapshot,
    }


def _extract_gold_pages(evidence_list) -> set:
    """Extract unique page indices from evidence list."""
    pages = set()
    if not evidence_list:
        return pages
    
    for ev in evidence_list:
        # Handle various formats: 'page_ix', 'page_idx', 'page'
        for key in ['page_ix', 'page_idx', 'page', 'page_number']:
            if key in ev:
                try:
                    val = ev[key]
                    if val is not None:
                        pages.add(int(val))
                    break 
                except:
                    pass
    return pages