from typing import List, Dict, Any
import logging
from collections import defaultdict
from .pdf_utils import load_pdf_with_fallback
from .vectorstore import build_chroma_store

logger = logging.getLogger(__name__)


def run_single_vector(experiment, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Run single vector store experiment (separate FAISS index per document).
    """
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING SINGLE VECTOR STORE EXPERIMENT (LangChain + FAISS)")
    logger.info("=" * 80)

    results = []
    doc_groups = defaultdict(list)
    
    for sample in data:
        doc_name = sample.get('doc_name', 'unknown')
        doc_groups[doc_name].append(sample)

    logger.info(f"Processing {len(doc_groups)} unique documents")

    for doc_name, samples in doc_groups.items():
        logger.info(f"\n{'='*80}")
        logger.info(f"Document: {doc_name} ({len(samples)} samples)")
        logger.info(f"{'='*80}")

        doc_link = samples[0].get('doc_link', '')
        
        # Use centralized PDF loading
        pdf_docs, pdf_source = load_pdf_with_fallback(
            doc_name=doc_name,
            doc_link=doc_link,
            local_dir=getattr(experiment, 'pdf_local_dir', None),
        )

        if not pdf_docs:
            logger.warning(f"No PDF pages for '{doc_name}'. Skipping {len(samples)} samples.")
            results.extend(_create_skipped_results(
                samples, doc_name, doc_link, pdf_source, 
                experiment.SINGLE_VECTOR, len(results)
            ))
            continue

        logger.info(f"PDF source: {pdf_source}")

        # Chunk the PDF text
        documents = experiment._chunk_text_langchain(
            pdf_docs,
            metadata={
                'doc_name': doc_name,
                'source': 'pdf',
                'doc_link': doc_link,
                'pdf_source': pdf_source
            }
        )

        try:
            retriever, _ = build_chroma_store(
                experiment,
                doc_name,
                documents=documents
            )
        except Exception as exc:
            logger.error(f"Chroma build failed for '{doc_name}': {exc}")
            results.extend(_create_skipped_results(
                samples, doc_name, doc_link, pdf_source,
                experiment.SINGLE_VECTOR, len(results)
            ))
            continue

        # Process each sample
        for i, sample in enumerate(samples):
            logger.info(f"\n--- Sample {i+1}/{len(samples)} for {doc_name} ---")
            
            result = _process_single_sample(
                experiment=experiment,
                sample=sample,
                doc_name=doc_name,
                doc_link=doc_link,
                pdf_source=pdf_source,
                retriever=retriever,
                sample_id=len(results)
            )
            
            results.append(result)
            logger.info(f"Completed sample {len(results)}")

    return results


def _process_single_sample(
    experiment,
    sample: Dict[str, Any],
    doc_name: str,
    doc_link: str,
    pdf_source: str,
    retriever,
    sample_id: int
) -> Dict[str, Any]:
    """Process a single sample with RetrievalQA and evaluation."""
    question = sample['question']
    reference_answer = sample['answer']
    gold_evidence = sample.get('evidence', '')

    gold_parts = experiment._normalize_evidence(gold_evidence)
    gold_evidence_str = "\n\n".join(gold_parts)

    generated_answer, retrieved_chunks = _run_retrieval_qa(
        experiment=experiment,
        question=question,
        retriever=retriever
    )

    context = "\n\n".join([chunk['text'] for chunk in retrieved_chunks])

    retrieval_eval = experiment.evaluator.compute_retrieval_metrics(
        [chunk['text'] for chunk in retrieved_chunks],
        gold_evidence_str
    )

    generation_eval = experiment.evaluator.evaluate_generation(
        generated_answer,
        reference_answer,
        question
    )

    return {
        'sample_id': sample_id,
        'doc_name': doc_name,
        'doc_link': doc_link,
        'question': question,
        'reference_answer': reference_answer,
        'gold_evidence': gold_evidence_str,
        'retrieved_chunks': retrieved_chunks,
        'num_retrieved': len(retrieved_chunks),
        'context_length': len(context),
        'generated_answer': generated_answer,
        'generation_length': len(generated_answer),
        'retrieval_evaluation': retrieval_eval,
        'generation_evaluation': generation_eval,
        'experiment_type': experiment.SINGLE_VECTOR,
        'vector_store_type': 'Chroma',
        'pdf_source': pdf_source
    }


def _run_retrieval_qa(experiment, question: str, retriever) -> tuple[str, List[Dict[str, Any]]]:
    """Run RetrievalQA using the experiment's LangChain LLM wrapper."""
    from langchain.chains import RetrievalQA

    if retriever is None:
        return "", []

    experiment.ensure_langchain_llm()
    qa = RetrievalQA.from_chain_type(
        llm=experiment.langchain_llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    result = qa(question)
    source_docs = result.get('source_documents', []) or []
    chunks = [
        {
            'rank': rank + 1,
            'text': _get_doc_text(doc),
            'score': None,
            'length': len(_get_doc_text(doc)),
            'metadata': getattr(doc, 'metadata', {})
        }
        for rank, doc in enumerate(source_docs)
    ]
    answer = result.get('result', '') or ""
    return answer.strip(), chunks


def _get_doc_text(doc) -> str:
    """Extract text content from document object."""
    return (getattr(doc, 'page_content', None) or 
            getattr(doc, 'content', None) or 
            str(doc))


def _create_skipped_results(
    samples: List[Dict[str, Any]],
    doc_name: str,
    doc_link: str,
    pdf_source: str,
    experiment_type: str,
    start_id: int
) -> List[Dict[str, Any]]:
    """Create skipped result entries for samples without PDF text."""
    results = []
    for i, sample in enumerate(samples):
        results.append({
            'sample_id': start_id + i,
            'doc_name': doc_name,
            'doc_link': doc_link,
            'question': sample.get('question', ''),
            'reference_answer': sample.get('answer', ''),
            'gold_evidence': '',
            'retrieved_chunks': [],
            'num_retrieved': 0,
            'context_length': 0,
            'generated_answer': '',
            'generation_length': 0,
            'retrieval_evaluation': {},
            'generation_evaluation': {},
            'experiment_type': experiment_type,
            'vector_store_type': 'FAISS',
            'skipped': True,
            'skipped_reason': 'no_pdf_text',
            'pdf_source': pdf_source
        })
    return results