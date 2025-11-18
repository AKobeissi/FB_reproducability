from typing import List, Dict, Any, Set
import logging
import os
from collections import Counter
from .pdf_utils import load_pdf_with_fallback
from .vectorstore import build_chroma_store, create_faiss_store, retrieve_faiss_chunks

logger = logging.getLogger(__name__)


def run_shared_vector(experiment, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Run shared vector store experiment (single FAISS index for all documents).
    """
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING SHARED VECTOR STORE EXPERIMENT (LangChain + Chroma preferred)")
    logger.info("=" * 80)

    # Collect unique documents
    unique_docs = {}
    for sample in data:
        doc_name = sample.get('doc_name', 'unknown')
        if doc_name not in unique_docs:
            unique_docs[doc_name] = sample.get('doc_link', '')

    logger.info(f"Collected {len(unique_docs)} unique documents")

    # Load and chunk all documents
    all_documents, available_docs, pdf_source_map = _load_all_documents(
        experiment, unique_docs
    )

    if not all_documents:
        logger.warning("No documents available. All samples will be skipped.")
        return _create_all_skipped_results(data, experiment.SHARED_VECTOR)

    logger.info(f"\nTotal chunks across all documents: {len(all_documents)}")
    _log_pdf_sources(pdf_source_map)

    # Create shared vector store (prefer Chroma). If Chroma fails and FAISS is available,
    # a fallback is attempted only if explicitly allowed via env var `ALLOW_FAISS_FALLBACK`.
    retriever = None
    shared_vector_store = None
    allow_faiss = os.environ.get('ALLOW_FAISS_FALLBACK', 'false').lower() in ('1', 'true', 'yes')
    try:
        retriever, shared_vector_store = build_chroma_store(
            experiment,
            "all",
            documents=all_documents,
        )

    except Exception as e:
        logger.warning(f"Chroma build failed: {e}")
        if allow_faiss:
            logger.info("Falling back to FAISS as configured by ALLOW_FAISS_FALLBACK")
            try:
                shared_vector_store = create_faiss_store(experiment, all_documents, index_name="shared_store")
            except Exception as e2:
                logger.error(f"FAISS fallback failed: {e2}")
                shared_vector_store = None
        else:
            logger.error("Chroma unavailable and FAISS fallback not allowed; aborting shared vector run.")
            return _create_all_skipped_results(data, experiment.SHARED_VECTOR)

    # Process samples
    results = []
    for i, sample in enumerate(data):
        logger.info(f"\n--- Sample {i+1}/{len(data)} ---")
        
        doc_name = sample.get('doc_name', 'unknown')
        
        if doc_name not in available_docs:
            logger.info(f"Skipping sample for '{doc_name}' (no PDF text)")
            results.append(_create_skipped_result(
                sample, i, doc_name, experiment.SHARED_VECTOR
            ))
            continue

        result = _process_shared_sample(
            experiment=experiment,
            sample=sample,
            sample_id=i,
            doc_name=doc_name,
            pdf_source=pdf_source_map.get(doc_name, 'none'),
            retriever=retriever,
            vectordb=shared_vector_store
        )
        
        results.append(result)
        logger.info(f"Completed sample {i+1}")

    return results


def _load_all_documents(
    experiment,
    unique_docs: Dict[str, str]
) -> tuple[List, Set[str], Dict[str, str]]:
    """
    Load and chunk all documents.
    
    Returns:
        (all_documents, available_docs, pdf_source_map)
    """
    all_documents = []
    available_docs = set()
    pdf_source_map = {}

    for doc_name, doc_link in unique_docs.items():
        logger.info(f"\nProcessing document: {doc_name}")
        
        pdf_text, pdf_source = load_pdf_with_fallback(
            doc_name=doc_name,
            doc_link=doc_link,
            local_dir=getattr(experiment, 'pdf_local_dir', None),
            pdf_loader_func=experiment._load_pdf_text
        )
        
        pdf_source_map[doc_name] = pdf_source

        if not pdf_text:
            logger.warning(f"No PDF text for '{doc_name}'. Excluding from shared store.")
            continue

        logger.info(f"Chunking (from {pdf_source}): {doc_name}")
        docs = experiment._chunk_text_langchain(
            pdf_text,
            metadata={
                'doc_name': doc_name,
                'source': 'pdf',
                'doc_link': doc_link,
                'pdf_source': pdf_source
            }
        )
        
        all_documents.extend(docs)
        available_docs.add(doc_name)

    return all_documents, available_docs, pdf_source_map


def _process_shared_sample(
    experiment,
    sample: Dict[str, Any],
    sample_id: int,
    doc_name: str,
    pdf_source: str,
    retriever,
    vectordb
) -> Dict[str, Any]:
    """Process a single sample in shared vector store mode."""
    question = sample.get('question', '')
    reference_answer = sample.get('answer', '')
    gold_evidence = sample.get('evidence', '')
    
    gold_parts = experiment._normalize_evidence(gold_evidence)
    gold_evidence_str = "\n\n".join(gold_parts)

    # Retrieve chunks
    retrieved_chunks = _retrieve_chunks_shared(
        experiment=experiment,
        question=question,
        retriever=retriever,
        vectordb=vectordb
    )

    context = "\n\n".join([chunk['text'] for chunk in retrieved_chunks])
    
    # Log source documents
    source_docs = [
        {
            'doc_name': chunk['metadata'].get('doc_name', 'unknown'),
            'doc_link': chunk['metadata'].get('doc_link', '')
        }
        for chunk in retrieved_chunks
    ]
    logger.info(f"Retrieved from documents: {source_docs}")

    # Evaluate retrieval
    retrieved_texts = [chunk['text'] for chunk in retrieved_chunks]
    retrieval_eval = experiment.evaluator.compute_retrieval_metrics(
        retrieved_texts,
        gold_evidence_str
    )

    # Generate answer
    generated_answer = experiment._generate_answer(question, context)

    # Evaluate generation
    generation_eval = experiment.evaluator.evaluate_generation(
        generated_answer,
        reference_answer,
        question
    )

    return {
        'sample_id': sample_id,
        'doc_name': doc_name,
        'question': question,
        'reference_answer': reference_answer,
        'gold_evidence': gold_evidence_str,
        'retrieved_chunks': retrieved_chunks,
        'retrieved_from_docs': source_docs,
        'num_retrieved': len(retrieved_chunks),
        'context_length': len(context),
        'generated_answer': generated_answer,
        'generation_length': len(generated_answer),
        'retrieval_evaluation': retrieval_eval,
        'generation_evaluation': generation_eval,
        'experiment_type': experiment.SHARED_VECTOR,
        'vector_store_type': 'Chroma' if retriever else 'FAISS',
        'pdf_source': pdf_source
    }


def _retrieve_chunks_shared(
    experiment,
    question: str,
    retriever,
    vectordb
) -> List[Dict[str, Any]]:
    """Retrieve chunks from shared vector store."""
    # Try RetrievalQA chain
    try:
        from langchain.chains import RetrievalQA
        if getattr(experiment, 'langchain_llm', None) and retriever:
            qa = RetrievalQA.from_chain_type(
                llm=experiment.langchain_llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
            )
            result = qa(question)
            source_docs = result.get('source_documents', []) or []
            return [
                {
                    'rank': rank + 1,
                    'text': _get_doc_text(doc),
                    'score': None,
                    'length': len(_get_doc_text(doc)),
                    'metadata': getattr(doc, 'metadata', {})
                }
                for rank, doc in enumerate(source_docs)
            ]
    except Exception:
        pass

    # Try retriever directly
    try:
        if retriever:
            docs = retriever.get_relevant_documents(question)
            return [
                {
                    'rank': rank + 1,
                    'text': _get_doc_text(doc),
                    'score': None,
                    'length': len(_get_doc_text(doc)),
                    'metadata': getattr(doc, 'metadata', {})
                }
                for rank, doc in enumerate(docs)
            ]
    except Exception:
        pass

    # Fallback to FAISS-like retrieval if retriever/vectordb not usable
    try:
        # Prefer new helper if available
        if 'retrieve_faiss_chunks' in globals() and vectordb is not None:
            return retrieve_faiss_chunks(experiment, question, vectordb)
        # Backwards-compat: try experiment method
        return experiment._retrieve_chunks_faiss(question, vectordb)
    except Exception as e:
        logger.warning(f"All retrieval methods failed: {e}")
        return []


def _get_doc_text(doc) -> str:
    """Extract text from document object."""
    return (getattr(doc, 'page_content', None) or 
            getattr(doc, 'content', None) or 
            str(doc))


def _create_skipped_result(
    sample: Dict[str, Any],
    sample_id: int,
    doc_name: str,
    experiment_type: str
) -> Dict[str, Any]:
    """Create a skipped result entry."""
    return {
        'sample_id': sample_id,
        'doc_name': doc_name,
        'doc_link': sample.get('doc_link', ''),
        'question': sample.get('question', ''),
        'reference_answer': sample.get('answer', ''),
        'gold_evidence': '',
        'retrieved_chunks': [],
        'retrieved_from_docs': [],
        'num_retrieved': 0,
        'context_length': 0,
        'generated_answer': '',
        'generation_length': 0,
        'retrieval_evaluation': {},
        'generation_evaluation': {},
        'experiment_type': experiment_type,
        'vector_store_type': 'FAISS',
        'skipped': True,
        'skipped_reason': 'no_pdf_text'
    }


def _create_all_skipped_results(
    data: List[Dict[str, Any]],
    experiment_type: str
) -> List[Dict[str, Any]]:
    """Create skipped results for all samples."""
    return [
        _create_skipped_result(sample, i, sample.get('doc_name', 'unknown'), experiment_type)
        for i, sample in enumerate(data)
    ]


def _log_pdf_sources(pdf_source_map: Dict[str, str]) -> None:
    """Log summary of PDF sources."""
    try:
        counts = Counter(pdf_source_map.values())
        logger.info(f"PDF source summary: {dict(counts)}")
    except Exception as e:
        logger.debug(f"Failed to log PDF sources: {e}")