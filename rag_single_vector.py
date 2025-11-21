from typing import List, Dict, Any
import logging
from collections import defaultdict
from .pdf_utils import load_pdf_with_fallback
from .vectorstore import build_chroma_store

try:
    from langchain.chains import RetrievalQA
except Exception:
    RetrievalQA = None

# Modern LangChain retrieval chain utilities (preferred in >=0.1.17)
create_retrieval_chain = None
create_stuff_documents_chain = None
try:
    from langchain.chains import create_retrieval_chain as _create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain as _create_stuff_documents_chain
    create_retrieval_chain = _create_retrieval_chain
    create_stuff_documents_chain = _create_stuff_documents_chain
except Exception:
    pass

ChatPromptTemplate = None
try:
    from langchain_core.prompts import ChatPromptTemplate as _ChatPromptTemplate
    ChatPromptTemplate = _ChatPromptTemplate
except Exception:
    try:
        from langchain.prompts import ChatPromptTemplate as _ChatPromptTemplate
        ChatPromptTemplate = _ChatPromptTemplate
    except Exception:
        ChatPromptTemplate = None

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
            skipped = _create_skipped_results(
                samples, doc_name, doc_link, pdf_source, 
                experiment.SINGLE_VECTOR, len(results)
            )
            results.extend(skipped)
            if hasattr(experiment, "notify_sample_complete"):
                experiment.notify_sample_complete(count=len(skipped), note=f"{doc_name} skipped (no pdf)")
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
            skipped = _create_skipped_results(
                samples, doc_name, doc_link, pdf_source,
                experiment.SINGLE_VECTOR, len(results)
            )
            results.extend(skipped)
            if hasattr(experiment, "notify_sample_complete"):
                experiment.notify_sample_complete(count=len(skipped), note=f"{doc_name} skipped (vector store)")
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
            if hasattr(experiment, "notify_sample_complete"):
                experiment.notify_sample_complete(note=f"{doc_name}")

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
    if retriever is None:
        return "", []

    if getattr(experiment, "use_api", False):
        return _fallback_retrieval_qa(experiment, question, retriever)

    experiment.ensure_langchain_llm()

    if _supports_modern_retrieval():
        try:
            prompt = _get_or_create_retrieval_prompt(experiment)
            doc_chain = _get_or_create_documents_chain(experiment, prompt)
            if prompt is not None and doc_chain is not None:
                chain = create_retrieval_chain(retriever, doc_chain)
                result = chain.invoke({"input": question})
                source_docs = (
                    result.get("context")
                    or result.get("source_documents")
                    or []
                )
                chunks = _build_chunks_from_docs(source_docs)
                answer = result.get("answer") or result.get("result") or ""
                if isinstance(answer, dict):
                    answer = answer.get("answer") or answer.get("result") or ""
                if not answer and isinstance(result, str):
                    answer = result
                return str(answer).strip(), chunks
        except Exception as exc:
            logger.warning("Modern retrieval chain failed (%s); falling back to legacy chain.", exc)

    if RetrievalQA is not None:
        try:
            qa = RetrievalQA.from_chain_type(
                llm=experiment.langchain_llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
            )

            result = qa(question)
            source_docs = result.get('source_documents', []) or []
            chunks = _build_chunks_from_docs(source_docs)
            answer = result.get('result', '') or ""
            return answer.strip(), chunks
        except Exception as exc:
            logger.warning("RetrievalQA chain failed (%s); falling back to manual retrieval.", exc)

    return _fallback_retrieval_qa(experiment, question, retriever)


def _get_doc_text(doc) -> str:
    """Extract text content from document object."""
    return (getattr(doc, 'page_content', None) or 
            getattr(doc, 'content', None) or 
            str(doc))


def _supports_modern_retrieval() -> bool:
    return all([
        create_retrieval_chain is not None,
        create_stuff_documents_chain is not None,
        ChatPromptTemplate is not None,
    ])


def _get_or_create_retrieval_prompt(experiment):
    if ChatPromptTemplate is None:
        return None
    prompt = getattr(experiment, "_lc_retrieval_prompt", None)
    if prompt is None:
        system_prompt = (
            "Use the provided context to answer the user's question. "
            "If the context is insufficient, reply with \"I don't know.\" "
            "Keep answers concise (<=3 sentences) and reference only the supplied context."
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt + " Context: {context}"),
                ("human", "{input}"),
            ]
        )
        experiment._lc_retrieval_prompt = prompt
    return prompt


def _get_or_create_documents_chain(experiment, prompt):
    if prompt is None or create_stuff_documents_chain is None:
        return None
    doc_chain = getattr(experiment, "_lc_documents_chain", None)
    doc_chain_llm = getattr(experiment, "_lc_documents_chain_llm", None)
    if doc_chain is None or doc_chain_llm is not experiment.langchain_llm:
        doc_chain = create_stuff_documents_chain(
            experiment.langchain_llm,
            prompt,
        )
        experiment._lc_documents_chain = doc_chain
        experiment._lc_documents_chain_llm = experiment.langchain_llm
    return doc_chain


def _build_chunks_from_docs(documents):
    chunks = []
    for rank, doc in enumerate(documents or []):
        text = _get_doc_text(doc)
        if not text:
            continue
        chunks.append(
            {
                'rank': rank + 1,
                'text': text,
                'score': getattr(doc, 'score', None),
                'length': len(text),
                'metadata': getattr(doc, 'metadata', {}) or {}
            }
        )
    return chunks


def _fallback_retrieval_qa(experiment, question: str, retriever):
    documents = _invoke_retriever(retriever, question)
    if not documents:
        logger.warning("Retriever returned no documents during fallback path.")
        return "", []

    chunks = _build_chunks_from_docs(documents)

    context = "\n\n".join(chunk['text'] for chunk in chunks)
    answer = experiment._generate_answer(question, context)
    return answer, chunks


def _invoke_retriever(retriever, question: str):
    for method_name in ("get_relevant_documents", "invoke"):
        method = getattr(retriever, method_name, None)
        if callable(method):
            try:
                docs = method(question)
                docs = _normalize_retriever_output(docs)
                if docs:
                    return docs
            except Exception as exc:
                logger.debug("Retriever method '%s' failed: %s", method_name, exc)
    try:
        docs = retriever(question)
        docs = _normalize_retriever_output(docs)
        if docs:
            return docs
    except Exception as exc:
        logger.debug("Retriever call failed: %s", exc)
    return []


def _normalize_retriever_output(docs):
    if docs is None:
        return []
    if isinstance(docs, dict):
        for key in ("documents", "source_documents", "result"):
            value = docs.get(key)
            if value:
                return value
        return []
    if isinstance(docs, list):
        return docs
    return [docs]


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