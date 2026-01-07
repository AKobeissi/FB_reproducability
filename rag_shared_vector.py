from typing import List, Dict, Any
import logging
import os
import json
from collections import Counter
try:
    from .pdf_utils import load_pdf_with_fallback
    from .vectorstore import build_chroma_store, populate_chroma_store, save_store_config, get_chroma_db_path
except ImportError:
    from pdf_utils import load_pdf_with_fallback
    from vectorstore import build_chroma_store, populate_chroma_store, save_store_config, get_chroma_db_path

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


def run_shared_vector(experiment, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Run shared vector store experiment (single FAISS index for all documents).
    """
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING SHARED VECTOR STORE EXPERIMENT (LangChain + Chroma preferred)")
    logger.info("=" * 80)

    # Collect unique documents
    unique_docs = {}
    
    # Option to include all PDFs from the local directory
    if getattr(experiment, "use_all_pdfs", False) and getattr(experiment, "pdf_local_dir", None):
        pdf_dir = str(experiment.pdf_local_dir)
        logger.info(f"Scanning {pdf_dir} for all PDF documents...")
        if os.path.exists(pdf_dir):
            import glob
            # Match both .pdf and .PDF
            pdf_files = glob.glob(os.path.join(pdf_dir, "**", "*.pdf"), recursive=True)
            pdf_files += glob.glob(os.path.join(pdf_dir, "**", "*.PDF"), recursive=True)
            
            logger.info(f"Found {len(pdf_files)} PDF files in local directory.")
            
            for pdf_path in pdf_files:
                # Use filename stem as doc_name (simplification, but aligns with _find_local_pdf logic)
                filename = os.path.basename(pdf_path)
                doc_name = os.path.splitext(filename)[0]
                # Store with empty link as we have local file
                unique_docs[doc_name] = ""
        else:
            logger.warning(f"PDF directory {pdf_dir} does not exist.")

    for sample in data:
        doc_name = sample.get('doc_name', 'unknown')
        if doc_name not in unique_docs:
            unique_docs[doc_name] = sample.get('doc_link', '')


    logger.info(f"Collected {len(unique_docs)} unique documents")

    # Initialize Chroma store (lazy load)
    try:
        retriever, vectordb, is_new = build_chroma_store(
            experiment,
            "all",
            lazy_load=True,
        )
    except Exception as e:
        logger.error("Chroma build failed for shared store: %s", e)
        skipped = _create_all_skipped_results(data, experiment.SHARED_VECTOR, experiment=experiment)
        if hasattr(experiment, "notify_sample_complete"):
            experiment.notify_sample_complete(count=len(skipped), note="shared store skipped (vector store)")
        return skipped

    # Define paths for metadata
    db_name, db_path = get_chroma_db_path(experiment, "all")
    meta_path = os.path.join(db_path, "shared_meta.json")

    available_docs = set()
    pdf_source_map = {}

    if not is_new:
        logger.info("Using existing shared vector store.")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                    available_docs = set(meta.get("available_docs", []))
                    pdf_source_map = meta.get("pdf_source_map", {})
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")

    # Identify documents that need to be ingested
    docs_to_process = {k: v for k, v in unique_docs.items() if k not in available_docs}

    if docs_to_process:
        if is_new:
            logger.info(f"Ingesting {len(docs_to_process)} documents (new store)...")
        else:
            logger.info(f"Ingesting {len(docs_to_process)} missing documents (incremental update)...")
        
        for i, (doc_name, doc_link) in enumerate(docs_to_process.items()):
            logger.info(f"\nProcessing document {i+1}/{len(docs_to_process)}: {doc_name}")
            
            pdf_docs, pdf_source = load_pdf_with_fallback(
                doc_name=doc_name,
                doc_link=doc_link,
                local_dir=getattr(experiment, 'pdf_local_dir', None),
            )
            
            pdf_source_map[doc_name] = pdf_source

            if not pdf_docs:
                logger.warning(f"No PDF pages for '{doc_name}'. Excluding from shared store.")
                continue

            logger.info(f"Chunking (from {pdf_source}): {doc_name}")
            chunks = experiment._chunk_text_langchain(
                pdf_docs,
                metadata={
                    'doc_name': doc_name,
                    'source': 'pdf',
                    'doc_link': doc_link,
                    'pdf_source': pdf_source
                }
            )
            
            if chunks:
                added = populate_chroma_store(experiment, vectordb, chunks, db_name)
                logger.info(f"Added {added} chunks for '{doc_name}'")
                available_docs.add(doc_name)
            
            # Clear memory
            del pdf_docs
            del chunks
        
        # Persist metadata and config
        save_store_config(experiment, db_path)
        try:
            with open(meta_path, 'w') as f:
                json.dump({
                    "available_docs": list(available_docs),
                    "pdf_source_map": pdf_source_map
                }, f)
        except Exception as e:
            logger.warning(f"Failed to save metadata to {meta_path}: {e}")

    _log_pdf_sources(pdf_source_map)

    # Process samples
    results = []
    for i, sample in enumerate(data):
        logger.info(f"\n--- Sample {i+1}/{len(data)} ---")
        
        doc_name = sample.get('doc_name', 'unknown')
        
        if doc_name not in available_docs:
            logger.info(f"Skipping sample for '{doc_name}' (no PDF text)")
            skipped = _create_skipped_result(
                sample, i, doc_name, experiment.SHARED_VECTOR, experiment=experiment
            )
            results.append(skipped)
            if hasattr(experiment, "notify_sample_complete"):
                experiment.notify_sample_complete(note=f"{doc_name} skipped (no pdf)")
            continue

        result = _process_shared_sample(
            experiment=experiment,
            sample=sample,
            sample_id=i,
            doc_name=doc_name,
            pdf_source=pdf_source_map.get(doc_name, 'none'),
            retriever=retriever,
        )
        
        results.append(result)
        logger.info(f"Completed sample {i+1}")
        if hasattr(experiment, "notify_sample_complete"):
            experiment.notify_sample_complete(note=f"{doc_name}")

    return results


def _process_shared_sample(
    experiment,
    sample: Dict[str, Any],
    sample_id: int,
    doc_name: str,
    pdf_source: str,
    retriever,
) -> Dict[str, Any]:
    """Process a single sample in shared vector store mode."""
    question = sample.get('question', '')
    reference_answer = sample.get('answer', '')
    gold_segments, gold_evidence_str = experiment._prepare_gold_evidence(sample.get('evidence', ''))

    generated_answer, retrieved_chunks, prompt_snapshot = _run_retrieval_qa(
        experiment=experiment,
        question=question,
        retriever=retriever
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

    return {
        'sample_id': sample_id,
        'doc_name': doc_name,
        'question': question,
        'reference_answer': reference_answer,
        'question_type': sample.get('question_type'),
        'question_reasoning': sample.get('question_reasoning'),
        'gold_evidence': gold_evidence_str,
        'retrieved_chunks': retrieved_chunks,
        'gold_evidence_segments': gold_segments,
        'retrieved_from_docs': source_docs,
        'num_retrieved': len(retrieved_chunks),
        'context_length': len(context),
        'generated_answer': generated_answer,
        'generation_length': len(generated_answer),
        'experiment_type': experiment.SHARED_VECTOR,
        'vector_store_type': 'Chroma',
        'pdf_source': pdf_source,
        'final_prompt': prompt_snapshot,
    }
 
 
def _run_retrieval_qa(experiment, question: str, retriever) -> tuple[str, List[Dict[str, Any]], str]:
    """Run RetrievalQA using the experiment's LangChain LLM wrapper."""
    if retriever is None:
        return "", [], ""

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
                context_text = "\n\n".join(chunk['text'] for chunk in chunks)
                prompt_snapshot = experiment._build_financebench_prompt(
                    question,
                    context_text,
                    mode=experiment.experiment_type,
                )
                return str(answer).strip(), chunks, prompt_snapshot
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
            context_text = "\n\n".join(chunk['text'] for chunk in chunks)
            prompt_snapshot = experiment._build_financebench_prompt(
                question,
                context_text,
                mode=experiment.experiment_type,
            )
            return answer.strip(), chunks, prompt_snapshot
        except Exception as exc:
            logger.warning("RetrievalQA chain failed (%s); falling back to manual retrieval.", exc)

    return _fallback_retrieval_qa(experiment, question, retriever)


def _get_doc_text(doc) -> str:
    """Extract text from document object."""
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
            "Keep answers concise (<=3 sentences) and cite only the supplied context."
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
        empty_prompt = experiment._build_financebench_prompt(question, "", mode=experiment.experiment_type)
        return "", [], empty_prompt

    chunks = _build_chunks_from_docs(documents)

    context = "\n\n".join(chunk['text'] for chunk in chunks)
    answer, prompt = experiment._generate_answer(question, context, return_prompt=True)
    if not prompt:
        prompt = experiment._build_financebench_prompt(question, context, mode=experiment.experiment_type)
    return answer, chunks, prompt


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


def _create_skipped_result(
    sample: Dict[str, Any],
    sample_id: int,
    doc_name: str,
    experiment_type: str,
    experiment=None,
) -> Dict[str, Any]:
    """Create a skipped result entry."""
    prompt = ""
    if experiment is not None:
        prompt = experiment._build_financebench_prompt(sample.get('question', ''), "", mode=experiment_type)
    return {
        'sample_id': sample_id,
        'doc_name': doc_name,
        'doc_link': sample.get('doc_link', ''),
        'question': sample.get('question', ''),
        'reference_answer': sample.get('answer', ''),
        'question_type': sample.get('question_type'),
        'question_reasoning': sample.get('question_reasoning'),
        'gold_evidence': '',
        'gold_evidence_segments': [],
        'retrieved_chunks': [],
        'retrieved_from_docs': [],
        'num_retrieved': 0,
        'context_length': 0,
        'generated_answer': '',
        'generation_length': 0,
        'retrieval_evaluation': {},
        'generation_evaluation': {},
        'experiment_type': experiment_type,
        'vector_store_type': 'Chroma',
        'skipped': True,
        'skipped_reason': 'no_pdf_text',
        'final_prompt': prompt
    }


def _create_all_skipped_results(
    data: List[Dict[str, Any]],
    experiment_type: str,
    experiment=None,
) -> List[Dict[str, Any]]:
    """Create skipped results for all samples."""
    return [
        _create_skipped_result(
            sample,
            i,
            sample.get('doc_name', 'unknown'),
            experiment_type,
            experiment=experiment,
        )
        for i, sample in enumerate(data)
    ]


def _log_pdf_sources(pdf_source_map: Dict[str, str]) -> None:
    """Log summary of PDF sources."""
    try:
        counts = Counter(pdf_source_map.values())
        logger.info(f"PDF source summary: {dict(counts)}")
    except Exception as e:
        logger.debug(f"Failed to log PDF sources: {e}")
