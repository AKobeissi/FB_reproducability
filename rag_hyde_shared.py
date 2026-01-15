"""
HyDE and Multi-HyDE Shared Vector Experiments.
Generates hypothetical financial passages to query the shared index.
"""
from typing import List, Dict, Any
import logging
import os
import json
import numpy as np

try:
    from .pdf_utils import load_pdf_with_fallback
    from .vectorstore import build_chroma_store, populate_chroma_store, save_store_config, get_chroma_db_path
    from .rag_shared_vector import (
        _get_or_create_retrieval_prompt, 
        _get_or_create_documents_chain, 
        _build_chunks_from_docs, 
        _fallback_retrieval_qa,
        _log_pdf_sources,
        _create_skipped_result,
        _create_all_skipped_results
    )
except ImportError:
    from pdf_utils import load_pdf_with_fallback
    from vectorstore import build_chroma_store, populate_chroma_store, save_store_config, get_chroma_db_path
    from rag_shared_vector import (
        _get_or_create_retrieval_prompt, 
        _get_or_create_documents_chain, 
        _build_chunks_from_docs, 
        _fallback_retrieval_qa,
        _log_pdf_sources,
        _create_skipped_result,
        _create_all_skipped_results
    )

logger = logging.getLogger(__name__)

def run_hyde_shared(experiment, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run standard HyDE (1 generation)."""
    return _run_hyde_base(experiment, data, mode="hyde", n_generations=1)

def run_multi_hyde_shared(experiment, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run Multi-HyDE (4 generations + vector averaging)."""
    return _run_hyde_base(experiment, data, mode="multi_hyde", n_generations=4)

def _run_hyde_base(experiment, data, mode, n_generations):
    logger.info("\n" + "=" * 80)
    logger.info(f"RUNNING {mode.upper()} EXPERIMENT ({n_generations} generation(s))")
    logger.info("=" * 80)

    # --- Step 1: Initialize Shared Index (Standard Logic) ---
    unique_docs = {}
    if getattr(experiment, "use_all_pdfs", False) and getattr(experiment, "pdf_local_dir", None):
        if os.path.exists(str(experiment.pdf_local_dir)):
            import glob
            p = str(experiment.pdf_local_dir)
            for f in glob.glob(os.path.join(p, "**", "*.pdf"), recursive=True):
                unique_docs[os.path.splitext(os.path.basename(f))[0]] = ""

    for sample in data:
        unique_docs.setdefault(sample.get('doc_name', 'unknown'), sample.get('doc_link', ''))

    try:
        # We need the vectordb object explicitly for vector search
        retriever, vectordb, is_new = build_chroma_store(experiment, "all", lazy_load=True)
    except Exception as e:
        logger.error("Chroma build failed: %s", e)
        return _create_all_skipped_results(data, mode, experiment=experiment)

    # Ingestion Logic (Standard shared store logic)
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
        except Exception: pass

    docs_to_process = {k: v for k, v in unique_docs.items() if k not in available_docs}
    if docs_to_process:
        logger.info(f"Ingesting {len(docs_to_process)} missing documents...")
        for doc_name, doc_link in docs_to_process.items():
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

    # --- Step 2: Process Samples with HyDE ---
    results = []
    for i, sample in enumerate(data):
        doc_name = sample.get('doc_name', 'unknown')
        if doc_name not in available_docs:
            results.append(_create_skipped_result(sample, i, doc_name, mode, experiment=experiment))
            continue

        logger.info(f"\n--- Sample {i+1}/{len(data)} ---")
        question = sample.get('question', '')
        
        # 1. Generate Hypothetical Document(s)
        hypotheticals = _generate_hypotheticals(experiment, question, n=n_generations)
        logger.info(f"Generated {len(hypotheticals)} hypothetical passage(s).")
        if hypotheticals:
            logger.debug(f"Hypothesis 1: {hypotheticals[0][:100]}...")

        # 2. Retrieve using HyDE (Robust Method)
        retrieved_chunks = _retrieve_with_hyde(experiment, vectordb, hypotheticals)
        
        # 3. Generate Final Answer (using Real Chunks + Original Question)
        context = "\n\n".join([c['text'] for c in retrieved_chunks])
        prompt = experiment._build_financebench_prompt(question, context, mode=mode)
        
        # Determine answer logic
        if experiment.use_api:
            ans = experiment._generate_answer(question, context)
        else:
            # We use the prompt built above
            out = experiment.llm_pipeline(prompt, max_new_tokens=experiment.max_new_tokens)
            ans = out[0]["generated_text"]
        
        ans = ans.strip()
        
        # 4. Metrics & Saving
        gold_segs, gold_str = experiment._prepare_gold_evidence(sample.get('evidence', ''))
        results.append({
            'sample_id': i,
            'doc_name': doc_name,
            'question': question,
            'hypothetical_passages': hypotheticals, 
            'reference_answer': sample.get('answer', ''),
            'generated_answer': ans,
            'retrieved_chunks': retrieved_chunks,
            'retrieved_from_docs': [{'doc_name': c['metadata'].get('doc_name'), 'score': c.get('score')} for c in retrieved_chunks],
            
            # Required Mixin Fields
            'question_type': sample.get('question_type'),
            'question_reasoning': sample.get('question_reasoning'),
            'gold_evidence': gold_str,
            'gold_evidence_segments': gold_segs,
            'num_retrieved': len(retrieved_chunks),
            'context_length': len(context),
            'generation_length': len(ans),
            'experiment_type': mode,
            'final_prompt': prompt,
            'pdf_source': pdf_source_map.get(doc_name, 'none')
        })
        logger.info(f"Completed sample {i+1}")

    return results

def _generate_hypotheticals(experiment, question: str, n: int) -> List[str]:
    """Generates N hypothetical financial passages answering the question."""
    # Finance-specific HyDE prompt
    hyde_prompt = (
        f"Please write a short financial report excerpt (1 paragraph) that precisely answers the question below. "
        f"Do not introduce the text, just write the excerpt.\n\n"
        f"Question: {question}\n\n"
        f"Passage:"
    )
    
    experiment.ensure_langchain_llm()
    hypotheticals = []
    
    for _ in range(n):
        try:
            if experiment.use_api:
                # Use the chat client directly for speed
                msgs = [{"role": "user", "content": hyde_prompt}]
                resp = experiment.api_client.chat.completions.create(
                    model=experiment.llm_model_name,
                    messages=msgs,
                    max_tokens=200,
                    temperature=0.7 # Need creativity for HyDE
                )
                text = resp.choices[0].message.content
            else:
                # Use local pipeline
                # Force sampling for diversity if n > 1
                do_sample = (n > 1) 
                outs = experiment.llm_pipeline(
                    hyde_prompt, 
                    max_new_tokens=200, 
                    do_sample=do_sample, 
                    temperature=0.7 if do_sample else None
                )
                text = outs[0]["generated_text"]
            
            # Strip prompt if pipeline returned it (common in some HF pipelines)
            if text.startswith(hyde_prompt):
                text = text[len(hyde_prompt):]
            
            hypotheticals.append(text.strip())
            
        except Exception as e:
            logger.warning(f"HyDE generation failed: {e}")
            hypotheticals.append(question) # Fallback to original question
            
    return hypotheticals

def _retrieve_with_hyde(experiment, vectordb, hypotheticals: List[str]):
    """Embeds hypotheticals -> Averages Vectors -> Searches Chroma."""
    try:
        # 1. Embed all hypotheticals
        embeddings = experiment.embeddings.embed_documents(hypotheticals)
        
        # 2. Average the vectors (Multi-HyDE Logic)
        if len(embeddings) > 1:
            avg_vector = np.mean(embeddings, axis=0).tolist()
        else:
            avg_vector = embeddings[0]
            
        # 3. Search using the vector directly
        # Attempt standard method
        try:
            docs_with_scores = vectordb.similarity_search_by_vector_with_score(
                avg_vector, 
                k=experiment.top_k
            )
        except AttributeError:
            # Fallback 1: Try without scores (older LangChain versions)
            # We assign a dummy score of 0.0 since HyDE is just for retrieval
            logger.debug("Falling back to similarity_search_by_vector (no scores).")
            docs = vectordb.similarity_search_by_vector(
                avg_vector,
                k=experiment.top_k
            )
            docs_with_scores = [(d, 0.0) for d in docs]
        
        # 4. Format chunks
        chunks = []
        for rank, (doc, score) in enumerate(docs_with_scores):
            chunks.append({
                'rank': rank + 1,
                'text': doc.page_content,
                'score': float(score), 
                'metadata': doc.metadata or {}
            })
        return chunks

    except Exception as e:
        logger.error(f"HyDE retrieval failed: {e}")
        return []