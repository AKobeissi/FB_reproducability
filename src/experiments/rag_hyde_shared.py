"""
HyDE and Multi-HyDE Shared Vector Experiments.
Generates hypothetical financial passages to query the shared index.
"""
import logging
from typing import Any, Dict, List

import numpy as np

from src.experiments.rag_shared_vector import _create_all_skipped_results, _create_skipped_result, _log_pdf_sources
from src.pipeline.shared_index import ensure_shared_chroma_index

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

    # --- Step 1: Ensure Shared Index (centralized helper) ---
    try:
        shared = ensure_shared_chroma_index(experiment, data)
        vectordb = shared.vectordb
        available_docs = shared.available_docs
        pdf_source_map = shared.pdf_source_map
    except Exception as e:
        logger.error("Shared index initialization failed: %s", e)
        return _create_all_skipped_results(data, mode, experiment=experiment)

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