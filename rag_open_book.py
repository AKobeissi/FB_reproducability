"""Open-book experiment using gold evidence."""
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def run_open_book(experiment, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Run open-book experiment using gold evidence as context.
    
    Args:
        experiment: RAGExperiment instance
        data: List of samples to process
        
    Returns:
        List of result dictionaries
    """
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING OPEN-BOOK EXPERIMENT (Gold Evidence)")
    logger.info("=" * 80)

    results = []

    for i, sample in enumerate(data):
        logger.info(f"\n--- Sample {i+1}/{len(data)} ---")

        question = sample['question']
        reference_answer = sample['answer']
        gold_segments, context = experiment._prepare_gold_evidence(sample.get('evidence', ''))
        gold_context_texts = [seg.get('text') for seg in gold_segments if seg.get('text')]

        logger.info(f"Using gold evidence (length: {len(context)} chars)")

        # Generate answer with gold evidence
        generated_answer, final_prompt = experiment._generate_answer(question, context, return_prompt=True)

        # Evaluate generation (no retrieval metrics for oracle test)
        generation_eval = experiment.evaluator.evaluate_generation(
            generated_answer,
            reference_answer,
            question,
            contexts=gold_context_texts,
            gold_contexts=gold_context_texts,
            langchain_llm=getattr(experiment, "langchain_llm", None),
        )

        result = {
            'sample_id': i,
            'question': question,
            'reference_answer': reference_answer,
            'gold_evidence': context,
            'gold_evidence_segments': gold_segments,
            'context_length': len(context),
            'generated_answer': generated_answer,
            'generation_length': len(generated_answer),
            'generation_evaluation': generation_eval,
            'experiment_type': experiment.OPEN_BOOK,
            'final_prompt': final_prompt,
        }

        results.append(result)
        logger.info(f"Completed sample {i+1}")
        if hasattr(experiment, "notify_sample_complete"):
            experiment.notify_sample_complete()

    return results