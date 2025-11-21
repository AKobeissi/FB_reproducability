"""Closed-book experiment (no retrieval)."""
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def run_closed_book(experiment, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Run closed-book experiment without any context retrieval.
    
    Args:
        experiment: RAGExperiment instance
        data: List of samples to process
        
    Returns:
        List of result dictionaries
    """
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING CLOSED-BOOK EXPERIMENT")
    logger.info("=" * 80)

    results = []

    for i, sample in enumerate(data):
        logger.info(f"\n--- Sample {i+1}/{len(data)} ---")

        question = sample['question']
        reference_answer = sample['answer']

        # Generate answer without context
        generated_answer = experiment._generate_answer(question, context=None)

        # Evaluate generation
        eval_results = experiment.evaluator.evaluate_generation(
            generated_answer,
            reference_answer,
            question
        )

        result = {
            'sample_id': i,
            'question': question,
            'reference_answer': reference_answer,
            'generated_answer': generated_answer,
            'generation_length': len(generated_answer),
            'generation_evaluation': eval_results,
            'experiment_type': experiment.CLOSED_BOOK
        }

        results.append(result)
        logger.info(f"Completed sample {i+1}")
        if hasattr(experiment, "notify_sample_complete"):
            experiment.notify_sample_complete()

    return results