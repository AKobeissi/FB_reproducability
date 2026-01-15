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
        generated_answer, final_prompt = experiment._generate_answer(question, context=None, return_prompt=True)

        result = {
            'sample_id': i,
            'doc_name': sample.get('doc_name'),
            'doc_link': sample.get('doc_link'),
            'question': question,
            'reference_answer': reference_answer,
            'question_type': sample.get('question_type'),
            'question_reasoning': sample.get('question_reasoning'),
            'generated_answer': generated_answer,
            'generation_length': len(generated_answer),
            'experiment_type': experiment.CLOSED_BOOK,
            'final_prompt': final_prompt,
            'gold_evidence_segments': [],
        }

        results.append(result)
        logger.info(f"Completed sample {i+1}")
        if hasattr(experiment, "notify_sample_complete"):
            experiment.notify_sample_complete()

    return results