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
        gold_entries = experiment._prepare_gold_evidence_payload(sample, i)
        context = experiment._gold_context_from_entries(gold_entries)

        logger.info(f"Using gold evidence (length: {len(context)} chars)")

        # Generate answer with gold evidence
        generated_answer = experiment._generate_answer(question, context)

        result = {
            'sample_id': i,
            'question': question,
            'reference_answer': reference_answer,
            'gold_evidence': gold_entries,
            'context_length': len(context),
            'generated_answer': generated_answer,
            'generation_length': len(generated_answer),
            'num_retrieved': 0,
            'retrieved_chunks': [],
            'experiment_type': experiment.OPEN_BOOK
        }

        results.append(result)
        logger.info(f"Completed sample {i+1}")

    return results