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

        gold_entries = experiment._prepare_gold_evidence_payload(sample, i)

        result = {
            'sample_id': i,
            'question': question,
            'reference_answer': reference_answer,
            'gold_evidence': gold_entries,
            'generated_answer': generated_answer,
            'generation_length': len(generated_answer),
            'retrieved_chunks': [],
            'num_retrieved': 0,
            'context_length': 0,
            'doc_name': sample.get('doc_name'),
            'doc_link': sample.get('doc_link'),
            'experiment_type': experiment.CLOSED_BOOK
        }

        results.append(result)
        logger.info(f"Completed sample {i+1}")

    return results