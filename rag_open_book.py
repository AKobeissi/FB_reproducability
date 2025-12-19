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
        doc_name = sample.get('doc_name')
        doc_link = sample.get('doc_link')
        reference_answer = sample['answer']
        gold_segments, context = experiment._prepare_gold_evidence(sample.get('evidence', ''))

        logger.info(f"Using gold evidence (length: {len(context)} chars)")

        # Generate answer with gold evidence
        generated_answer, final_prompt = experiment._generate_answer(question, context, return_prompt=True)

        result = {
            'sample_id': i,
            'doc_name': doc_name,
            'doc_link': doc_link,
            'question': question,
            'reference_answer': reference_answer,
            'question_type': sample.get('question_type'),
            'question_reasoning': sample.get('question_reasoning'),
            'gold_evidence': context,
            'gold_evidence_segments': gold_segments,
            'context_length': len(context),
            'generated_answer': generated_answer,
            'generation_length': len(generated_answer),
            'experiment_type': experiment.OPEN_BOOK,
            'final_prompt': final_prompt,
        }

        results.append(result)
        logger.info(f"Completed sample {i+1}")
        if hasattr(experiment, "notify_sample_complete"):
            experiment.notify_sample_complete()

    return results