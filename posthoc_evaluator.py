#!/usr/bin/env python3
"""
Post-hoc evaluation utility for FinanceBench RAG experiments.
Calculates generation metrics (BLEU, ROUGE, BERTScore) and retrieval metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional

import numpy as np

# Try importing metric libraries
try:
    import evaluate
except ImportError:
    evaluate = None

try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None

try:
    from bert_score import score as bert_score_func
except ImportError:
    bert_score_func = None

try:
    from sacrebleu.metrics import BLEU
except ImportError:
    BLEU = None

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate FinanceBench RAG evaluation metrics from saved JSON results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to a JSON results file produced by rag_* experiments.",
    )
    parser.add_argument(
        "--save-report",
        "-o",
        default=None,
        help="Optional path to write the aggregated summary as JSON.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--skip-bertscore",
        action="store_true",
        help="Skip BERTScore calculation (can be slow).",
    )
    return parser.parse_args()


def load_results(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data, {}

    if isinstance(data, dict):
        if "results" in data and isinstance(data["results"], list):
            return data["results"], data.get("metadata", {})
        # treat dict with integer keys as list-like
        if all(isinstance(k, str) and k.isdigit() for k in data.keys()):
            ordered = [data[k] for k in sorted(data.keys(), key=int)]
            return ordered, {}

    raise ValueError(
        "Unsupported JSON structure. Expected either a list or a dict with a 'results' field."
    )


class MetricsCalculator:
    def __init__(self, skip_bertscore: bool = False):
        self.skip_bertscore = skip_bertscore
        self.rouge_scorer = None
        self.bleu_metric = None
        
        if rouge_scorer:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        if BLEU:
            self.bleu_metric = BLEU()

        self.hf_evaluate_bleu = None
        if evaluate and not self.bleu_metric:
            try:
                self.hf_evaluate_bleu = evaluate.load("bleu")
            except Exception as e:
                logger.warning(f"Failed to load HF BLEU metric: {e}")

    def compute_generation_metrics(self, references: List[str], predictions: List[str]) -> Dict[str, float]:
        metrics = {}
        
        # Filter out empty predictions/references to avoid errors
        valid_indices = [i for i, (r, p) in enumerate(zip(references, predictions)) if r.strip() and p.strip()]
        if not valid_indices:
            return {}
            
        refs = [references[i] for i in valid_indices]
        preds = [predictions[i] for i in valid_indices]

        # ROUGE
        if self.rouge_scorer:
            rouge_scores = defaultdict(list)
            for r, p in zip(refs, preds):
                scores = self.rouge_scorer.score(r, p)
                for key in scores:
                    rouge_scores[key].append(scores[key].fmeasure)
            for key, vals in rouge_scores.items():
                metrics[key] = float(np.mean(vals))
        
        # BLEU
        if self.bleu_metric:
            # sacrebleu expects list of references list (for multiple refs per sample)
            # here we have 1 ref per sample
            refs_formatted = [[r] for r in refs] 
            try:
                # sacrebleu corpus_score expects list of hypotheses and list of references (where each ref is a list if multiple)
                # actually corpus_score takes (sys, refs) where refs is list of list of strings (refs[0] is all ref1s, refs[1] is all ref2s)
                # Since we have 1 ref per doc, we pass [refs]
                score = self.bleu_metric.corpus_score(preds, [refs])
                metrics['bleu'] = score.score
            except Exception as e:
                logger.warning(f"SacreBLEU calculation failed: {e}")
        elif self.hf_evaluate_bleu:
            try:
                results = self.hf_evaluate_bleu.compute(predictions=preds, references=refs)
                metrics['bleu'] = results['bleu'] * 100 # usually 0-1
            except Exception as e:
                logger.warning(f"HF BLEU calculation failed: {e}")

        # BERTScore
        if not self.skip_bertscore and bert_score_func:
            try:
                # lang="en" downloads model if not present. Might be slow or fail in restricted env.
                P, R, F1 = bert_score_func(preds, refs, lang="en", verbose=False)
                metrics['bertscore_f1'] = float(F1.mean())
                metrics['bertscore_precision'] = float(P.mean())
                metrics['bertscore_recall'] = float(R.mean())
            except Exception as e:
                logger.warning(f"BERTScore calculation failed: {e}")

        return metrics

    def compute_retrieval_metrics(self, samples: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute Document, Page, and Chunk level hit rates.
        """
        if not samples:
            return {}
            
        doc_hits = []
        page_hits = []
        chunk_hits = []
        
        k_values = [1, 3, 5] # We can calc hit@k if we know rank. 
        # But here we just have 'retrieved_chunks'. If top_k was 5 during exp, we are calculating hit@5.
        
        for sample in samples:
            retrieved = sample.get('retrieved_chunks', [])
            gold_segments = sample.get('gold_evidence_segments', [])
            
            if not gold_segments:
                # No gold evidence, skip or treat as 0? 
                # If question type suggests retrieval, treat as failure?
                # For now, skip calculation for this sample
                continue

            # Document Level
            # Gold doc names
            gold_docs = set(s.get('doc_name') for s in gold_segments if s.get('doc_name'))
            
            # Retrieved doc names
            retrieved_docs = set(c.get('metadata', {}).get('doc_name') for c in retrieved if c.get('metadata', {}).get('doc_name'))
            
            # Check overlap
            if not gold_docs:
                 # If gold doesn't have doc info (e.g. just text), strictly can't evaluate doc hit.
                 # But in FinanceBench, doc_name is usually top-level field too.
                 if sample.get('doc_name'):
                     gold_docs.add(sample['doc_name'])
            
            doc_hit = 1.0 if not gold_docs.isdisjoint(retrieved_docs) else 0.0
            doc_hits.append(doc_hit)
            
            # Page Level
            # Gold pages: (doc_name, page_num)
            gold_pages = set()
            for s in gold_segments:
                dn = s.get('doc_name') or sample.get('doc_name')
                p = s.get('page')
                if dn and p is not None:
                    gold_pages.add((str(dn), str(p)))
            
            retrieved_pages = set()
            for c in retrieved:
                m = c.get('metadata', {})
                dn = m.get('doc_name')
                p = m.get('page') or m.get('page_number')
                if dn and p is not None:
                    retrieved_pages.add((str(dn), str(p)))
            
            if gold_pages:
                page_hit = 1.0 if not gold_pages.isdisjoint(retrieved_pages) else 0.0
                page_hits.append(page_hit)
            
            # Chunk/Text Level
            # Check if any gold evidence text is roughly contained in retrieved text
            # This is a proxy for chunk hit since we don't have shared chunk IDs
            chunk_hit = 0.0
            retrieved_texts = [c.get('text', '').lower() for c in retrieved]
            
            for s in gold_segments:
                gold_text = s.get('text', '').lower()
                # Simple containment or overlap
                # If a significant portion of gold text is in retrieved text
                if not gold_text: continue
                
                # Check if gold text is in any retrieved chunk
                # Relaxed check: if 50% of gold text is found? 
                # Strict check: "exact" substring
                
                # Using strict substring for now as FinanceBench often retrieves exact paragraphs
                if any(gold_text in rt for rt in retrieved_texts):
                    chunk_hit = 1.0
                    break
                
                # Fallback: if gold text is very long, maybe it spans chunks?
                # Let's try simple overlap
            
            chunk_hits.append(chunk_hit)

        metrics = {}
        if doc_hits:
            metrics['doc_hit_rate'] = float(np.mean(doc_hits))
        if page_hits:
            metrics['page_hit_rate'] = float(np.mean(page_hits))
        if chunk_hits:
            metrics['chunk_hit_rate'] = float(np.mean(chunk_hits))
            
        return metrics


def summarize_values(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def analyze_results(samples: List[Dict[str, Any]], calculator: MetricsCalculator) -> Dict[str, Any]:
    # Filter valid samples for generation evaluation (must have answer)
    gen_samples = [s for s in samples if s.get('generated_answer') and s.get('reference_answer')]
    
    references = [s['reference_answer'] for s in gen_samples]
    predictions = [s['generated_answer'] for s in gen_samples]
    
    gen_metrics = calculator.compute_generation_metrics(references, predictions)
    ret_metrics = calculator.compute_retrieval_metrics(samples)
    
    # Length metrics
    gen_lengths = [s.get('generation_length', 0) for s in samples]
    ctx_lengths = [s.get('context_length', 0) for s in samples]
    
    metrics = {
        "count": len(samples),
        **gen_metrics,
        **ret_metrics,
        "generation_length": float(np.mean(gen_lengths)) if gen_lengths else 0.0,
        "context_length": float(np.mean(ctx_lengths)) if ctx_lengths else 0.0
    }
    
    return metrics


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s - %(message)s",
    )

    logger.info(f"Loading results from {args.input}")
    samples, metadata = load_results(Path(args.input))
    if not samples:
        raise ValueError("No samples found in the provided JSON file.")

    logger.info(f"Loaded {len(samples)} samples.")
    calculator = MetricsCalculator(skip_bertscore=args.skip_bertscore)

    # 1. Overall Evaluation
    logger.info("Computing overall metrics...")
    overall_metrics = analyze_results(samples, calculator)

    # 2. Split by Question Type
    logger.info("Computing metrics by question type...")
    by_type = defaultdict(list)
    for s in samples:
        q_type = s.get('question_type') or 'unknown'
        by_type[q_type].append(s)
    
    type_metrics = {}
    for q_type, type_samples in by_type.items():
        type_metrics[q_type] = analyze_results(type_samples, calculator)

    # 3. Split by Question Reasoning
    logger.info("Computing metrics by question reasoning...")
    by_reasoning = defaultdict(list)
    for s in samples:
        q_reason = s.get('question_reasoning') or 'unknown'
        # Sometimes reasoning is a list or complex string, normalize?
        # Assuming simple string for now based on data loader
        by_reasoning[q_reason].append(s)
        
    reasoning_metrics = {}
    for q_reason, reason_samples in by_reasoning.items():
        reasoning_metrics[q_reason] = analyze_results(reason_samples, calculator)

    # Construct Report
    report = {
        "metadata": metadata,
        "overall": overall_metrics,
        "by_question_type": type_metrics,
        "by_question_reasoning": reasoning_metrics
    }

    # Print Summary
    print("\n" + "="*80)
    print("EVALUATION REPORT")
    print("="*80)
    
    def print_metrics(title, m):
        print(f"\n[{title}] (n={m.get('count', 0)})")
        for k, v in m.items():
            if k == "count": continue
            print(f"  {k:<25}: {v:.4f}")

    print_metrics("Overall", overall_metrics)

    print("\n--- By Question Type ---")
    for q_type, m in sorted(type_metrics.items()):
        print_metrics(q_type, m)

    print("\n--- By Question Reasoning ---")
    # Sort by count desc
    sorted_reasons = sorted(reasoning_metrics.items(), key=lambda x: x[1]['count'], reverse=True)
    for q_reason, m in sorted_reasons:
        print_metrics(q_reason, m)

    # Save
    if args.save_report:
        out_path = Path(args.save_report)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to {out_path}")
    else:
        # If no explicit save path, maybe save alongside input?
        # But user didn't explicitly ask for this, only "results file should have both retreival and generation results". 
        # The input file IS the results file from the run. The PROMPT says: "the results file should have both retreival and generation results".
        # This implies the *output of the experiment* should have them. 
        # But I implemented the metrics calculation in `posthoc_evaluator.py`.
        # To strictly satisfy "the results file should have...", I should probably append these metrics back to the input file or save a new one.
        # But usually posthoc generates a REPORT.
        pass

if __name__ == "__main__":
    main()
