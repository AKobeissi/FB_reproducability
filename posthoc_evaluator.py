#!/usr/bin/env python3
"""
Post-hoc evaluation utility for FinanceBench RAG experiments.
Calculates generation metrics (BLEU, ROUGE, BERTScore) and retrieval metrics.
Also computes a programmatic exact match accuracy for numeric answers.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
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
            refs_formatted = [[r] for r in refs] 
            try:
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
        
        for sample in samples:
            retrieved = sample.get('retrieved_chunks', [])
            gold_segments = sample.get('gold_evidence_segments', [])
            
            if not gold_segments:
                continue

            # Document Level
            gold_docs = set(s.get('doc_name') for s in gold_segments if s.get('doc_name'))
            retrieved_docs = set(c.get('metadata', {}).get('doc_name') for c in retrieved if c.get('metadata', {}).get('doc_name'))
            if not gold_docs:
                 if sample.get('doc_name'):
                     gold_docs.add(sample['doc_name'])
            
            doc_hit = 1.0 if not gold_docs.isdisjoint(retrieved_docs) else 0.0
            doc_hits.append(doc_hit)
            
            # Page Level
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
            chunk_hit = 0.0
            retrieved_texts = [c.get('text', '').lower() for c in retrieved]
            for s in gold_segments:
                gold_text = s.get('text', '').lower()
                if not gold_text: continue
                if any(gold_text in rt for rt in retrieved_texts):
                    chunk_hit = 1.0
                    break
            chunk_hits.append(chunk_hit)

        metrics = {}
        if doc_hits:
            metrics['doc_hit_rate'] = float(np.mean(doc_hits))
        if page_hits:
            metrics['page_hit_rate'] = float(np.mean(page_hits))
        if chunk_hits:
            metrics['chunk_hit_rate'] = float(np.mean(chunk_hits))
            
        return metrics

    def compute_numeric_accuracy(self, samples: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute correctness of numeric answers by extracting numbers from
        the reference answer and checking if they appear in the generated answer.
        Primarily useful for 'metrics-generated' questions.
        """
        if not samples:
            return {}
            
        matches = []
        valid_samples = 0
        
        for sample in samples:
            # We specifically target metrics-generated questions or just generally attempt
            # parsing if the question type is relevant.
            # However, the user request says "metrics-generated", so we can be strict or loose.
            # Being loose allows us to evaluate "domain-relevant" too if they happen to be numeric.
            
            ref = sample.get('reference_answer', '')
            gen = sample.get('generated_answer', '')
            
            if not ref or not gen:
                continue

            ref_nums = self._extract_numbers(ref)
            if not ref_nums:
                # If reference has no numbers, we skip this metric for this sample
                # or treat as N/A.
                continue
                
            valid_samples += 1
            gen_nums = self._extract_numbers(gen)
            
            # Check if all numbers in reference are present in generation (with tolerance)
            # Typically FinanceBench questions ask for a specific value.
            # If the reference is "$1,577", numbers are [1577.0].
            # If generated is "$1,577 million", numbers are [1577.0].
            # If generated is "1577", numbers are [1577.0].
            
            # We use a relaxed containment check: is the KEY reference number in the generated numbers?
            # If reference has multiple numbers (rare for single metric Q), we require all of them?
            # Or at least one?
            # Let's require ALL reference numbers to be present in the generated answer (precision).
            
            is_match = True
            for r_val in ref_nums:
                # Check if r_val matches any g_val within tolerance
                found = False
                for g_val in gen_nums:
                    if self._is_close(r_val, g_val):
                        found = True
                        break
                if not found:
                    is_match = False
                    break
            
            matches.append(1.0 if is_match else 0.0)
            
        if not matches:
            return {}
            
        return {
            "numeric_correctness": float(np.mean(matches)),
            "numeric_eval_count": len(matches)
        }

    def _extract_numbers(self, text: str) -> List[float]:
        """
        Extract numbers from text, handling commas, currency, percentages.
        Returns a list of floats.
        Example: "$1,234.56" -> [1234.56]
                 "5.1%" -> [5.1]
        """
        # Remove commas
        clean_text = text.replace(',', '')
        # Regex for float-like numbers
        # Matches: 123, -123, 123.45, .45
        pattern = r"[-+]?\d*\.\d+|[-+]?\d+"
        matches = re.findall(pattern, clean_text)
        
        nums = []
        for m in matches:
            try:
                # remove typical non-numeric chars if any slipped through (like trailing .)
                val = float(m)
                nums.append(val)
            except ValueError:
                pass
        return nums

    def _is_close(self, a: float, b: float, tol: float = 0.01) -> bool:
        """Check if two floats are close within a percentage tolerance."""
        if a == 0:
            return abs(b) < 1e-6
        # Allow 1% error margin
        return abs(a - b) / abs(a) <= tol


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
    num_metrics = calculator.compute_numeric_accuracy(samples)
    
    # Length metrics
    gen_lengths = [s.get('generation_length', 0) for s in samples]
    ctx_lengths = [s.get('context_length', 0) for s in samples]
    
    metrics = {
        "count": len(samples),
        **gen_metrics,
        **ret_metrics,
        **num_metrics,
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

if __name__ == "__main__":
    main()
