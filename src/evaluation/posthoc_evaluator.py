#!/usr/bin/env python3
"""
Post-hoc evaluation utility for FinanceBench RAG experiments.
Injects ground-truth metadata, calculates per-sample metrics,
and aggregates results with Mean and Standard Deviation.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

# --- Optional Metric Libraries ---
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
        description="Aggregate FinanceBench RAG evaluation metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to a JSON results file produced by rag_* experiments.",
    )
    parser.add_argument(
        "--dataset", "-d", 
        default="data/financebench_open_source.jsonl",
        help="Path to the original FinanceBench dataset (for metadata injection).",
    )
    parser.add_argument(
        "--save-report", "-o", default=None,
        help="Path to write the full JSON report.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable debug logging."
    )
    parser.add_argument(
        "--skip-bertscore", action="store_true", help="Skip BERTScore calculation."
    )
    return parser.parse_args()

def load_jsonl(path: Path) -> List[Dict]:
    data = []
    if not path.exists():
        logger.warning(f"Dataset file not found at {path}. Metadata injection will be skipped.")
        return []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

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
        # handle dict with integer keys
        if all(isinstance(k, str) and k.isdigit() for k in data.keys()):
            ordered = [data[k] for k in sorted(data.keys(), key=int)]
            return ordered, {}
    raise ValueError("Unsupported JSON structure.")

def normalize_text(s: str) -> str:
    """Lower text and remove punctuation/extra whitespace."""
    s = str(s).lower()
    s = re.sub(r'[^\w\s]', '', s)
    return " ".join(s.split())

def extract_numbers(text: str) -> List[float]:
    """Extracts numbers from text, handling commas and currency symbols."""
    clean_text = re.sub(r'[,$]', '', str(text))
    matches = re.findall(r'-?\d*\.?\d+', clean_text)
    nums = []
    for m in matches:
        try:
            if m in {'.', '-'}:
                continue
            nums.append(float(m))
        except ValueError:
            pass
    return nums

def check_numeric_match(reference: str, prediction: str) -> float:
    """
    Checks if numbers in reference appear in prediction.
    Returns 1.0 if the key number from reference is found in prediction.
    """
    ref_nums = extract_numbers(reference)
    pred_nums = extract_numbers(prediction)
    
    if not ref_nums:
        return 0.0 
    
    # Check if *any* number in Ref exists in Pred with small tolerance
    for r_num in ref_nums:
        for p_num in pred_nums:
            if np.isclose(r_num, p_num, atol=0.01) or np.isclose(r_num, p_num, rtol=0.05):
                return 1.0
    return 0.0

def token_overlap(text1: str, text2: str) -> float:
    tokens1 = set(normalize_text(text1).split())
    tokens2 = set(normalize_text(text2).split())
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1.intersection(tokens2)
    denominator = len(tokens2) 
    return len(intersection) / denominator if denominator > 0 else 0.0

class MetricsCalculator:
    def __init__(self, skip_bertscore: bool = False):
        self.skip_bertscore = skip_bertscore
        self.r_scorer = None
        if rouge_scorer:
            self.r_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        self.bleu_metric = None
        if BLEU:
            self.bleu_metric = BLEU()

    def compute_text_metrics(self, reference: str, prediction: str, q_type: str) -> Dict[str, float]:
        metrics = {
            "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "bleu": 0.0, 
            "numeric_match": 0.0
        }
        if not reference or not prediction:
            return metrics

        # ROUGE
        if self.r_scorer:
            scores = self.r_scorer.score(reference, prediction)
            metrics["rouge1"] = scores["rouge1"].fmeasure
            metrics["rouge2"] = scores["rouge2"].fmeasure
            metrics["rougeL"] = scores["rougeL"].fmeasure

        # BLEU
        if self.bleu_metric:
            try:
                res = self.bleu_metric.sentence_score(prediction, [reference])
                metrics["bleu"] = res.score
            except Exception:
                pass

        # Numeric Match
        if "metrics" in q_type.lower() or "generated" in q_type.lower():
            metrics["numeric_match"] = check_numeric_match(reference, prediction)
        
        return metrics

    def compute_retrieval_metrics(self, retrieved: List[Dict], gold_segments: List[Dict], ref_answer: str) -> Dict[str, float]:
        metrics = {
            "doc_hit": 0.0,
            "page_hit": 0.0,
            "recall_evidence_70": 0.0,
            "recall_answer_70": 0.0
        }
        
        if not gold_segments:
            return metrics

        # Sets for Hit calculations
        gold_docs = set()
        gold_pages = set()
        
        for s in gold_segments:
            dn = s.get('doc_name')
            p = s.get('page')
            if dn:
                gold_docs.add(dn)
                if p is not None:
                    gold_pages.add((str(dn), str(p)))

        retrieved_docs = set()
        retrieved_pages = set()
        
        for c in retrieved:
            m = c.get('metadata', {})
            dn = m.get('doc_name')
            p = m.get('page') or m.get('page_number')
            
            if dn:
                retrieved_docs.add(dn)
                if p is not None:
                    retrieved_pages.add((str(dn), str(p)))

        # Doc Hit
        if not gold_docs.isdisjoint(retrieved_docs):
            metrics["doc_hit"] = 1.0

        # Page Hit
        if not gold_pages.isdisjoint(retrieved_pages):
            metrics["page_hit"] = 1.0

        # Chunk Recall > 70% on Evidence
        evidence_hit = False
        for s in gold_segments:
            gold_text = s.get('text') or s.get('evidence_text') or ""
            if not gold_text:
                continue
            for c in retrieved:
                chunk_text = c.get('text') or c.get('page_content') or ""
                if token_overlap(chunk_text, gold_text) >= 0.70:
                    evidence_hit = True
                    break
            if evidence_hit:
                break
        metrics["recall_evidence_70"] = 1.0 if evidence_hit else 0.0

        # Chunk Recall > 70% on Answer
        answer_hit = False
        if ref_answer:
            for c in retrieved:
                chunk_text = c.get('text') or c.get('page_content') or ""
                if token_overlap(chunk_text, ref_answer) >= 0.70:
                    answer_hit = True
                    break
        metrics["recall_answer_70"] = 1.0 if answer_hit else 0.0

        return metrics

    def compute_batch_bertscore(self, refs, preds):
        results = [{"bertscore_f1": 0.0} for _ in refs]
        if self.skip_bertscore or not bert_score_func:
            return results
        
        valid_idxs = [i for i, (r, p) in enumerate(zip(refs, preds)) if r.strip() and p.strip()]
        if not valid_idxs:
            return results
        
        try:
            v_refs = [refs[i] for i in valid_idxs]
            v_preds = [preds[i] for i in valid_idxs]
            P, R, F1 = bert_score_func(v_preds, v_refs, lang="en", verbose=False)
            for idx, f1 in zip(valid_idxs, F1):
                results[idx]["bertscore_f1"] = float(f1)
        except Exception as e:
            logger.error(f"BERTScore error: {e}")
        return results

def aggregate_metrics(metrics_list: List[Dict]) -> Dict[str, float]:
    """Computes Mean and Std Dev for each metric key."""
    if not metrics_list:
        return {}
    agg = {}
    keys = metrics_list[0].keys()
    
    for k in keys:
        vals = [m[k] for m in metrics_list if k in m and isinstance(m[k], (int, float))]
        if vals:
            agg[k] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))
        else:
            agg[k] = 0.0
            agg[f"{k}_std"] = 0.0
            
    agg["count"] = len(metrics_list)
    return agg

def main():
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # 1. Load Data
    logger.info(f"Loading results: {args.input}")
    samples, meta = load_results(Path(args.input))
    
    # 2. Load Ground Truth Metadata
    logger.info(f"Loading ground truth: {args.dataset}")
    gt_data = load_jsonl(Path(args.dataset))
    
    # 3. Process
    calc = MetricsCalculator(skip_bertscore=args.skip_bertscore)
    detailed_results = []
    
    refs, preds = [], []

    for i, s in enumerate(samples):
        # Merge metadata (1-1 mapping)
        gt_item = gt_data[i] if i < len(gt_data) else {}
        
        f_id = s.get('financebench_id') or s.get('sample_id') or gt_item.get('financebench_id') or f"id_{i}"
        q_type = gt_item.get('question_type', 'unknown')
        q_reasoning = gt_item.get('question_reasoning', 'unknown')
        
        s['financebench_id'] = f_id
        s['question_type'] = q_type
        s['question_reasoning'] = q_reasoning
        
        refs.append(s.get('reference_answer', ''))
        preds.append(s.get('generated_answer', ''))

    # Batch BERTScore
    logger.info("Computing BERTScore...")
    bert_scores = calc.compute_batch_bertscore(refs, preds)

    # Individual Metrics
    logger.info("Computing per-sample metrics...")
    for i, s in enumerate(samples):
        ref = refs[i]
        pred = preds[i]
        
        t_metrics = calc.compute_text_metrics(ref, pred, s['question_type'])
        t_metrics.update(bert_scores[i])
        
        retrieved = s.get('retrieved_chunks', [])
        gold = s.get('gold_evidence_segments', [])
        if gold and isinstance(gold[0], str):
            gold = [{"text": g} for g in gold]
        
        r_metrics = calc.compute_retrieval_metrics(retrieved, gold, ref)
        
        full = {**t_metrics, **r_metrics}
        
        detailed_results.append({
            "financebench_id": s['financebench_id'],
            "question_type": s['question_type'],
            "question_reasoning": s['question_reasoning'],
            "metrics": full
        })

    # 4. Aggregation
    all_mets = [d['metrics'] for d in detailed_results]
    overall = aggregate_metrics(all_mets)
    
    by_type = defaultdict(list)
    for d in detailed_results:
        by_type[d['question_type']].append(d['metrics'])
    agg_type = {k: aggregate_metrics(v) for k, v in by_type.items()}
    
    by_reas = defaultdict(list)
    for d in detailed_results:
        by_reas[d['question_reasoning']].append(d['metrics'])
    agg_reas = {k: aggregate_metrics(v) for k, v in by_reas.items()}

    # 5. Output
    report = {
        "summary": {
            "overall": overall,
            "by_question_type": agg_type,
            "by_question_reasoning": agg_reas
        },
        "detailed_results": detailed_results
    }

    # Print Summary
    print("\n" + "="*60)
    print("EVALUATION REPORT")
    print("="*60)
    
    def print_m(name, m):
        print(f"\n--- {name} (n={m.get('count', 0)}) ---")
        for k in sorted(m.keys()):
            if k == "count":
                continue
            val = m[k]
            print(f"{k:<30}: {val:.4f}")

    print_m("Overall", overall)
    
    print("\n[By Question Type]")
    for k, v in sorted(agg_type.items()):
        print_m(k, v)

    if args.save_report:
        with open(args.save_report, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {args.save_report}")

if __name__ == "__main__":
    main()