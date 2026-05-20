#!/usr/bin/env python3
"""
recompute_metrics_with_precision.py
====================================
Re-reads all saved prediction files and recomputes metrics using the updated
RetrievalEvaluator that now includes precision@k metrics.

Also fixes the 'Earnings' doc_type (was saved as 'unknown' in older runs).

Run from repo root:
    python3 baselines/recompute_metrics_with_precision.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.retrieval_evaluator import RetrievalEvaluator

PREDICTIONS_DIR = PROJECT_ROOT / "baselines/results/predictions"
METRICS_DIR     = PROJECT_ROOT / "baselines/results/metrics"
DOC_INFO_PATH   = PROJECT_ROOT / "data/financebench_document_information.jsonl"
K_VALUES        = [1, 3, 5, 10, 20]

# Variants that have a retrieval prediction file
VARIANTS = [
    "bm25",
    "splade",
    "hybrid_25_75",
    "hybrid_50_50",
    "hybrid_75_25",
    "dense_bge_m3",
    "parent_child",
    "query_expansion",
    "hyde",
    "multi_hyde",
    "bge_reranker",
    "multi_hyde_reranker",
    "dense_bge_m3_ft_reranker",
    "multi_hyde_ft_reranker",
]

# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def load_doc_info() -> dict:
    doc_info = {}
    if DOC_INFO_PATH.exists():
        with open(DOC_INFO_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                doc_info[d["doc_name"]] = d
    return doc_info


def load_predictions(variant: str) -> list:
    path = PREDICTIONS_DIR / f"{variant}_retrieval.json"
    if not path.exists():
        print(f"  [SKIP] {path.name} not found")
        return []
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    return data.get("results", [])


# ---------------------------------------------------------------------------
# Generative metrics (answer ROUGE-L + numeric match)
# ---------------------------------------------------------------------------

def _numeric_match(reference: str, prediction: str) -> float:
    import re
    def extract(text):
        clean = re.sub(r"[,$]", "", str(text))
        matches = re.findall(r"-?\d*\.?\d+", clean)
        try:
            return [float(m) for m in matches]
        except ValueError:
            return []
    ref_nums = extract(reference)
    pred_nums = extract(prediction)
    if not ref_nums or not pred_nums:
        return 0.0
    for r in ref_nums:
        for p in pred_nums:
            if np.isclose(r, p, atol=0.03, rtol=0.03):
                return 1.0
    return 0.0


def compute_generative_metrics(samples: list) -> dict:
    try:
        from rouge_score import rouge_scorer as rs_mod
        scorer = rs_mod.RougeScorer(["rougeL"], use_stemmer=True)
    except ImportError:
        return {}

    rougeL_scores = []
    numeric_matches = []
    for s in samples:
        gen = s.get("generated_answer") or s.get("predicted_answer") or ""
        ref = s.get("reference_answer", "")
        if ref and gen:
            score = scorer.score(ref, gen)["rougeL"].fmeasure
        else:
            score = 0.0
        rougeL_scores.append(score)

        if s.get("question_type") == "metrics-generated":
            numeric_matches.append(_numeric_match(gen, ref))

    return {
        "answer_rougeL":  float(np.mean(rougeL_scores)) if rougeL_scores else 0.0,
        "numeric_match":  float(np.mean(numeric_matches)) if numeric_matches else 0.0,
    }


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_by_group(evaluator: RetrievalEvaluator, samples: list, key_fn) -> dict:
    groups = defaultdict(list)
    for s in samples:
        groups[key_fn(s)].append(s)

    results = {}
    for group, group_samples in groups.items():
        ret = evaluator.compute_metrics(group_samples, K_VALUES)
        gen = compute_generative_metrics(group_samples)
        results[group] = {**ret, **gen}
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading doc_info …")
    doc_info = load_doc_info()

    evaluator = RetrievalEvaluator()
    QUESTION_TYPES = ["metrics-generated", "domain-relevant", "novel-generated"]

    for variant in VARIANTS:
        print(f"\nProcessing: {variant}")
        samples = load_predictions(variant)
        if not samples:
            continue

        # Attach doc_type from doc_info (fixes Earnings/unknown issue)
        for s in samples:
            doc_name = s.get("doc_name", "")
            info = doc_info.get(doc_name, {})
            s["doc_type"] = info.get("doc_type", "unknown")

        # Also try to load generated answers and attach
        gen_path = PREDICTIONS_DIR / f"{variant}_generated.json"
        if gen_path.exists():
            gen_data = json.loads(gen_path.read_text())
            gen_list = gen_data if isinstance(gen_data, list) else gen_data.get("results", [])
            gen_map = {s.get("financebench_id"): s for s in gen_list if s.get("financebench_id")}
            for s in samples:
                fid = s.get("financebench_id")
                if fid and fid in gen_map:
                    s["generated_answer"] = gen_map[fid].get("generated_answer", "")
                    s["predicted_answer"]  = gen_map[fid].get("predicted_answer", "")

        # Overall metrics
        overall = evaluator.compute_metrics(samples, K_VALUES)
        gen_metrics = compute_generative_metrics(samples)
        overall.update(gen_metrics)

        # Breakdowns
        by_qt     = aggregate_by_group(evaluator, samples, lambda s: s.get("question_type", "unknown"))
        by_dt     = aggregate_by_group(evaluator, samples, lambda s: s.get("doc_type", "unknown"))
        by_qt_dt  = aggregate_by_group(evaluator, samples,
                        lambda s: f"{s.get('question_type','?')}|{s.get('doc_type','?')}")

        metrics_out = {
            "overall":                       overall,
            "by_question_type":              by_qt,
            "by_doc_type":                   by_dt,
            "by_question_type_x_doc_type":   by_qt_dt,
        }

        out_path = METRICS_DIR / f"{variant}_metrics.json"
        out_path.write_text(json.dumps(metrics_out, indent=2))
        pr5 = overall.get("page_recall@5", 0)
        pp5 = overall.get("page_precision@5", 0)
        dr5 = overall.get("doc_recall@5", 0)
        print(f"  PageRec@5={pr5:.3f}  PagePrec@5={pp5:.3f}  DocRec@5={dr5:.3f}  → saved {out_path.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
