"""
Evaluate uncertainty quality: ECE, Brier score, AUROC/AUPRC for failure detection.
Works on scored_pairs output from evaluate_ranking.py.
"""
import argparse
import pathlib
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

import numpy as np
from src.evaluation.metrics import (
    expected_calibration_error, brier_score,
    compute_failure_detection_metrics, reliability_diagram_data,
    recall_at_k,
)
from src.utils.io import load_jsonl, save_json
from src.utils.logging import get_logger


def compute_question_level_uncertainty(scored_pairs: list[dict], top_k: int = 5) -> dict[str, dict]:
    by_qid: dict[str, list[dict]] = defaultdict(list)
    for p in scored_pairs:
        by_qid[p["qid"]].append(p)

    qid_metrics = {}
    for qid, pairs in by_qid.items():
        pairs_sorted = sorted(pairs, key=lambda x: x["score"], reverse=True)
        top = pairs_sorted[:top_k]

        uncs = [p["uncertainty"] for p in top if not (isinstance(p["uncertainty"], float) and p["uncertainty"] != p["uncertainty"])]
        p_rels = [p["p_relevant"] for p in top]
        scores = [p["score"] for p in top]

        qid_metrics[qid] = {
            "top1_uncertainty": uncs[0] if uncs else float("nan"),
            "mean_top5_uncertainty": float(np.mean(uncs)) if uncs else float("nan"),
            "max_top5_uncertainty": float(np.max(uncs)) if uncs else float("nan"),
            "top1_relevance": p_rels[0] if p_rels else float("nan"),
            "mean_top5_relevance": float(np.mean(p_rels)) if p_rels else float("nan"),
            "score_gap_top1_top2": float(scores[0] - scores[1]) if len(scores) >= 2 else 0.0,
            "entropy_top5": float(-np.sum([s * np.log(s + 1e-9) for s in scores if s > 0])) if scores else float("nan"),
        }
    return qid_metrics


def evaluate_uncertainty(
    scored_pairs_path: str,
    out_dir: str,
    tag: str = "eval",
    failure_k: int = 5,
    n_ece_bins: int = 10,
) -> dict:
    logger = get_logger("eval_uncertainty")
    pairs = load_jsonl(scored_pairs_path)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pair-level calibration
    probs = np.array([p["p_relevant"] for p in pairs])
    labels = np.array([p["label"] for p in pairs], dtype=float)
    valid = ~np.isnan(probs)

    calibration = {}
    if valid.sum() > 0:
        ece = expected_calibration_error(probs[valid], labels[valid], n_ece_bins)
        bs = brier_score(probs[valid], labels[valid])
        rel_data = reliability_diagram_data(probs[valid], labels[valid], n_ece_bins)
        calibration = {"ece": ece, "brier_score": bs, "reliability": rel_data}
        logger.info(f"Calibration: ECE={ece:.4f}, Brier={bs:.4f}")

    # Question-level failure detection
    by_qid: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        by_qid[p["qid"]].append(p)

    q_metrics = compute_question_level_uncertainty(pairs, top_k=failure_k)
    qids = sorted(q_metrics.keys())

    failure_labels = []
    uncertainty_scores = []
    for qid in qids:
        qpairs = sorted(by_qid[qid], key=lambda x: x["score"], reverse=True)
        ranked = [p["candidate_id"] for p in qpairs]
        gold = {p["candidate_id"] for p in qpairs if p["label"] == 1}
        failed = 1 if recall_at_k(ranked, gold, failure_k) == 0 else 0
        failure_labels.append(failed)
        uncertainty_scores.append(q_metrics[qid]["top1_uncertainty"])

    failure_labels = np.array(failure_labels)
    uncertainty_scores = np.array(uncertainty_scores)
    valid_q = ~np.isnan(uncertainty_scores)

    failure_detection = {}
    if valid_q.sum() > 0:
        fd = compute_failure_detection_metrics(uncertainty_scores[valid_q], failure_labels[valid_q])
        failure_detection = fd
        logger.info(f"Failure detection @{failure_k}: AUROC={fd.get('auroc', 'nan'):.4f}, AUPRC={fd.get('auprc', 'nan'):.4f}")

    results = {
        "tag": tag,
        "n_pairs": len(pairs),
        "n_questions": len(qids),
        "failure_rate": float(failure_labels.mean()),
        "calibration": calibration,
        "failure_detection": failure_detection,
    }
    save_json(results, out_dir / f"{tag}_uncertainty_metrics.json")
    save_json(q_metrics, out_dir / f"{tag}_question_uncertainty.json")
    logger.info(f"Saved uncertainty results to {out_dir}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored_pairs", required=True)
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--tag", default="eval")
    parser.add_argument("--failure_k", type=int, default=5)
    args = parser.parse_args()
    evaluate_uncertainty(args.scored_pairs, args.out_dir, args.tag, args.failure_k)
