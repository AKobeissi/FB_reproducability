"""
Selective retrieval: at each coverage level, keep questions with lowest uncertainty
and report Recall@k. Generates risk-coverage curve data.
"""
import argparse
import pathlib
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

import numpy as np
from src.evaluation.metrics import selective_recall, recall_at_k
from src.utils.io import load_jsonl, load_json, save_json
from src.utils.logging import get_logger


def run_selective_eval(
    scored_pairs_path: str,
    out_dir: str,
    tag: str = "eval",
    coverage_levels: list[float] | None = None,
    recall_k: int = 5,
) -> list[dict]:
    if coverage_levels is None:
        coverage_levels = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

    logger = get_logger("eval_selective")
    pairs = load_jsonl(scored_pairs_path)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_qid: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        by_qid[p["qid"]].append(p)

    qid_to_ranked: dict[str, list[str]] = {}
    qid_to_gold: dict[str, set[str]] = {}
    qid_to_uncertainty: dict[str, float] = {}

    for qid, qpairs in by_qid.items():
        qpairs_sorted = sorted(qpairs, key=lambda x: x["score"], reverse=True)
        qid_to_ranked[qid] = [p["candidate_id"] for p in qpairs_sorted]
        qid_to_gold[qid] = {p["candidate_id"] for p in qpairs if p["label"] == 1}
        # Use top-1 uncertainty as question-level uncertainty
        unc = qpairs_sorted[0]["uncertainty"] if qpairs_sorted else float("nan")
        qid_to_uncertainty[qid] = unc if not (isinstance(unc, float) and np.isnan(unc)) else 1.0

    results = selective_recall(
        qid_to_ranked, qid_to_gold, qid_to_uncertainty, coverage_levels, k=recall_k
    )

    logger.info(f"Selective recall @{recall_k}:")
    for r in results:
        logger.info(f"  Coverage {r['coverage']:.0%}: Recall@{recall_k}={r[f'recall@{recall_k}']:.4f}, N={r['n_questions']}")

    import csv
    csv_path = out_dir / f"{tag}_selective_recall.csv"
    with open(csv_path, "w", newline="") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    save_json(results, out_dir / f"{tag}_selective_recall.json")
    logger.info(f"Saved to {out_dir}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored_pairs", required=True)
    parser.add_argument("--out_dir", default="results/tables")
    parser.add_argument("--tag", default="eval")
    parser.add_argument("--recall_k", type=int, default=5)
    parser.add_argument("--coverage_levels", nargs="+", type=float, default=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    args = parser.parse_args()
    run_selective_eval(args.scored_pairs, args.out_dir, args.tag, args.coverage_levels, args.recall_k)
