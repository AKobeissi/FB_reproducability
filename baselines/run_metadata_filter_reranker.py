#!/usr/bin/env python3
"""
run_metadata_filter_reranker.py
================================
Pipeline: BGE-M3 dense retrieval → doc-name metadata filter → FT cross-encoder reranker.

Motivation
----------
Pure dense retrieval (BGE-M3) retrieves the right company's documents but also
pulls in same-company documents from other fiscal years. MultiHyDE can do the
opposite: its hypothetical documents are generic enough to match cash-flow or
balance-sheet pages from entirely different companies, poisoning the reranker
input pool.

Inserting a cheap, exact doc_name filter *between* dense retrieval and the
reranker fixes both problems:
  - keeps only chunks whose metadata.doc_name matches the question's known doc,
  - leaves the reranker a clean, company-scoped candidate pool to work on.

Pipeline
--------
  dense_bge_m3_retrieval.json  (top-20 candidates per question)
       └─ doc-name filter         (keep chunks where chunk.doc_name == sample.doc_name)
            └─ fallback           (if < MIN_FILTER_K survive, use full top-20)
                 └─ FT reranker   (checkpoints/ft_cross_encoder)
                      └─ top-20 reranked saved as new retrieval JSON

Output
------
  baselines/results/predictions/dense_bge_m3_meta_filter_ft_reranker_retrieval.json
  baselines/results/metrics/dense_bge_m3_meta_filter_ft_reranker_metrics.json
"""

import copy
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("meta_filter_reranker")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHECKPOINT   = PROJECT_ROOT / "checkpoints/ft_cross_encoder"
PREDS_DIR    = PROJECT_ROOT / "baselines/results/predictions"
METRICS_DIR  = PROJECT_ROOT / "baselines/results/metrics"

INPUT_FILE   = PREDS_DIR / "dense_bge_m3_retrieval.json"
OUTPUT_FILE  = PREDS_DIR / "dense_bge_m3_meta_filter_ft_reranker_retrieval.json"
METRICS_FILE = METRICS_DIR / "dense_bge_m3_meta_filter_ft_reranker_metrics.json"

# Metadata filter: if fewer than this many candidates match, fall back to full pool
MIN_FILTER_K = 5
RERANK_TOP_N = 20   # score all retrieved chunks, keep all reranked
BATCH_SIZE   = 64   # cross-encoder inference batch size


# ---------------------------------------------------------------------------
# Metadata filter
# ---------------------------------------------------------------------------

def apply_metadata_filter(
    samples: List[Dict],
    min_k: int = MIN_FILTER_K,
) -> List[Dict]:
    """
    For each sample, keep only retrieved chunks whose metadata.doc_name matches
    sample['doc_name'].  If fewer than *min_k* chunks survive, fall back to the
    full original pool (no filtering).

    Adds a filter_stats dict to each sample for analysis:
      n_before, n_after, fallback (bool)
    """
    results = copy.deepcopy(samples)
    n_filtered = 0
    n_fallback = 0
    n_zero     = 0

    for sample in results:
        target_doc = sample.get("doc_name", "")
        chunks     = sample.get("retrieved_chunks", [])

        if not target_doc:
            # No doc_name on this sample — cannot filter, keep as-is
            sample["filter_stats"] = {"n_before": len(chunks), "n_after": len(chunks),
                                       "fallback": True, "reason": "no_doc_name"}
            n_fallback += 1
            continue

        filtered = [c for c in chunks if c.get("metadata", {}).get("doc_name") == target_doc]

        if len(filtered) == 0:
            n_zero += 1

        if len(filtered) >= min_k:
            sample["retrieved_chunks"] = filtered
            sample["filter_stats"] = {
                "n_before": len(chunks),
                "n_after":  len(filtered),
                "fallback": False,
                "reason":   "filtered",
            }
            n_filtered += 1
        else:
            # Fall back: keep original pool so the reranker still has enough candidates
            sample["filter_stats"] = {
                "n_before":  len(chunks),
                "n_after":   len(chunks),
                "n_matched": len(filtered),
                "fallback":  True,
                "reason":    "too_few" if len(filtered) > 0 else "zero_match",
            }
            n_fallback += 1

    logger.info(
        f"Metadata filter: {n_filtered} filtered, {n_fallback} fallback "
        f"(of which {n_zero} had zero matching chunks)"
    )
    return results


# ---------------------------------------------------------------------------
# Reranking  (mirrors run_ft_reranker_baselines.py)
# ---------------------------------------------------------------------------

def rerank_predictions(
    samples: List[Dict],
    model,
    top_n: int = RERANK_TOP_N,
    batch_size: int = BATCH_SIZE,
) -> List[Dict]:
    """
    Re-score retrieved_chunks[:top_n] with the cross-encoder; re-sort in place.
    Stores _reranker_score on each chunk (and overwrites _score for consistency
    with the rest of the pipeline, matching existing conventions).
    """
    results = copy.deepcopy(samples)

    for sample in results:
        chunks = sample.get("retrieved_chunks", [])[:top_n]
        if not chunks:
            continue

        query = sample.get("question", "")
        pairs = [(query, c.get("text", "")) for c in chunks]

        scores: List[float] = []
        for i in range(0, len(pairs), batch_size):
            batch_scores = model.predict(pairs[i : i + batch_size])
            scores.extend(
                batch_scores.tolist() if hasattr(batch_scores, "tolist") else list(batch_scores)
            )

        for chunk, score in zip(chunks, scores):
            chunk["_dense_score"]    = chunk.get("_score")
            chunk["_reranker_score"] = float(score)
            chunk["_score"]          = float(score)

        chunks.sort(key=lambda c: c["_reranker_score"], reverse=True)
        for i, chunk in enumerate(chunks):
            chunk["rank"] = i + 1
        sample["retrieved_chunks"] = chunks

    return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _infer_doc_type(sample: Dict) -> str:
    doc = str(sample.get("doc_name", "")).lower()
    if "10k" in doc or "10-k" in doc:
        return "10k"
    if "10q" in doc or "10-q" in doc:
        return "10q"
    if "8k" in doc or "8-k" in doc:
        return "8k"
    return "unknown"


def compute_and_save_metrics(
    samples: List[Dict],
    metrics_path: Path,
    label: str,
) -> Dict:
    from evaluation.retrieval_evaluator import RetrievalEvaluator

    evaluator = RetrievalEvaluator()
    overall   = evaluator.compute_metrics(samples)

    by_qtype: Dict[str, List] = defaultdict(list)
    by_dtype: Dict[str, List] = defaultdict(list)
    by_qt_dt: Dict[str, List] = defaultdict(list)

    for s in samples:
        qt = str(s.get("question_type") or "unknown").strip().lower()
        dt = str(s.get("doc_type") or _infer_doc_type(s)).strip()
        by_qtype[qt].append(s)
        by_dtype[dt].append(s)
        by_qt_dt[f"{qt}|{dt}"].append(s)

    metrics_out = {
        "overall":                     overall,
        "by_question_type":            {k: evaluator.compute_metrics(v) for k, v in by_qtype.items()},
        "by_doc_type":                 {k: evaluator.compute_metrics(v) for k, v in by_dtype.items()},
        "by_question_type_x_doc_type": {k: evaluator.compute_metrics(v) for k, v in by_qt_dt.items()},
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    logger.info(f"[{label}] metrics → {metrics_path}")

    ov = overall
    logger.info(
        f"[{label}]  page_recall@5={ov.get('page_recall@5', 0):.4f}  "
        f"chunk_recall@5={ov.get('chunk_recall@5', 0):.4f}  "
        f"mrr={ov.get('mrr', 0):.4f}"
    )
    return metrics_out


# ---------------------------------------------------------------------------
# Filter-effect analysis
# ---------------------------------------------------------------------------

def print_filter_stats(samples: List[Dict]) -> None:
    """Log aggregate filter statistics."""
    total      = len(samples)
    filtered   = sum(1 for s in samples if not s.get("filter_stats", {}).get("fallback", True))
    fallback   = total - filtered
    zero_match = sum(1 for s in samples
                     if s.get("filter_stats", {}).get("reason") == "zero_match")
    too_few    = sum(1 for s in samples
                     if s.get("filter_stats", {}).get("reason") == "too_few")

    n_before_avg = sum(s["filter_stats"]["n_before"] for s in samples
                       if "filter_stats" in s) / max(total, 1)
    n_after_avg  = sum(s["filter_stats"]["n_after"]  for s in samples
                       if "filter_stats" in s) / max(total, 1)

    logger.info("=" * 60)
    logger.info("Filter statistics")
    logger.info(f"  Total questions      : {total}")
    logger.info(f"  Filtered (strict)    : {filtered}  ({100*filtered/total:.1f}%)")
    logger.info(f"  Fallback (too few)   : {too_few}")
    logger.info(f"  Fallback (zero match): {zero_match}")
    logger.info(f"  Avg candidates before: {n_before_avg:.1f}")
    logger.info(f"  Avg candidates after : {n_after_avg:.1f}")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="BGE-M3 → metadata filter → FT reranker")
    parser.add_argument("--input",       type=Path, default=INPUT_FILE,
                        help="Dense retrieval JSON to filter+rerank")
    parser.add_argument("--output",      type=Path, default=OUTPUT_FILE,
                        help="Output retrieval JSON")
    parser.add_argument("--metrics",     type=Path, default=METRICS_FILE,
                        help="Output metrics JSON")
    parser.add_argument("--checkpoint",  type=Path, default=CHECKPOINT,
                        help="FT cross-encoder checkpoint directory")
    parser.add_argument("--min-filter-k", type=int, default=MIN_FILTER_K,
                        help="Min matching chunks before falling back to full pool")
    args = parser.parse_args()

    # --- Load data ---
    if not args.input.exists():
        raise FileNotFoundError(f"Input retrieval file not found: {args.input}")
    logger.info(f"Loading retrieval results from {args.input}")
    samples = json.load(open(args.input))
    logger.info(f"Loaded {len(samples)} samples  ({samples[0].get('question','')[:60]}…)")

    # --- Stage 1: metadata filter ---
    logger.info(f"Applying doc-name metadata filter (min_k={args.min_filter_k})")
    filtered_samples = apply_metadata_filter(samples, min_k=args.min_filter_k)
    print_filter_stats(filtered_samples)

    # --- Load FT reranker ---
    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"FT cross-encoder checkpoint not found: {args.checkpoint}\n"
            "Run scripts/run_ft_reranker.sh first."
        )
    logger.info(f"Loading FT cross-encoder from {args.checkpoint}")
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(str(args.checkpoint), max_length=512)

    # --- Stage 2: rerank ---
    logger.info("Re-ranking filtered candidates with FT cross-encoder…")
    reranked_samples = rerank_predictions(filtered_samples, model)

    # --- Save retrieval output ---
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(reranked_samples, f, indent=2)
    logger.info(f"Retrieval results → {args.output}")

    # --- Compute and save metrics ---
    label = "BGE-M3 + MetaFilter + FT-Reranker"
    metrics = compute_and_save_metrics(reranked_samples, args.metrics, label)

    # --- Baseline comparison (print page recall for context) ---
    baseline_files = {
        "BGE-M3 (dense)":        METRICS_DIR / "dense_bge_m3_metrics.json",
        "BGE-M3 + FT-Reranker":  METRICS_DIR / "dense_bge_m3_ft_reranker_metrics.json",
        "MH + FT-Reranker":      METRICS_DIR / "multi_hyde_ft_reranker_metrics.json",
    }
    logger.info("\n--- Page Recall@5 comparison ---")
    for name, path in baseline_files.items():
        if path.exists():
            m = json.load(open(path))
            pr5 = m.get("overall", {}).get("page_recall@5", float("nan"))
            logger.info(f"  {name:<35}: {pr5:.4f}")
    new_pr5 = metrics["overall"].get("page_recall@5", float("nan"))
    logger.info(f"  {label:<35}: {new_pr5:.4f}  ← new")

    logger.info("Done.")


if __name__ == "__main__":
    main()
