#!/usr/bin/env python3
"""
run_ft_reranker_baselines.py
==============================
Applies the fine-tuned cross-encoder reranker to existing retrieval outputs
and evaluates on FinanceBench.

Pipeline
--------
  existing_retrieval.json
       └─ top-20 chunks per query
            └─ CrossEncoder re-scores each (query, chunk) pair
                 └─ re-sorted top-20 saved as new prediction file
                      └─ evaluated with RetrievalEvaluator

Variants produced
-----------------
  dense_bge_m3   + FT reranker  →  dense_bge_m3_ft_reranker_retrieval.json
  multi_hyde     + FT reranker  →  multi_hyde_ft_reranker_retrieval.json

Metrics and comparison plots are saved to baselines/results/.
"""

import copy
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ft_reranker_baselines")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHECKPOINT   = PROJECT_ROOT / "checkpoints/ft_cross_encoder"
PREDS_DIR    = PROJECT_ROOT / "baselines/results/predictions"
METRICS_DIR  = PROJECT_ROOT / "baselines/results/metrics"
PLOTS_DIR    = PROJECT_ROOT / "baselines/results/plots"

# (label, input_retrieval_file, output_retrieval_file, output_metrics_file)
# Variant 1: BGE-M3 retrieval  → FT reranker
# Variant 2: BGE-M3 + MultiHyDE retrieval → FT reranker
#            (multi_hyde_retrieval.json already encodes BGE-M3 + MultiHyDE
#             query expansion; the FT reranker is the third stage on top)
VARIANTS = [
    (
        "BGE-M3 + FT-Reranker",
        PREDS_DIR / "dense_bge_m3_retrieval.json",
        PREDS_DIR / "dense_bge_m3_ft_reranker_retrieval.json",
        METRICS_DIR / "dense_bge_m3_ft_reranker_metrics.json",
    ),
    (
        "BGE-M3 + MultiHyDE + FT-Reranker",
        PREDS_DIR / "multi_hyde_retrieval.json",
        PREDS_DIR / "multi_hyde_ft_reranker_retrieval.json",
        METRICS_DIR / "multi_hyde_ft_reranker_metrics.json",
    ),
]

# Baselines to include in comparison plot (must already exist)
COMPARISON_BASELINES = {
    "BGE-M3":                       METRICS_DIR / "dense_bge_m3_metrics.json",
    "BGE-M3+Reranker":              METRICS_DIR / "bge_reranker_metrics.json",
    "MultiHyDE":                    METRICS_DIR / "multi_hyde_metrics.json",
    "BGE-M3+FT-Reranker":           METRICS_DIR / "dense_bge_m3_ft_reranker_metrics.json",
    "BGE-M3+MultiHyDE+FT-Reranker": METRICS_DIR / "multi_hyde_ft_reranker_metrics.json",
}

RERANK_TOP_N = 20   # number of retrieved chunks to re-score
BATCH_SIZE   = 64   # cross-encoder inference batch size


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------

def rerank_predictions(
    samples: List[Dict],
    model,
    top_n: int = RERANK_TOP_N,
    batch_size: int = BATCH_SIZE,
) -> List[Dict]:
    """
    For each sample, score all retrieved_chunks[:top_n] with the cross-encoder
    and return a new sample list with chunks re-sorted by reranker score.
    """
    results = copy.deepcopy(samples)

    for sample in results:
        chunks = sample.get("retrieved_chunks", [])[:top_n]
        if not chunks:
            continue

        query   = sample.get("question", "")
        pairs   = [(query, c.get("text", "")) for c in chunks]

        # Score in batches
        scores = []
        for i in range(0, len(pairs), batch_size):
            batch_scores = model.predict(pairs[i : i + batch_size])
            # predict() returns numpy array or list
            scores.extend(batch_scores.tolist() if hasattr(batch_scores, "tolist") else list(batch_scores))

        # Attach reranker score and re-sort
        for chunk, score in zip(chunks, scores):
            chunk["_dense_score"]    = chunk.get("_score")   # preserve original dense score
            chunk["_reranker_score"] = float(score)
            chunk["_score"]          = float(score)  # overwrite for consistency

        chunks.sort(key=lambda c: c["_reranker_score"], reverse=True)
        for i, chunk in enumerate(chunks):
            chunk["rank"] = i + 1   # update rank to reflect post-reranker position
        sample["retrieved_chunks"] = chunks

    return results


# ---------------------------------------------------------------------------
# Metrics (delegates to existing RetrievalEvaluator)
# ---------------------------------------------------------------------------

def compute_and_save_metrics(
    samples: List[Dict],
    metrics_path: Path,
    label: str,
) -> Dict:
    from evaluation.retrieval_evaluator import RetrievalEvaluator

    evaluator = RetrievalEvaluator()
    overall   = evaluator.compute_metrics(samples)

    # Break down by question_type and doc_type (mirrors existing format)
    from collections import defaultdict

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
        "overall":                  overall,
        "by_question_type":         {k: evaluator.compute_metrics(v) for k, v in by_qtype.items()},
        "by_doc_type":              {k: evaluator.compute_metrics(v) for k, v in by_dtype.items()},
        "by_question_type_x_doc_type": {k: evaluator.compute_metrics(v) for k, v in by_qt_dt.items()},
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    logger.info(f"[{label}] metrics → {metrics_path}")

    # Print key numbers
    ov = overall
    logger.info(
        f"[{label}]  page_recall@5={ov.get('page_recall@5', 0):.4f}  "
        f"chunk_recall@5={ov.get('chunk_recall@5', 0):.4f}  "
        f"mrr={ov.get('mrr', 0):.4f}"
    )
    return metrics_out


def _infer_doc_type(sample: Dict) -> str:
    """Guess doc type from doc_name if 'doc_type' key is absent."""
    doc = str(sample.get("doc_name", "")).lower()
    if "10k" in doc or "10-k" in doc:
        return "10k"
    if "10q" in doc or "10-q" in doc:
        return "10q"
    if "8k" in doc or "8-k" in doc:
        return "8k"
    return "unknown"


# ---------------------------------------------------------------------------
# Comparison plots
# ---------------------------------------------------------------------------

def save_comparison_plots(comparison_metrics: Dict[str, Dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping plots.")
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    labels   = list(comparison_metrics.keys())
    n        = len(labels)
    x        = np.arange(n)

    # ── Plot 1: Page and Chunk Recall @1,5,10 ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    k_vals    = [1, 5, 10]
    colors    = ["#3F51B5", "#E91E63", "#FF9800"]

    for ax, metric_base, title in [
        (axes[0], "page_recall",  "Page Recall@k"),
        (axes[1], "chunk_recall", "Chunk Recall@k"),
    ]:
        width = 0.25
        for ki, (k, color) in enumerate(zip(k_vals, colors)):
            key  = f"{metric_base}@{k}"
            vals = [comparison_metrics[l]["overall"].get(key, 0) for l in labels]
            offset = (ki - 1) * width
            bars = ax.bar(x + offset, vals, width, label=f"@{k}", color=color, alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=6,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel("Score")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.set_ylim(0, min(1.0, ax.get_ylim()[1] * 1.15))
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "FinanceBench — FT Cross-Encoder Reranker vs Baselines", fontsize=12
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(PLOTS_DIR / f"ft_reranker_comparison.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {PLOTS_DIR}/ft_reranker_comparison.*")

    # ── Plot 2: MRR comparison ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    mrr_vals = [comparison_metrics[l]["overall"].get("mrr", 0) for l in labels]
    bar_colors = ["#9E9E9E"] * (n - 2) + ["#E91E63", "#3F51B5"]  # highlight new variants
    bars = ax.bar(labels, mrr_vals, color=bar_colors, alpha=0.85, width=0.5)
    for bar, v in zip(bars, mrr_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{v:.3f}", ha="center", va="bottom", fontsize=9,
        )
    ax.set_ylabel("MRR")
    ax.set_title("MRR — FT Cross-Encoder Reranker vs Baselines")
    ax.set_ylim(0, min(1.0, max(mrr_vals) * 1.25 + 0.05))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(PLOTS_DIR / f"ft_reranker_mrr.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {PLOTS_DIR}/ft_reranker_mrr.*")

    # ── Plot 3: Question-type breakdown (page_recall@5) ────────────────────
    qtypes   = ["metrics-generated", "domain-relevant", "novel-generated"]
    qt_short = ["Metrics", "Domain", "Novel"]
    fig, ax  = plt.subplots(figsize=(10, 5))
    width    = 0.15
    for li, label in enumerate(labels):
        vals = [
            comparison_metrics[label]
            .get("by_question_type", {})
            .get(qt, {})
            .get("page_recall@5", 0)
            for qt in qtypes
        ]
        offset = (li - n / 2 + 0.5) * width
        bars = ax.bar(
            np.arange(len(qtypes)) + offset, vals, width,
            label=label, alpha=0.85,
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{v:.2f}", ha="center", va="bottom", fontsize=6,
            )
    ax.set_xticks(np.arange(len(qtypes)))
    ax.set_xticklabels(qt_short, fontsize=11)
    ax.set_ylabel("Page Recall@5")
    ax.set_title("Page Recall@5 by Question Type")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_ylim(0, min(1.0, ax.get_ylim()[1] * 1.15))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(
            PLOTS_DIR / f"ft_reranker_qtype_breakdown.{ext}",
            bbox_inches="tight", dpi=150,
        )
    plt.close(fig)
    logger.info(f"Saved: {PLOTS_DIR}/ft_reranker_qtype_breakdown.*")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ── Load cross-encoder ────────────────────────────────────────────────
    if not CHECKPOINT.exists():
        logger.error(
            f"Checkpoint not found at {CHECKPOINT}. "
            "Run train_cross_encoder_reranker.py first."
        )
        sys.exit(1)

    try:
        from sentence_transformers.cross_encoder import CrossEncoder
    except ImportError as e:
        logger.error(f"sentence-transformers not available: {e}")
        sys.exit(1)

    logger.info(f"Loading fine-tuned CrossEncoder from {CHECKPOINT}")
    model = CrossEncoder(str(CHECKPOINT), max_length=512)

    all_metrics: Dict[str, Dict] = {}

    # ── Process each variant ──────────────────────────────────────────────
    for label, input_path, output_path, metrics_path in VARIANTS:
        if not input_path.exists():
            logger.warning(f"Input not found, skipping {label}: {input_path}")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Variant: {label}")
        logger.info(f"  Input : {input_path}")
        logger.info(f"  Output: {output_path}")

        samples = json.load(open(input_path))
        logger.info(f"  Loaded {len(samples)} samples")

        # Re-rank
        logger.info(f"  Re-ranking top-{RERANK_TOP_N} chunks …")
        reranked = rerank_predictions(samples, model, top_n=RERANK_TOP_N)

        # Save predictions
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(reranked, f, indent=2)
        logger.info(f"  Saved predictions → {output_path}")

        # Compute metrics
        metrics = compute_and_save_metrics(reranked, metrics_path, label)
        all_metrics[label] = metrics

    # ── Load existing baseline metrics for comparison ──────────────────────
    comparison: Dict[str, Dict] = {}
    for name, path in COMPARISON_BASELINES.items():
        if path.exists():
            comparison[name] = json.load(open(path))
        else:
            logger.warning(f"Baseline metrics not found (skipping plot): {path}")

    if len(comparison) >= 2:
        save_comparison_plots(comparison)
    else:
        logger.warning("Not enough baselines loaded for comparison plots.")

    # ── Print summary table ───────────────────────────────────────────────
    print("\n" + "=" * 90)
    header = f"{'Variant':<30}  {'page_rec@1':>10}  {'page_rec@5':>10}  {'chunk_rec@5':>11}  {'mrr':>8}"
    print(header)
    print("-" * 90)
    for name, m in comparison.items():
        ov = m.get("overall", {})
        print(
            f"{name:<30}  "
            f"{ov.get('page_recall@1', 0):>10.4f}  "
            f"{ov.get('page_recall@5', 0):>10.4f}  "
            f"{ov.get('chunk_recall@5', 0):>11.4f}  "
            f"{ov.get('mrr', 0):>8.4f}"
        )
    print("=" * 90 + "\n")

    logger.info("Done.")


if __name__ == "__main__":
    main()
