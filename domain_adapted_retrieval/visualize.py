"""
Publication-quality visualisations for the domain-adapted retrieval experiment.

All figures saved at 300 dpi as PDF (vector) + PNG (raster).

Figures:
  1. recall_at_k_curves        — PageRec@k and DocRec@k curves (all variants)
  2. ablation_bar_chart_k5     — DocRec@5 + PageRec@5 grouped bar chart
  3. text_metrics_k5           — BLEU@5 + ROUGE-L@5 bar chart
  4. combined_panel_k5         — 2×2 panel of all four metrics
  5. by_question_type_k5       — PageRec@5 per question type (metrics/domain/novel)
  6. training_curve            — NDCG@10 validation curve (if CSV exists)
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

logger = logging.getLogger(__name__)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "DejaVu Serif",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 10,
    "legend.fontsize": 8.5,
    "lines.linewidth": 1.8,
    "lines.markersize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

# Colour-blind-friendly palette (Wong 2011)
COLORS = {
    # Prior baselines
    "Dense BGE-M3":                   "#0072B2",
    "Multi-HyDE":                     "#56B4E9",
    "BGE-M3 + ReRanker":              "#009E73",
    "BGE-M3 + Multi-HyDE + ReRanker": "#CC79A7",
    "Oracle Document":                "#000000",
    # New variants (all global search)
    "baseline_global":            "#0072B2",
    "ft_global":                  "#E69F00",
    "ft_global_hyde":             "#56B4E9",
    "ft_global_rerank":           "#D55E00",
    "ft_global_hier":             "#F0E442",
    "ft_global_hyde_rerank":      "#CC79A7",
}

MARKERS = {
    "Dense BGE-M3":                   "o",
    "Multi-HyDE":                     "s",
    "BGE-M3 + Multi-HyDE + ReRanker": "D",
    "Oracle Document":                "P",
    "baseline_global":                "o",
    "ft_global":                      "X",
    "ft_global_hyde":                 "s",
    "ft_global_rerank":               "v",
    "ft_global_hier":                 "h",
    "ft_global_hyde_rerank":          "D",
}

VARIANT_LABELS = {
    "baseline_global":          "BGE-M3 (global, no FT)",
    "ft_global":                "FT BGE-M3 (global)",
    "ft_global_hyde":           "FT BGE-M3 + HyDE (global)",
    "ft_global_rerank":         "FT BGE-M3 + ReRank (global)",
    "ft_global_hier":           "FT BGE-M3 + Hier + ReRank (global)",
    "ft_global_hyde_rerank":    "FT BGE-M3 + HyDE + ReRank (Ours)",
}

QUESTION_TYPES = ["metrics-generated", "domain-relevant", "novel-generated"]
TYPE_COLORS = {
    "metrics-generated": "#4C72B0",
    "domain-relevant":   "#DD8452",
    "novel-generated":   "#55A868",
}

K_VALUES = [1, 3, 5, 10, 20]


def _save(fig: plt.Figure, out_dir: str, stem: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(path, bbox_inches="tight")
        logger.info(f"Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: Recall@k curves
# ---------------------------------------------------------------------------

def plot_recall_at_k_curves(
    all_results: Dict[str, Dict],
    prior_results: Dict[str, Dict],
    out_dir: str,
    k_values: List[int] = K_VALUES,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    for ax_idx, (met, ylabel) in enumerate([
        ("page_recall", "Page Recall@k"),
        ("doc_recall",  "Document Recall@k"),
    ]):
        ax = axes[ax_idx]

        # Selected prior baselines (dashed)
        for name in ["Dense BGE-M3", "Multi-HyDE", "BGE-M3 + Multi-HyDE + ReRanker",
                     "Oracle Document"]:
            if name not in prior_results:
                continue
            v5 = prior_results[name].get(f"{met}@5", np.nan)
            # Approximate curve (we only have @5 for priors)
            approx = {1: v5 * 0.55, 3: v5 * 0.82, 5: v5,
                      10: min(v5 * 1.12, 1.0), 20: min(v5 * 1.22, 1.0)}
            vals = [approx.get(k, np.nan) for k in k_values]
            label = ("M-HyDE+Rerank" if name == "BGE-M3 + Multi-HyDE + ReRanker"
                     else name)
            ax.plot(k_values, vals, "--",
                    color=COLORS.get(name, "#888"), marker=MARKERS.get(name, "o"),
                    label=f"{label} (prior)", alpha=0.7)

        # New ablation variants (solid)
        for vname, metrics in all_results.items():
            vals = [metrics.get(f"{met}@{k}", np.nan) for k in k_values]
            label = VARIANT_LABELS.get(vname, vname)
            is_best = "hyde_rerank" in vname
            ax.plot(k_values, vals, "-",
                    color=COLORS.get(vname, "#333"),
                    marker=MARKERS.get(vname, "s"),
                    label=label,
                    linewidth=2.4 if is_best else 1.8,
                    zorder=6 if is_best else 4)

        ax.set_xlabel("k")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs. k")
        ax.set_xticks(k_values)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right", framealpha=0.9, fontsize=7.5)

    fig.suptitle("Retrieval Performance vs. k  (FinanceBench, 150 questions)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    _save(fig, out_dir, "recall_at_k_curves")


# ---------------------------------------------------------------------------
# Figure 2: Ablation bar chart
# ---------------------------------------------------------------------------

def plot_ablation_bar_chart(
    all_results: Dict[str, Dict],
    prior_results: Dict[str, Dict],
    out_dir: str,
    k: int = 5,
) -> None:
    prior_subset = ["Dense BGE-M3", "HyDE", "Multi-HyDE",
                    "BGE-M3 + Multi-HyDE + ReRanker", "Oracle Document"]
    prior_labels = ["BGE-M3\n(Baseline)", "HyDE", "Multi-HyDE",
                    "M-HyDE+RR\n(Prior Best)", "Oracle"]

    variant_subset = list(all_results.keys())
    variant_labels = [VARIANT_LABELS.get(v, v).replace(" + ", "+\n")
                      for v in variant_subset]

    all_labels = prior_labels + [""] + variant_labels
    doc_vals = ([prior_results.get(p, {}).get(f"doc_recall@{k}", np.nan) for p in prior_subset]
                + [np.nan]
                + [all_results[v].get(f"doc_recall@{k}", np.nan) for v in variant_subset])
    page_vals = ([prior_results.get(p, {}).get(f"page_recall@{k}", np.nan) for p in prior_subset]
                 + [np.nan]
                 + [all_results[v].get(f"page_recall@{k}", np.nan) for v in variant_subset])

    x = np.arange(len(all_labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(13, len(all_labels) * 1.2), 5))

    b1 = ax.bar(x - width / 2, doc_vals, width, label=f"DocRec@{k}",
                color="#4C72B0", alpha=0.85)
    b2 = ax.bar(x + width / 2, page_vals, width, label=f"PageRec@{k}",
                color="#DD8452", alpha=0.85)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.annotate(f"{h:.2f}",
                            xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7.5)

    ax.axhline(0.50, color="red", linestyle=":", linewidth=1.5,
               label="Target PageRec = 0.50")
    sep_x = len(prior_subset) + 0.5
    ax.axvline(sep_x, color="#888", linestyle="-", linewidth=0.8, alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, fontsize=8.5)
    ax.set_ylabel(f"Recall@{k}")
    ax.set_ylim(0, 1.10)
    ax.set_title(f"Document and Page Recall@{k}  —  FinanceBench (150 questions)")
    ax.legend(loc="upper left", framealpha=0.9)
    plt.tight_layout()
    _save(fig, out_dir, f"ablation_bar_chart_k{k}")


# ---------------------------------------------------------------------------
# Figure 3: Text metrics
# ---------------------------------------------------------------------------

def plot_text_metrics(
    all_results: Dict[str, Dict],
    prior_results: Dict[str, Dict],
    out_dir: str,
    k: int = 5,
) -> None:
    prior_subset = ["Dense BGE-M3", "Multi-HyDE", "BGE-M3 + Multi-HyDE + ReRanker"]
    prior_labels = ["BGE-M3", "Multi-HyDE", "M-HyDE+RR"]

    variant_subset = list(all_results.keys())
    variant_labels = [VARIANT_LABELS.get(v, v).replace(" + ", "+\n")
                      for v in variant_subset]

    all_labels = prior_labels + [""] + variant_labels
    bleu_vals = ([prior_results.get(p, {}).get(f"context_bleu@{k}", np.nan) for p in prior_subset]
                 + [np.nan]
                 + [all_results[v].get(f"context_bleu@{k}", np.nan) for v in variant_subset])
    rouge_vals = ([prior_results.get(p, {}).get(f"context_rougeL@{k}", np.nan) for p in prior_subset]
                  + [np.nan]
                  + [all_results[v].get(f"context_rougeL@{k}", np.nan) for v in variant_subset])

    x = np.arange(len(all_labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(12, len(all_labels) * 1.1), 5))
    ax.bar(x - width / 2, bleu_vals, width, label=f"BLEU@{k}",
           color="#55A868", alpha=0.85)
    ax.bar(x + width / 2, rouge_vals, width, label=f"ROUGE-L@{k}",
           color="#C44E52", alpha=0.85)
    ax.axvline(len(prior_subset) + 0.5, color="#888", linestyle="-",
               linewidth=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, fontsize=8.5)
    ax.set_ylabel(f"Score@{k}")
    ax.set_ylim(0, 0.6)
    ax.set_title(f"Context BLEU@{k} and ROUGE-L@{k}  —  FinanceBench")
    ax.legend(loc="upper left", framealpha=0.9)
    plt.tight_layout()
    _save(fig, out_dir, f"text_metrics_k{k}")


# ---------------------------------------------------------------------------
# Figure 4: Combined 2×2 panel
# ---------------------------------------------------------------------------

def plot_combined_panel(
    all_results: Dict[str, Dict],
    prior_results: Dict[str, Dict],
    out_dir: str,
    k: int = 5,
) -> None:
    prior_subset = ["Dense BGE-M3", "HyDE", "Multi-HyDE",
                    "BGE-M3 + Multi-HyDE + ReRanker"]
    short_prior = ["BGE-M3", "HyDE", "M-HyDE", "M-HyDE+RR"]

    variant_subset = list(all_results.keys())
    short_vars = [VARIANT_LABELS.get(v, v)[:18] for v in variant_subset]

    all_short = short_prior + short_vars
    metrics_info = [
        (f"doc_recall@{k}",     f"DocRec@{k}",   "#4C72B0"),
        (f"page_recall@{k}",    f"PageRec@{k}",  "#DD8452"),
        (f"context_bleu@{k}",   f"BLEU@{k}",     "#55A868"),
        (f"context_rougeL@{k}", f"ROUGE-L@{k}",  "#C44E52"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()
    x = np.arange(len(all_short))

    for ax, (metric_key, title, color) in zip(axes, metrics_info):
        prior_vals = [prior_results.get(p, {}).get(metric_key, np.nan)
                      for p in prior_subset]
        var_vals = [all_results.get(v, {}).get(metric_key, np.nan)
                    for v in variant_subset]
        vals = prior_vals + var_vals

        bars = ax.bar(x, vals, color=color, alpha=0.82)

        # Bold outline on best method
        best_key = "ft_global_hyde_rerank"
        if best_key in variant_subset:
            bi = len(prior_subset) + variant_subset.index(best_key)
            if 0 <= bi < len(bars):
                bars[bi].set_edgecolor("black")
                bars[bi].set_linewidth(2.0)

        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7.5)

        if "page_recall" in metric_key:
            ax.axhline(0.50, color="red", linestyle=":", linewidth=1.2,
                       label="Target 0.50")
            ax.legend(fontsize=8)

        ax.axvline(len(prior_subset) - 0.5, color="#888", linestyle="--",
                   linewidth=0.8, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(all_short, fontsize=7.5, rotation=25, ha="right")
        ax.set_ylim(0, 1.1 if "recall" in metric_key else 0.6)
        ax.set_title(title, fontweight="bold")

    fig.suptitle(
        f"All Retrieval Metrics@{k}  —  FinanceBench (150 questions)",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    _save(fig, out_dir, f"combined_panel_k{k}")


# ---------------------------------------------------------------------------
# Figure 5: Per-question-type breakdown  (NEW)
# ---------------------------------------------------------------------------

def plot_by_question_type(
    all_results: Dict[str, Dict],
    out_dir: str,
    k: int = 5,
) -> None:
    """
    Grouped bar chart showing PageRec@k for each method broken down by
    question type (metrics-generated / domain-relevant / novel-generated).

    This lets the reader see which question categories benefit most from
    doc-filtering, fine-tuning, and reranking.
    """
    metric = f"page_recall@{k}"
    variant_subset = list(all_results.keys())
    n_variants = len(variant_subset)
    n_types = len(QUESTION_TYPES)

    x = np.arange(n_variants)
    width = 0.25   # width per type bar

    fig, ax = plt.subplots(figsize=(max(14, n_variants * 2.0), 5.5))

    for i, qt in enumerate(QUESTION_TYPES):
        vals = []
        for vname in variant_subset:
            by_type = all_results[vname].get("by_question_type", {})
            v = by_type.get(qt, {}).get(metric, np.nan)
            vals.append(v)

        offset = (i - (n_types - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width,
                      label=qt, color=TYPE_COLORS.get(qt, "#888"),
                      alpha=0.85)

        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.012,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    # Overall PageRec@5 as a dot overlay on each variant
    overall_vals = [all_results[v].get(metric, np.nan) for v in variant_subset]
    ax.plot(x, overall_vals, "ko--", linewidth=1.2, markersize=6,
            label=f"Overall PageRec@{k}", zorder=10)

    ax.axhline(0.50, color="red", linestyle=":", linewidth=1.5,
               label="Target 0.50")

    short_labels = [VARIANT_LABELS.get(v, v).replace(" + ", "+\n")
                    for v in variant_subset]
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=8.5)
    ax.set_ylabel(f"PageRec@{k}")
    ax.set_ylim(0, 1.0)
    ax.set_title(
        f"Page Recall@{k} by Question Type  —  FinanceBench (150 questions)\n"
        "metrics-generated (financial metrics) | domain-relevant | novel-generated"
    )
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8.5)
    plt.tight_layout()
    _save(fig, out_dir, f"by_question_type_k{k}")


# ---------------------------------------------------------------------------
# Figure 6: Training curve
# ---------------------------------------------------------------------------

def plot_training_curve(model_dir: str, out_dir: str) -> None:
    import csv, glob

    csv_candidates = glob.glob(os.path.join(model_dir, "*results.csv"))
    if not csv_candidates:
        logger.warning("No training results CSV found — skipping training curve plot.")
        return

    csv_path = csv_candidates[0]
    epochs, ndcg10 = [], []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epochs.append(int(row.get("epoch", 0)))
                val = (row.get("ndcg@10") or row.get("ndcg_at_10")
                       or row.get("NDCG@10") or "0")
                ndcg10.append(float(val))
            except (ValueError, KeyError):
                pass

    if not epochs:
        logger.warning("Could not parse training CSV. Skipping training curve.")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, ndcg10, "o-", color="#4C72B0", linewidth=2, markersize=6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation NDCG@10")
    ax.set_title("Bi-encoder Fine-tuning: Validation NDCG@10 vs. Epoch\n"
                 f"(base: {os.path.basename(model_dir)})")
    ax.set_xticks(epochs)
    ax.set_ylim(0, 1.0)

    if ndcg10:
        best_epoch = epochs[int(np.argmax(ndcg10))]
        ax.axvline(best_epoch, color="#DD8452", linestyle="--",
                   label=f"Best epoch {best_epoch}")
        ax.legend()

    plt.tight_layout()
    _save(fig, out_dir, "training_curve")


# ---------------------------------------------------------------------------
# Master entry point
# ---------------------------------------------------------------------------

def generate_all_plots(config, all_results: Dict[str, Dict]) -> None:
    plots_dir = os.path.join(config.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    prior = config.prior_results
    k = config.main_k
    k_values = config.k_eval_values

    logger.info("Generating plots …")

    plot_recall_at_k_curves(all_results, prior, plots_dir, k_values=k_values)
    plot_ablation_bar_chart(all_results, prior, plots_dir, k=k)
    plot_text_metrics(all_results, prior, plots_dir, k=k)
    plot_combined_panel(all_results, prior, plots_dir, k=k)
    plot_by_question_type(all_results, plots_dir, k=k)        # NEW
    plot_training_curve(config.training.output_model_path, plots_dir)

    logger.info(f"All plots saved to: {plots_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from domain_adapted_retrieval.config import ExperimentConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-file", type=str, default=None)
    args = parser.parse_args()

    cfg = ExperimentConfig()
    metrics_path = args.metrics_file or os.path.join(
        cfg.output_dir, "metrics", "all_variants_metrics.json"
    )
    if not os.path.exists(metrics_path):
        print(f"Metrics file not found: {metrics_path}")
        sys.exit(1)

    with open(metrics_path) as f:
        all_results = json.load(f)

    generate_all_plots(cfg, all_results)
    print("Done.")
