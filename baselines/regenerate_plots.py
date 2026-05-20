#!/usr/bin/env python3
"""
regenerate_plots.py
===================
Loads all saved per-variant metrics JSONs (including FT-reranker) and
regenerates every publication-quality plot in baselines/results/plots/.

Run from the repo root:
    python3 baselines/regenerate_plots.py
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR  = PROJECT_ROOT / "baselines/results/metrics"
PLOTS_DIR    = PROJECT_ROOT / "baselines/results/plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Variant ordering + labels  (add FT-reranker variants here)
# ---------------------------------------------------------------------------
VARIANTS_ORDERED = [
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

VARIANT_LABELS = {
    "bm25":                       "BM25",
    "splade":                     "SPLADE",
    "hybrid_25_75":               "Hybrid RRF 25/75 (sparse-heavy)",
    "hybrid_50_50":               "Hybrid RRF 50/50",
    "hybrid_75_25":               "Hybrid RRF 75/25 (dense-heavy)",
    "dense_bge_m3":               "Dense BGE-M3",
    "parent_child":               "Parent-Child",
    "query_expansion":            "Query Expansion",
    "hyde":                       "HyDE",
    "multi_hyde":                 "Multi-HyDE",
    "bge_reranker":               "BGE-M3 + ReRanker",
    "multi_hyde_reranker":        "BGE-M3 + Multi-HyDE + ReRanker",
    "dense_bge_m3_ft_reranker":   "Dense BGE-M3 + FT-ReRanker",
    "multi_hyde_ft_reranker":     "Multi-HyDE + FT-ReRanker  ★",
}

# Highlight tier colours for the two FT-reranker variants
FT_HIGHLIGHT = {"dense_bge_m3_ft_reranker", "multi_hyde_ft_reranker"}

K_VALUES     = [1, 3, 5, 10, 20]
MAIN_K       = 5
QUESTION_TYPES = ["metrics-generated", "domain-relevant", "novel-generated"]
DOC_TYPES    = ["10k", "10q", "8k", "Earnings"]  # canonical; 'Earnings' replaces old 'unknown'


# ---------------------------------------------------------------------------
# Load metrics
# ---------------------------------------------------------------------------
def load_all_metrics():
    """
    Returns three dicts:
      all_results[name]        = overall metrics dict
      by_type[name]            = {question_type: metrics}
      by_doctype[name]         = {doc_type: metrics}
    """
    all_results, by_type, by_doctype = {}, {}, {}

    for name in VARIANTS_ORDERED:
        path = METRICS_DIR / f"{name}_metrics.json"
        if not path.exists():
            print(f"  [SKIP] {path.name} not found")
            continue
        data = json.loads(path.read_text())
        all_results[name] = data.get("overall", {})
        by_type[name]     = data.get("by_question_type", {})
        by_doctype[name]  = data.get("by_doc_type", {})
        pr5 = all_results[name].get('page_recall@5', 0)
        pp5 = all_results[name].get('page_precision@5', 0)
        print(f"  Loaded {name}: page_recall@5={pr5:.3f}  page_precision@5={pp5:.3f}")

    return all_results, by_type, by_doctype


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _save_fig(fig, path: Path):
    path = Path(path)
    fig.savefig(str(path), bbox_inches="tight", dpi=150)
    fig.savefig(str(path.with_suffix(".png")), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved → {path.name}")


def _variant_colors(names):
    """Assign colours; FT-reranker variants get vivid reds to stand out."""
    base_colors = cm.tab20(np.linspace(0, 1, len(VARIANTS_ORDERED)))
    color_map = {n: base_colors[i] for i, n in enumerate(VARIANTS_ORDERED)}
    color_map["dense_bge_m3_ft_reranker"] = (0.85, 0.20, 0.10, 1.0)  # vivid red
    color_map["multi_hyde_ft_reranker"]   = (0.60, 0.05, 0.50, 1.0)  # vivid purple
    return [color_map[n] for n in names]


# ---------------------------------------------------------------------------
# Plot 1 — Recall@k curves (PageRec + DocRec)
# ---------------------------------------------------------------------------
def plot_recall_at_k(all_results, plots_dir):
    present = [n for n in VARIANTS_ORDERED if n in all_results]
    colors  = _variant_colors(present)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, metric_prefix, ylabel in [
        (axes[0], "page_recall", "Page Recall@k"),
        (axes[1], "doc_recall",  "Doc Recall@k"),
    ]:
        for name, color in zip(present, colors):
            vals = [all_results[name].get(f"{metric_prefix}@{k}", 0) for k in K_VALUES]
            lw   = 2.5 if name in FT_HIGHLIGHT else 1.5
            ls   = "-"
            zorder = 5 if name in FT_HIGHLIGHT else 2
            ax.plot(K_VALUES, vals, marker="o", label=VARIANT_LABELS[name],
                    color=color, linewidth=lw, linestyle=ls, zorder=zorder)
        ax.set_xlabel("k", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel + " curves (global search, 84 docs)", fontsize=11)
        ax.set_xticks(K_VALUES)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6.5, loc="lower right")

    fig.tight_layout()
    _save_fig(fig, plots_dir / "recall_at_k_curves.pdf")


# ---------------------------------------------------------------------------
# Plot 2 — Bar chart at k=5
# ---------------------------------------------------------------------------
def plot_retrieval_bar_k5(all_results, plots_dir):
    present = [n for n in VARIANTS_ORDERED if n in all_results]
    labels  = [VARIANT_LABELS[n] for n in present]
    colors  = _variant_colors(present)

    page_vals = [all_results[n].get(f"page_recall@{MAIN_K}", 0) for n in present]
    doc_vals  = [all_results[n].get(f"doc_recall@{MAIN_K}",  0) for n in present]

    x   = np.arange(len(present))
    w   = 0.38
    fig, ax = plt.subplots(figsize=(max(13, len(present) * 1.1), 5))
    bars1 = ax.bar(x - w/2, page_vals, w, label=f"PageRec@{MAIN_K}", color=colors, alpha=0.85)
    bars2 = ax.bar(x + w/2, doc_vals,  w, label=f"DocRec@{MAIN_K}",  color=colors, alpha=0.45)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Recall", fontsize=11)
    ax.set_title(f"Retrieval Performance at k={MAIN_K} (global search, 84 docs)", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9)

    # Value labels on page_recall bars
    for bar, val in zip(bars1, page_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=6.5)

    fig.tight_layout()
    _save_fig(fig, plots_dir / "retrieval_bar_k5.pdf")


# ---------------------------------------------------------------------------
# Plot 3 — Heatmap: methods × question types
# ---------------------------------------------------------------------------
def plot_heatmap_question_type(by_type, plots_dir):
    present = [n for n in VARIANTS_ORDERED if n in by_type]
    types   = QUESTION_TYPES
    matrix  = np.zeros((len(present), len(types)))
    for i, name in enumerate(present):
        for j, qt in enumerate(types):
            matrix[i, j] = by_type[name].get(qt, {}).get(f"page_recall@{MAIN_K}", 0)

    fig, ax = plt.subplots(figsize=(8, max(4, len(present) * 0.58)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels([t.replace("-", "\n") for t in types], fontsize=9)
    ax.set_yticks(range(len(present)))
    ylabels = [VARIANT_LABELS[n] for n in present]
    ax.set_yticklabels(ylabels, fontsize=8)
    for i in range(len(present)):
        for j in range(len(types)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if matrix[i, j] < 0.6 else "white")
    plt.colorbar(im, ax=ax, label=f"PageRec@{MAIN_K}")
    ax.set_title(f"PageRec@{MAIN_K} by Question Type", fontsize=11)
    fig.tight_layout()
    _save_fig(fig, plots_dir / "heatmap_by_question_type.pdf")


# ---------------------------------------------------------------------------
# Plot 4 — Heatmap: methods × doc types
# ---------------------------------------------------------------------------
def plot_heatmap_doc_type(by_doctype, plots_dir):
    present   = [n for n in VARIANTS_ORDERED if n in by_doctype]
    available = {dt for m in by_doctype.values() for dt in m}
    # Use canonical order; fall back to sorted() for any extra types
    doc_types = [dt for dt in DOC_TYPES if dt in available]
    doc_types += sorted(available - set(DOC_TYPES))
    if not present or not doc_types:
        return
    matrix = np.zeros((len(present), len(doc_types)))
    for i, name in enumerate(present):
        for j, dt in enumerate(doc_types):
            matrix[i, j] = by_doctype[name].get(dt, {}).get(f"page_recall@{MAIN_K}", 0)

    fig, ax = plt.subplots(figsize=(max(6, len(doc_types) * 1.5), max(4, len(present) * 0.58)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(doc_types)))
    ax.set_xticklabels([dt.replace("_", " ") for dt in doc_types], rotation=25, ha="right", fontsize=8)
    ax.set_yticks(range(len(present)))
    ax.set_yticklabels([VARIANT_LABELS[n] for n in present], fontsize=8)
    for i in range(len(present)):
        for j in range(len(doc_types)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if matrix[i, j] < 0.6 else "white")
    plt.colorbar(im, ax=ax, label=f"PageRec@{MAIN_K}")
    ax.set_title(f"PageRec@{MAIN_K} by Document Type", fontsize=11)
    fig.tight_layout()
    _save_fig(fig, plots_dir / "heatmap_by_doc_type.pdf")


# ---------------------------------------------------------------------------
# Plot 5 — Question-type grouped bar (PageRec@5 per type per method)
# ---------------------------------------------------------------------------
def plot_question_type_breakdown(by_type, plots_dir):
    present = [n for n in VARIANTS_ORDERED if n in by_type]
    types   = QUESTION_TYPES
    colors  = _variant_colors(present)

    x = np.arange(len(types))
    w = 0.8 / len(present)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (name, color) in enumerate(zip(present, colors)):
        vals = [by_type[name].get(qt, {}).get(f"page_recall@{MAIN_K}", 0) for qt in types]
        lw   = 1.5 if name in FT_HIGHLIGHT else 0.8
        ax.bar(x + i*w - (len(present)-1)*w/2, vals, w,
               label=VARIANT_LABELS[name], color=color,
               edgecolor="black" if name in FT_HIGHLIGHT else "none",
               linewidth=lw)

    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("-", "\n") for t in types], fontsize=10)
    ax.set_ylabel(f"PageRec@{MAIN_K}", fontsize=11)
    ax.set_title(f"PageRec@{MAIN_K} by Question Type", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=6, loc="upper right", ncol=2)
    fig.tight_layout()
    _save_fig(fig, plots_dir / "question_type_breakdown.pdf")


# ---------------------------------------------------------------------------
# Plot 6 — MRR bar chart
# ---------------------------------------------------------------------------
def plot_mrr(all_results, plots_dir):
    present = [n for n in VARIANTS_ORDERED if n in all_results]
    colors  = _variant_colors(present)
    labels  = [VARIANT_LABELS[n] for n in present]
    mrr_vals = [all_results[n].get("mrr", 0) for n in present]

    x   = np.arange(len(present))
    fig, ax = plt.subplots(figsize=(max(13, len(present)*1.1), 4))
    bars = ax.bar(x, mrr_vals, color=colors, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("MRR", fontsize=11)
    ax.set_title("Mean Reciprocal Rank (page-level)", fontsize=11)
    ax.set_ylim(0, max(mrr_vals) * 1.15)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, mrr_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    _save_fig(fig, plots_dir / "mrr_bar.pdf")


# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------
def print_summary_table(all_results):
    present = [n for n in VARIANTS_ORDERED if n in all_results]
    hdr = (f"{'Method':<42} {'DocRec@5':>9} {'PageRec@5':>9} "
           f"{'PagePrec@5':>10} {'DocPrec@5':>9} {'MRR':>7}")
    print("\n" + "="*len(hdr))
    print(hdr)
    print("-"*len(hdr))
    for name in present:
        m = all_results[name]
        marker = " ★" if name in FT_HIGHLIGHT else ""
        print(f"{VARIANT_LABELS[name]+marker:<42} "
              f"{m.get('doc_recall@5',0):>9.3f} "
              f"{m.get('page_recall@5',0):>9.3f} "
              f"{m.get('page_precision@5',0):>10.3f} "
              f"{m.get('doc_precision@5',0):>9.3f} "
              f"{m.get('mrr',0):>7.3f}")
    print("="*len(hdr))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    print("Loading metrics…")
    all_results, by_type, by_doctype = load_all_metrics()

    print(f"\nGenerating plots → {PLOTS_DIR}")
    plot_recall_at_k(all_results, PLOTS_DIR)
    plot_retrieval_bar_k5(all_results, PLOTS_DIR)
    plot_heatmap_question_type(by_type, PLOTS_DIR)
    plot_heatmap_doc_type(by_doctype, PLOTS_DIR)
    plot_question_type_breakdown(by_type, PLOTS_DIR)
    plot_mrr(all_results, PLOTS_DIR)

    print_summary_table(all_results)
    print("\nDone.")
