#!/usr/bin/env python3
"""
generate_thesis_plots.py
========================
Generates clean, focused paper-ready plots for the thesis.

Focused variant set: key baselines + progressive additions + best FT-CE-RR result.
Produces both Recall and Precision metrics, breakdowns by question type and doc type.

Run from repo root:
    python3 baselines/generate_thesis_plots.py
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR  = PROJECT_ROOT / "baselines/results/metrics"
PLOTS_DIR    = PROJECT_ROOT / "baselines/results/plots/thesis"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Focused variant set for paper (not too cluttered)
# ---------------------------------------------------------------------------
FOCUSED_VARIANTS = [
    "bm25",
    "dense_bge_m3",
    "splade",
    "hybrid_75_25",
    "parent_child",
    "query_expansion",
    "hyde",
    "multi_hyde",
    "bge_reranker",
    "multi_hyde_reranker",
    "multi_hyde_ft_reranker",   # Best result: ~0.55 PageRec@5
]

VARIANT_LABELS = {
    "bm25":                    "BM25",
    "dense_bge_m3":            "Dense BGE-M3",
    "splade":                  "SPLADE",
    "hybrid_75_25":            "Hybrid RRF (dense-heavy)",
    "parent_child":            "Parent-Child",
    "query_expansion":         "Query Expansion",
    "hyde":                    "HyDE",
    "multi_hyde":              "Multi-HyDE",
    "bge_reranker":            "BGE-M3 + ReRanker",
    "multi_hyde_reranker":     "Multi-HyDE + ReRanker",
    "multi_hyde_ft_reranker":  "Multi-HyDE + FT-ReRanker ★",
}

# Short labels for heatmaps / tight spaces
SHORT_LABELS = {
    "bm25":                    "BM25",
    "dense_bge_m3":            "BGE-M3",
    "splade":                  "SPLADE",
    "hybrid_75_25":            "Hybrid",
    "parent_child":            "Par-Child",
    "query_expansion":         "QE",
    "hyde":                    "HyDE",
    "multi_hyde":              "Multi-HyDE",
    "bge_reranker":            "BGE+RR",
    "multi_hyde_reranker":     "MH+RR",
    "multi_hyde_ft_reranker":  "MH+FT-RR ★",
}

K_VALUES   = [1, 3, 5, 10, 20]
MAIN_K     = 5
QUESTION_TYPES = ["metrics-generated", "domain-relevant", "novel-generated"]
DOC_TYPES  = ["10k", "10q", "8k", "Earnings"]   # canonical order

# Colours: gradient from blue→green for baselines, vivid purple for FT best
def _make_palette(names):
    base = cm.tab10(np.linspace(0, 0.9, len(names)))
    palette = {n: base[i] for i, n in enumerate(names)}
    palette["multi_hyde_ft_reranker"] = (0.55, 0.05, 0.55, 1.0)
    palette["multi_hyde_reranker"]    = (0.20, 0.55, 0.80, 1.0)
    return palette

# ---------------------------------------------------------------------------
# Load metrics
# ---------------------------------------------------------------------------
def load_all_metrics():
    overall, by_qt, by_dt = {}, {}, {}
    for name in FOCUSED_VARIANTS:
        path = METRICS_DIR / f"{name}_metrics.json"
        if not path.exists():
            print(f"  [SKIP] {path.name} not found")
            continue
        data = json.loads(path.read_text())
        overall[name] = data.get("overall", {})
        by_qt[name]   = data.get("by_question_type", {})
        by_dt[name]   = data.get("by_doc_type", {})
        pr5 = overall[name].get("page_recall@5", 0)
        pp5 = overall[name].get("page_precision@5", 0)
        print(f"  {name:<35} PageRec@5={pr5:.3f}  PagePrec@5={pp5:.3f}")
    return overall, by_qt, by_dt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _save(fig, stem: str):
    pdf_path = PLOTS_DIR / f"{stem}.pdf"
    png_path = PLOTS_DIR / f"{stem}.png"
    fig.savefig(str(pdf_path), bbox_inches="tight", dpi=150)
    fig.savefig(str(png_path), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved → {stem}.pdf / .png")


def _present(names, overall):
    return [n for n in names if n in overall]


# ---------------------------------------------------------------------------
# Plot 1 — PageRec@k + PagePrec@k curves (focused)
# ---------------------------------------------------------------------------
def plot_rec_prec_curves(overall, plots_dir=PLOTS_DIR):
    names   = _present(FOCUSED_VARIANTS, overall)
    palette = _make_palette(names)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (metric, ylabel) in zip(axes, [
        ("page_recall",    "Page Recall@k"),
        ("page_precision", "Page Precision@k"),
    ]):
        for name in names:
            vals = [overall[name].get(f"{metric}@{k}", 0) for k in K_VALUES]
            lw = 2.5 if name == "multi_hyde_ft_reranker" else 1.5
            ax.plot(K_VALUES, vals, marker="o",
                    label=SHORT_LABELS[name], color=palette[name],
                    linewidth=lw, zorder=5 if name == "multi_hyde_ft_reranker" else 2)
        ax.set_xlabel("k", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel, fontsize=11)
        ax.set_xticks(K_VALUES)
        ax.set_ylim(0, 1.0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower right" if metric == "page_recall" else "upper right",
                  ncol=1)

    fig.tight_layout()
    _save(fig, "rec_prec_at_k_curves")


# ---------------------------------------------------------------------------
# Plot 2 — Recall@k curves: doc + page  (focused)
# ---------------------------------------------------------------------------
def plot_recall_curves(overall, plots_dir=PLOTS_DIR):
    names   = _present(FOCUSED_VARIANTS, overall)
    palette = _make_palette(names)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (metric, ylabel) in zip(axes, [
        ("page_recall", "Page Recall@k"),
        ("doc_recall",  "Doc Recall@k"),
    ]):
        for name in names:
            vals = [overall[name].get(f"{metric}@{k}", 0) for k in K_VALUES]
            lw = 2.5 if name == "multi_hyde_ft_reranker" else 1.5
            ax.plot(K_VALUES, vals, marker="o",
                    label=SHORT_LABELS[name], color=palette[name],
                    linewidth=lw, zorder=5 if name == "multi_hyde_ft_reranker" else 2)
        ax.set_xlabel("k", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel, fontsize=11)
        ax.set_xticks(K_VALUES)
        ax.set_ylim(0, 1.0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower right", ncol=1)

    fig.tight_layout()
    _save(fig, "recall_at_k_curves_focused")


# ---------------------------------------------------------------------------
# Plot 3 — Bar chart: PageRec@5 + PagePrec@5 side by side (focused)
# ---------------------------------------------------------------------------
def plot_rec_prec_bar(overall, plots_dir=PLOTS_DIR):
    names   = _present(FOCUSED_VARIANTS, overall)
    palette = _make_palette(names)
    colors  = [palette[n] for n in names]
    labels  = [VARIANT_LABELS[n] for n in names]

    rec_vals  = [overall[n].get(f"page_recall@{MAIN_K}", 0) for n in names]
    prec_vals = [overall[n].get(f"page_precision@{MAIN_K}", 0) for n in names]

    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(13, 5))
    bars_r = ax.bar(x - w/2, rec_vals,  w, label=f"PageRec@{MAIN_K}",  color=colors, alpha=0.90)
    bars_p = ax.bar(x + w/2, prec_vals, w, label=f"PagePrec@{MAIN_K}", color=colors, alpha=0.45)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(f"Page-Level Recall vs Precision at k={MAIN_K}", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9)

    for bar, val in zip(bars_r, rec_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=7)
    for bar, val in zip(bars_p, prec_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    _save(fig, "rec_prec_bar_k5")


# ---------------------------------------------------------------------------
# Plot 4 — Precision–Recall scatter at k=5
# ---------------------------------------------------------------------------
def plot_prec_recall_scatter(overall, plots_dir=PLOTS_DIR):
    names   = _present(FOCUSED_VARIANTS, overall)
    palette = _make_palette(names)

    fig, ax = plt.subplots(figsize=(7, 6))
    for name in names:
        rec  = overall[name].get(f"page_recall@{MAIN_K}", 0)
        prec = overall[name].get(f"page_precision@{MAIN_K}", 0)
        ms = 120 if name == "multi_hyde_ft_reranker" else 80
        ax.scatter(rec, prec, s=ms, color=palette[name],
                   zorder=5 if name == "multi_hyde_ft_reranker" else 2,
                   edgecolors="black" if name == "multi_hyde_ft_reranker" else "none",
                   linewidth=1.0)
        offset_x = 0.005
        offset_y = 0.005
        ax.annotate(SHORT_LABELS[name], (rec, prec),
                    textcoords="offset points", xytext=(5, 4),
                    fontsize=7.5, color=palette[name])

    ax.set_xlabel(f"Page Recall@{MAIN_K}", fontsize=11)
    ax.set_ylabel(f"Page Precision@{MAIN_K}", fontsize=11)
    ax.set_title(f"Precision vs Recall @ k={MAIN_K} (page level)", fontsize=11)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 0.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "prec_recall_scatter_k5")


# ---------------------------------------------------------------------------
# Plot 5 — Heatmap: methods × question types (PageRec@5)
# ---------------------------------------------------------------------------
def plot_heatmap_question_type(by_qt, overall, plots_dir=PLOTS_DIR):
    names = _present(FOCUSED_VARIANTS, overall)
    if not names:
        return

    matrix = np.zeros((len(names), len(QUESTION_TYPES)))
    for i, name in enumerate(names):
        for j, qt in enumerate(QUESTION_TYPES):
            matrix[i, j] = by_qt.get(name, {}).get(qt, {}).get(f"page_recall@{MAIN_K}", 0)

    fig, ax = plt.subplots(figsize=(8, max(3.5, len(names) * 0.55)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(QUESTION_TYPES)))
    ax.set_xticklabels([t.replace("-", "\n") for t in QUESTION_TYPES], fontsize=9)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([SHORT_LABELS[n] for n in names], fontsize=8.5)
    for i in range(len(names)):
        for j in range(len(QUESTION_TYPES)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if matrix[i, j] > 0.6 else "black")
    plt.colorbar(im, ax=ax, label=f"PageRec@{MAIN_K}")
    ax.set_title(f"PageRec@{MAIN_K} by Question Type", fontsize=11)
    fig.tight_layout()
    _save(fig, "heatmap_question_type")


# ---------------------------------------------------------------------------
# Plot 6 — Heatmap: methods × doc types (PageRec@5)
# ---------------------------------------------------------------------------
def plot_heatmap_doc_type(by_dt, overall, plots_dir=PLOTS_DIR):
    names = _present(FOCUSED_VARIANTS, overall)
    if not names:
        return

    # Only include doc types that have data
    available_dt = set()
    for name in names:
        available_dt.update(by_dt.get(name, {}).keys())
    doc_types = [dt for dt in DOC_TYPES if dt in available_dt]
    # Also check for 'unknown' (legacy Earnings label)
    if not doc_types:
        return

    matrix = np.zeros((len(names), len(doc_types)))
    for i, name in enumerate(names):
        for j, dt in enumerate(doc_types):
            matrix[i, j] = by_dt.get(name, {}).get(dt, {}).get(f"page_recall@{MAIN_K}", 0)

    fig, ax = plt.subplots(figsize=(max(6, len(doc_types) * 1.6), max(3.5, len(names) * 0.55)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(doc_types)))
    ax.set_xticklabels(doc_types, rotation=20, ha="right", fontsize=10)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([SHORT_LABELS[n] for n in names], fontsize=8.5)
    for i in range(len(names)):
        for j in range(len(doc_types)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if matrix[i, j] > 0.6 else "black")
    plt.colorbar(im, ax=ax, label=f"PageRec@{MAIN_K}")
    ax.set_title(f"PageRec@{MAIN_K} by Document Type", fontsize=11)
    fig.tight_layout()
    _save(fig, "heatmap_doc_type")


# ---------------------------------------------------------------------------
# Plot 7 — Heatmap: methods × doc types (PagePrec@5)
# ---------------------------------------------------------------------------
def plot_heatmap_doc_type_prec(by_dt, overall, plots_dir=PLOTS_DIR):
    names = _present(FOCUSED_VARIANTS, overall)
    if not names:
        return

    available_dt = set()
    for name in names:
        available_dt.update(by_dt.get(name, {}).keys())
    doc_types = [dt for dt in DOC_TYPES if dt in available_dt]
    if not doc_types:
        return

    matrix = np.zeros((len(names), len(doc_types)))
    for i, name in enumerate(names):
        for j, dt in enumerate(doc_types):
            matrix[i, j] = by_dt.get(name, {}).get(dt, {}).get(f"page_precision@{MAIN_K}", 0)

    fig, ax = plt.subplots(figsize=(max(6, len(doc_types) * 1.6), max(3.5, len(names) * 0.55)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=0.4)
    ax.set_xticks(range(len(doc_types)))
    ax.set_xticklabels(doc_types, rotation=20, ha="right", fontsize=10)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([SHORT_LABELS[n] for n in names], fontsize=8.5)
    for i in range(len(names)):
        for j in range(len(doc_types)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if matrix[i, j] > 0.25 else "black")
    plt.colorbar(im, ax=ax, label=f"PagePrec@{MAIN_K}")
    ax.set_title(f"PagePrec@{MAIN_K} by Document Type", fontsize=11)
    fig.tight_layout()
    _save(fig, "heatmap_doc_type_precision")


# ---------------------------------------------------------------------------
# Plot 8 — Grouped bar by doc type (PageRec@5, focused)
# ---------------------------------------------------------------------------
def plot_doc_type_bar(by_dt, overall, plots_dir=PLOTS_DIR):
    names = _present(FOCUSED_VARIANTS, overall)
    if not names:
        return

    available_dt = set()
    for name in names:
        available_dt.update(by_dt.get(name, {}).keys())
    doc_types = [dt for dt in DOC_TYPES if dt in available_dt]
    if not doc_types:
        return

    palette = _make_palette(names)
    x = np.arange(len(doc_types))
    w = 0.8 / len(names)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, name in enumerate(names):
        vals = [by_dt.get(name, {}).get(dt, {}).get(f"page_recall@{MAIN_K}", 0)
                for dt in doc_types]
        lw = 1.5 if name == "multi_hyde_ft_reranker" else 0.8
        ax.bar(x + i*w - (len(names)-1)*w/2, vals, w,
               label=SHORT_LABELS[name], color=palette[name],
               edgecolor="black" if name == "multi_hyde_ft_reranker" else "none",
               linewidth=lw)

    ax.set_xticks(x)
    ax.set_xticklabels(doc_types, fontsize=11)
    ax.set_ylabel(f"PageRec@{MAIN_K}", fontsize=11)
    ax.set_title(f"PageRec@{MAIN_K} by Document Type", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    fig.tight_layout()
    _save(fig, "doc_type_bar_recall")


# ---------------------------------------------------------------------------
# Plot 9 — Grouped bar by question type (PageRec@5, focused)
# ---------------------------------------------------------------------------
def plot_question_type_bar(by_qt, overall, plots_dir=PLOTS_DIR):
    names = _present(FOCUSED_VARIANTS, overall)
    if not names:
        return

    palette = _make_palette(names)
    x = np.arange(len(QUESTION_TYPES))
    w = 0.8 / len(names)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, name in enumerate(names):
        vals = [by_qt.get(name, {}).get(qt, {}).get(f"page_recall@{MAIN_K}", 0)
                for qt in QUESTION_TYPES]
        lw = 1.5 if name == "multi_hyde_ft_reranker" else 0.8
        ax.bar(x + i*w - (len(names)-1)*w/2, vals, w,
               label=SHORT_LABELS[name], color=palette[name],
               edgecolor="black" if name == "multi_hyde_ft_reranker" else "none",
               linewidth=lw)

    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("-", "\n") for t in QUESTION_TYPES], fontsize=10)
    ax.set_ylabel(f"PageRec@{MAIN_K}", fontsize=11)
    ax.set_title(f"PageRec@{MAIN_K} by Question Type", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    fig.tight_layout()
    _save(fig, "question_type_bar_recall")


# ---------------------------------------------------------------------------
# Plot 10 — MRR bar (focused)
# ---------------------------------------------------------------------------
def plot_mrr_bar(overall, plots_dir=PLOTS_DIR):
    names   = _present(FOCUSED_VARIANTS, overall)
    palette = _make_palette(names)
    colors  = [palette[n] for n in names]
    labels  = [VARIANT_LABELS[n] for n in names]
    mrr_vals = [overall[n].get("mrr", 0) for n in names]

    fig, ax = plt.subplots(figsize=(13, 4))
    bars = ax.bar(np.arange(len(names)), mrr_vals, color=colors, alpha=0.88)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel("MRR", fontsize=11)
    ax.set_title("Mean Reciprocal Rank (chunk-level)", fontsize=11)
    ax.set_ylim(0, max(mrr_vals) * 1.2 if mrr_vals else 1)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, mrr_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)
    fig.tight_layout()
    _save(fig, "mrr_bar_focused")


# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------
def print_summary(overall):
    present = _present(FOCUSED_VARIANTS, overall)
    hdr = f"{'Method':<32} {'DocRec@5':>9} {'PageRec@5':>9} {'PagePrec@5':>10} {'DocPrec@5':>9} {'MRR':>7}"
    print("\n" + "="*len(hdr))
    print(hdr)
    print("-"*len(hdr))
    for name in present:
        m = overall[name]
        star = " ★" if name == "multi_hyde_ft_reranker" else ""
        print(f"{VARIANT_LABELS[name]+star:<32} "
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
    overall, by_qt, by_dt = load_all_metrics()
    print(f"\nLoaded {len(overall)} variants")

    print(f"\nGenerating thesis plots → {PLOTS_DIR}")
    plot_rec_prec_curves(overall)
    plot_recall_curves(overall)
    plot_rec_prec_bar(overall)
    plot_prec_recall_scatter(overall)
    plot_heatmap_question_type(by_qt, overall)
    plot_heatmap_doc_type(by_dt, overall)
    plot_heatmap_doc_type_prec(by_dt, overall)
    plot_doc_type_bar(by_dt, overall)
    plot_question_type_bar(by_qt, overall)
    plot_mrr_bar(overall)
    print_summary(overall)
    print("\nDone.")
