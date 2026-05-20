#!/usr/bin/env python3
"""
plot_gen_thesis.py
==================
Thesis-quality plots for the generative evaluation results.

Reads from:
  outputs/gen_comparison_{condition}/by_doc_type.json
  outputs/gen_comparison_{condition}/by_question_type.json
  outputs/gen_comparison_{condition}/summary.json

Produces 6 plots saved to outputs/gen_plots_thesis/:
  DOC TYPE
    plot1_numatch_heatmap_doctype.pdf/.png
    plot2_rougel_doctype_retrieval.pdf/.png
    plot3_bertscore_doctype.pdf/.png

  QUESTION TYPE
    plot4_numatch_qtype_retrieval.pdf/.png
    plot5_rougel_qtype_breakdown.pdf/.png
    plot6_metric_disconnect.pdf/.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          11,
    "axes.titlesize":     12,
    "axes.labelsize":     11,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    9,
    "legend.framealpha":  0.85,
    "figure.dpi":         150,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "axes.grid.axis":     "y",
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
})

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

CONDITIONS = {
    "Dense BGE-M3":          "gen_comparison_dense_bge_m3",
    "Multi-HyDE + RR":       "gen_comparison_multi_hyde_rr",
    "Multi-HyDE + FT-RR":    "gen_comparison_multi_hyde_ftrr",
    "Oracle Doc":            "gen_comparison_oracle_doc",
    "Oracle Page":           "gen_comparison_oracle_page",
}
MODELS       = ["Qwen2.5-7B", "Llama3.1-8B"]
MODEL_LABELS = ["Qwen2.5-7B", "Llama 3.1-8B"]
DOC_TYPES    = ["10k", "10q", "8k", "Earnings"]
Q_TYPES      = ["metrics-generated", "domain-relevant", "novel-generated"]
Q_LABELS     = ["Metrics", "Domain", "Novel"]

# Palette
COND_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
MODEL_COLORS = ["#2166AC", "#D6604D"]
MODEL_HATCHES = ["", "//"]

OUT_DIR = ROOT / "outputs" / "gen_plots_thesis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load(condition_dir: str) -> dict:
    base = ROOT / "outputs" / condition_dir
    return {
        "summary":    json.loads((base / "summary.json").read_text()),
        "by_doc":     json.loads((base / "by_doc_type.json").read_text()),
        "by_qtype":   json.loads((base / "by_question_type.json").read_text()),
    }


data = {label: load(d) for label, d in CONDITIONS.items()}

def save(fig, name: str):
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)


# ===========================================================================
# DOC TYPE — PLOT 1
# Heatmap: NumMatch(metrics) — rows = retrieval condition, cols = doc type
# Two side-by-side panels, one per model
# ===========================================================================
def plot1_numatch_heatmap():
    cond_labels = list(CONDITIONS.keys())
    n_cond = len(cond_labels)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    fig.suptitle(
        "Numeric Match Rate (±3%) by Retrieval Condition and Document Type",
        fontsize=13, fontweight="bold", y=1.01
    )

    for col, (model, model_lbl) in enumerate(zip(MODELS, MODEL_LABELS)):
        ax = axes[col]
        mat = np.zeros((n_cond, len(DOC_TYPES)))
        for i, cond in enumerate(cond_labels):
            for j, dt in enumerate(DOC_TYPES):
                v = data[cond]["by_doc"].get(model, {}).get(dt, {}).get("numeric_match", 0)
                mat[i, j] = v

        # Mask doc types with no metrics questions (value always 0)
        display = mat.copy()
        display[:, 1:] = np.nan  # 10q, 8k, Earnings have n_metrics_qs=0

        im = ax.imshow(display, cmap="Blues", vmin=0, vmax=0.85, aspect="auto")

        # Annotate all cells
        for i in range(n_cond):
            for j in range(len(DOC_TYPES)):
                v = mat[i, j]
                if j == 0:
                    txt = f"{v:.2f}"
                    fc  = "white" if v > 0.5 else "black"
                else:
                    txt = "N/A"
                    fc  = "#aaaaaa"
                ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=fc)

        ax.set_xticks(range(len(DOC_TYPES)))
        ax.set_xticklabels([dt.upper() for dt in DOC_TYPES], fontsize=10)
        ax.set_title(model_lbl, fontsize=11, pad=8)

        if col == 0:
            ax.set_yticks(range(n_cond))
            ax.set_yticklabels(cond_labels, fontsize=9)
        ax.tick_params(left=False, bottom=False)

        # Remove grid for heatmap
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Shared colorbar
    cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label("NumMatch (metrics-generated)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    fig.text(
        0.5, -0.04,
        "N/A = doc type contains no metrics-generated questions",
        ha="center", fontsize=9, color="grey", style="italic"
    )

    save(fig, "plot1_numatch_heatmap_doctype")
    print("  Saved plot1_numatch_heatmap_doctype")


# ===========================================================================
# DOC TYPE — PLOT 2
# Grouped bar: ROUGE-L by doc type, all 5 retrieval conditions
# Two rows (one per model), 4 groups (doc types)
# ===========================================================================
def plot2_rougel_doctype():
    cond_labels = list(CONDITIONS.keys())
    n_cond = len(cond_labels)
    n_dt   = len(DOC_TYPES)
    width  = 0.14
    x      = np.arange(n_dt)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle(
        "ROUGE-L by Document Type across Retrieval Conditions",
        fontsize=13, fontweight="bold", y=1.01
    )

    for col, (model, model_lbl) in enumerate(zip(MODELS, MODEL_LABELS)):
        ax = axes[col]
        offsets = np.linspace(-(n_cond - 1) / 2, (n_cond - 1) / 2, n_cond) * width

        for i, (cond, color) in enumerate(zip(cond_labels, COND_COLORS)):
            vals = [data[cond]["by_doc"].get(model, {}).get(dt, {}).get("rougeL", 0)
                    for dt in DOC_TYPES]
            bars = ax.bar(x + offsets[i], vals, width * 0.92, label=cond,
                          color=color, alpha=0.88, zorder=3)
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.006,
                            f"{v:.2f}", ha="center", va="bottom",
                            fontsize=6.5, rotation=90, color="#333333")

        ax.set_xticks(x)
        ax.set_xticklabels([dt.upper() for dt in DOC_TYPES], fontsize=10)
        ax.set_title(model_lbl, fontsize=11, pad=8)
        ax.set_ylabel("ROUGE-L" if col == 0 else "")
        ax.set_ylim(0, 0.55)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.zorder = 2

        if col == 1:
            ax.legend(loc="upper right", framealpha=0.9, fontsize=8.5)

    fig.text(
        0.5, -0.03,
        "Lower ROUGE-L on 10K is expected — those queries are numeric (metrics-generated) "
        "and answers don't overlap with reference text.",
        ha="center", fontsize=9, color="grey", style="italic"
    )

    save(fig, "plot2_rougel_doctype_retrieval")
    print("  Saved plot2_rougel_doctype_retrieval")


# ===========================================================================
# DOC TYPE — PLOT 3
# Dot / lollipop: BERTScore-F1 across doc types, comparing models
# One panel per retrieval condition (select 3: Dense, FT-RR, OraclePage)
# ===========================================================================
def plot3_bertscore_doctype():
    show = ["Dense BGE-M3", "Multi-HyDE + FT-RR", "Oracle Page"]
    n_dt = len(DOC_TYPES)
    y    = np.arange(n_dt)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True, sharex=True)
    fig.suptitle(
        "BERTScore-F1 by Document Type — Qwen vs Llama (select conditions)",
        fontsize=13, fontweight="bold", y=1.01
    )

    for ax, cond in zip(axes, show):
        for m_idx, (model, color, marker) in enumerate(
            zip(MODELS, MODEL_COLORS, ["o", "s"])
        ):
            vals = [data[cond]["by_doc"].get(model, {}).get(dt, {}).get("bertscore_f1", 0)
                    for dt in DOC_TYPES]
            # lollipop stems
            for j, v in enumerate(vals):
                ax.hlines(y[j] + m_idx * 0.22 - 0.11, 0, v,
                          color=color, lw=1.4, alpha=0.6)
            ax.scatter(vals, y + m_idx * 0.22 - 0.11,
                       color=color, s=60, zorder=5, marker=marker,
                       label=MODEL_LABELS[m_idx])

        ax.axvline(0, color="black", lw=0.8, alpha=0.4)
        ax.set_title(cond, fontsize=10, pad=6)
        ax.set_xlabel("BERTScore-F1")
        ax.grid(axis="x", alpha=0.3, linestyle="--")
        ax.grid(axis="y", alpha=0)
        ax.spines["left"].set_visible(False)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([dt.upper() for dt in DOC_TYPES], fontsize=10)
    axes[0].set_ylabel("Document Type")

    handles = [
        mpatches.Patch(color=MODEL_COLORS[0], label=MODEL_LABELS[0]),
        mpatches.Patch(color=MODEL_COLORS[1], label=MODEL_LABELS[1]),
    ]
    axes[2].legend(handles=handles, loc="lower right", fontsize=9)

    fig.text(
        0.5, -0.04,
        "Negative BERTScore on metrics questions is expected — numeric answers share almost "
        "no semantic content with reference text embeddings.",
        ha="center", fontsize=9, color="grey", style="italic"
    )

    save(fig, "plot3_bertscore_doctype")
    print("  Saved plot3_bertscore_doctype")


# ===========================================================================
# QUESTION TYPE — PLOT 4
# Bar chart: NumMatch(metrics) across retrieval conditions
# Only metrics-generated questions (the only group where it's defined)
# With a "retrieval ceiling" annotation
# ===========================================================================
def plot4_numatch_qtype():
    cond_labels = list(CONDITIONS.keys())
    x      = np.arange(len(cond_labels))
    width  = 0.32

    fig, ax = plt.subplots(figsize=(10, 5))

    for m_idx, (model, color, hatch) in enumerate(
        zip(MODELS, MODEL_COLORS, MODEL_HATCHES)
    ):
        vals = [data[cond]["by_qtype"].get(model, {})
                .get("metrics-generated", {}).get("numeric_match", 0)
                for cond in cond_labels]
        offset = (m_idx - 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.92, label=MODEL_LABELS[m_idx],
                      color=color, alpha=0.85, hatch=hatch, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9.5,
                    fontweight="bold", color="#222222")

    # Retrieval ceiling arrow annotation (Oracle Page is upper bound)
    oracle_q = max(
        data["Oracle Page"]["by_qtype"]["Qwen2.5-7B"]["metrics-generated"]["numeric_match"],
        data["Oracle Page"]["by_qtype"]["Llama3.1-8B"]["metrics-generated"]["numeric_match"],
    )
    ax.axhline(oracle_q, color="#8172B2", ls="--", lw=1.5, alpha=0.7, zorder=2)
    ax.text(len(cond_labels) - 0.05, oracle_q + 0.015,
            f"Oracle Page ceiling ({oracle_q:.2f})",
            ha="right", va="bottom", fontsize=9, color="#8172B2", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, fontsize=10)
    ax.set_ylabel("Numeric Match Rate (±3% tolerance)", fontsize=11)
    ax.set_title(
        "Numeric Match on Metrics-Generated Questions by Retrieval Condition",
        fontsize=13, fontweight="bold", pad=10
    )
    ax.set_ylim(0, 0.95)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_xlabel("Retrieval Condition", fontsize=11)

    fig.text(
        0.5, -0.03,
        "NumMatch defined only for the 50 metrics-generated questions (n=50); "
        "domain-relevant and novel-generated always score 0 by definition.",
        ha="center", fontsize=9, color="grey", style="italic"
    )

    save(fig, "plot4_numatch_qtype_retrieval")
    print("  Saved plot4_numatch_qtype_retrieval")


# ===========================================================================
# QUESTION TYPE — PLOT 5
# Grouped bars: ROUGE-L by question type, all 5 retrieval conditions
# One panel per model
# ===========================================================================
def plot5_rougel_qtype():
    cond_labels = list(CONDITIONS.keys())
    n_cond = len(cond_labels)
    n_qt   = len(Q_TYPES)
    width  = 0.14
    x      = np.arange(n_qt)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle(
        "ROUGE-L by Question Type across Retrieval Conditions",
        fontsize=13, fontweight="bold", y=1.01
    )

    for col, (model, model_lbl) in enumerate(zip(MODELS, MODEL_LABELS)):
        ax = axes[col]
        offsets = np.linspace(-(n_cond - 1) / 2, (n_cond - 1) / 2, n_cond) * width

        for i, (cond, color) in enumerate(zip(cond_labels, COND_COLORS)):
            vals = [data[cond]["by_qtype"].get(model, {}).get(qt, {}).get("rougeL", 0)
                    for qt in Q_TYPES]
            bars = ax.bar(x + offsets[i], vals, width * 0.92, label=cond,
                          color=color, alpha=0.88, zorder=3)
            for bar, v in zip(bars, vals):
                if v > 0.01:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.005,
                            f"{v:.2f}", ha="center", va="bottom",
                            fontsize=6.5, rotation=90, color="#333333")

        ax.set_xticks(x)
        ax.set_xticklabels(Q_LABELS, fontsize=11)
        ax.set_title(model_lbl, fontsize=11, pad=8)
        ax.set_ylabel("ROUGE-L" if col == 0 else "")
        ax.set_ylim(0, 0.35)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

        if col == 1:
            ax.legend(loc="upper right", framealpha=0.9, fontsize=8.5)

    fig.text(
        0.5, -0.03,
        "Metrics questions score near-zero ROUGE-L even when numerically correct — "
        "LLM rephrasing does not overlap with reference text.",
        ha="center", fontsize=9, color="grey", style="italic"
    )

    save(fig, "plot5_rougel_qtype_breakdown")
    print("  Saved plot5_rougel_qtype_breakdown")


# ===========================================================================
# QUESTION TYPE — PLOT 6
# "Metric disconnect" scatter: NumMatch(metrics) vs ROUGE-L (novel-generated)
# per retrieval condition — highlights the two axes of generative quality
# One plot, both models, conditions as points with labels
# ===========================================================================
def plot6_metric_disconnect():
    fig, ax = plt.subplots(figsize=(8, 6))

    for m_idx, (model, color, marker) in enumerate(
        zip(MODELS, MODEL_COLORS, ["o", "s"])
    ):
        xs, ys, labels = [], [], []
        for cond in CONDITIONS:
            nm  = data[cond]["by_qtype"].get(model, {}) \
                      .get("metrics-generated", {}).get("numeric_match", 0)
            rl  = data[cond]["by_qtype"].get(model, {}) \
                      .get("novel-generated", {}).get("rougeL", 0)
            xs.append(nm)
            ys.append(rl)
            labels.append(cond)

        sc = ax.scatter(xs, ys, color=color, s=120, zorder=5, marker=marker,
                        label=MODEL_LABELS[m_idx], edgecolors="white", linewidth=0.8)

        for x_v, y_v, lbl in zip(xs, ys, labels):
            short = lbl.replace("Multi-HyDE + ", "MH+").replace("Oracle ", "Ora.")
            ax.annotate(
                short,
                (x_v, y_v),
                textcoords="offset points",
                xytext=(6, 4) if m_idx == 0 else (6, -12),
                fontsize=8,
                color=color,
                alpha=0.9,
            )

    ax.set_xlabel("NumMatch — Metrics Questions (±3% tol.)", fontsize=11)
    ax.set_ylabel("ROUGE-L — Novel Questions", fontsize=11)
    ax.set_title(
        "Numeric Accuracy vs Textual Quality by Retrieval Condition\n"
        "(two orthogonal axes of generative performance)",
        fontsize=12, fontweight="bold", pad=10
    )

    # Quadrant shading
    xm = ax.get_xlim()
    ym = ax.get_ylim()
    ax.set_xlim(0.2, 0.88)
    ax.set_ylim(0.17, 0.28)
    ax.axvline(0.5, color="grey", lw=0.8, ls=":", alpha=0.5)
    ax.axhline(0.22, color="grey", lw=0.8, ls=":", alpha=0.5)

    ax.text(0.22, 0.274, "Low Numeric\nHigh Textual",
            fontsize=8, color="grey", va="top", ha="left", style="italic")
    ax.text(0.72, 0.274, "High Numeric\nHigh Textual",
            fontsize=8, color="grey", va="top", ha="left", style="italic")
    ax.text(0.72, 0.172, "High Numeric\nLow Textual",
            fontsize=8, color="grey", va="bottom", ha="left", style="italic")

    ax.legend(loc="lower left", framealpha=0.9, fontsize=9)

    fig.text(
        0.5, -0.03,
        "Better retrieval improves numeric accuracy (x-axis) but not necessarily "
        "novel-question textual quality (y-axis) — the two are largely orthogonal.",
        ha="center", fontsize=9, color="grey", style="italic"
    )

    save(fig, "plot6_metric_disconnect")
    print("  Saved plot6_metric_disconnect")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Output directory: {OUT_DIR}\n")

    print("Generating Doc-Type plots...")
    plot1_numatch_heatmap()
    plot2_rougel_doctype()
    plot3_bertscore_doctype()

    print("\nGenerating Question-Type plots...")
    plot4_numatch_qtype()
    plot5_rougel_qtype()
    plot6_metric_disconnect()

    print(f"\nAll 6 plots saved to: {OUT_DIR}")
