#!/usr/bin/env python
"""
ColPali-v1.3 + Qwen2-VL-7B Visual RAG Baseline
================================================
A fully visual retrieval-augmented generation pipeline for FinanceBench.

Architecture
------------
  Stage 1 — Indexing   : byaldi indexes every PDF page as a ColPali multi-vector embedding.
  Stage 2 — Retrieval  : ColPali MaxSim search returns top-K *pages* (with images).
  Stage 3 — Generation : Qwen2-VL-7B-Instruct answers the question from the page images.
  Stage 4 — Evaluation : Same RetrievalEvaluator as all other baselines (page/doc recall).
  Stage 5 — Plots      : Standalone plots + side-by-side comparison with text baselines.

Memory plan (L40S 46 GB)
-------------------------
  Stage 1–2  ColPali-v1.3 fp16  ≈  7 GB  →  freed before Stage 3
  Stage 3    Qwen2-VL-7B bf16   ≈ 15 GB  →  freed before Stage 4
  Stage 4–5  CPU only

Outputs (under --output-dir, default baselines/results)
-------------------------------------------------------
  predictions/colpali_byaldi_retrieval.json   full predictions (retrieval + generation)
  metrics/colpali_byaldi_metrics.json         full metrics breakdown
  metrics/all_variants_metrics.json           merged with existing text baselines (if present)
  plots/colpali_byaldi/                       standalone plots
  plots/colpali_byaldi/comparison_*.pdf/png   side-by-side vs text baselines
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup — allow importing from project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("colpali_byaldi")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
METHOD_NAME   = "colpali_byaldi"
METHOD_LABEL  = "CLIP ViT-L/14 + Qwen2-VL (Visual RAG)"

COLPALI_MODEL = "openai/clip-vit-large-patch14"
QWEN_VL_MODEL = "Qwen/Qwen2-VL-7B-Instruct"

K_VALUES = [1, 3, 5, 10, 20]
MAIN_K   = 5
RETRIEVE_TOP_K = 20   # retrieve 20 pages, evaluate at k ∈ {1,3,5,10,20}

QUESTION_TYPES = ["metrics-generated", "domain-relevant", "novel-generated"]

# Text baselines we compare against in the comparison plots
COMPARISON_VARIANTS = {
    "dense_bge_m3":          "Dense BGE-M3",
    "multi_hyde":             "Multi-HyDE",
    "bge_reranker":           "BGE-M3 + Reranker",
    "multi_hyde_reranker":    "BGE-M3 + Multi-HyDE + Reranker",
    "multi_hyde_ft_reranker": "BGE-M3 + Multi-HyDE + FT-Reranker",
}


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_data(
    fb_path: str,
    doc_info_path: Optional[str],
) -> Tuple[List[Dict], Dict[str, Dict]]:
    """Load FinanceBench JSONL samples and optional document metadata."""
    samples: List[Dict] = []
    with open(fb_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            gold_segs = []
            for ev in raw.get("evidence", []):
                gold_segs.append(
                    {
                        "text":     ev.get("evidence_text", ""),
                        "doc_name": ev.get("doc_name", raw.get("doc_name", "")),
                        "page":     ev.get("evidence_page_num", -1),
                    }
                )
            samples.append(
                {
                    "financebench_id":       raw.get("financebench_id", ""),
                    "question":              raw.get("question", ""),
                    "reference_answer":      raw.get("answer", ""),
                    "question_type":         raw.get("question_type", "unknown"),
                    "doc_name":              raw.get("doc_name", ""),
                    "doc_link":              raw.get("doc_link", ""),
                    "gold_evidence_segments": gold_segs,
                    "retrieved_chunks":      [],
                    "generated_answer":      "",
                }
            )

    doc_info: Dict[str, Dict] = {}
    if doc_info_path and os.path.exists(doc_info_path):
        with open(doc_info_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                doc_info[d["doc_name"]] = d

    type_counts = {
        qt: sum(1 for s in samples if s["question_type"] == qt)
        for qt in QUESTION_TYPES
    }
    logger.info("Loaded %d samples — %s", len(samples), type_counts)
    return samples, doc_info


# ---------------------------------------------------------------------------
# 2. Retrieval phase
# ---------------------------------------------------------------------------

def run_retrieval(
    samples: List[Dict],
    retriever,
    top_k: int = RETRIEVE_TOP_K,
) -> List[Dict]:
    """Search ColPali for every question; populate retrieved_chunks."""
    logger.info("Running ColPali retrieval (top_k=%d) …", top_k)

    for sample in tqdm(samples, desc="ColPali search"):
        query = sample["question"]
        try:
            results = retriever.search(query, top_k=top_k)
        except Exception as exc:
            logger.warning("Search failed for '%s': %s", query[:60], exc)
            results = []

        chunks = []
        for rank, r in enumerate(results, start=1):
            chunks.append(
                {
                    # text field: placeholder — visual RAG has no text chunks.
                    # The evaluator uses metadata.page / metadata.doc_name for
                    # page_recall / doc_recall, which is what we care about.
                    "text": f"[ColPali page image: {r['doc_name']} p.{r['page']}]",
                    "metadata": {
                        "doc_name":  r["doc_name"],
                        "page":      r["page"],
                        "chunk_idx": rank - 1,
                    },
                    "_score":       r["score"],
                    "rank":         rank,
                    "base64_image": r.get("base64_image"),  # stored for generation
                }
            )
        sample["retrieved_chunks"] = chunks

    return samples


# ---------------------------------------------------------------------------
# 3. Generation phase
# ---------------------------------------------------------------------------

def run_generation(
    samples: List[Dict],
    generator,
    top_k_images: int = 2,
) -> List[Dict]:
    """Generate answers using Qwen2-VL over the top-K retrieved page images."""
    logger.info("Running Qwen2-VL generation (top_%d images) …", top_k_images)

    for sample in tqdm(samples, desc="Qwen2-VL generate"):
        # Collect base64 images from top-K chunks
        images = []
        for chunk in sample["retrieved_chunks"][:top_k_images]:
            b64 = chunk.get("base64_image")
            if b64:
                images.append(b64)

        if not images:
            sample["generated_answer"] = (
                "I cannot determine the answer from the provided document pages."
            )
            continue

        try:
            answer = generator.generate_answer(
                query=sample["question"],
                base64_images=images,
            )
        except Exception as exc:
            logger.warning(
                "Generation failed for '%s': %s", sample["question"][:60], exc
            )
            answer = ""

        sample["generated_answer"] = answer

    return samples


# ---------------------------------------------------------------------------
# 4. Evaluation helpers  (mirror run_baselines.py)
# ---------------------------------------------------------------------------

def compute_retrieval_metrics(samples: List[Dict], k_values: List[int] = K_VALUES) -> Dict:
    from src.evaluation.retrieval_evaluator import RetrievalEvaluator

    ev = RetrievalEvaluator()
    return ev.compute_metrics(samples, k_values=k_values)


def compute_generative_metrics(samples: List[Dict]) -> Dict:
    from rouge_score import rouge_scorer as rs_module

    scorer = rs_module.RougeScorer(["rougeL"], use_stemmer=True)
    rougeL_scores = []
    numeric_matches = []

    for s in samples:
        gen = s.get("generated_answer", "")
        ref = s.get("reference_answer", "")
        score = scorer.score(ref, gen)["rougeL"].fmeasure if (gen and ref) else 0.0
        rougeL_scores.append(score)
        if s.get("question_type") == "metrics-generated":
            numeric_matches.append(_numeric_match(gen, ref))

    return {
        "answer_rougeL": float(np.mean(rougeL_scores)) if rougeL_scores else 0.0,
        "numeric_match":  float(np.mean(numeric_matches)) if numeric_matches else 0.0,
        "n_samples":      len(samples),
        "n_metrics_qs":   len(numeric_matches),
    }


def _numeric_match(pred: str, ref: str, rtol: float = 0.03) -> float:
    def _extract(text: str) -> Optional[float]:
        text = re.sub(r"[$,€£%]", "", text)
        ms = re.findall(r"-?\d[\d,]*\.?\d*", text)
        if not ms:
            return None
        try:
            return float(ms[0].replace(",", ""))
        except ValueError:
            return None

    pn, rn = _extract(pred), _extract(ref)
    if pn is None or rn is None:
        return 0.0
    if rn == 0:
        return 1.0 if pn == 0 else 0.0
    return 1.0 if abs(pn - rn) / abs(rn) <= rtol else 0.0


def aggregate_by_group(
    samples: List[Dict], key_fn, k_values: List[int] = K_VALUES
) -> Dict:
    groups: Dict[str, List] = defaultdict(list)
    for s in samples:
        groups[key_fn(s)].append(s)
    results = {}
    for group, gs in groups.items():
        ret = compute_retrieval_metrics(gs, k_values)
        gen = compute_generative_metrics(gs)
        results[group] = {**ret, **gen}
    return results


# ---------------------------------------------------------------------------
# 5. Predictions serialisation (strip base64 for compact storage)
# ---------------------------------------------------------------------------

def predictions_for_disk(samples: List[Dict]) -> List[Dict]:
    """Return a copy of samples without base64 images (for compact JSON storage).

    The base64 images are large (>100KB each) and not needed by the evaluator.
    We store a flag 'has_image': True so it's clear the image existed.
    """
    import copy

    clean = []
    for s in copy.deepcopy(s for s in samples):
        for chunk in s.get("retrieved_chunks", []):
            if chunk.get("base64_image"):
                chunk["has_image"] = True
                del chunk["base64_image"]
        clean.append(s)
    return clean


def predictions_for_disk_v2(samples: List[Dict]) -> List[Dict]:
    """Compact version that removes base64 without deepcopy overhead."""
    out = []
    for s in samples:
        new_chunks = []
        for chunk in s.get("retrieved_chunks", []):
            c = {k: v for k, v in chunk.items() if k != "base64_image"}
            if chunk.get("base64_image"):
                c["has_image"] = True
            new_chunks.append(c)
        out.append({**s, "retrieved_chunks": new_chunks})
    return out


# ---------------------------------------------------------------------------
# 6. Plotting
# ---------------------------------------------------------------------------

def _save_fig(fig, path: str) -> None:
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", path)


def plot_recall_curves(our_metrics: Dict, plots_dir: str) -> None:
    """PageRec@k and DocRec@k curves for our method."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, prefix, ylabel in [
        (axes[0], "page_recall", "PageRec@k"),
        (axes[1], "doc_recall",  "DocRec@k"),
    ]:
        overall = our_metrics.get("overall", our_metrics)
        vals = [overall.get(f"{prefix}@{k}", 0) for k in K_VALUES]
        ax.plot(K_VALUES, vals, marker="o", linewidth=2, color="#9C27B0", label=METHOD_LABEL)
        ax.set_xlabel("k")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} — {METHOD_LABEL}")
        ax.set_xticks(K_VALUES)
        ax.set_ylim(0, 1.0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    _save_fig(fig, os.path.join(plots_dir, "recall_at_k_curves.pdf"))


def plot_question_type_breakdown(our_metrics: Dict, plots_dir: str) -> None:
    """Bar chart: PageRec@5 and DocRec@5 broken down by question type."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_type = our_metrics.get("by_question_type", {})
    qtypes = [qt for qt in QUESTION_TYPES if qt in by_type]
    if not qtypes:
        return

    x = np.arange(len(qtypes))
    width = 0.35
    page_vals = [by_type[qt].get(f"page_recall@{MAIN_K}", 0) for qt in qtypes]
    doc_vals  = [by_type[qt].get(f"doc_recall@{MAIN_K}",  0) for qt in qtypes]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - width / 2, page_vals, width, label="PageRec@5", color="#9C27B0", alpha=0.85)
    ax.bar(x + width / 2, doc_vals,  width, label="DocRec@5",  color="#7B1FA2", alpha=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels([qt.replace("-", "\n") for qt in qtypes], fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"{METHOD_LABEL}\nPageRec@5 / DocRec@5 by Question Type")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, os.path.join(plots_dir, "question_type_breakdown.pdf"))


def plot_doc_type_breakdown(our_metrics: Dict, plots_dir: str) -> None:
    """Bar chart: PageRec@5 broken down by document type."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_doc = our_metrics.get("by_doc_type", {})
    if not by_doc:
        return

    dtypes = sorted(by_doc.keys())
    x = np.arange(len(dtypes))
    page_vals = [by_doc[dt].get(f"page_recall@{MAIN_K}", 0) for dt in dtypes]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(x, page_vals, color="#9C27B0", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([dt.upper() for dt in dtypes], fontsize=10)
    ax.set_ylabel("PageRec@5")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"{METHOD_LABEL} — PageRec@5 by Document Type")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, page_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.01,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9,
        )
    fig.tight_layout()
    _save_fig(fig, os.path.join(plots_dir, "doc_type_breakdown.pdf"))


def plot_comparison_with_baselines(
    our_metrics: Dict,
    baseline_metrics_dir: str,
    plots_dir: str,
) -> None:
    """Side-by-side comparison of our method vs existing text baselines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Collect all methods
    all_methods: Dict[str, Tuple[str, Dict]] = {}
    for key, label in COMPARISON_VARIANTS.items():
        mpath = Path(baseline_metrics_dir) / f"{key}_metrics.json"
        if mpath.exists():
            try:
                with open(mpath) as fh:
                    m = json.load(fh)
                all_methods[key] = (label, m.get("overall", m))
            except Exception:
                pass

    # Add our method
    all_methods[METHOD_NAME] = (METHOD_LABEL, our_metrics.get("overall", our_metrics))

    if len(all_methods) < 2:
        logger.warning("Not enough baselines found in '%s' for comparison plot.", baseline_metrics_dir)
        return

    # --- Plot 1: PageRec@5 and DocRec@5 bar chart ---
    methods_ordered = [k for k in COMPARISON_VARIANTS if k in all_methods] + [METHOD_NAME]
    labels  = [all_methods[k][0] for k in methods_ordered]
    pr5     = [all_methods[k][1].get(f"page_recall@{MAIN_K}", 0) for k in methods_ordered]
    dr5     = [all_methods[k][1].get(f"doc_recall@{MAIN_K}",  0) for k in methods_ordered]

    x     = np.arange(len(methods_ordered))
    width = 0.38
    # Colour our method distinctively
    colors_pr = ["#2196F3"] * (len(methods_ordered) - 1) + ["#9C27B0"]
    colors_dr = ["#90CAF9"] * (len(methods_ordered) - 1) + ["#CE93D8"]

    fig, ax = plt.subplots(figsize=(max(10, len(methods_ordered) * 1.6), 5))
    ax.bar(x - width / 2, pr5, width, label="PageRec@5", color=colors_pr, alpha=0.9)
    ax.bar(x + width / 2, dr5, width, label="DocRec@5",  color=colors_dr, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Retrieval Performance Comparison (PageRec@5 / DocRec@5)\n"
                 f"ColPali-v1.3 Visual RAG vs. Text Baselines")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    # Annotate our bar
    our_idx = methods_ordered.index(METHOD_NAME)
    ax.annotate(
        f"{pr5[our_idx]:.3f}",
        xy=(our_idx - width / 2, pr5[our_idx]),
        xytext=(0, 4), textcoords="offset points",
        ha="center", fontsize=8, color="#9C27B0", fontweight="bold",
    )
    fig.tight_layout()
    _save_fig(fig, os.path.join(plots_dir, "comparison_bar_k5.pdf"))

    # --- Plot 2: PageRec@k curves for selected methods ---
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    base_colors = plt.cm.tab10(np.linspace(0, 0.8, len(methods_ordered) - 1))

    for i, key in enumerate(methods_ordered[:-1]):  # text baselines
        label, m = all_methods[key]
        vals = [m.get(f"page_recall@{k}", 0) for k in K_VALUES]
        ax2.plot(K_VALUES, vals, marker="o", linewidth=1.4,
                 color=base_colors[i], label=label, alpha=0.75)

    # Our method highlighted
    our_label, our_m = all_methods[METHOD_NAME]
    our_vals = [our_m.get(f"page_recall@{k}", 0) for k in K_VALUES]
    ax2.plot(K_VALUES, our_vals, marker="D", linewidth=2.5,
             color="#9C27B0", label=our_label, zorder=10)

    ax2.set_xlabel("k")
    ax2.set_ylabel("PageRec@k")
    ax2.set_title("PageRec@k Curves — Visual RAG vs. Text Baselines")
    ax2.set_xticks(K_VALUES)
    ax2.set_ylim(0, 1.0)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=7.5, loc="lower right")
    fig2.tight_layout()
    _save_fig(fig2, os.path.join(plots_dir, "comparison_recall_curves.pdf"))

    # --- Plot 3: Answer quality bar chart (if answer_rougeL available) ---
    methods_with_gen = [k for k in methods_ordered
                        if "answer_rougeL" in all_methods[k][1]]
    if len(methods_with_gen) >= 2:
        labels_g = [all_methods[k][0] for k in methods_with_gen]
        rouge_g  = [all_methods[k][1].get("answer_rougeL", 0) for k in methods_with_gen]
        numm_g   = [all_methods[k][1].get("numeric_match",  0) for k in methods_with_gen]
        xg = np.arange(len(methods_with_gen))
        colors_r = ["#4CAF50"] * (len(methods_with_gen) - 1) + ["#9C27B0"]
        colors_n = ["#A5D6A7"] * (len(methods_with_gen) - 1) + ["#CE93D8"]

        fig3, ax3 = plt.subplots(figsize=(max(9, len(methods_with_gen) * 1.6), 5))
        ax3.bar(xg - width / 2, rouge_g, width, label="AnswerROUGE-L", color=colors_r, alpha=0.9)
        ax3.bar(xg + width / 2, numm_g,  width, label="NumericMatch",  color=colors_n, alpha=0.9)
        ax3.set_xticks(xg)
        ax3.set_xticklabels(labels_g, rotation=35, ha="right", fontsize=8)
        ax3.set_ylabel("Score")
        ax3.set_ylim(0, 1.05)
        ax3.set_title("Generative Quality — Visual RAG vs. Text Baselines")
        ax3.legend()
        ax3.grid(axis="y", alpha=0.3)
        fig3.tight_layout()
        _save_fig(fig3, os.path.join(plots_dir, "comparison_generative.pdf"))


def plot_heatmap_question_type(our_metrics: Dict, plots_dir: str) -> None:
    """Heatmap: question types × k values, coloured by PageRec@k."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_type = our_metrics.get("by_question_type", {})
    qtypes = [qt for qt in QUESTION_TYPES if qt in by_type]
    if not qtypes:
        return

    matrix = np.zeros((len(qtypes), len(K_VALUES)))
    for i, qt in enumerate(qtypes):
        for j, k in enumerate(K_VALUES):
            matrix[i, j] = by_type[qt].get(f"page_recall@{k}", 0)

    fig, ax = plt.subplots(figsize=(8, max(3, len(qtypes) * 0.9)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(K_VALUES)))
    ax.set_xticklabels([f"k={k}" for k in K_VALUES], fontsize=9)
    ax.set_yticks(range(len(qtypes)))
    ax.set_yticklabels([qt.replace("-", "\n") for qt in qtypes], fontsize=8)
    for i in range(len(qtypes)):
        for j in range(len(K_VALUES)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black" if matrix[i, j] < 0.6 else "white")
    plt.colorbar(im, ax=ax, label="PageRec@k")
    ax.set_title(f"{METHOD_LABEL}\nPageRec@k by Question Type")
    fig.tight_layout()
    _save_fig(fig, os.path.join(plots_dir, "heatmap_question_type.pdf"))


def plot_heatmap_doc_type(our_metrics: Dict, plots_dir: str) -> None:
    """Heatmap: doc types × k values, coloured by PageRec@k."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_doc = our_metrics.get("by_doc_type", {})
    if not by_doc:
        return

    dtypes = sorted(by_doc.keys())
    matrix = np.zeros((len(dtypes), len(K_VALUES)))
    for i, dt in enumerate(dtypes):
        for j, k in enumerate(K_VALUES):
            matrix[i, j] = by_doc[dt].get(f"page_recall@{k}", 0)

    fig, ax = plt.subplots(figsize=(8, max(3, len(dtypes) * 0.9)))
    im = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(K_VALUES)))
    ax.set_xticklabels([f"k={k}" for k in K_VALUES], fontsize=9)
    ax.set_yticks(range(len(dtypes)))
    ax.set_yticklabels([dt.upper() for dt in dtypes], fontsize=9)
    for i in range(len(dtypes)):
        for j in range(len(K_VALUES)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black" if matrix[i, j] < 0.6 else "white")
    plt.colorbar(im, ax=ax, label="PageRec@k")
    ax.set_title(f"{METHOD_LABEL}\nPageRec@k by Document Type")
    fig.tight_layout()
    _save_fig(fig, os.path.join(plots_dir, "heatmap_doc_type.pdf"))


# ---------------------------------------------------------------------------
# 7. Merge our metrics into the shared all_variants_metrics.json
# ---------------------------------------------------------------------------

def merge_into_all_variants(our_metrics: Dict, metrics_dir: str) -> None:
    """Add our result to the shared metrics file used by generate_thesis_plots.py."""
    all_path = Path(metrics_dir) / "all_variants_metrics.json"
    all_metrics: Dict = {}
    if all_path.exists():
        try:
            with open(all_path) as fh:
                all_metrics = json.load(fh)
        except Exception:
            pass

    all_metrics[METHOD_NAME] = our_metrics.get("overall", our_metrics)

    with open(all_path, "w") as fh:
        json.dump(all_metrics, fh, indent=2)
    logger.info("Merged '%s' into %s", METHOD_NAME, all_path)


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ColPali-v1.3 + Qwen2-VL Visual RAG Baseline for FinanceBench"
    )
    p.add_argument("--data-dir",     default="data",
                   help="Directory containing financebench_open_source.jsonl")
    p.add_argument("--pdf-dir",      default="pdfs",
                   help="Directory of FinanceBench PDFs")
    p.add_argument("--output-dir",   default="baselines/results",
                   help="Root output directory (predictions/, metrics/, plots/)")
    p.add_argument("--index-root",   default=".clip_index",
                   help="Where the CLIP FAISS index is persisted")
    p.add_argument("--index-name",   default="financebench_clip_vl",
                   help="Index name within index-root")
    p.add_argument("--colpali-model", default=COLPALI_MODEL,
                   help="CLIP model name (passed to CLIPVisualRetriever)")
    p.add_argument("--qwen-model",    default=QWEN_VL_MODEL)
    p.add_argument("--top-k",        type=int, default=RETRIEVE_TOP_K,
                   help="Number of pages to retrieve (evaluated at k ∈ {1,3,5,10,20})")
    p.add_argument("--top-k-images", type=int, default=2,
                   help="Number of top-retrieved page images fed to the VLM")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--overwrite-index", action="store_true",
                   help="Force re-indexing even if index already exists")
    p.add_argument("--skip-generation", action="store_true",
                   help="Skip Qwen2-VL generation (retrieval evaluation only)")
    p.add_argument("--skip-plots",   action="store_true")
    p.add_argument("--resume",       action="store_true",
                   help="Load cached predictions and skip already-completed phases")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Output dirs ───────────────────────────────────────────────────────────
    out_dir   = Path(args.output_dir)
    pred_dir  = out_dir / "predictions"
    met_dir   = out_dir / "metrics"
    plot_dir  = out_dir / "plots" / METHOD_NAME
    for d in [pred_dir, met_dir, plot_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    fb_path       = Path(args.data_dir) / "financebench_open_source.jsonl"
    doc_info_path = Path(args.data_dir) / "financebench_document_information.jsonl"

    samples, doc_info = load_data(str(fb_path), str(doc_info_path) if doc_info_path.exists() else None)

    # Attach doc_type for breakdowns
    for s in samples:
        s["doc_type"]    = doc_info.get(s["doc_name"], {}).get("doc_type", "unknown")
        s["gics_sector"] = doc_info.get(s["doc_name"], {}).get("gics_sector", "unknown")

    # ── Paths ─────────────────────────────────────────────────────────────────
    retrieval_pred_path = pred_dir / f"{METHOD_NAME}_retrieval.json"
    final_pred_path     = pred_dir / f"{METHOD_NAME}_final.json"

    # ── Phase 1 + 2: Retrieval ────────────────────────────────────────────────
    if args.resume and retrieval_pred_path.exists():
        logger.info(">>> PHASE 1-2: Loading cached retrieval from %s", retrieval_pred_path)
        with open(retrieval_pred_path) as fh:
            samples_with_retrieval = json.load(fh)
        # Re-attach doc_type (not stored in compact predictions)
        for s in samples_with_retrieval:
            if "doc_type" not in s:
                s["doc_type"]    = doc_info.get(s["doc_name"], {}).get("doc_type", "unknown")
                s["gics_sector"] = doc_info.get(s["doc_name"], {}).get("gics_sector", "unknown")
    else:
        logger.info("\n>>> PHASE 1: Building ColPali index")
        t0 = time.time()
        from colpali_byaldi_retriever import CLIPVisualRetriever as ColPaliByaldiRetriever

        retriever = ColPaliByaldiRetriever(
            pdf_dir=args.pdf_dir,
            index_name=args.index_name,
            index_root=args.index_root,
            clip_model=args.colpali_model,
        )
        retriever.build_index(overwrite=args.overwrite_index)
        logger.info("Index ready in %.1fs", time.time() - t0)

        logger.info("\n>>> PHASE 2: ColPali Retrieval")
        t1 = time.time()
        samples_retrieved = run_retrieval(samples, retriever, top_k=args.top_k)
        logger.info("Retrieval done in %.1fs", time.time() - t1)

        # Save compact predictions (no base64 images)
        with open(retrieval_pred_path, "w") as fh:
            json.dump(predictions_for_disk_v2(samples_retrieved), fh, indent=2)
        logger.info("Retrieval predictions saved → %s", retrieval_pred_path)

        # Also save in-memory version with images for generation
        samples_with_retrieval = samples_retrieved

        # Free ColPali before loading Qwen2-VL
        retriever.free()

    # ── Phase 3: Generation ───────────────────────────────────────────────────
    if args.skip_generation:
        logger.info("\n>>> PHASE 3: Skipped (--skip-generation)")
        samples_final = samples_with_retrieval
    elif args.resume and final_pred_path.exists():
        logger.info(">>> PHASE 3: Loading cached final predictions from %s", final_pred_path)
        with open(final_pred_path) as fh:
            samples_final = json.load(fh)
        for s in samples_final:
            if "doc_type" not in s:
                s["doc_type"]    = doc_info.get(s["doc_name"], {}).get("doc_type", "unknown")
                s["gics_sector"] = doc_info.get(s["doc_name"], {}).get("gics_sector", "unknown")
    else:
        # When resuming retrieval from cache, we don't have base64 images in memory.
        # We need to re-fetch them for generation.
        if args.resume and retrieval_pred_path.exists():
            logger.info(
                ">>> PHASE 3: Cache hit for retrieval but images stripped — "
                "re-running retrieval in image-only mode for generation inputs."
            )
            from colpali_byaldi_retriever import CLIPVisualRetriever as ColPaliByaldiRetriever

            retriever = ColPaliByaldiRetriever(
                pdf_dir=args.pdf_dir,
                index_name=args.index_name,
                index_root=args.index_root,
                clip_model=args.colpali_model,
            )
            retriever.build_index(overwrite=False)
            samples_with_retrieval = run_retrieval(
                samples, retriever, top_k=args.top_k
            )
            retriever.free()

        logger.info("\n>>> PHASE 3: Qwen2-VL Generation")
        from colpali_byaldi_generator import QwenVLGenerator

        generator = QwenVLGenerator(
            model_name=args.qwen_model,
            max_new_tokens=args.max_new_tokens,
        )
        generator.load()
        t2 = time.time()
        samples_final = run_generation(
            samples_with_retrieval, generator, top_k_images=args.top_k_images
        )
        logger.info("Generation done in %.1fs", time.time() - t2)
        generator.free()

        # Save final predictions (no base64)
        with open(final_pred_path, "w") as fh:
            json.dump(predictions_for_disk_v2(samples_final), fh, indent=2)
        logger.info("Final predictions saved → %s", final_pred_path)

    # ── Phase 4: Evaluation ───────────────────────────────────────────────────
    logger.info("\n>>> PHASE 4: Evaluation")

    overall_ret = compute_retrieval_metrics(samples_final)
    overall_gen = compute_generative_metrics(samples_final) if not args.skip_generation else {}
    overall     = {**overall_ret, **overall_gen}

    by_qt    = aggregate_by_group(samples_final, lambda s: s.get("question_type", "unknown"))
    by_dt    = aggregate_by_group(samples_final, lambda s: s.get("doc_type", "unknown"))
    by_qt_dt = aggregate_by_group(
        samples_final,
        lambda s: f"{s.get('question_type','?')}|{s.get('doc_type','?')}",
    )

    full_metrics = {
        "overall": overall,
        "by_question_type": by_qt,
        "by_doc_type": by_dt,
        "by_question_type_x_doc_type": by_qt_dt,
    }

    metrics_path = met_dir / f"{METHOD_NAME}_metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(full_metrics, fh, indent=2)
    logger.info("Metrics saved → %s", metrics_path)

    # Print headline numbers
    pr5  = overall.get(f"page_recall@{MAIN_K}", 0)
    dr5  = overall.get(f"doc_recall@{MAIN_K}",  0)
    bl5  = overall.get(f"context_bleu@{MAIN_K}", 0)
    rl5  = overall.get(f"context_rougeL@{MAIN_K}", 0)
    ans  = overall.get("answer_rougeL", 0)
    num  = overall.get("numeric_match",  0)
    logger.info(
        "[%s]  DocRec@5=%.3f  PageRec@5=%.3f  BLEU@5=%.3f  "
        "ROUGE-L@5=%.3f  AnsROUGE=%.3f  NumMatch=%.3f",
        METHOD_LABEL, dr5, pr5, bl5, rl5, ans, num,
    )

    # Merge into shared all_variants_metrics.json
    merge_into_all_variants(full_metrics, str(met_dir))

    # ── Phase 5: Plots ────────────────────────────────────────────────────────
    if not args.skip_plots:
        logger.info("\n>>> PHASE 5: Generating plots")
        plot_recall_curves(full_metrics, str(plot_dir))
        plot_question_type_breakdown(full_metrics, str(plot_dir))
        plot_doc_type_breakdown(full_metrics, str(plot_dir))
        plot_heatmap_question_type(full_metrics, str(plot_dir))
        plot_heatmap_doc_type(full_metrics, str(plot_dir))
        plot_comparison_with_baselines(full_metrics, str(met_dir), str(plot_dir))
        logger.info("All plots saved to %s", plot_dir)

    logger.info(
        "\n========================================\n"
        "  DONE  —  %s\n"
        "  PageRec@5 = %.3f   DocRec@5 = %.3f\n"
        "========================================",
        METHOD_LABEL, pr5, dr5,
    )


if __name__ == "__main__":
    main()
