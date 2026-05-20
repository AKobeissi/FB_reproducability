"""
Generate all paper-ready figures:
  1. Reliability diagram (calibration)
  2. Risk-coverage curve
  3. Uncertainty distribution: success vs failure
  4. Score vs uncertainty scatterplot
  5. Recall@k bar chart comparing models
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from src.utils.io import load_jsonl, load_json, save_json
from src.utils.logging import get_logger

sns.set_theme(style="whitegrid", font_scale=1.2)
PALETTE = sns.color_palette("colorblind")
logger = get_logger("make_plots")


def plot_reliability_diagram(reliability_data: dict, out_path: str, title: str = "Reliability Diagram") -> None:
    fop = reliability_data["fraction_of_positives"]
    mp = reliability_data["mean_predicted"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", alpha=0.6)
    ax.bar(mp, fop, width=0.08, alpha=0.7, color=PALETTE[0], label="Model", edgecolor="white")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved reliability diagram: {out_path}")


def plot_risk_coverage(
    coverage_data_by_model: dict[str, list[dict]],
    recall_k: int,
    out_path: str,
    title: str = "Risk-Coverage Curve",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (model_name, data) in enumerate(coverage_data_by_model.items()):
        coverages = [d["coverage"] for d in data]
        recalls = [d.get(f"recall@{recall_k}", 0.0) for d in data]
        ax.plot(coverages, recalls, marker="o", label=model_name, color=PALETTE[i % len(PALETTE)])
    ax.set_xlabel("Coverage (fraction of questions answered)")
    ax.set_ylabel(f"Recall@{recall_k}")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(0.45, 1.05)
    plt.tight_layout()
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved risk-coverage curve: {out_path}")


def plot_uncertainty_distribution(
    scored_pairs: list[dict],
    out_path: str,
    k: int = 5,
    title: str = "Uncertainty: Success vs Failure",
) -> None:
    by_qid: dict[str, list[dict]] = defaultdict(list)
    for p in scored_pairs:
        by_qid[p["qid"]].append(p)

    success_unc = []
    failure_unc = []
    from src.evaluation.metrics import recall_at_k
    for qid, pairs in by_qid.items():
        pairs_sorted = sorted(pairs, key=lambda x: x["score"], reverse=True)
        gold = {p["candidate_id"] for p in pairs if p["label"] == 1}
        ranked = [p["candidate_id"] for p in pairs_sorted]
        unc = pairs_sorted[0]["uncertainty"]
        if isinstance(unc, float) and np.isnan(unc):
            continue
        if recall_at_k(ranked, gold, k) > 0:
            success_unc.append(unc)
        else:
            failure_unc.append(unc)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(success_unc, bins=30, alpha=0.6, label=f"Success (Recall@{k}>0)", color=PALETTE[2], density=True)
    ax.hist(failure_unc, bins=30, alpha=0.6, label=f"Failure (Recall@{k}=0)", color=PALETTE[3], density=True)
    ax.set_xlabel("Uncertainty (vacuity)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved uncertainty distribution: {out_path}")


def plot_score_vs_uncertainty(scored_pairs: list[dict], out_path: str, n_sample: int = 2000) -> None:
    valid = [p for p in scored_pairs if not (isinstance(p.get("uncertainty"), float) and np.isnan(p.get("uncertainty", 0)))]
    if len(valid) > n_sample:
        import random
        valid = random.sample(valid, n_sample)
    scores = np.array([p["p_relevant"] for p in valid])
    uncs = np.array([p["uncertainty"] for p in valid])
    labels = np.array([p["label"] for p in valid])

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(scores, uncs, c=labels, cmap="RdYlGn", alpha=0.5, s=10)
    plt.colorbar(scatter, ax=ax, label="Label (1=relevant)")
    ax.set_xlabel("P(relevant)")
    ax.set_ylabel("Uncertainty")
    ax.set_title("Score vs Uncertainty (green=relevant)")
    plt.tight_layout()
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved scatter: {out_path}")


def plot_recall_bar(
    model_metrics: dict[str, dict],
    metric: str = "recall@5",
    out_path: str = "results/plots/recall_bar.png",
    title: str = "Recall@5 by Model",
) -> None:
    models = list(model_metrics.keys())
    values = [model_metrics[m].get(metric, 0.0) for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, values, color=PALETTE[:len(models)])
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved bar chart: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--out_dir", default="results/plots")
    args = parser.parse_args()

    results_dir = pathlib.Path(args.results_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find all scored_pairs files and plot
    for scored_path in results_dir.rglob("*_scored_pairs.jsonl"):
        tag = scored_path.stem.replace("_scored_pairs", "")
        pairs = load_jsonl(scored_path)
        if any("uncertainty" in p and not np.isnan(float(p.get("uncertainty", "nan") or "nan")) for p in pairs):
            plot_uncertainty_distribution(pairs, str(out_dir / f"{tag}_uncertainty_dist.png"))
            plot_score_vs_uncertainty(pairs, str(out_dir / f"{tag}_score_vs_uncertainty.png"))

    # Reliability diagrams
    for unc_path in results_dir.rglob("*_uncertainty_metrics.json"):
        tag = unc_path.stem.replace("_uncertainty_metrics", "")
        data = load_json(unc_path)
        rel = data.get("calibration", {}).get("reliability")
        if rel:
            plot_reliability_diagram(rel, str(out_dir / f"{tag}_reliability.png"), title=f"Reliability Diagram ({tag})")

    # Risk-coverage
    selective_files = list(results_dir.rglob("*_selective_recall.json"))
    if selective_files:
        coverage_data = {}
        for sf in selective_files:
            model_tag = sf.stem.replace("_selective_recall", "")
            coverage_data[model_tag] = load_json(sf)
        plot_risk_coverage(coverage_data, 5, str(out_dir / "risk_coverage_curve.png"))
