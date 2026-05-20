"""
Generate CSV and LaTeX tables from experiment results.
Tables:
  1. Main reranking results (all models, all metrics)
  2. Calibration table
  3. Selective retrieval table
  4. Ablation table
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

import json
import csv
from src.utils.io import load_json
from src.utils.logging import get_logger

logger = get_logger("make_tables")


def load_all_results(results_dir: pathlib.Path) -> dict[str, dict]:
    """Scan results dir for ranking_metrics and uncertainty_metrics JSON files."""
    model_results = {}
    for rm_path in results_dir.rglob("*_ranking_metrics.json"):
        tag = rm_path.stem.replace("_ranking_metrics", "")
        data = load_json(rm_path)
        model_results.setdefault(tag, {})["ranking"] = data

    for um_path in results_dir.rglob("*_uncertainty_metrics.json"):
        tag = um_path.stem.replace("_uncertainty_metrics", "")
        data = load_json(um_path)
        model_results.setdefault(tag, {})["uncertainty"] = data

    for sr_path in results_dir.rglob("*_selective_recall.json"):
        tag = sr_path.stem.replace("_selective_recall", "")
        data = load_json(sr_path)
        model_results.setdefault(tag, {})["selective"] = data

    return model_results


def make_main_table(model_results: dict, out_dir: pathlib.Path) -> None:
    """Main reranking results table."""
    rows = []
    metrics_order = ["recall@1", "recall@3", "recall@5", "recall@10", "mrr@10", "ndcg@5", "ndcg@10", "map@10"]
    for model_tag, results in sorted(model_results.items()):
        ranking = results.get("ranking", {})
        if not ranking:
            continue
        row = {"model": model_tag}
        for m in metrics_order:
            row[m] = f"{ranking.get(m, float('nan')):.4f}"
        rows.append(row)

    if not rows:
        logger.warning("No ranking results found")
        return

    csv_path = out_dir / "main_reranking_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model"] + metrics_order)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Saved {csv_path}")

    # LaTeX
    tex_path = out_dir / "main_reranking_results.tex"
    with open(tex_path, "w") as f:
        cols = "l" + "r" * len(metrics_order)
        f.write(f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n")
        header = " & ".join(["Model"] + [m.replace("@", "@") for m in metrics_order]) + " \\\\\n\\midrule\n"
        f.write(header)
        for row in rows:
            line = " & ".join([row["model"]] + [row[m] for m in metrics_order]) + " \\\\\n"
            f.write(line)
        f.write("\\bottomrule\n\\end{tabular}\n")
    logger.info(f"Saved {tex_path}")


def make_calibration_table(model_results: dict, out_dir: pathlib.Path) -> None:
    rows = []
    for model_tag, results in sorted(model_results.items()):
        unc = results.get("uncertainty", {})
        cal = unc.get("calibration", {})
        fd = unc.get("failure_detection", {})
        if not cal and not fd:
            continue
        rows.append({
            "model": model_tag,
            "ECE": f"{cal.get('ece', float('nan')):.4f}",
            "Brier": f"{cal.get('brier_score', float('nan')):.4f}",
            "failure_rate": f"{unc.get('failure_rate', float('nan')):.4f}",
            "AUROC": f"{fd.get('auroc', float('nan')):.4f}",
            "AUPRC": f"{fd.get('auprc', float('nan')):.4f}",
        })
    if not rows:
        return
    csv_path = out_dir / "calibration_uncertainty_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Saved {csv_path}")

    tex_path = out_dir / "calibration_uncertainty_results.tex"
    with open(tex_path, "w") as f:
        cols = "l" + "r" * (len(rows[0]) - 1)
        f.write(f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n")
        f.write(" & ".join(rows[0].keys()) + " \\\\\n\\midrule\n")
        for row in rows:
            f.write(" & ".join(row.values()) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    logger.info(f"Saved {tex_path}")


def make_selective_table(model_results: dict, out_dir: pathlib.Path, recall_k: int = 5) -> None:
    coverages = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    all_rows = []
    for model_tag, results in sorted(model_results.items()):
        selective = results.get("selective", [])
        if not selective:
            continue
        cov_to_row = {r["coverage"]: r for r in selective}
        row = {"model": model_tag}
        for cov in coverages:
            r = cov_to_row.get(cov, {})
            row[f"cov={cov:.0%}"] = f"{r.get(f'recall@{recall_k}', float('nan')):.4f}"
        all_rows.append(row)
    if not all_rows:
        return
    csv_path = out_dir / f"selective_recall@{recall_k}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    logger.info(f"Saved {csv_path}")


def write_summary_markdown(model_results: dict, out_dir: pathlib.Path) -> None:
    lines = ["# Experiment Results Summary\n"]
    for model_tag, results in sorted(model_results.items()):
        ranking = results.get("ranking", {})
        unc = results.get("uncertainty", {})
        fd = unc.get("failure_detection", {})
        r5 = ranking.get("recall@5", float("nan"))
        auroc = fd.get("auroc", float("nan"))
        lines.append(f"## {model_tag}\n")
        lines.append(f"**Recall@5**: {r5:.4f}  |  **AUROC (failure)**: {auroc:.4f}\n")
        if ranking:
            for k, v in sorted(ranking.items()):
                lines.append(f"- {k}: {v:.4f}")
        lines.append("")
    with open(out_dir / "results_summary.md", "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved results_summary.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--out_dir", default="results/tables")
    args = parser.parse_args()
    results_dir = pathlib.Path(args.results_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = load_all_results(results_dir)
    logger.info(f"Found results for: {list(all_results.keys())}")
    make_main_table(all_results, out_dir)
    make_calibration_table(all_results, out_dir)
    make_selective_table(all_results, out_dir)
    write_summary_markdown(all_results, out_dir)
