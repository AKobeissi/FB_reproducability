"""
Auto-generate paper-ready markdown summaries for each experiment.
One file per model/comparison.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

from src.utils.io import load_json
from src.utils.logging import get_logger

logger = get_logger("write_paper_summaries")


def write_summary(model_tag: str, ranking: dict, uncertainty: dict, selective: list[dict], out_dir: pathlib.Path) -> None:
    r5 = ranking.get("recall@5", float("nan"))
    r1 = ranking.get("recall@1", float("nan"))
    mrr = ranking.get("mrr@10", float("nan"))
    ndcg5 = ranking.get("ndcg@5", float("nan"))
    cal = uncertainty.get("calibration", {})
    fd = uncertainty.get("failure_detection", {})
    ece = cal.get("ece", float("nan"))
    brier = cal.get("brier_score", float("nan"))
    auroc = fd.get("auroc", float("nan"))
    auprc = fd.get("auprc", float("nan"))
    failure_rate = uncertainty.get("failure_rate", float("nan"))

    cov_100 = next((r.get("recall@5", float("nan")) for r in selective if r.get("coverage") == 1.0), float("nan"))
    cov_70 = next((r.get("recall@5", float("nan")) for r in selective if r.get("coverage") == 0.7), float("nan"))
    cov_50 = next((r.get("recall@5", float("nan")) for r in selective if r.get("coverage") == 0.5), float("nan"))

    text = f"""## Experiment: {model_tag}

**What was tested:** {model_tag} evaluated on FinanceBench/FinQA for page-level evidence retrieval.

**Key results:**
- Recall@5: {r5:.4f} | Recall@1: {r1:.4f} | MRR@10: {mrr:.4f} | nDCG@5: {ndcg5:.4f}
- ECE: {ece:.4f} | Brier score: {brier:.4f}
- Failure detection AUROC: {auroc:.4f} | AUPRC: {auprc:.4f}
- Failure rate (Recall@5=0): {failure_rate:.2%}

**Selective retrieval (Recall@5):**
- Coverage 100%: {cov_100:.4f}
- Coverage 70%: {cov_70:.4f}
- Coverage 50%: {cov_50:.4f}

**Interpretation:**
The model achieves Recall@5 of {r5:.3f} on the evaluation set, with a failure rate of {failure_rate:.1%}.
{"The uncertainty estimates achieve AUROC of " + f"{auroc:.3f}" + " for predicting retrieval failure, suggesting the uncertainty signal is " + ("informative." if auroc > 0.6 else "weak and may need further tuning.") if auroc == auroc else ""}
{"Selective retrieval at 70% coverage improves Recall@5 from " + f"{cov_100:.3f} to {cov_70:.3f}" + f" ({(cov_70-cov_100):.3f} absolute gain), " + ("confirming that uncertainty-based abstention helps." if cov_70 > cov_100 else "suggesting the coverage-accuracy tradeoff is not yet favorable.") if cov_70 == cov_70 else ""}

**Possible paper sentence:**
"{model_tag} achieves Recall@5 of {r5:.3f} on FinanceBench/FinQA, with uncertainty estimates that detect retrieval failure at AUROC {auroc:.3f}. Selective answering at 70% coverage yields Recall@5 of {cov_70:.3f}, a {(cov_70-cov_100)*100:.1f} point improvement over full coverage."
"""
    out_path = out_dir / f"{model_tag}_summary.md"
    with open(out_path, "w") as f:
        f.write(text)
    logger.info(f"Saved: {out_path}")


def run(results_dir: str, out_dir: str) -> None:
    results_dir = pathlib.Path(results_dir)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find all ranking result files
    for rm_path in sorted(results_dir.rglob("*_ranking_metrics.json")):
        tag = rm_path.stem.replace("_ranking_metrics", "")
        ranking = load_json(rm_path)
        unc_path = rm_path.parent / f"{tag}_uncertainty_metrics.json"
        uncertainty = load_json(unc_path) if unc_path.exists() else {}
        sel_path = rm_path.parent / f"{tag}_selective_recall.json"
        selective = load_json(sel_path) if sel_path.exists() else []
        write_summary(tag, ranking, uncertainty, selective, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--out_dir", default="results/paper_summaries")
    args = parser.parse_args()
    run(args.results_dir, args.out_dir)
