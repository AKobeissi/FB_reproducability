"""
End-to-end pipeline runner.
Orchestrates: data prep → retrieval → training → evaluation → analysis.
Each stage is idempotent (skips if output already exists unless --force).
"""
import argparse
import pathlib
import subprocess
import sys
import os

ROOT = pathlib.Path(__file__).parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from src.utils.logging import get_logger
from src.utils.io import load_json, save_json

logger = get_logger("pipeline")


def run_cmd(cmd: list[str], desc: str, skip_if: pathlib.Path | None = None, force: bool = False) -> bool:
    if skip_if and skip_if.exists() and not force:
        logger.info(f"SKIP (exists): {desc} → {skip_if}")
        return True
    logger.info(f"RUN: {desc}")
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        logger.error(f"FAILED: {desc}")
        return False
    return True


def main(args):
    force = args.force
    py = sys.executable

    stages = []

    # ── Stage 1: Extract FinQA pages ──────────────────────────────────────────
    if not args.skip_data:
        stages.append((
            [py, "src/data/build_finqa_pages.py",
             "--gold_pages", str(args.finqa_gold_pages),
             "--pdf_dirs", str(args.pdf_dir_finqa), str(args.pdf_dir_finqa2),
             "--out_dir", "data/processed/pages",
             "--min_chars", "50"],
            "Build FinQA pages",
            pathlib.Path("data/processed/pages/finqa_pages.jsonl"),
        ))

        stages.append((
            [py, "src/data/build_financebench_pages.py",
             "--fb_path", str(args.fb_path),
             "--docinfo_path", str(args.fb_docinfo),
             "--pdf_dir", str(args.pdf_dir_fb),
             "--out_dir", "data/processed/pages"],
            "Build FinanceBench pages",
            pathlib.Path("data/processed/pages/financebench_pages.jsonl"),
        ))

        stages.append((
            [py, "src/data/make_splits.py",
             "--gold_pages", str(args.finqa_gold_pages),
             "--out_dir", "data/processed/splits"],
            "Make FinQA splits",
            pathlib.Path("data/processed/splits/finqa_splits.json"),
        ))

        stages.append((
            [py, "src/data/generate_hard_negatives.py",
             "--finqa_pages", "data/processed/pages/finqa_pages.jsonl",
             "--splits", "data/processed/splits/finqa_splits.json",
             "--out", "data/processed/pairs/finqa_hard_negatives.jsonl"],
            "Generate hard negatives",
            pathlib.Path("data/processed/pairs/finqa_hard_negatives.jsonl"),
        ))

        stages.append((
            [py, "src/data/make_pairs.py",
             "--pages", "data/processed/pages/finqa_pages.jsonl",
             "--splits", "data/processed/splits/finqa_splits.json",
             "--hard_negatives", "data/processed/pairs/finqa_hard_negatives.jsonl",
             "--out_dir", "data/processed/pairs"],
            "Build training pairs",
            pathlib.Path("data/processed/pairs/finqa_train_pairs.jsonl"),
        ))

    # ── Stage 2: Training ─────────────────────────────────────────────────────
    if not args.skip_training:
        stages.append((
            [py, "src/training/train_cross_encoder.py",
             "--train_pairs", "data/processed/pairs/finqa_train_pairs.jsonl",
             "--val_pairs", "data/processed/pairs/finqa_val_pairs.jsonl",
             "--backbone", args.backbone,
             "--batch_size", str(args.batch_size),
             "--epochs", str(args.epochs),
             "--run_dir", "results/runs"],
            "Train standard cross-encoder",
            None,  # always runs unless skipped
        ))

        stages.append((
            [py, "src/training/train_evidential.py",
             "--train_pairs", "data/processed/pairs/finqa_train_pairs.jsonl",
             "--val_pairs", "data/processed/pairs/finqa_val_pairs.jsonl",
             "--backbone", args.backbone,
             "--batch_size", str(args.batch_size),
             "--epochs", str(args.epochs),
             "--lambda_kl", str(args.lambda_kl),
             "--run_dir", "results/runs"],
            "Train evidential cross-encoder",
            None,
        ))

    for cmd, desc, skip_path in stages:
        ok = run_cmd(cmd, desc, skip_path, force)
        if not ok and not args.continue_on_error:
            logger.error(f"Pipeline halted at: {desc}")
            sys.exit(1)

    logger.info("Pipeline complete. Run evaluation scripts manually with trained model paths.")
    logger.info("  python src/evaluation/evaluate_ranking.py --model_dir results/runs/<run>/best_model ...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end UQ_EDL pipeline")
    parser.add_argument("--finqa_gold_pages", default="../data/finqa_test_gold_pages.jsonl")
    parser.add_argument("--fb_path", default="../data/financebench_open_source.jsonl")
    parser.add_argument("--fb_docinfo", default="../data/financebench_document_information.jsonl")
    parser.add_argument("--pdf_dir_finqa", default="../pdfs-extended-v4")
    parser.add_argument("--pdf_dir_finqa2", default="../pdfs")
    parser.add_argument("--pdf_dir_fb", default="../pdfs")
    parser.add_argument("--backbone", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lambda_kl", type=float, default=0.001)
    parser.add_argument("--skip_data", action="store_true")
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rerun even if outputs exist")
    parser.add_argument("--continue_on_error", action="store_true")
    args = parser.parse_args()
    main(args)
