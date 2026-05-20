"""
Create train/val/test splits for FinQA by company to avoid leakage.
FinanceBench is eval-only — never split into train.
"""
import argparse
import pathlib
import json
import random
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

from src.utils.io import load_jsonl, save_jsonl, save_json
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger("make_splits")


def make_finqa_splits(
    gold_pages_path: str,
    finqa_pages_dir: str,
    out_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[str]]:
    set_seed(seed)
    records = load_jsonl(gold_pages_path)
    logger.info(f"Loaded {len(records)} FinQA records")

    # Group by company (ticker = first part of doc_name)
    company_to_qids: dict[str, list[str]] = defaultdict(list)
    for rec in records:
        evidences = rec.get("evidences_updated", rec.get("evidences", []))
        if not evidences:
            continue
        doc_name = evidences[0]["doc_name"]
        company = doc_name.split("_")[0]
        company_to_qids[company].append(rec["qid"])

    companies = sorted(company_to_qids.keys())
    random.shuffle(companies)

    n = len(companies)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_companies = companies[:n_train]
    val_companies = companies[n_train: n_train + n_val]
    test_companies = companies[n_train + n_val:]

    splits = {
        "train": [qid for c in train_companies for qid in company_to_qids[c]],
        "val": [qid for c in val_companies for qid in company_to_qids[c]],
        "test": [qid for c in test_companies for qid in company_to_qids[c]],
    }

    for split_name, qids in splits.items():
        logger.info(f"{split_name}: {len(qids)} questions")

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_json(splits, out_dir / "finqa_splits.json")
    save_json(
        {
            "train_companies": train_companies,
            "val_companies": val_companies,
            "test_companies": test_companies,
        },
        out_dir / "finqa_company_splits.json",
    )

    # Save per-split record lists for convenience
    qid_to_rec = {r["qid"]: r for r in records}
    for split_name, qids in splits.items():
        split_records = [qid_to_rec[q] for q in qids if q in qid_to_rec]
        save_jsonl(split_records, out_dir / f"finqa_{split_name}.jsonl")
        logger.info(f"Saved finqa_{split_name}.jsonl with {len(split_records)} records")

    return splits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_pages", default="../data/finqa_test_gold_pages.jsonl")
    parser.add_argument("--finqa_pages_dir", default="data/processed/pages")
    parser.add_argument("--out_dir", default="data/processed/splits")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    make_finqa_splits(
        args.gold_pages, args.finqa_pages_dir, args.out_dir,
        args.train_ratio, args.val_ratio, args.seed
    )
