"""
Generate finance-aware hard negatives for FinQA training.

Hard negative types:
  - same_company_wrong_year: same company, different fiscal year
  - same_doc_wrong_page: same filing, nearby page (not gold)
  - same_metric_wrong_table: page with same keywords but not the gold page
  - wrong_company_same_year: different company, same fiscal year
  - boilerplate_page: pages dominated by legal/boilerplate text

These simulate realistic financial RAG retrieval errors.
"""
import argparse
import pathlib
import random
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

from src.utils.io import load_jsonl, save_jsonl, load_json
from src.utils.text import normalize_text
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger("generate_hard_negatives")

BOILERPLATE_KEYWORDS = [
    "forward-looking statements", "risk factors", "safe harbor",
    "cautionary note", "except as required by law", "market risk",
    "quantitative and qualitative disclosures",
]


def is_boilerplate(text: str) -> bool:
    norm = normalize_text(text)
    return sum(1 for kw in BOILERPLATE_KEYWORDS if kw in norm) >= 2


def generate_hard_negatives(
    finqa_pages_path: str,
    splits_path: str,
    out_path: str,
    max_per_type: int = 2,
    seed: int = 42,
) -> list[dict]:
    set_seed(seed)
    pages = load_jsonl(finqa_pages_path)
    splits = load_json(splits_path)
    train_qids = set(splits.get("train", []))

    # Index pages
    by_doc: dict[str, list[dict]] = defaultdict(list)
    by_company_year: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_company: dict[str, list[dict]] = defaultdict(list)

    for p in pages:
        doc_id = p["doc_id"]
        meta = p.get("metadata", {})
        company = meta.get("ticker", "")
        year = meta.get("fiscal_year", "")
        by_doc[doc_id].append(p)
        by_company_year[(company, year)].append(p)
        by_company[company].append(p)

    # Load FinQA questions and their gold pages
    splits_dir = pathlib.Path(splits_path).parent
    train_records = load_jsonl(splits_dir / "finqa_train.jsonl")
    qid_to_rec = {r["qid"]: r for r in train_records if r["qid"] in train_qids}

    hard_negatives = []

    for qid, rec in qid_to_rec.items():
        evidences = rec.get("evidences_updated", rec.get("evidences", []))
        if not evidences:
            continue

        gold_pages_by_doc: dict[str, set[int]] = defaultdict(set)
        for ev in evidences:
            gold_pages_by_doc[ev["doc_name"]].add(ev["page_num"])

        for ev in evidences:
            doc_name = ev["doc_name"]
            gold_page_num = ev["page_num"]
            doc_pages = by_doc.get(doc_name, [])
            meta = parse_doc_name_meta(doc_name)
            company = meta["ticker"]
            year = meta["fiscal_year"]

            # 1. same_doc_wrong_page
            non_gold = [p for p in doc_pages if p["page_number"] not in gold_pages_by_doc[doc_name] and p["char_count"] > 100]
            # Prefer pages near the gold page
            non_gold_sorted = sorted(non_gold, key=lambda p: abs(p["page_number"] - gold_page_num))
            for p in non_gold_sorted[:max_per_type]:
                hard_negatives.append(_make_hn(qid, rec["question"], p, "same_doc_wrong_page"))

            # 2. same_company_wrong_year: same company, different year
            other_years = [
                (c, y) for (c, y) in by_company_year.keys()
                if c == company and y != year
            ]
            for (c, y) in random.sample(other_years, min(max_per_type, len(other_years))):
                pool = [p for p in by_company_year[(c, y)] if p["label"] == 0]
                if pool:
                    p = random.choice(pool)
                    hard_negatives.append(_make_hn(qid, rec["question"], p, "same_company_wrong_year"))

            # 3. wrong_company_same_year
            other_companies_same_year = [
                (c, y) for (c, y) in by_company_year.keys()
                if c != company and y == year
            ]
            for (c, y) in random.sample(other_companies_same_year, min(max_per_type, len(other_companies_same_year))):
                pool = [p for p in by_company_year[(c, y)] if p["label"] == 0]
                if pool:
                    p = random.choice(pool)
                    hard_negatives.append(_make_hn(qid, rec["question"], p, "wrong_company_same_year"))

            # 4. boilerplate pages from same doc
            bp_pages = [p for p in doc_pages if is_boilerplate(p["text"]) and p["page_number"] not in gold_pages_by_doc[doc_name]]
            for p in bp_pages[:max_per_type]:
                hard_negatives.append(_make_hn(qid, rec["question"], p, "boilerplate_same_doc"))

    logger.info(f"Generated {len(hard_negatives)} hard negatives")
    neg_types = defaultdict(int)
    for hn in hard_negatives:
        neg_types[hn["negative_type"]] += 1
    for t, cnt in sorted(neg_types.items()):
        logger.info(f"  {t}: {cnt}")

    if out_path:
        save_jsonl(hard_negatives, out_path)
        logger.info(f"Saved to {out_path}")

    return hard_negatives


def _make_hn(qid: str, question: str, page: dict, neg_type: str) -> dict:
    meta = page.get("metadata", {})
    return {
        "qid": qid,
        "question": question,
        "candidate_id": page["candidate_id"],
        "doc_id": page["doc_id"],
        "page_number": page["page_number"],
        "candidate_text": page["text"],
        "metadata": meta,
        "label": 0,
        "negative_type": neg_type,
        "source_dataset": "finqa",
    }


def parse_doc_name_meta(doc_name: str) -> dict:
    import re
    parts = doc_name.split("_")
    ticker = parts[0] if parts else ""
    period = parts[1] if len(parts) > 1 else ""
    year = re.findall(r"\d{4}", period)
    return {"ticker": ticker, "fiscal_year": year[0] if year else ""}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--finqa_pages", default="data/processed/pages/finqa_pages.jsonl")
    parser.add_argument("--splits", default="data/processed/splits/finqa_splits.json")
    parser.add_argument("--out", default="data/processed/pairs/finqa_hard_negatives.jsonl")
    parser.add_argument("--max_per_type", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_hard_negatives(args.finqa_pages, args.splits, args.out, args.max_per_type, args.seed)
