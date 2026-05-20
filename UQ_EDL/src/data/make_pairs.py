"""
Build training/evaluation pairs from pages + candidates.
Each pair = (question, candidate_page) with a binary label.

Strategy:
  - Positives: gold evidence pages
  - Easy negatives: pages from unrelated docs
  - Retriever negatives: BM25/dense top-k non-gold
  - Hard negatives: from generate_hard_negatives.py
"""
import argparse
import pathlib
import random
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

from src.utils.io import load_jsonl, load_json, save_jsonl
from src.utils.text import build_model_input
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger("make_pairs")


def build_pairs(
    pages_path: str,
    splits_path: str,
    candidates_path: str | None,
    hard_negatives_path: str | None,
    out_dir: str,
    negative_ratio: int = 6,
    hard_neg_ratio: int = 3,
    seed: int = 42,
) -> None:
    set_seed(seed)
    pages = load_jsonl(pages_path)
    splits = load_json(splits_path)
    logger.info(f"Loaded {len(pages)} pages")

    # Index pages
    page_map: dict[str, dict] = {p["candidate_id"]: p for p in pages}
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for p in pages:
        by_doc[p["doc_id"]].append(p)

    gold_pages_per_qid: dict[str, list[dict]] = defaultdict(list)
    for p in pages:
        if p["label"] == 1:
            gold_pages_per_qid[p["doc_id"]].append(p)

    # Load FinQA question records per split
    splits_dir = pathlib.Path(splits_path).parent
    all_docs = sorted(set(p["doc_id"] for p in pages))

    # Load candidates if available
    candidates_by_qid: dict[str, list[dict]] = defaultdict(list)
    if candidates_path and pathlib.Path(candidates_path).exists():
        cands = load_jsonl(candidates_path)
        for c in cands:
            candidates_by_qid[c["qid"]].append(c)
        logger.info(f"Loaded candidates for {len(candidates_by_qid)} questions")

    # Load hard negatives
    hard_neg_by_qid: dict[str, list[dict]] = defaultdict(list)
    if hard_negatives_path and pathlib.Path(hard_negatives_path).exists():
        hns = load_jsonl(hard_negatives_path)
        for hn in hns:
            hard_neg_by_qid[hn["qid"]].append(hn)
        logger.info(f"Loaded {len(hns)} hard negatives")

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ["train", "val", "test"]:
        split_records_path = splits_dir / f"finqa_{split_name}.jsonl"
        if not split_records_path.exists():
            logger.warning(f"Missing {split_records_path}, skipping")
            continue
        split_records = load_jsonl(split_records_path)
        pairs = []

        for rec in split_records:
            qid = rec["qid"]
            question = rec["question"]
            evidences = rec.get("evidences_updated", rec.get("evidences", []))
            if not evidences:
                continue

            gold_candidate_ids = set()
            positive_pairs = []

            for ev in evidences:
                doc_name = ev["doc_name"]
                page_num = ev["page_num"]
                candidate_id = f"{doc_name}_page_{page_num}"
                if candidate_id in page_map:
                    p = page_map[candidate_id]
                    gold_candidate_ids.add(candidate_id)
                    positive_pairs.append(_make_pair(qid, question, p, 1, None, "finqa"))

            if not positive_pairs:
                continue

            # Negatives
            neg_pairs = []

            # 1. Hard negatives (finance-aware)
            hns = hard_neg_by_qid.get(qid, [])
            used_ids = set(gold_candidate_ids)
            for hn in hns[:hard_neg_ratio]:
                if hn["candidate_id"] not in used_ids:
                    neg_pairs.append({
                        "qid": qid,
                        "question": question,
                        "candidate_id": hn["candidate_id"],
                        "doc_id": hn["doc_id"],
                        "page_number": hn["page_number"],
                        "candidate_text": hn["candidate_text"],
                        "metadata": hn.get("metadata", {}),
                        "label": 0,
                        "negative_type": hn["negative_type"],
                        "source_dataset": "finqa",
                    })
                    used_ids.add(hn["candidate_id"])

            # 2. Retriever negatives (from first-stage candidates)
            retriever_negs = [
                c for c in candidates_by_qid.get(qid, [])
                if c.get("candidate_id") not in used_ids and c.get("label", 0) == 0
            ]
            random.shuffle(retriever_negs)
            for c in retriever_negs[:max(0, negative_ratio - len(neg_pairs))]:
                neg_pairs.append(_make_pair(qid, question, c, 0, "retriever_negative", "finqa"))
                used_ids.add(c["candidate_id"])

            # 3. Easy negatives (random unrelated pages) if still short
            if len(neg_pairs) < negative_ratio:
                doc_ids_used = set(ev["doc_name"] for ev in evidences)
                easy_pool = [
                    p for p in pages
                    if p["doc_id"] not in doc_ids_used and p["candidate_id"] not in used_ids
                ]
                random.shuffle(easy_pool)
                for p in easy_pool[:negative_ratio - len(neg_pairs)]:
                    neg_pairs.append(_make_pair(qid, question, p, 0, "easy_negative", "finqa"))
                    used_ids.add(p["candidate_id"])

            pairs.extend(positive_pairs)
            pairs.extend(neg_pairs)

        out_path = out_dir / f"finqa_{split_name}_pairs.jsonl"
        save_jsonl(pairs, out_path)
        positives = sum(p["label"] for p in pairs)
        logger.info(f"{split_name}: {len(pairs)} pairs, {positives} pos, {len(pairs)-positives} neg → {out_path}")

    # FinanceBench eval pairs (all candidates, no training)
    _build_financebench_eval_pairs(out_dir, candidates_by_qid)


def _build_financebench_eval_pairs(out_dir: pathlib.Path, candidates_by_qid: dict) -> None:
    """If FB candidates exist, save them as eval pairs."""
    fb_cands = {qid: cands for qid, cands in candidates_by_qid.items() if "financebench" in qid.lower()}
    if not fb_cands:
        logger.info("No FinanceBench candidates found in candidates file; skipping FB eval pairs")
        return
    pairs = []
    for qid, cands in fb_cands.items():
        pairs.extend(cands)
    save_jsonl(pairs, out_dir / "financebench_eval_pairs.jsonl")
    logger.info(f"Saved {len(pairs)} FinanceBench eval pairs")


def _make_pair(qid: str, question: str, page: dict, label: int, neg_type: str | None, source: str) -> dict:
    return {
        "qid": qid,
        "question": question,
        "candidate_id": page.get("candidate_id", ""),
        "doc_id": page.get("doc_id", ""),
        "page_number": page.get("page_number", -1),
        "candidate_text": page.get("text", page.get("candidate_text", "")),
        "metadata": page.get("metadata", {}),
        "label": label,
        "negative_type": neg_type,
        "source_dataset": source,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", default="data/processed/pages/finqa_pages.jsonl")
    parser.add_argument("--splits", default="data/processed/splits/finqa_splits.json")
    parser.add_argument("--candidates", default=None)
    parser.add_argument("--hard_negatives", default="data/processed/pairs/finqa_hard_negatives.jsonl")
    parser.add_argument("--out_dir", default="data/processed/pairs")
    parser.add_argument("--negative_ratio", type=int, default=6)
    parser.add_argument("--hard_neg_ratio", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_pairs(
        args.pages, args.splits, args.candidates, args.hard_negatives,
        args.out_dir, args.negative_ratio, args.hard_neg_ratio, args.seed
    )
