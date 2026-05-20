"""
Build FinQA evaluation pairs for val/test splits.
Uses ALL pages from each question's document as candidates (no sampling).
This gives a realistic evaluation of the reranker.
"""
import argparse
import pathlib
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

from src.utils.io import load_jsonl, save_jsonl
from src.utils.logging import get_logger

logger = get_logger("build_finqa_eval_pairs")


def build_eval_pairs(
    questions_path: str,
    pages_path: str,
    out_path: str,
) -> None:
    questions = load_jsonl(questions_path)
    pages = load_jsonl(pages_path)
    logger.info(f"Loaded {len(questions)} questions, {len(pages)} pages")

    by_doc: dict[str, list[dict]] = defaultdict(list)
    for p in pages:
        by_doc[p["doc_id"]].append(p)

    pairs = []
    for rec in questions:
        qid = rec["qid"]
        question = rec["question"]
        evidences = rec.get("evidences_updated", rec.get("evidences", []))
        if not evidences:
            continue
        doc_name = evidences[0]["doc_name"]
        gold_pages = {ev["page_num"] for ev in evidences}
        doc_pages = by_doc.get(doc_name, [])
        if not doc_pages:
            logger.warning(f"No pages for {doc_name}")
            continue
        for p in doc_pages:
            label = 1 if p["page_number"] in gold_pages else 0
            pairs.append({
                "qid": qid,
                "question": question,
                "candidate_id": p["candidate_id"],
                "doc_id": p["doc_id"],
                "page_number": p["page_number"],
                "candidate_text": p["text"],
                "metadata": p.get("metadata", {}),
                "label": label,
                "negative_type": None if label == 1 else "document_page",
                "source_dataset": "finqa",
            })

    positives = sum(p["label"] for p in pairs)
    logger.info(f"Pairs: {len(pairs)}, pos: {positives}, neg: {len(pairs)-positives}")
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(pairs, out_path)
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--pages", default="data/processed/pages/finqa_pages.jsonl")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    build_eval_pairs(args.questions, args.pages, args.out)
