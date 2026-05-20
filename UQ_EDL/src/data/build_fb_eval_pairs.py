"""
Build FinanceBench evaluation pairs.
For each question, all pages from its document are candidates.
Gold pages (label=1) are identified via evidence alignment from build_financebench_pages.py.
"""
import argparse
import pathlib
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

from src.utils.io import load_jsonl, save_jsonl
from src.utils.logging import get_logger

logger = get_logger("build_fb_eval_pairs")


def build_fb_eval_pairs(
    fb_questions_path: str,
    fb_pages_path: str,
    out_path: str,
) -> None:
    questions = load_jsonl(fb_questions_path)
    pages = load_jsonl(fb_pages_path)
    logger.info(f"Loaded {len(questions)} FB questions, {len(pages)} pages")

    # Index pages by doc_id
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for p in pages:
        by_doc[p["doc_id"]].append(p)

    pairs = []
    for rec in questions:
        qid = rec["financebench_id"]
        question = rec["question"]
        doc_name = rec["doc_name"]
        doc_pages = by_doc.get(doc_name, [])
        if not doc_pages:
            logger.warning(f"No pages found for {doc_name}")
            continue
        for p in doc_pages:
            pairs.append({
                "qid": qid,
                "question": question,
                "candidate_id": p["candidate_id"],
                "doc_id": p["doc_id"],
                "page_number": p["page_number"],
                "candidate_text": p["text"],
                "metadata": p.get("metadata", {}),
                "label": p["label"],
                "negative_type": None if p["label"] == 1 else "document_page",
                "source_dataset": "financebench",
            })

    positives = sum(p["label"] for p in pairs)
    logger.info(f"Total pairs: {len(pairs)}, positives: {positives}, negatives: {len(pairs)-positives}")

    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(pairs, out_path)
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fb_questions", default="../data/financebench_open_source.jsonl")
    parser.add_argument("--fb_pages", default="data/processed/pages/financebench_pages.jsonl")
    parser.add_argument("--out", default="data/processed/pairs/financebench_eval_pairs.jsonl")
    args = parser.parse_args()
    build_fb_eval_pairs(args.fb_questions, args.fb_pages, args.out)
