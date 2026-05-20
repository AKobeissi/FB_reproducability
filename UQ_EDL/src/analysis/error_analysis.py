"""
Error analysis: categorize retrieval failures on FinanceBench.
For each failed question, save the gold evidence, top retrieved pages, scores, and suspected error category.
"""
import argparse
import pathlib
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

import numpy as np
from src.evaluation.metrics import recall_at_k
from src.utils.io import load_jsonl, load_json, save_jsonl, save_json
from src.utils.text import normalize_text
from src.utils.logging import get_logger

logger = get_logger("error_analysis")

ERROR_CATEGORIES = [
    "wrong_fiscal_year",
    "wrong_quarter",
    "wrong_company",
    "wrong_metric",
    "correct_page_not_in_candidates",
    "correct_page_reranked_too_low",
    "correct_page_high_uncertainty",
    "irrelevant_page_overconfident",
    "table_extraction_failure",
    "annotation_ambiguity",
]


def classify_error(
    qid: str,
    question: str,
    gold_pages: list[dict],
    top_pages: list[dict],
    gold_in_candidates: bool,
    failure_type: str,
) -> str:
    """Heuristic error classification. Can be manually corrected later."""
    if not gold_in_candidates:
        return "correct_page_not_in_candidates"

    if not gold_pages:
        return "annotation_ambiguity"

    # Check if gold page has very short text (table extraction issue)
    if any(p.get("char_count", 0) < 100 for p in gold_pages):
        return "table_extraction_failure"

    q_lower = question.lower()
    # Fiscal year mismatch
    if any(word in q_lower for word in ["year", "annual", "fy", "fiscal"]):
        if top_pages and gold_pages:
            top_meta = top_pages[0].get("metadata", {})
            gold_meta = gold_pages[0].get("metadata", {})
            if top_meta.get("fiscal_year") and gold_meta.get("fiscal_year"):
                if top_meta["fiscal_year"] != gold_meta["fiscal_year"]:
                    return "wrong_fiscal_year"

    if any(word in q_lower for word in ["quarter", "q1", "q2", "q3", "q4"]):
        return "wrong_quarter"

    if failure_type == "reranked_too_low":
        # Check if top page has high uncertainty
        if top_pages and top_pages[0].get("uncertainty", 0) > 0.7:
            return "correct_page_high_uncertainty"
        return "correct_page_reranked_too_low"

    if top_pages and top_pages[0].get("p_relevant", 0) > 0.8:
        return "irrelevant_page_overconfident"

    return "wrong_metric"


def run_error_analysis(
    scored_pairs_path: str,
    fb_questions_path: str,
    out_dir: str,
    top_k: int = 5,
    max_qualitative: int = 50,
) -> list[dict]:
    logger.info(f"Loading scored pairs from {scored_pairs_path}")
    pairs = load_jsonl(scored_pairs_path)
    fb_questions = {r["financebench_id"]: r for r in load_jsonl(fb_questions_path)}

    by_qid: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        by_qid[p["qid"]].append(p)

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    error_records = []
    category_counts: dict[str, int] = defaultdict(int)

    for qid, qpairs in by_qid.items():
        qpairs_sorted = sorted(qpairs, key=lambda x: x["score"], reverse=True)
        gold = {p["candidate_id"] for p in qpairs if p["label"] == 1}
        gold_pages = [p for p in qpairs if p["label"] == 1]

        if not gold:
            continue

        ranked = [p["candidate_id"] for p in qpairs_sorted]
        rec_at_k = recall_at_k(ranked, gold, top_k)
        if rec_at_k > 0:
            continue  # not a failure

        gold_in_candidates = len(gold) > 0
        gold_ranks = [i + 1 for i, p in enumerate(qpairs_sorted) if p["candidate_id"] in gold]

        failure_type = "correct_page_not_in_candidates" if not gold_in_candidates else "reranked_too_low"
        fb_q = fb_questions.get(qid, {})
        category = classify_error(
            qid,
            fb_q.get("question", qpairs_sorted[0].get("question", "")),
            gold_pages,
            qpairs_sorted[:3],
            gold_in_candidates,
            failure_type,
        )
        category_counts[category] += 1

        error_records.append({
            "qid": qid,
            "question": fb_q.get("question", qpairs_sorted[0].get("question", "")),
            "answer": fb_q.get("answer", ""),
            "failure_type": failure_type,
            "error_category": category,
            "gold_ranks": gold_ranks,
            "gold_in_top5": any(r <= 5 for r in gold_ranks),
            "gold_pages": [
                {"candidate_id": p["candidate_id"], "page_number": p.get("page_number"), "char_count": p.get("char_count")}
                for p in gold_pages[:3]
            ],
            "top3_retrieved": [
                {
                    "rank": i + 1,
                    "candidate_id": p["candidate_id"],
                    "page_number": p.get("page_number"),
                    "score": p.get("score"),
                    "p_relevant": p.get("p_relevant"),
                    "uncertainty": p.get("uncertainty"),
                    "text_snippet": p.get("candidate_text", "")[:200],
                }
                for i, p in enumerate(qpairs_sorted[:3])
            ],
        })

    logger.info(f"Failures: {len(error_records)}")
    for cat, cnt in sorted(category_counts.items()):
        logger.info(f"  {cat}: {cnt}")

    save_jsonl(error_records[:max_qualitative], out_dir / "error_analysis.jsonl")
    save_json(dict(category_counts), out_dir / "error_category_counts.json")

    # Write human-readable report
    with open(out_dir / "error_analysis_report.txt", "w") as f:
        f.write(f"Error Analysis Report\n{'='*50}\n\n")
        f.write(f"Total failures (Recall@{top_k}=0): {len(error_records)}\n\n")
        f.write("Error categories:\n")
        for cat, cnt in sorted(category_counts.items(), key=lambda x: -x[1]):
            f.write(f"  {cat}: {cnt}\n")
        f.write("\n" + "─" * 50 + "\n\n")
        for rec in error_records[:30]:
            f.write(f"QID: {rec['qid']}\n")
            f.write(f"Q: {rec['question']}\n")
            f.write(f"A: {rec['answer']}\n")
            f.write(f"Category: {rec['error_category']}\n")
            f.write(f"Gold ranks: {rec['gold_ranks']}\n")
            if rec["top3_retrieved"]:
                top = rec["top3_retrieved"][0]
                f.write(f"Top-1 score={top['score']:.3f} unc={top['uncertainty']}\n")
                f.write(f"Text: {top['text_snippet'][:100]}...\n")
            f.write("\n")

    logger.info(f"Saved error analysis to {out_dir}")
    return error_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored_pairs", required=True)
    parser.add_argument("--fb_questions", default="../data/financebench_open_source.jsonl")
    parser.add_argument("--out_dir", default="results/qualitative")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()
    run_error_analysis(args.scored_pairs, args.fb_questions, args.out_dir, args.top_k)
