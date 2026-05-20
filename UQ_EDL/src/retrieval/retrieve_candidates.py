"""
Run first-stage retrieval (BM25, BGE-M3, RRF) for all questions.
Saves top-k candidate lists per question.
"""
import argparse
import pathlib
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

from src.utils.io import load_jsonl, save_jsonl, load_json
from src.utils.logging import get_logger
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense_bge import DenseRetriever
from src.retrieval.rrf import reciprocal_rank_fusion

logger = get_logger("retrieve_candidates")


def retrieve_for_dataset(
    questions: list[dict],
    pages: list[dict],
    retriever_type: str,
    top_k: int = 100,
    index_dir: str | None = None,
    model_name: str = "BAAI/bge-m3",
    per_doc: bool = True,
) -> list[dict]:
    """
    Run retrieval and return flat list of candidate records.
    If per_doc=True, retrieval is done within each question's document.
    Otherwise, retrieval is done over the full corpus.
    """
    # Build or load retriever
    if retriever_type == "bm25":
        retriever = BM25Retriever()
        idx_path = pathlib.Path(index_dir) / "bm25.pkl" if index_dir else None
        if idx_path and idx_path.exists():
            retriever = BM25Retriever.load(idx_path)
        else:
            retriever.build(pages)
            if idx_path:
                retriever.save(idx_path)

    elif retriever_type in ("bge_m3", "dense"):
        retriever = DenseRetriever(model_name=model_name)
        idx_path = pathlib.Path(index_dir) if index_dir else None
        if idx_path and (idx_path / "faiss.index").exists():
            retriever = DenseRetriever.load(idx_path)
        else:
            retriever.build(pages)
            if idx_path:
                retriever.save(idx_path)

    all_candidates = []
    for rec in questions:
        qid = rec["qid"]
        question = rec["question"]
        evidences = rec.get("evidences_updated", rec.get("evidences", []))
        gold_ids = set()
        for ev in evidences:
            doc_name = ev["doc_name"]
            page_num = ev["page_num"]
            gold_ids.add(f"{doc_name}_page_{page_num}")

        if per_doc and evidences:
            # Retrieve from the question's document only
            doc_name = evidences[0]["doc_name"]
            if retriever_type == "bm25":
                results = retriever.retrieve(question, top_k=top_k, doc_id=doc_name)
            else:
                results = retriever.retrieve(question, top_k=top_k, doc_id=doc_name)
        else:
            if retriever_type == "bm25":
                results = retriever.retrieve(question, top_k=top_k)
            else:
                results = retriever.retrieve(question, top_k=top_k)

        for r in results:
            r["qid"] = qid
            r["question"] = question
            r["label"] = 1 if r["candidate_id"] in gold_ids else 0

        all_candidates.extend(results)

    return all_candidates


def retrieve_rrf(
    questions: list[dict],
    bm25_candidates: dict[str, list[dict]],
    dense_candidates: dict[str, list[dict]],
    top_k: int = 100,
) -> list[dict]:
    all_candidates = []
    for rec in questions:
        qid = rec["qid"]
        bm25 = bm25_candidates.get(qid, [])
        dense = dense_candidates.get(qid, [])
        merged = reciprocal_rank_fusion(
            [bm25, dense], ["bm25_score", "dense_score"], top_k=top_k
        )
        for r in merged:
            r["qid"] = qid
            r["question"] = rec["question"]
        all_candidates.extend(merged)
    return all_candidates


def group_by_qid(candidates: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for c in candidates:
        grouped[c["qid"]].append(c)
    return dict(grouped)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True, help="JSONL of questions (finqa or fb)")
    parser.add_argument("--pages", required=True, help="JSONL of pages")
    parser.add_argument("--out_dir", default="data/processed/candidates")
    parser.add_argument("--retrievers", nargs="+", default=["bm25", "bge_m3", "rrf"])
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--index_dir", default="data/processed/candidates/indices")
    parser.add_argument("--dataset", default="finqa")
    parser.add_argument("--per_doc", action="store_true", default=True)
    args = parser.parse_args()

    questions = load_jsonl(args.questions)
    pages = load_jsonl(args.pages)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bm25_by_qid = {}
    dense_by_qid = {}

    for retriever in args.retrievers:
        if retriever == "rrf":
            continue
        logger.info(f"Running {retriever} retrieval...")
        index_dir = pathlib.Path(args.index_dir) / retriever
        index_dir.mkdir(parents=True, exist_ok=True)
        cands = retrieve_for_dataset(
            questions, pages, retriever, args.top_k, str(index_dir), per_doc=args.per_doc
        )
        out_path = out_dir / f"{args.dataset}_{retriever}_top{args.top_k}.jsonl"
        save_jsonl(cands, out_path)
        logger.info(f"Saved {len(cands)} candidates to {out_path}")
        if retriever == "bm25":
            bm25_by_qid = group_by_qid(cands)
        elif retriever in ("bge_m3", "dense"):
            dense_by_qid = group_by_qid(cands)

    if "rrf" in args.retrievers and bm25_by_qid and dense_by_qid:
        logger.info("Running RRF fusion...")
        rrf_cands = retrieve_rrf(questions, bm25_by_qid, dense_by_qid, args.top_k)
        out_path = out_dir / f"{args.dataset}_rrf_top{args.top_k}.jsonl"
        save_jsonl(rrf_cands, out_path)
        logger.info(f"Saved {len(rrf_cands)} RRF candidates to {out_path}")
