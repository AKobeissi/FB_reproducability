#!/usr/bin/env python3
"""
run_finqa_ft_reranker.py
=========================
Runs the fine-tuned cross-encoder reranker on the FinQA *test* set.

Pipeline
--------
  1. Ingest FinQA test PDFs from Final-PDF/ into a ChromaDB collection.
  2. Retrieve top-20 chunks per question with BGE-M3 dense retrieval.
  3. Re-rank with the FT cross-encoder.
  4. Evaluate with RetrievalEvaluator (page_recall@k, chunk_recall@k, mrr).

Input
-----
  data/finqa_test_gold_pages.jsonl     — 530 FinQA test questions with gold evidence
  Final-PDF/                            — PDF corpus (all 115 FinQA test docs present)
  checkpoints/ft_cross_encoder/        — trained reranker checkpoint

Output
------
  baselines/results/predictions/finqa_bge_m3_retrieval.json
  baselines/results/predictions/finqa_ft_reranker_retrieval.json
  baselines/results/metrics/finqa_bge_m3_metrics.json
  baselines/results/metrics/finqa_ft_reranker_metrics.json
"""

import copy
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("finqa_ft_reranker")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FINQA_TEST   = PROJECT_ROOT / "data" / "finqa_test_gold_pages.jsonl"
PDF_DIR      = PROJECT_ROOT / "Final-PDF"
VS_DIR       = PROJECT_ROOT / "vector_stores" / "finqa_test"
CHECKPOINT   = PROJECT_ROOT / "checkpoints" / "ft_cross_encoder"
PREDS_DIR    = PROJECT_ROOT / "baselines" / "results" / "predictions"
METRICS_DIR  = PROJECT_ROOT / "baselines" / "results" / "metrics"

EMBED_MODEL   = "BAAI/bge-m3"
CHUNK_SIZE    = 1024
CHUNK_OVERLAP = 128
RETRIEVE_K    = 20   # retrieve top-20 for dense; reranker rescores all
RERANK_TOP_N  = 20
BATCH_SIZE    = 64   # reranker inference batch size
COLLECTION    = "finqa_test_dense_tok1024_ol128"


# ---------------------------------------------------------------------------
# Helpers (mirrored from run_baselines.py)
# ---------------------------------------------------------------------------

def make_recursive_splitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    from transformers import AutoTokenizer
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL, use_fast=True)
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def load_pdf(doc_name: str, pdf_dir: str):
    try:
        from src.ingestion.pdf_utils import load_pdf_with_fallback
        docs, _ = load_pdf_with_fallback(doc_name, "", str(pdf_dir))
        return docs or []
    except Exception as e:
        logger.warning("PDF load failed for %s: %s", doc_name, e)
        return []


def _get_or_create_chroma(persist_dir: str, collection_name: str, embed_fn):
    import chromadb
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )


def build_dense_index(doc_names: List[str], pdf_dir: str, vs_dir: str) -> "chromadb.Collection":
    """Build (or load) ChromaDB collection for FinQA test docs."""
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL, device="cuda")
    collection = _get_or_create_chroma(vs_dir, COLLECTION, ef)
    existing_ids = set(collection.get(include=[])["ids"])
    splitter = make_recursive_splitter()

    docs_to_add, meta_to_add, ids_to_add = [], [], []

    for doc_name in tqdm(doc_names, desc="Indexing FinQA test docs"):
        marker_id = f"{doc_name}__page_0__chunk_0"
        if marker_id in existing_ids:
            continue

        pages = load_pdf(doc_name, pdf_dir)
        if not pages:
            logger.warning("No pages loaded for %s", doc_name)
            continue

        for page_doc in pages:
            meta = page_doc.metadata or {}
            page_num = meta.get("page", -1)
            for chunk_text in splitter.split_text(page_doc.page_content):
                chunk_text = chunk_text.strip()
                if not chunk_text:
                    continue
                idx = len(docs_to_add)
                cid = f"{doc_name}__page_{page_num}__chunk_{idx}"
                if cid in existing_ids:
                    continue
                docs_to_add.append(chunk_text)
                meta_to_add.append({"doc_name": doc_name, "page": int(page_num)})
                ids_to_add.append(cid)

                if len(docs_to_add) >= 2000:
                    collection.add(documents=docs_to_add, metadatas=meta_to_add, ids=ids_to_add)
                    docs_to_add, meta_to_add, ids_to_add = [], [], []

    if docs_to_add:
        collection.add(documents=docs_to_add, metadatas=meta_to_add, ids=ids_to_add)

    logger.info("Dense index '%s': %d chunks", COLLECTION, collection.count())
    return collection


def retrieve_dense(
    question: str, embed_model, collection, k: int, doc_filter: Optional[List[str]] = None
) -> List[Dict]:
    emb = embed_model.encode([question], normalize_embeddings=True, show_progress_bar=False)[0]
    n = min(k, collection.count())
    if n == 0:
        return []

    kwargs = dict(
        query_embeddings=[emb.tolist()],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )
    if doc_filter:
        if len(doc_filter) == 1:
            kwargs["where"] = {"doc_name": doc_filter[0]}
        else:
            kwargs["where"] = {"doc_name": {"$in": doc_filter}}

    try:
        res = collection.query(**kwargs)
    except Exception as e:
        logger.warning("ChromaDB query failed: %s", e)
        return []

    chunks = []
    for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        chunks.append({
            "text": text,
            "metadata": meta,
            "_score": float(1.0 - dist),
        })
    return chunks


def rerank_chunks(
    question: str, chunks: List[Dict], model, batch_size: int = BATCH_SIZE
) -> List[Dict]:
    if not chunks:
        return chunks
    pairs = [(question, c["text"]) for c in chunks]
    scores = []
    for i in range(0, len(pairs), batch_size):
        batch_scores = model.predict(pairs[i : i + batch_size])
        scores.extend(batch_scores.tolist() if hasattr(batch_scores, "tolist") else list(batch_scores))

    reranked = copy.deepcopy(chunks)
    for chunk, score in zip(reranked, scores):
        chunk["_dense_score"]    = chunk.get("_score")
        chunk["_reranker_score"] = float(score)
        chunk["_score"]          = float(score)
    reranked.sort(key=lambda c: c["_reranker_score"], reverse=True)
    for i, chunk in enumerate(reranked):
        chunk["rank"] = i + 1
    return reranked


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_and_save_metrics(samples: List[Dict], metrics_path: Path, label: str) -> Dict:
    from evaluation.retrieval_evaluator import RetrievalEvaluator
    from collections import defaultdict

    evaluator = RetrievalEvaluator()
    overall = evaluator.compute_metrics(samples)

    by_qtype: Dict[str, List] = defaultdict(list)
    for s in samples:
        qt = str(s.get("question_type") or "unknown").strip().lower()
        by_qtype[qt].append(s)

    metrics_out = {
        "overall": overall,
        "by_question_type": {k: evaluator.compute_metrics(v) for k, v in by_qtype.items()},
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    logger.info("[%s] metrics → %s", label, metrics_path)

    ov = overall
    logger.info(
        "[%s]  page_recall@5=%.4f  chunk_recall@5=%.4f  mrr=%.4f",
        label,
        ov.get("page_recall@5", 0),
        ov.get("chunk_recall@5", 0),
        ov.get("mrr", 0),
    )
    return metrics_out


def finqa_to_prediction_format(sample: Dict) -> Dict:
    """Convert a finqa_test_gold_pages entry to the prediction-file format."""
    gold_segments = []
    for ev in sample.get("evidences_updated", []):
        gold_segments.append({
            "text":     ev.get("evidence_text", ""),
            "doc_name": ev.get("doc_name", ""),
            "page":     ev.get("page_num", -1),
        })
    return {
        "qid":                  sample["qid"],
        "question":             sample["question"],
        "reference_answer":     sample.get("answer", ""),
        "question_type":        "unknown",
        "doc_name":             gold_segments[0]["doc_name"] if gold_segments else "",
        "gold_evidence_segments": gold_segments,
        "retrieved_chunks":     [],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ── Load FinQA test data ───────────────────────────────────────────────
    logger.info("Loading FinQA test from %s", FINQA_TEST)
    with open(FINQA_TEST) as f:
        raw_samples = [json.loads(line) for line in f]
    logger.info("Loaded %d FinQA test samples", len(raw_samples))

    # Convert to prediction format
    samples = [finqa_to_prediction_format(s) for s in raw_samples]

    # Collect unique doc names for indexing
    doc_names = sorted({s["doc_name"] for s in samples if s["doc_name"]})
    logger.info("Unique docs: %d", len(doc_names))

    # ── Build dense index ─────────────────────────────────────────────────
    VS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Building dense index in %s", VS_DIR)
    collection = build_dense_index(doc_names, str(PDF_DIR), str(VS_DIR))

    # ── Load embed model ──────────────────────────────────────────────────
    logger.info("Loading embed model: %s", EMBED_MODEL)
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer(EMBED_MODEL, device="cuda")

    # ── Stage A: BGE-M3 dense retrieval ──────────────────────────────────
    logger.info("Running BGE-M3 dense retrieval (top-%d)…", RETRIEVE_K)
    dense_samples = copy.deepcopy(samples)
    for s in tqdm(dense_samples, desc="Dense retrieval"):
        s["retrieved_chunks"] = retrieve_dense(
            s["question"], embed_model, collection, k=RETRIEVE_K
        )

    PREDS_DIR.mkdir(parents=True, exist_ok=True)
    bge_pred_path = PREDS_DIR / "finqa_bge_m3_retrieval.json"
    with open(bge_pred_path, "w") as f:
        json.dump(dense_samples, f, indent=2)
    logger.info("Saved BGE-M3 predictions → %s", bge_pred_path)

    bge_metrics = compute_and_save_metrics(
        dense_samples,
        METRICS_DIR / "finqa_bge_m3_metrics.json",
        "FinQA BGE-M3",
    )

    # ── Stage B: FT reranker ──────────────────────────────────────────────
    if not CHECKPOINT.exists():
        logger.error("Checkpoint not found at %s — skipping reranker stage", CHECKPOINT)
        return

    logger.info("Loading FT cross-encoder from %s", CHECKPOINT)
    from sentence_transformers.cross_encoder import CrossEncoder
    reranker = CrossEncoder(str(CHECKPOINT), max_length=512)

    logger.info("Re-ranking top-%d chunks…", RERANK_TOP_N)
    reranked_samples = copy.deepcopy(dense_samples)
    for s in tqdm(reranked_samples, desc="FT reranking"):
        chunks = s["retrieved_chunks"][:RERANK_TOP_N]
        s["retrieved_chunks"] = rerank_chunks(s["question"], chunks, reranker)

    ft_pred_path = PREDS_DIR / "finqa_ft_reranker_retrieval.json"
    with open(ft_pred_path, "w") as f:
        json.dump(reranked_samples, f, indent=2)
    logger.info("Saved FT-reranker predictions → %s", ft_pred_path)

    ft_metrics = compute_and_save_metrics(
        reranked_samples,
        METRICS_DIR / "finqa_ft_reranker_metrics.json",
        "FinQA BGE-M3 + FT-Reranker",
    )

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 85)
    print(f"{'Variant':<35}  {'page_rec@1':>10}  {'page_rec@5':>10}  {'chunk_rec@5':>11}  {'mrr':>8}")
    print("-" * 85)
    for label, m in [("FinQA BGE-M3", bge_metrics), ("FinQA BGE-M3 + FT-Reranker", ft_metrics)]:
        ov = m.get("overall", {})
        print(
            f"{label:<35}  "
            f"{ov.get('page_recall@1', 0):>10.4f}  "
            f"{ov.get('page_recall@5', 0):>10.4f}  "
            f"{ov.get('chunk_recall@5', 0):>11.4f}  "
            f"{ov.get('mrr', 0):>8.4f}"
        )
    print("=" * 85 + "\n")
    logger.info("Done.")


if __name__ == "__main__":
    main()
