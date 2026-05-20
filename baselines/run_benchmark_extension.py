#!/usr/bin/env python3
"""
run_benchmark_extension.py
============================
Full baseline suite on the COMBINED benchmark:
  FinanceBench  (150 questions, 84 unique docs)
  + FinQA gold-pages test set (530 questions, 122 unique docs)

Index: ONE global Chroma collection over ALL PDFs from both corpora.

Methods
-------
  14 standard baselines (dense/bm25/splade/hybrid/parent-child/hyde/multi-hyde/reranker)
  + dense_bge_m3_ft_reranker    — FT CrossEncoder on BGE-M3 top-20
  + multi_hyde_ft_reranker      — FT CrossEncoder on Multi-HyDE top-20  ← 0.56 PageRec@5
  + oracle_doc                  — dense BGE-M3 restricted to gold document
  + oracle_page                 — dense BGE-M3 restricted to gold page(s)

Results reported at THREE levels:
  global     — all 680 questions over the unified search space
  financebench — FinanceBench questions only (subset of global results)
  finqa      — FinQA questions only (subset of global results)

Corpus size stats (chunk counts, question counts, PDF counts) printed at end.

Usage
-----
  # Full run (GPU required for HyDE and generation)
  python baselines/run_benchmark_extension.py

  # Retrieval only (no LLM generation)
  python baselines/run_benchmark_extension.py --skip-generation

  # Skip SPLADE (saves ~30 min)
  python baselines/run_benchmark_extension.py --skip-splade --skip-generation

  # Specific variants only
  python baselines/run_benchmark_extension.py --variants dense_bge_m3 bm25 multi_hyde_ft_reranker

  # Resume after partial run
  python baselines/run_benchmark_extension.py --resume --skip-generation
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import pickle
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "baselines"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark_ext")

# ---------------------------------------------------------------------------
# Import reusable components from run_baselines
# ---------------------------------------------------------------------------
from run_baselines import (  # noqa: E402
    EMBED_MODEL, BGE_BASE_MODEL, QWEN_MODEL, RERANKER_MODEL, SPLADE_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP, PARENT_SIZE, PARENT_OVERLAP, CHILD_SIZE, CHILD_OVERLAP,
    K_VALUES, MAIN_K, CANDIDATE_K, RRF_K,
    make_recursive_splitter, chunk_docs,
    _get_or_create_chroma,
    _finance_tokenize,
    _embed_texts, _rrf_merge, _chunk_id,
    apply_reranker,
    pre_generate_hypotheticals,
    compute_retrieval_metrics, aggregate_by_group,
    plot_retrieval_bar_k5, plot_recall_at_k_curves,
    save_metrics_table,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FT_CHECKPOINT   = PROJECT_ROOT / "checkpoints" / "ft_cross_encoder"

FB_DATA_PATH    = PROJECT_ROOT / "data" / "financebench_open_source.jsonl"
FB_DOCINFO_PATH = PROJECT_ROOT / "data" / "financebench_document_information.jsonl"
FINQA_DATA_PATH = PROJECT_ROOT / "data" / "finqa_test_gold_pages.jsonl"

FB_PDF_DIR      = PROJECT_ROOT / "pdfs"
FINQA_PDF_DIR   = PROJECT_ROOT / "Final-PDF"

GLOBAL_COLLECTION = "benchmark_ext_global_bge_m3_tok1024_ol128"

QUESTION_TYPES = ["metrics-generated", "domain-relevant", "novel-generated", "finqa"]

VARIANTS_NAMES = [
    "dense_bge_m3",
    "dense_bge_base",
    "dense_investopedia",
    "bm25",
    "splade",
    "hybrid_50_50",
    "hybrid_75_25",
    "hybrid_25_75",
    "parent_child",
    "query_expansion",
    "hyde",
    "multi_hyde",
    "bge_reranker",
    "multi_hyde_reranker",
    "dense_bge_m3_ft_reranker",
    "multi_hyde_ft_reranker",
    "oracle_doc",
    "oracle_page",
]

VARIANT_LABELS = {
    "dense_bge_m3":              "Dense BGE-M3",
    "dense_bge_base":            "Dense BGE-Base",
    "dense_investopedia":        "Dense Investopedia",
    "bm25":                      "BM25",
    "splade":                    "SPLADE",
    "hybrid_50_50":              "Hybrid RRF 50/50",
    "hybrid_75_25":              "Hybrid RRF 75/25 (dense-heavy)",
    "hybrid_25_75":              "Hybrid RRF 25/75 (sparse-heavy)",
    "parent_child":              "Parent-Child",
    "query_expansion":           "Query Expansion",
    "hyde":                      "HyDE",
    "multi_hyde":                "Multi-HyDE",
    "bge_reranker":              "BGE-M3 + ReRanker",
    "multi_hyde_reranker":       "BGE-M3 + Multi-HyDE + ReRanker",
    "dense_bge_m3_ft_reranker":  "BGE-M3 + FT-Reranker",
    "multi_hyde_ft_reranker":    "BGE-M3 + Multi-HyDE + FT-Reranker",
    "oracle_doc":                "Oracle-Doc (upper bound, doc known)",
    "oracle_page":               "Oracle-Page (upper bound, page known)",
}

INVESTOPEDIA_MODEL = "FinLang/finance-embeddings-investopedia"

# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_financebench(fb_path: Path, docinfo_path: Path) -> Tuple[List[Dict], Dict]:
    """Load FinanceBench JSONL → unified sample schema."""
    doc_info: Dict[str, Dict] = {}
    if docinfo_path.exists():
        with open(docinfo_path) as f:
            for line in f:
                d = json.loads(line.strip())
                if d:
                    doc_info[d["doc_name"]] = d

    samples = []
    with open(fb_path) as f:
        for line in f:
            raw = json.loads(line.strip())
            if not raw:
                continue
            gold_segs = []
            for ev in raw.get("evidence", []):
                gold_segs.append({
                    "text":     ev.get("evidence_text", ""),
                    "doc_name": ev.get("doc_name", raw.get("doc_name", "")),
                    "page":     ev.get("evidence_page_num", -1),
                })
            dm = doc_info.get(raw.get("doc_name", ""), {})
            samples.append({
                "id":                   raw.get("financebench_id", ""),
                "question":             raw.get("question", ""),
                "reference_answer":     raw.get("answer", ""),
                "question_type":        raw.get("question_type", "unknown"),
                "doc_name":             raw.get("doc_name", ""),
                "doc_link":             raw.get("doc_link", ""),
                "doc_type":             dm.get("doc_type", _infer_doc_type(raw.get("doc_name", ""))),
                "dataset":              "financebench",
                "gold_evidence_segments": gold_segs,
                "retrieved_chunks":     [],
                "generated_answer":     "",
            })
    logger.info(f"FinanceBench: {len(samples)} questions from {len({s['doc_name'] for s in samples})} docs")
    return samples, doc_info


def load_finqa(finqa_path: Path) -> List[Dict]:
    """Load FinQA gold-pages JSONL → unified sample schema."""
    samples = []
    with open(finqa_path) as f:
        for line in f:
            raw = json.loads(line.strip())
            if not raw:
                continue

            gold_segs = []
            for ev in raw.get("evidences_updated", raw.get("evidences", [])):
                gold_segs.append({
                    "text":     ev.get("evidence_text", ev.get("pre_text", "")),
                    "doc_name": ev.get("doc_name", ""),
                    "page":     int(ev.get("page_num", -1)),
                })
            if not gold_segs:
                continue

            doc_name = gold_segs[0]["doc_name"]
            samples.append({
                "id":                   raw.get("qid", ""),
                "question":             raw.get("question", ""),
                "reference_answer":     str(raw.get("answer", "")),
                "question_type":        "finqa",
                "doc_name":             doc_name,
                "doc_link":             "",          # PDF is local
                "doc_type":             _infer_doc_type(doc_name),
                "dataset":              "finqa",
                "gold_evidence_segments": gold_segs,
                "retrieved_chunks":     [],
                "generated_answer":     "",
            })
    logger.info(f"FinQA: {len(samples)} questions from {len({s['doc_name'] for s in samples})} docs")
    return samples


def _infer_doc_type(doc_name: str) -> str:
    d = doc_name.lower()
    if "10q" in d or "10-q" in d:
        return "10q"
    if "8k" in d or "8-k" in d:
        return "8k"
    if "10k" in d or "10-k" in d:
        return "10k"
    if "earnings" in d or "earn" in d:
        return "earnings"
    return "unknown"


# ---------------------------------------------------------------------------
# 2. Multi-directory PDF loader
# ---------------------------------------------------------------------------

def load_pdf_multidir(doc_name: str, doc_link: str,
                      pdf_dirs: List[str]) -> List:
    """Try each pdf_dir in order, return first successful load."""
    from src.ingestion.pdf_utils import load_pdf_with_fallback
    for pdf_dir in pdf_dirs:
        try:
            docs, _ = load_pdf_with_fallback(doc_name, doc_link, pdf_dir)
            if docs:
                return docs
        except Exception as e:
            logger.debug(f"  {pdf_dir}: {e}")
    logger.warning(f"  Could not load PDF for {doc_name} from any dir")
    return []


# ---------------------------------------------------------------------------
# 3. Global index building
# ---------------------------------------------------------------------------

def build_global_dense_index(
    all_samples: List[Dict],
    pdf_dirs: List[str],
    vs_dir: str,
    embed_model_name: str = EMBED_MODEL,
    collection_name: Optional[str] = None,
) -> "chromadb.Collection":
    """Build or load a global ChromaDB dense index over all docs in both corpora."""
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    if collection_name is None:
        if embed_model_name == EMBED_MODEL:
            collection_name = GLOBAL_COLLECTION
        else:
            tag = re.sub(r"[^a-z0-9]", "_", embed_model_name.lower().split("/")[-1])
            collection_name = f"benchmark_ext_global_{tag}_tok1024_ol128"

    ef = SentenceTransformerEmbeddingFunction(model_name=embed_model_name, device="cuda")
    collection = _get_or_create_chroma(vs_dir, collection_name, ef)

    existing_ids = set(collection.get(include=[])["ids"])
    unique_docs  = {s["doc_name"]: s["doc_link"] for s in all_samples}
    splitter     = make_recursive_splitter()

    docs_to_add, meta_to_add, ids_to_add = [], [], []

    for doc_name, doc_link in tqdm(unique_docs.items(), desc=f"Indexing {collection_name}"):
        marker = f"{doc_name}__page_0__chunk_0"
        if marker in existing_ids:
            continue
        pages = load_pdf_multidir(doc_name, doc_link, pdf_dirs)
        if not pages:
            continue
        chunks = chunk_docs(pages, splitter)
        for idx, c in enumerate(chunks):
            cid = f"{doc_name}__page_{c['page']}__chunk_{idx}"
            if cid in existing_ids:
                continue
            docs_to_add.append(c["text"])
            meta_to_add.append({"doc_name": c["doc_name"], "page": int(c["page"]), "chunk_idx": idx})
            ids_to_add.append(cid)
            if len(docs_to_add) >= 2000:
                collection.add(documents=docs_to_add, metadatas=meta_to_add, ids=ids_to_add)
                docs_to_add, meta_to_add, ids_to_add = [], [], []

    if docs_to_add:
        collection.add(documents=docs_to_add, metadatas=meta_to_add, ids=ids_to_add)

    logger.info(f"Dense index '{collection_name}': {collection.count()} chunks")
    return collection


def build_global_bm25_index(
    all_samples: List[Dict],
    pdf_dirs: List[str],
    cache_dir: str,
    cache_suffix: str = "global",
) -> Tuple[List[Dict], object]:
    """Build or load a BM25 index over all docs from both corpora."""
    from rank_bm25 import BM25Okapi

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"bm25_{cache_suffix}_sz{CHUNK_SIZE}_ol{CHUNK_OVERLAP}.pkl")

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        logger.info(f"BM25 index loaded from cache ({len(data['chunks'])} chunks)")
        return data["chunks"], BM25Okapi([c["tokens"] for c in data["chunks"]])

    unique_docs = {s["doc_name"]: s["doc_link"] for s in all_samples}
    splitter    = make_recursive_splitter()
    all_chunks  = []

    for doc_name, doc_link in tqdm(unique_docs.items(), desc="Building global BM25"):
        pages = load_pdf_multidir(doc_name, doc_link, pdf_dirs)
        if not pages:
            continue
        for c in chunk_docs(pages, splitter):
            all_chunks.append({**c, "tokens": _finance_tokenize(c["text"])})

    bm25 = BM25Okapi([c["tokens"] for c in all_chunks])
    with open(cache_path, "wb") as f:
        pickle.dump({"chunks": all_chunks}, f)
    logger.info(f"BM25 index built: {len(all_chunks)} chunks")
    return all_chunks, bm25


def build_global_parent_child_index(
    all_samples: List[Dict],
    pdf_dirs: List[str],
    vs_dir: str,
) -> Tuple["chromadb.Collection", Dict[str, str]]:
    """Build parent-child index over global corpus."""
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    child_col_name = "benchmark_ext_global_parent_child_child"
    ef   = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL, device="cuda")
    child_col = _get_or_create_chroma(vs_dir, child_col_name, ef)

    parent_splitter = make_recursive_splitter(chunk_size=PARENT_SIZE, chunk_overlap=PARENT_OVERLAP)
    child_splitter  = make_recursive_splitter(chunk_size=CHILD_SIZE,  chunk_overlap=CHILD_OVERLAP)

    existing_ids = set(child_col.get(include=[])["ids"])
    unique_docs  = {s["doc_name"]: s["doc_link"] for s in all_samples}
    parent_map: Dict[str, str] = {}

    docs_to_add, meta_to_add, ids_to_add = [], [], []

    for doc_name, doc_link in tqdm(unique_docs.items(), desc="Parent-child index"):
        marker = f"{doc_name}__child_page_0__child_0"
        if marker in existing_ids:
            continue
        pages = load_pdf_multidir(doc_name, doc_link, pdf_dirs)
        if not pages:
            continue
        parent_chunks = chunk_docs(pages, parent_splitter)
        child_chunks  = chunk_docs(pages, child_splitter)

        for pidx, pc in enumerate(parent_chunks):
            pid = f"{doc_name}__parent_{pidx}"
            parent_map[pid] = pc["text"]

        for cidx, cc in enumerate(child_chunks):
            cid = f"{doc_name}__child_page_{cc['page']}__child_{cidx}"
            if cid in existing_ids:
                continue
            parent_idx = min(int(cidx * CHILD_SIZE / PARENT_SIZE), len(parent_chunks) - 1)
            pid = f"{doc_name}__parent_{parent_idx}"
            docs_to_add.append(cc["text"])
            meta_to_add.append({
                "doc_name": cc["doc_name"], "page": int(cc["page"]),
                "chunk_idx": cidx, "parent_id": pid,
            })
            ids_to_add.append(cid)
            if len(docs_to_add) >= 2000:
                child_col.add(documents=docs_to_add, metadatas=meta_to_add, ids=ids_to_add)
                docs_to_add, meta_to_add, ids_to_add = [], [], []

    if docs_to_add:
        child_col.add(documents=docs_to_add, metadatas=meta_to_add, ids=ids_to_add)

    logger.info(f"Parent-child index: {child_col.count()} children, {len(parent_map)} parents")
    return child_col, parent_map


def build_global_splade_index(
    all_samples: List[Dict],
    pdf_dirs: List[str],
    cache_dir: str,
    cache_suffix: str = "global",
):
    """Build SPLADE index over global corpus."""
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"splade_{cache_suffix}_sz{CHUNK_SIZE}_ol{CHUNK_OVERLAP}.pkl")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(SPLADE_MODEL)
    model = AutoModelForMaskedLM.from_pretrained(SPLADE_MODEL).to(device).eval()

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        logger.info(f"SPLADE index loaded from cache ({len(data['chunks'])} chunks)")
        return model, tokenizer, data["chunks"], data["postings"]

    unique_docs = {s["doc_name"]: s["doc_link"] for s in all_samples}
    splitter    = make_recursive_splitter()
    all_chunks  = []

    for doc_name, doc_link in tqdm(unique_docs.items(), desc="Chunking for SPLADE"):
        pages = load_pdf_multidir(doc_name, doc_link, pdf_dirs)
        if not pages:
            continue
        all_chunks.extend(chunk_docs(pages, splitter))

    postings: Dict[int, Tuple[List[int], List[float]]] = defaultdict(lambda: ([], []))
    batch_size = 32
    with torch.no_grad():
        for start in tqdm(range(0, len(all_chunks), batch_size), desc="SPLADE encode"):
            batch = [c["text"][:512] for c in all_chunks[start:start + batch_size]]
            enc   = tokenizer(batch, return_tensors="pt", padding=True,
                              truncation=True, max_length=512).to(device)
            out   = model(**enc)
            sparse = torch.log1p(torch.relu(out.logits)).max(dim=1).values
            sparse = sparse.cpu().float().numpy()
            for offset, vec in enumerate(sparse):
                doc_id   = start + offset
                nz_ids   = np.nonzero(vec)[0]
                for tid in nz_ids:
                    postings[int(tid)][0].append(doc_id)
                    postings[int(tid)][1].append(float(vec[tid]))

    postings_final = {tid: (np.array(ids), np.array(weights))
                      for tid, (ids, weights) in postings.items()}
    with open(cache_path, "wb") as f:
        pickle.dump({"chunks": all_chunks, "postings": postings_final}, f)
    logger.info(f"SPLADE index: {len(all_chunks)} chunks, {len(postings_final)} terms")
    return model, tokenizer, all_chunks, postings_final


# ---------------------------------------------------------------------------
# 4. Retrieval helpers (extending run_baselines retrieval functions)
# ---------------------------------------------------------------------------

def _chroma_query_safe(collection, embedding: np.ndarray, n_results: int,
                       where: Optional[Dict] = None) -> List[Dict]:
    """ChromaDB query with safe handling for where-filtered result sets."""
    if where is None:
        n = min(n_results, collection.count())
    else:
        # Count items matching the filter
        try:
            matched = collection.get(where=where, include=[])
            n = min(n_results, len(matched["ids"]))
        except Exception:
            n = n_results
    if n == 0:
        return []
    try:
        kwargs = dict(
            query_embeddings=[embedding.tolist()],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where
        res = collection.query(**kwargs)
    except Exception as e:
        logger.warning(f"ChromaDB query failed: {e}")
        return []
    candidates = []
    for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        candidates.append({"text": text, "metadata": meta, "_score": float(1.0 - dist)})
    return candidates


def _make_oracle_where(sample: Dict, mode: str) -> Optional[Dict]:
    """Build ChromaDB where-filter for oracle conditions."""
    segs     = sample.get("gold_evidence_segments", [])
    if not segs:
        return None
    gold_doc = segs[0]["doc_name"]

    if mode == "oracle_doc":
        return {"doc_name": {"$eq": gold_doc}}

    if mode == "oracle_page":
        gold_pages = list({int(s["page"]) for s in segs if s.get("page", -1) >= 0})
        if not gold_pages:
            return {"doc_name": {"$eq": gold_doc}}
        if len(gold_pages) == 1:
            return {"$and": [
                {"doc_name": {"$eq": gold_doc}},
                {"page":     {"$eq": gold_pages[0]}},
            ]}
        return {"$and": [
            {"doc_name": {"$eq": gold_doc}},
            {"page":     {"$in": gold_pages}},
        ]}
    return None


def retrieve_oracle(sample: Dict, embed_model, collection,
                    k: int, mode: str) -> List[Dict]:
    """
    Oracle retrieval using a get+rank approach so ALL gold-doc/gold-page
    chunks are considered (not just those surfaced by ANN search).
    """
    where = _make_oracle_where(sample, mode)
    if where is None:
        return []

    try:
        result = collection.get(where=where, include=["documents", "metadatas", "embeddings"])
    except Exception as e:
        logger.warning(f"Oracle get failed ({mode}): {e}")
        return []

    ids       = result.get("ids", [])
    documents = result.get("documents", [])
    metadatas = result.get("metadatas", [])
    embeddings_list = result.get("embeddings", [])

    if not ids:
        return []

    q_emb = _embed_texts([sample["question"]], embed_model)[0]

    if embeddings_list is not None and len(embeddings_list) > 0:
        doc_embs = np.array(embeddings_list, dtype=np.float32)
        # Cosine similarity (embeddings already normalised by SentenceTransformer)
        norms = np.linalg.norm(doc_embs, axis=1, keepdims=True)
        doc_embs_norm = doc_embs / np.where(norms > 0, norms, 1)
        q_norm = q_emb / (np.linalg.norm(q_emb) or 1.0)
        sims = doc_embs_norm @ q_norm
    else:
        sims = np.zeros(len(ids))

    top_idxs = np.argsort(sims)[::-1][:k]
    candidates = []
    for rank, idx in enumerate(top_idxs, start=1):
        candidates.append({
            "text":     documents[idx],
            "metadata": metadatas[idx],
            "_score":   float(sims[idx]),
            "rank":     rank,
        })
    return candidates


def retrieve_dense_ext(sample: Dict, embed_model, collection,
                        k: int = MAIN_K) -> List[Dict]:
    q_emb      = _embed_texts([sample["question"]], embed_model)[0]
    candidates = _chroma_query_safe(collection, q_emb, k)
    for i, c in enumerate(candidates):
        c["rank"] = i + 1
    return candidates


def retrieve_bm25_ext(sample: Dict, bm25_chunks: List[Dict], bm25_index,
                      k: int = MAIN_K) -> List[Dict]:
    tokens = _finance_tokenize(sample["question"])
    scores = bm25_index.get_scores(tokens)
    top    = np.argsort(scores)[::-1][:k]
    results = []
    for rank, idx in enumerate(top, start=1):
        if scores[int(idx)] <= 0.0:
            break
        c = bm25_chunks[int(idx)]
        results.append({
            "text":     c["text"],
            "metadata": {"doc_name": c["doc_name"], "page": c["page"]},
            "_score":   float(scores[int(idx)]),
            "rank":     rank,
        })
    return results


def retrieve_hybrid_ext(sample: Dict, embed_model, collection,
                         bm25_chunks: List[Dict], bm25_index,
                         alpha: float = 0.5, k: int = MAIN_K) -> List[Dict]:
    """RRF hybrid: dense + BM25."""
    pool = max(k, CANDIDATE_K)

    q_emb        = _embed_texts([sample["question"]], embed_model)[0]
    dense_cands  = _chroma_query_safe(collection, q_emb, pool)
    bm25_tokens  = _finance_tokenize(sample["question"])
    bm25_scores  = bm25_index.get_scores(bm25_tokens)
    bm25_top_idx = np.argsort(bm25_scores)[::-1][:pool]

    scores_map: Dict[str, Dict] = {}
    dense_ids: List[str] = []
    bm25_ids:  List[str] = []

    for c in dense_cands:
        cid = _chunk_id(c)
        scores_map[cid] = c
        dense_ids.append(cid)

    for idx in bm25_top_idx:
        if bm25_scores[int(idx)] <= 0:
            continue
        c = bm25_chunks[int(idx)]
        cid = f"{c['doc_name']}__page_{c['page']}__score_{bm25_scores[int(idx)]:.4f}"
        if cid not in scores_map:
            scores_map[cid] = {
                "text":     c["text"],
                "metadata": {"doc_name": c["doc_name"], "page": c["page"]},
                "_score":   float(bm25_scores[int(idx)]),
            }
        bm25_ids.append(cid)

    merged = _rrf_merge([dense_ids, bm25_ids], scores_map, k=RRF_K, top_n=k)
    for i, c in enumerate(merged):
        c["rank"] = i + 1
    return merged


def retrieve_splade_ext(sample: Dict, splade_model, splade_tokenizer,
                         splade_chunks: List[Dict], splade_postings: Dict,
                         k: int = MAIN_K) -> List[Dict]:
    import torch
    device = next(splade_model.parameters()).device
    enc    = splade_tokenizer([sample["question"]], return_tensors="pt",
                              truncation=True, max_length=512, padding=True).to(device)
    with torch.no_grad():
        out   = splade_model(**enc)
        q_vec = torch.log1p(torch.relu(out.logits)).max(dim=1).values.squeeze(0)
        q_vec = q_vec.cpu().float().numpy()

    scores = np.zeros(len(splade_chunks))
    for tid in np.nonzero(q_vec)[0]:
        if int(tid) in splade_postings:
            doc_ids, weights = splade_postings[int(tid)]
            scores[doc_ids] += float(q_vec[tid]) * weights

    top = np.argsort(scores)[::-1][:k]
    results = []
    for rank, idx in enumerate(top, start=1):
        if scores[int(idx)] <= 0:
            break
        c = splade_chunks[int(idx)]
        results.append({
            "text":     c["text"],
            "metadata": {"doc_name": c["doc_name"], "page": c["page"]},
            "_score":   float(scores[int(idx)]),
            "rank":     rank,
        })
    return results


def retrieve_parent_child_ext(sample: Dict, embed_model, child_col,
                               parent_map: Dict[str, str],
                               k: int = MAIN_K) -> List[Dict]:
    q_emb    = _embed_texts([sample["question"]], embed_model)[0]
    children = _chroma_query_safe(child_col, q_emb, k)
    results  = []
    seen_parents: set = set()
    for rank, c in enumerate(children, start=1):
        pid  = c["metadata"].get("parent_id", "")
        text = parent_map.get(pid, c["text"])
        if pid and pid in seen_parents:
            continue
        if pid:
            seen_parents.add(pid)
        results.append({
            "text":     text,
            "metadata": c["metadata"],
            "_score":   c["_score"],
            "rank":     rank,
        })
    return results[:k]


def retrieve_hyde_ext(sample: Dict, embed_model, collection,
                       hyde_cache: Dict[str, List[str]],
                       hyde_n: int = 1, k: int = MAIN_K) -> List[Dict]:
    hyps = hyde_cache.get(sample["question"], [])[:hyde_n]
    if not hyps:
        return retrieve_dense_ext(sample, embed_model, collection, k)

    embs = _embed_texts(hyps, embed_model)
    avg  = embs.mean(axis=0)
    avg  /= max(np.linalg.norm(avg), 1e-9)

    candidates = _chroma_query_safe(collection, avg, k)
    for i, c in enumerate(candidates):
        c["rank"] = i + 1
    return candidates


def retrieve_query_expansion_ext(sample: Dict, embed_model, collection,
                                  k: int = MAIN_K) -> List[Dict]:
    try:
        from src.experiments.query_expansion import process_query_for_experiment
        expanded, _ = process_query_for_experiment(sample["question"])
    except Exception:
        expanded = sample["question"]
    q_emb = _embed_texts([expanded], embed_model)[0]
    candidates = _chroma_query_safe(collection, q_emb, k)
    for i, c in enumerate(candidates):
        c["rank"] = i + 1
    return candidates


# ---------------------------------------------------------------------------
# 5. Corpus size statistics
# ---------------------------------------------------------------------------

def compute_corpus_stats(
    fb_samples: List[Dict],
    finqa_samples: List[Dict],
    fb_pdf_dir: str,
    finqa_pdf_dir: str,
    vs_dir: str,
) -> Dict:
    """Count unique docs, questions, and index chunks per corpus."""
    import chromadb

    fb_docs   = set(s["doc_name"] for s in fb_samples)
    finqa_docs = set(s["doc_name"] for s in finqa_samples)
    all_docs   = fb_docs | finqa_docs

    stats = {
        "financebench_questions": len(fb_samples),
        "finqa_questions":        len(finqa_samples),
        "total_questions":        len(fb_samples) + len(finqa_samples),
        "financebench_docs":      len(fb_docs),
        "finqa_only_docs":        len(finqa_docs - fb_docs),
        "overlap_docs":           len(fb_docs & finqa_docs),
        "total_unique_docs":      len(all_docs),
    }

    # Chunk counts from ChromaDB if available
    try:
        client = chromadb.PersistentClient(path=vs_dir)
        existing_names = {c.name for c in client.list_collections()}

        # FB-only collection (from run_baselines.py)
        fb_col_name = "baselines_dense_tok1024_ol128"
        if fb_col_name in existing_names:
            fb_col = client.get_collection(fb_col_name)
            stats["financebench_chunks"] = fb_col.count()

        # Global collection
        if GLOBAL_COLLECTION in existing_names:
            global_col = client.get_collection(GLOBAL_COLLECTION)
            stats["global_chunks"] = global_col.count()
            if "financebench_chunks" in stats:
                stats["finqa_added_chunks"] = stats["global_chunks"] - stats["financebench_chunks"]
    except Exception as e:
        logger.debug(f"Could not read chunk counts: {e}")

    return stats


def print_corpus_stats(stats: Dict) -> None:
    print("\n" + "=" * 65)
    print("CORPUS STATISTICS")
    print("=" * 65)
    print(f"  {'':35s}  {'FinanceBench':>12}  {'Global':>8}")
    print(f"  {'Questions':35s}  {stats.get('financebench_questions',0):>12,d}  {stats.get('total_questions',0):>8,d}")
    print(f"  {'Unique docs':35s}  {stats.get('financebench_docs',0):>12,d}  {stats.get('total_unique_docs',0):>8,d}")
    if "financebench_chunks" in stats:
        print(f"  {'Index chunks (BGE-M3 tok1024/ol128)':35s}  {stats.get('financebench_chunks',0):>12,d}  {stats.get('global_chunks',0):>8,d}")
    print()
    added_chunks = f"{stats['finqa_added_chunks']:,}" if 'finqa_added_chunks' in stats else '?'
    print(f"  FinQA adds: {stats.get('finqa_questions',0):,} questions, "
          f"{stats.get('finqa_only_docs',0):,} new docs, "
          f"{added_chunks} new chunks")
    print(f"  Docs in both corpora: {stats.get('overlap_docs',0)}")
    print("=" * 65)


# ---------------------------------------------------------------------------
# 6. Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_subset(
    retrieval_results: Dict[str, List[Dict]],
    dataset_filter: Optional[str],
    label: str,
) -> Dict[str, Dict]:
    """Compute metrics for all variants on an optional subset of samples."""
    all_metrics: Dict[str, Dict] = {}
    for name, samples in retrieval_results.items():
        if dataset_filter:
            sub = [s for s in samples if s.get("dataset") == dataset_filter]
        else:
            sub = samples
        if not sub:
            continue
        overall = compute_retrieval_metrics(sub)
        by_qt   = aggregate_by_group(sub, lambda s: s.get("question_type", "unknown"))
        by_dt   = aggregate_by_group(sub, lambda s: s.get("doc_type",      "unknown"))
        by_qtdt = aggregate_by_group(sub, lambda s: f"{s.get('question_type','?')}|{s.get('doc_type','?')}")
        all_metrics[name] = {
            "overall":                      overall,
            "by_question_type":             by_qt,
            "by_doc_type":                  by_dt,
            "by_question_type_x_doc_type":  by_qtdt,
            "n":                            len(sub),
        }
        pr5 = overall.get(f"page_recall@{MAIN_K}", 0)
        dr5 = overall.get(f"doc_recall@{MAIN_K}",  0)
        logger.info(f"  [{label}][{VARIANT_LABELS.get(name, name):<42}] "
                    f"DocRec@5={dr5:.3f}  PageRec@5={pr5:.3f}  n={len(sub)}")
    return all_metrics


def print_summary_table(
    all_metrics_by_level: Dict[str, Dict[str, Dict]],
    variant_order: List[str],
) -> None:
    print()
    for level, all_metrics in all_metrics_by_level.items():
        label_width = 44
        col_head = "  ".join(f"PR@{k:>2}" for k in K_VALUES)
        print(f"\n{'='*110}")
        print(f"RESULTS — {level.upper()}")
        print(f"{'Method':<{label_width}}  {col_head}  {'PR@5':>6}  {'DR@5':>6}  {'MRR':>6}")
        print("-" * 110)
        for name in variant_order:
            if name not in all_metrics:
                continue
            r   = all_metrics[name]["overall"]
            lbl = VARIANT_LABELS.get(name, name)
            pr  = "  ".join(f"{r.get(f'page_recall@{k}', 0):>5.3f}" for k in K_VALUES)
            pr5 = r.get(f"page_recall@{MAIN_K}", 0)
            dr5 = r.get(f"doc_recall@{MAIN_K}",  0)
            mrr = r.get("mrr", 0)
            print(f"{lbl:<{label_width}}  {pr}  {pr5:>6.3f}  {dr5:>6.3f}  {mrr:>6.3f}")
        print("=" * 110)


def save_all_metrics(
    all_metrics: Dict[str, Dict],
    output_dir: str,
    level: str,
) -> None:
    path = os.path.join(output_dir, "metrics", f"all_variants_{level}.json")
    with open(path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Metrics saved → {path}")


def save_csv_summary(
    all_metrics_by_level: Dict[str, Dict[str, Dict]],
    output_dir: str,
    variant_order: List[str],
) -> None:
    import csv
    path = os.path.join(output_dir, "metrics", "benchmark_extension_summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        header = (["Level", "Method"]
                  + [f"PageRec@{k}" for k in K_VALUES]
                  + [f"DocRec@{k}"  for k in K_VALUES]
                  + ["MRR", "N"])
        w.writerow(header)
        for level, all_metrics in all_metrics_by_level.items():
            for name in variant_order:
                if name not in all_metrics:
                    continue
                r   = all_metrics[name]["overall"]
                n   = all_metrics[name].get("n", 0)
                lbl = VARIANT_LABELS.get(name, name)
                w.writerow(
                    [level, lbl]
                    + [f"{r.get(f'page_recall@{k}', 0):.4f}" for k in K_VALUES]
                    + [f"{r.get(f'doc_recall@{k}',  0):.4f}" for k in K_VALUES]
                    + [f"{r.get('mrr', 0):.4f}", n]
                )
    logger.info(f"Summary CSV → {path}")


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Benchmark Extension: FinanceBench + FinQA")
    p.add_argument("--fb-data",      default=str(FB_DATA_PATH))
    p.add_argument("--fb-docinfo",   default=str(FB_DOCINFO_PATH))
    p.add_argument("--finqa-data",   default=str(FINQA_DATA_PATH))
    p.add_argument("--fb-pdf-dir",   default=str(FB_PDF_DIR))
    p.add_argument("--finqa-pdf-dir",default=str(FINQA_PDF_DIR))
    p.add_argument("--vs-dir",       default="vector_stores/benchmark_extension")
    p.add_argument("--output-dir",   default="baselines/results/benchmark_extension")
    p.add_argument("--hyde-cache",   default="baselines/benchmark_ext_hyde_cache.json")
    p.add_argument("--variants",     nargs="*", default=None,
                   help="Run only these variants (default: all)")
    p.add_argument("--skip-generation", action="store_true")
    p.add_argument("--skip-plots",      action="store_true")
    p.add_argument("--skip-splade",     action="store_true",
                   help="Skip SPLADE indexing and retrieval (saves ~30min)")
    p.add_argument("--skip-hyde",       action="store_true",
                   help="Skip HyDE/Multi-HyDE variants (no LLM needed)")
    p.add_argument("--resume",          action="store_true",
                   help="Skip variants whose prediction JSON already exists")
    return p.parse_args()


def main():
    args   = parse_args()
    t_start = time.time()

    def abspath(p):
        return str(PROJECT_ROOT / p) if not os.path.isabs(p) else p

    fb_data_path    = abspath(args.fb_data)
    fb_docinfo_path = abspath(args.fb_docinfo)
    finqa_data_path = abspath(args.finqa_data)
    fb_pdf_dir      = abspath(args.fb_pdf_dir)
    finqa_pdf_dir   = abspath(args.finqa_pdf_dir)
    vs_dir          = abspath(args.vs_dir)
    output_dir      = abspath(args.output_dir)
    hyde_cache_path = abspath(args.hyde_cache)

    pdf_dirs = [fb_pdf_dir, finqa_pdf_dir]

    for d in [vs_dir, output_dir,
              os.path.join(output_dir, "metrics"),
              os.path.join(output_dir, "plots"),
              os.path.join(output_dir, "predictions")]:
        os.makedirs(d, exist_ok=True)

    # ── Select active variants ────────────────────────────────────────────────
    all_names = VARIANTS_NAMES.copy()
    if args.skip_splade:
        all_names = [n for n in all_names if "splade" not in n]
    if args.skip_hyde:
        all_names = [n for n in all_names if "hyde" not in n]
    if args.variants:
        all_names = [n for n in all_names if n in args.variants]

    logger.info("=" * 70)
    logger.info("Benchmark Extension: FinanceBench + FinQA")
    logger.info(f"  Variants : {all_names}")
    logger.info(f"  FT ckpt  : {FT_CHECKPOINT}")
    logger.info(f"  Output   : {output_dir}")
    logger.info("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("\n>>> Loading data")
    fb_samples, doc_info = load_financebench(
        Path(fb_data_path), Path(fb_docinfo_path)
    )
    finqa_samples = load_finqa(Path(finqa_data_path))
    all_samples   = fb_samples + finqa_samples
    questions     = [s["question"] for s in all_samples]
    logger.info(f"  Combined: {len(all_samples)} questions, "
                f"{len({s['doc_name'] for s in all_samples})} unique docs")

    # ── PHASE 0: Corpus stats ────────────────────────────────────────────────
    stats = compute_corpus_stats(fb_samples, finqa_samples, fb_pdf_dir, finqa_pdf_dir, vs_dir)
    print_corpus_stats(stats)

    # ── Determine what we need to build ──────────────────────────────────────
    needs_dense   = any(n in all_names for n in [
        "dense_bge_m3", "dense_bge_base", "dense_investopedia",
        "hybrid_50_50", "hybrid_75_25", "hybrid_25_75",
        "query_expansion", "hyde", "multi_hyde",
        "bge_reranker", "multi_hyde_reranker",
        "dense_bge_m3_ft_reranker", "multi_hyde_ft_reranker",
        "oracle_doc", "oracle_page",
    ])
    needs_bm25    = any(n in all_names for n in ["bm25","hybrid_50_50","hybrid_75_25","hybrid_25_75"])
    needs_splade  = "splade" in all_names
    needs_pc      = "parent_child" in all_names
    needs_hyde    = any("hyde" in n for n in all_names)
    needs_rerank  = any(n in all_names for n in ["bge_reranker","multi_hyde_reranker"])
    needs_ft_rerank = any(n in all_names for n in ["dense_bge_m3_ft_reranker","multi_hyde_ft_reranker"])

    # Collect unique embed models needed
    embed_model_map: Dict[str, str] = {
        "dense_bge_m3":        EMBED_MODEL,
        "dense_bge_base":      BGE_BASE_MODEL,
        "dense_investopedia":  INVESTOPEDIA_MODEL,
        "hybrid_50_50":        EMBED_MODEL,
        "hybrid_75_25":        EMBED_MODEL,
        "hybrid_25_75":        EMBED_MODEL,
        "query_expansion":     EMBED_MODEL,
        "hyde":                EMBED_MODEL,
        "multi_hyde":          EMBED_MODEL,
        "bge_reranker":        EMBED_MODEL,
        "multi_hyde_reranker": EMBED_MODEL,
        "dense_bge_m3_ft_reranker":  EMBED_MODEL,
        "multi_hyde_ft_reranker":    EMBED_MODEL,
        "parent_child":        EMBED_MODEL,
        "oracle_doc":          EMBED_MODEL,
        "oracle_page":         EMBED_MODEL,
    }
    dense_embed_models = list(dict.fromkeys(
        embed_model_map[n] for n in all_names if n in embed_model_map
    ))

    # ── PHASE 1: Build indexes ────────────────────────────────────────────────
    logger.info("\n>>> PHASE 1: Building global indexes")

    dense_collections: Dict[str, "chromadb.Collection"] = {}
    if needs_dense:
        for em in dense_embed_models:
            cname = GLOBAL_COLLECTION if em == EMBED_MODEL else None
            dense_collections[em] = build_global_dense_index(
                all_samples, pdf_dirs, vs_dir, embed_model_name=em, collection_name=cname
            )
    dense_col = dense_collections.get(EMBED_MODEL)

    bm25_chunks, bm25_index = None, None
    if needs_bm25:
        bm25_chunks, bm25_index = build_global_bm25_index(
            all_samples, pdf_dirs, os.path.join(vs_dir, "bm25_cache"), "global"
        )

    splade_model = splade_tokenizer = splade_chunks = splade_postings = None
    if needs_splade:
        splade_model, splade_tokenizer, splade_chunks, splade_postings = build_global_splade_index(
            all_samples, pdf_dirs, os.path.join(vs_dir, "splade_cache"), "global"
        )

    pc_col, parent_map = None, {}
    if needs_pc:
        pc_col, parent_map = build_global_parent_child_index(all_samples, pdf_dirs, vs_dir)

    # Update stats with chunk counts now that indexes are built
    stats = compute_corpus_stats(fb_samples, finqa_samples, fb_pdf_dir, finqa_pdf_dir, vs_dir)
    print_corpus_stats(stats)

    # ── PHASE 2: HyDE pre-generation ─────────────────────────────────────────
    hyde_cache: Dict[str, List[str]] = {}
    if needs_hyde:
        import torch
        torch.cuda.empty_cache()
        logger.info("\n>>> PHASE 2: Pre-generating HyDE hypotheticals (Qwen 7B only)")
        max_hyps  = 3 if any("multi_hyde" in n for n in all_names) else 1
        hyde_cache = pre_generate_hypotheticals(questions, max_hyps, hyde_cache_path)
    else:
        logger.info("\n>>> PHASE 2: Skipped (no HyDE variants)")

    # ── PHASE 3: Retrieval ────────────────────────────────────────────────────
    logger.info("\n>>> PHASE 3: Retrieval (global search space)")
    from sentence_transformers import SentenceTransformer, CrossEncoder

    cross_encoder    = None
    if needs_rerank:
        logger.info(f"Loading cross-encoder: {RERANKER_MODEL}")
        cross_encoder = CrossEncoder(RERANKER_MODEL, max_length=512)

    ft_cross_encoder = None
    if needs_ft_rerank:
        if FT_CHECKPOINT.exists():
            logger.info(f"Loading FT cross-encoder from {FT_CHECKPOINT}")
            ft_cross_encoder = CrossEncoder(str(FT_CHECKPOINT), max_length=512)
        else:
            logger.warning(f"FT checkpoint not found at {FT_CHECKPOINT}; skipping FT variants")
            all_names = [n for n in all_names if "ft_reranker" not in n]

    retrieval_results: Dict[str, List[Dict]] = {}
    MAX_K = max(K_VALUES)

    # Group variants by embed model to avoid reloading SentenceTransformer
    _SPARSE_KEY    = "__sparse__"
    _FT_RERANK_KEY = "__ft_rerank__"

    def _em_group(name):
        if name in ("bm25", "splade"):
            return _SPARSE_KEY
        if name in ("dense_bge_m3_ft_reranker", "multi_hyde_ft_reranker"):
            return _FT_RERANK_KEY
        return embed_model_map.get(name, EMBED_MODEL)

    em_groups: Dict[str, List[str]] = {}
    for name in all_names:
        if name in ("dense_bge_m3_ft_reranker", "multi_hyde_ft_reranker"):
            continue   # handled after base retrieval
        key = _em_group(name)
        em_groups.setdefault(key, []).append(name)

    for em_key, names in em_groups.items():
        if em_key == _SPARSE_KEY:
            embed_model = None
        else:
            logger.info(f"\n>>> Loading embed model: {em_key}")
            embed_model = SentenceTransformer(em_key, device="cuda")

        for name in names:
            pred_path = os.path.join(output_dir, "predictions", f"{name}_retrieval.json")
            if args.resume and os.path.exists(pred_path):
                with open(pred_path) as f:
                    retrieval_results[name] = json.load(f)
                logger.info(f"  [{name}] loaded from cache")
                continue

            logger.info(f"  Running: {name}")
            samples = copy.deepcopy(all_samples)
            t0 = time.time()

            for s in tqdm(samples, desc=name, leave=False):
                if name == "dense_bge_m3" or (name.startswith("dense_") and "ft_reranker" not in name):
                    chunks = retrieve_dense_ext(s, embed_model,
                                                dense_collections.get(em_key, dense_col), MAX_K)
                elif name == "bm25":
                    chunks = retrieve_bm25_ext(s, bm25_chunks, bm25_index, MAX_K)
                elif name == "splade":
                    chunks = retrieve_splade_ext(s, splade_model, splade_tokenizer,
                                                 splade_chunks, splade_postings, MAX_K)
                elif name in ("hybrid_50_50","hybrid_75_25","hybrid_25_75"):
                    alpha_map = {"hybrid_50_50": 0.50, "hybrid_75_25": 0.75, "hybrid_25_75": 0.25}
                    chunks = retrieve_hybrid_ext(s, embed_model, dense_col,
                                                 bm25_chunks, bm25_index,
                                                 alpha=alpha_map[name], k=MAX_K)
                elif name == "parent_child":
                    chunks = retrieve_parent_child_ext(s, embed_model, pc_col, parent_map, MAX_K)
                elif name == "query_expansion":
                    chunks = retrieve_query_expansion_ext(s, embed_model, dense_col, MAX_K)
                elif name == "hyde":
                    chunks = retrieve_hyde_ext(s, embed_model, dense_col,
                                               hyde_cache, hyde_n=1, k=MAX_K)
                elif name == "multi_hyde":
                    chunks = retrieve_hyde_ext(s, embed_model, dense_col,
                                               hyde_cache, hyde_n=3, k=MAX_K)
                elif name == "bge_reranker":
                    chunks = retrieve_dense_ext(s, embed_model, dense_col, CANDIDATE_K)
                    if cross_encoder:
                        chunks = apply_reranker(s, chunks, cross_encoder, k=MAX_K)
                    else:
                        chunks = chunks[:MAX_K]
                elif name == "multi_hyde_reranker":
                    chunks = retrieve_hyde_ext(s, embed_model, dense_col,
                                               hyde_cache, hyde_n=3, k=CANDIDATE_K)
                    if cross_encoder:
                        chunks = apply_reranker(s, chunks, cross_encoder, k=MAX_K)
                    else:
                        chunks = chunks[:MAX_K]
                elif name in ("oracle_doc", "oracle_page"):
                    chunks = retrieve_oracle(s, embed_model, dense_col, MAX_K, name)
                else:
                    chunks = []

                s["retrieved_chunks"] = chunks[:MAX_K]

            retrieval_results[name] = samples
            with open(pred_path, "w") as f:
                json.dump(samples, f)
            logger.info(f"  [{name}] {len(samples)} q in {time.time()-t0:.0f}s — saved {pred_path}")

        if embed_model is not None:
            del embed_model
            import torch; torch.cuda.empty_cache()

    # Free standard cross-encoder before FT reranker
    if cross_encoder is not None:
        del cross_encoder
        import torch; torch.cuda.empty_cache()

    # ── PHASE 4: FT Reranker (post-processing on dense/multi_hyde results) ────
    if ft_cross_encoder is not None:
        logger.info("\n>>> PHASE 4: FT Reranker post-processing")
        for name, base_name in [
            ("dense_bge_m3_ft_reranker",  "dense_bge_m3"),
            ("multi_hyde_ft_reranker",    "multi_hyde"),
        ]:
            if name not in all_names:
                continue
            pred_path = os.path.join(output_dir, "predictions", f"{name}_retrieval.json")
            if args.resume and os.path.exists(pred_path):
                with open(pred_path) as f:
                    retrieval_results[name] = json.load(f)
                logger.info(f"  [{name}] loaded from cache")
                continue

            base_samples = retrieval_results.get(base_name)
            if not base_samples:
                logger.warning(f"  Base retrieval '{base_name}' missing — skipping {name}")
                continue

            logger.info(f"  Reranking: {name}  (base={base_name})")
            reranked = []
            for s in tqdm(base_samples, desc=name, leave=False):
                sr = copy.deepcopy(s)
                sr["retrieved_chunks"] = apply_reranker(
                    sr, sr["retrieved_chunks"][:CANDIDATE_K],
                    ft_cross_encoder, k=MAX_K,
                )
                reranked.append(sr)

            retrieval_results[name] = reranked
            with open(pred_path, "w") as f:
                json.dump(reranked, f)
            logger.info(f"  [{name}] saved {pred_path}")

    # Free FT cross-encoder
    if ft_cross_encoder is not None:
        del ft_cross_encoder
        import torch; torch.cuda.empty_cache()

    # ── PHASE 5: Evaluation ───────────────────────────────────────────────────
    logger.info("\n>>> PHASE 5: Evaluation")

    all_metrics_by_level: Dict[str, Dict[str, Dict]] = {}

    variant_order = [n for n in all_names if n in retrieval_results]

    for level, dataset_filter in [
        ("global",       None),
        ("financebench", "financebench"),
        ("finqa",        "finqa"),
    ]:
        logger.info(f"\n  --- {level.upper()} ---")
        metrics = evaluate_subset(retrieval_results, dataset_filter, level)
        all_metrics_by_level[level] = metrics
        save_all_metrics(metrics, output_dir, level)

    # ── PHASE 6: Tables and plots ─────────────────────────────────────────────
    logger.info("\n>>> PHASE 6: Saving tables and plots")
    print_summary_table(all_metrics_by_level, variant_order)
    save_csv_summary(all_metrics_by_level, output_dir, variant_order)

    if not args.skip_plots:
        for level, metrics in all_metrics_by_level.items():
            plots_dir = os.path.join(output_dir, "plots", level)
            os.makedirs(plots_dir, exist_ok=True)
            # Flatten to {variant: overall_metrics} for existing plot functions
            flat = {n: m["overall"] for n, m in metrics.items()}
            try:
                plot_retrieval_bar_k5(flat, plots_dir)
                plot_recall_at_k_curves(flat, plots_dir)
                save_metrics_table(flat, plots_dir)
            except Exception as e:
                logger.warning(f"Plotting failed for {level}: {e}")

    elapsed = (time.time() - t_start) / 60
    logger.info(f"\nTotal wall-clock time: {elapsed:.1f} min")
    logger.info(f"Results directory: {output_dir}")

    # Final compact stats
    print_corpus_stats(stats)

    # Final summary (global only)
    global_m = all_metrics_by_level.get("global", {})
    print(f"\n{'='*90}")
    print("HEADLINE NUMBERS — GLOBAL (FinanceBench + FinQA, n="
          f"{stats.get('total_questions',0)})")
    print(f"{'Method':<44}  PR@1   PR@3   PR@5   PR@10  PR@20  MRR")
    print("-" * 90)
    for name in variant_order:
        if name not in global_m:
            continue
        r   = global_m[name]["overall"]
        lbl = VARIANT_LABELS.get(name, name)
        prs = [r.get(f"page_recall@{k}", 0) for k in K_VALUES]
        mrr = r.get("mrr", 0)
        print(f"{lbl:<44}  "
              + "  ".join(f"{v:.3f}" for v in prs)
              + f"  {mrr:.3f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
