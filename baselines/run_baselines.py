#!/usr/bin/env python
"""
FinanceBench Baseline Methods — Full Experiment Runner
======================================================
Reproduces the retrieval baseline table from the thesis and adds
generative evaluation, per-question-type and per-doc-type breakdowns,
and publication-quality plots.

Methods
-------
  dense_bge_m3          Dense BGE-M3 (baseline)
  bm25                  BM25 sparse retrieval
  splade                SPLADE sparse retrieval
  hybrid_50_50          Dense + BM25 RRF, alpha=0.50
  hybrid_75_25          Dense + BM25 RRF, alpha=0.75 (dense-heavy)
  hybrid_25_75          Dense + BM25 RRF, alpha=0.25 (sparse-heavy)
  parent_child          Parent-child chunking + dense
  query_expansion       Financial term expansion + dense
  hyde                  HyDE  (1 hypothetical)
  multi_hyde            Multi-HyDE (3 hypotheticals)
  bge_reranker          Dense → cross-encoder rerank
  multi_hyde_reranker   Multi-HyDE → cross-encoder rerank

Chunking
--------
  RecursiveCharacterTextSplitter, chunk_size=1024, chunk_overlap=30
  (matches original experiment settings for reproducibility)

Evaluation
----------
  Retrieval : DocRec@k, PageRec@k  for k ∈ {1,3,5,10,20}
  Context   : BLEU@5, ROUGE-L@5 vs gold evidence
  Generative: ROUGE-L of generated answer vs reference answer
              Numeric match for metrics-generated questions
  Breakdowns: by question_type (metrics/domain/novel)
              by question_type × doc_type (10k/10q/8k…)

Memory plan (L40S 46 GB)
-------------------------
  Phase 1  Index building        BGE-M3 (~2.3 GB) + SPLADE (~0.5 GB)
  Phase 2  HyDE pre-generation   Qwen 7B 4-bit (~4.5 GB) only → free
  Phase 3  Retrieval             BGE-M3 + CrossEncoder (~4.6 GB max)
  Phase 4  Generation            Qwen 7B 4-bit (~4.5 GB) only → free
  Phase 5  Evaluation + plots    CPU only
"""

import argparse
import copy
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# Project root so we can import existing evaluators
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("baselines")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMBED_MODEL   = "BAAI/bge-m3"
QWEN_MODEL    = "Qwen/Qwen2.5-7B-Instruct"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
SPLADE_MODEL  = "naver/splade-cocondenser-ensembledistil"

# All sizes are in TOKENS (BGE-M3 tokenizer), not characters.
CHUNK_SIZE    = 1024   # tokens
CHUNK_OVERLAP = 128    # tokens  (was 30 chars)

# Parent-child sizes (tokens)
PARENT_SIZE    = 2048
PARENT_OVERLAP = 256
CHILD_SIZE     = 512
CHILD_OVERLAP  = 64

K_VALUES = [1, 3, 5, 10, 20]
MAIN_K   = 5

CANDIDATE_K  = 20   # pool before reranking
RRF_K        = 60   # constant in RRF formula

QUESTION_TYPES = ["metrics-generated", "domain-relevant", "novel-generated"]

# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------
VARIANTS = [
    {"name": "dense_bge_m3",        "index": "dense", "hyde_n": 0, "rerank": False, "alpha": None, "parent_child": False, "qexp": False},
    {"name": "bm25",                "index": "bm25",  "hyde_n": 0, "rerank": False, "alpha": None, "parent_child": False, "qexp": False},
    {"name": "splade",              "index": "splade","hyde_n": 0, "rerank": False, "alpha": None, "parent_child": False, "qexp": False},
    {"name": "hybrid_50_50",        "index": "hybrid","hyde_n": 0, "rerank": False, "alpha": 0.50, "parent_child": False, "qexp": False},
    {"name": "hybrid_75_25",        "index": "hybrid","hyde_n": 0, "rerank": False, "alpha": 0.75, "parent_child": False, "qexp": False},
    {"name": "hybrid_25_75",        "index": "hybrid","hyde_n": 0, "rerank": False, "alpha": 0.25, "parent_child": False, "qexp": False},
    {"name": "parent_child",        "index": "parent_child","hyde_n": 0, "rerank": False, "alpha": None, "parent_child": True, "qexp": False},
    {"name": "query_expansion",     "index": "dense", "hyde_n": 0, "rerank": False, "alpha": None, "parent_child": False, "qexp": True},
    {"name": "hyde",                "index": "dense", "hyde_n": 1, "rerank": False, "alpha": None, "parent_child": False, "qexp": False},
    {"name": "multi_hyde",          "index": "dense", "hyde_n": 3, "rerank": False, "alpha": None, "parent_child": False, "qexp": False},
    {"name": "bge_reranker",        "index": "dense", "hyde_n": 0, "rerank": True,  "alpha": None, "parent_child": False, "qexp": False},
    {"name": "multi_hyde_reranker", "index": "dense", "hyde_n": 3, "rerank": True,  "alpha": None, "parent_child": False, "qexp": False},
]

VARIANT_LABELS = {
    "dense_bge_m3":        "Dense BGE-M3",
    "bm25":                "BM25",
    "splade":              "SPLADE",
    "hybrid_50_50":        "Hybrid RRF 50/50",
    "hybrid_75_25":        "Hybrid RRF 75/25 (dense-heavy)",
    "hybrid_25_75":        "Hybrid RRF 25/75 (sparse-heavy)",
    "parent_child":        "Parent-Child",
    "query_expansion":     "Query Expansion",
    "hyde":                "HyDE",
    "multi_hyde":          "Multi-HyDE",
    "bge_reranker":        "BGE-M3 + ReRanker",
    "multi_hyde_reranker": "BGE-M3 + Multi-HyDE + ReRanker",
}

# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_data(fb_path: str, doc_info_path: str) -> Tuple[List[Dict], Dict[str, Dict]]:
    """
    Load FinanceBench samples and document metadata.
    Returns (samples, doc_info_map) where doc_info_map: doc_name → {doc_type, gics_sector, company, …}.
    """
    samples = []
    with open(fb_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            gold_segs = []
            for ev in raw.get("evidence", []):
                gold_segs.append({
                    "text":      ev.get("evidence_text", ""),
                    "doc_name":  ev.get("doc_name", raw.get("doc_name", "")),
                    "page":      ev.get("evidence_page_num", -1),
                })
            samples.append({
                "financebench_id":     raw.get("financebench_id", ""),
                "question":            raw.get("question", ""),
                "reference_answer":    raw.get("answer", ""),
                "question_type":       raw.get("question_type", "unknown"),
                "doc_name":            raw.get("doc_name", ""),
                "doc_link":            raw.get("doc_link", ""),
                "gold_evidence_segments": gold_segs,
                "retrieved_chunks":    [],
                "generated_answer":    "",
            })

    doc_info: Dict[str, Dict] = {}
    if doc_info_path and os.path.exists(doc_info_path):
        with open(doc_info_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                doc_info[d["doc_name"]] = d

    type_counts = {qt: sum(1 for s in samples if s["question_type"] == qt) for qt in QUESTION_TYPES}
    logger.info(f"Loaded {len(samples)} samples — {type_counts}")
    return samples, doc_info


# ---------------------------------------------------------------------------
# 2. PDF loading & chunking utilities
# ---------------------------------------------------------------------------

def load_pdf(doc_name: str, doc_link: str, pdf_dir: str):
    """Return list of LangChain Documents (one per page)."""
    try:
        from src.ingestion.pdf_utils import load_pdf_with_fallback
        docs, _ = load_pdf_with_fallback(doc_name, doc_link, pdf_dir)
        return docs or []
    except Exception as e:
        logger.warning(f"PDF load failed for {doc_name}: {e}")
        return []


def make_recursive_splitter(chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    """Token-based recursive splitter using the BGE-M3 tokenizer."""
    from transformers import AutoTokenizer
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL, use_fast=True)
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_docs(pages, splitter) -> List[Dict]:
    """Split LangChain Document pages into chunks with metadata."""
    chunks = []
    for doc in pages:
        meta = doc.metadata or {}
        for chunk_text in splitter.split_text(doc.page_content):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            chunks.append({
                "text":     chunk_text,
                "doc_name": meta.get("doc_name", ""),
                "page":     meta.get("page", -1),
                "source":   meta.get("source", ""),
            })
    return chunks


# ---------------------------------------------------------------------------
# 3. Index building (Phase 1)
# ---------------------------------------------------------------------------

def _get_or_create_chroma(persist_dir: str, collection_name: str, embed_fn):
    import chromadb
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )


def build_dense_index(samples: List[Dict], pdf_dir: str, vs_dir: str,
                      collection_name: str = "baselines_dense_tok1024_ol128") -> "chromadb.Collection":
    """
    Build (or load) ChromaDB with RecursiveCharacterTextSplitter chunks.
    Returns the chromadb Collection.
    """
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL, device="cuda")
    collection = _get_or_create_chroma(vs_dir, collection_name, ef)

    # Check which docs are already indexed
    existing_ids = set(collection.get(include=[])["ids"])
    unique_docs = {s["doc_name"]: s["doc_link"] for s in samples}
    splitter = make_recursive_splitter()

    docs_to_add = []
    meta_to_add = []
    ids_to_add  = []

    for doc_name, doc_link in tqdm(unique_docs.items(), desc=f"Indexing {collection_name}"):
        # Check if this doc is already indexed (any chunk with this doc_name)
        marker_id = f"{doc_name}__page_0__chunk_0"
        if marker_id in existing_ids:
            continue

        pages = load_pdf(doc_name, doc_link, pdf_dir)
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

            # Flush inside chunk loop to stay under ChromaDB's max batch size
            if len(docs_to_add) >= 2000:
                collection.add(documents=docs_to_add, metadatas=meta_to_add, ids=ids_to_add)
                docs_to_add, meta_to_add, ids_to_add = [], [], []

    if docs_to_add:
        collection.add(documents=docs_to_add, metadatas=meta_to_add, ids=ids_to_add)

    logger.info(f"Dense index '{collection_name}': {collection.count()} chunks")
    return collection


def build_parent_child_index(samples: List[Dict], pdf_dir: str,
                             vs_dir: str) -> Tuple["chromadb.Collection", Dict[str, str]]:
    """
    Build a child-level ChromaDB and a mapping child_id → parent_text.
    Children (512 chars) are indexed; on retrieval we return parent text (2048 chars).
    """
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL, device="cuda")
    collection = _get_or_create_chroma(vs_dir, "baselines_parent_child", ef)

    parent_text_cache_path = os.path.join(vs_dir, "parent_child_texts.json")
    if os.path.exists(parent_text_cache_path):
        with open(parent_text_cache_path) as f:
            parent_map = json.load(f)
        if collection.count() > 0:
            logger.info(f"Parent-child index: {collection.count()} children (loaded from cache)")
            return collection, parent_map

    parent_splitter = make_recursive_splitter(PARENT_SIZE, PARENT_OVERLAP)
    child_splitter  = make_recursive_splitter(CHILD_SIZE,  CHILD_OVERLAP)
    unique_docs = {s["doc_name"]: s["doc_link"] for s in samples}
    parent_map: Dict[str, str] = {}

    docs_to_add, meta_to_add, ids_to_add = [], [], []
    existing_ids = set(collection.get(include=[])["ids"])

    for doc_name, doc_link in tqdm(unique_docs.items(), desc="Indexing parent-child"):
        if f"{doc_name}__parent_0__child_0" in existing_ids:
            continue
        pages = load_pdf(doc_name, doc_link, pdf_dir)
        if not pages:
            continue
        parent_chunks = chunk_docs(pages, parent_splitter)
        for p_idx, parent in enumerate(parent_chunks):
            parent_id_prefix = f"{doc_name}__parent_{p_idx}"
            parent_map[parent_id_prefix] = parent["text"]
            child_texts = child_splitter.split_text(parent["text"])
            for c_idx, child_text in enumerate(child_texts):
                child_text = child_text.strip()
                if not child_text:
                    continue
                cid = f"{parent_id_prefix}__child_{c_idx}"
                docs_to_add.append(child_text)
                meta_to_add.append({
                    "doc_name":   doc_name,
                    "page":       int(parent["page"]),
                    "parent_key": parent_id_prefix,
                    "chunk_idx":  c_idx,
                })
                ids_to_add.append(cid)

            # Flush inside the parent loop so a single large document never
            # accumulates more than CHROMA_MAX_BATCH items before insertion.
            if len(docs_to_add) >= 2000:
                collection.add(documents=docs_to_add, metadatas=meta_to_add, ids=ids_to_add)
                docs_to_add, meta_to_add, ids_to_add = [], [], []

    if docs_to_add:
        collection.add(documents=docs_to_add, metadatas=meta_to_add, ids=ids_to_add)

    with open(parent_text_cache_path, "w") as f:
        json.dump(parent_map, f)
    logger.info(f"Parent-child index: {collection.count()} children")
    return collection, parent_map


def build_bm25_index(samples: List[Dict], pdf_dir: str,
                     cache_dir: str) -> Tuple[List[Dict], object]:
    """
    Build (or load) a BM25Okapi index over all chunks.
    Returns (chunks_list, bm25_index).
    """
    import pickle
    from rank_bm25 import BM25Okapi

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"bm25_sz{CHUNK_SIZE}_ol{CHUNK_OVERLAP}.pkl")

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        logger.info(f"BM25 index loaded from cache ({len(data['chunks'])} chunks)")
        return data["chunks"], BM25Okapi([c["tokens"] for c in data["chunks"]])

    unique_docs = {s["doc_name"]: s["doc_link"] for s in samples}
    splitter = make_recursive_splitter()
    all_chunks = []

    for doc_name, doc_link in tqdm(unique_docs.items(), desc="Building BM25 index"):
        pages = load_pdf(doc_name, doc_link, pdf_dir)
        if not pages:
            continue
        chunks = chunk_docs(pages, splitter)
        for c in chunks:
            tokens = _finance_tokenize(c["text"])
            all_chunks.append({**c, "tokens": tokens})

    bm25 = BM25Okapi([c["tokens"] for c in all_chunks])
    with open(cache_path, "wb") as f:
        pickle.dump({"chunks": all_chunks}, f)
    logger.info(f"BM25 index built: {len(all_chunks)} chunks")
    return all_chunks, bm25


def _finance_tokenize(text: str) -> List[str]:
    """Tokenize preserving financial symbols like $, %, numbers."""
    text = re.sub(r"[$,€£]", " ", text.lower())
    text = re.sub(r"%", " percent ", text)
    tokens = re.findall(r"\b\w[\w.]*\b", text)
    return tokens or text.lower().split()


def build_splade_index(samples: List[Dict], pdf_dir: str, cache_dir: str):
    """
    Build SPLADE inverted index.
    Returns (splade_model, splade_tokenizer, chunks_list, postings_dict).
    postings_dict: term_id → (doc_ids array, weights array)
    """
    import pickle
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"splade_sz{CHUNK_SIZE}_ol{CHUNK_OVERLAP}.pkl")

    logger.info(f"Loading SPLADE model: {SPLADE_MODEL}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(SPLADE_MODEL)
    model = AutoModelForMaskedLM.from_pretrained(SPLADE_MODEL).to(device).eval()

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        logger.info(f"SPLADE index loaded from cache ({len(data['chunks'])} chunks)")
        return model, tokenizer, data["chunks"], data["postings"]

    unique_docs = {s["doc_name"]: s["doc_link"] for s in samples}
    splitter = make_recursive_splitter()
    all_chunks = []

    for doc_name, doc_link in tqdm(unique_docs.items(), desc="Chunking for SPLADE"):
        pages = load_pdf(doc_name, doc_link, pdf_dir)
        if not pages:
            continue
        all_chunks.extend(chunk_docs(pages, splitter))

    # Build inverted index
    postings: Dict[int, Tuple[List[int], List[float]]] = defaultdict(lambda: ([], []))
    logger.info(f"Building SPLADE inverted index over {len(all_chunks)} chunks…")
    batch_size = 32

    with torch.no_grad():
        for start in tqdm(range(0, len(all_chunks), batch_size), desc="SPLADE encoding"):
            batch = [c["text"][:512] for c in all_chunks[start:start + batch_size]]
            enc = tokenizer(batch, return_tensors="pt", padding=True,
                            truncation=True, max_length=512).to(device)
            out = model(**enc)
            # SPLADE sparse representation: max(0, log(1+x)) pooled over seq
            sparse = torch.log1p(torch.relu(out.logits)).max(dim=1).values
            sparse = sparse.cpu().float().numpy()
            for doc_offset, vec in enumerate(sparse):
                doc_id = start + doc_offset
                nonzero_ids = np.nonzero(vec)[0]
                for tid in nonzero_ids:
                    postings[int(tid)][0].append(doc_id)
                    postings[int(tid)][1].append(float(vec[tid]))

    # Convert lists → arrays for fast retrieval
    postings_final = {tid: (np.array(ids), np.array(weights))
                      for tid, (ids, weights) in postings.items()}

    with open(cache_path, "wb") as f:
        pickle.dump({"chunks": all_chunks, "postings": postings_final}, f)
    logger.info(f"SPLADE index built: {len(all_chunks)} chunks, {len(postings_final)} terms")
    return model, tokenizer, all_chunks, postings_final


# ---------------------------------------------------------------------------
# 4. HyDE pre-generation (Phase 2) — Qwen alone on GPU
# ---------------------------------------------------------------------------

HYDE_PROMPT = (
    "You are a financial analyst reviewing SEC 10-K and 10-Q filings. "
    "Given the following question, write a short passage (3–4 sentences) "
    "that would directly answer the question, as if excerpted from a "
    "company's annual report. Include specific financial figures and "
    "terminology as they would appear in the filing.\n\n"
    "Question: {question}\n\n"
    "Passage from annual report:"
)


def pre_generate_hypotheticals(questions: List[str], max_hypotheticals: int,
                               cache_path: str) -> Dict[str, List[str]]:
    """
    Load Qwen 7B 4-bit, generate `max_hypotheticals` passages per question,
    save to JSON cache, then free GPU memory.

    Returns: {question: [hyp1, hyp2, ...]}
    """
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        # Check if all questions are present with enough hypotheticals
        if all(len(cached.get(q, [])) >= max_hypotheticals for q in questions):
            logger.info(f"HyDE cache loaded from {cache_path} ({len(cached)} questions)")
            return cached

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    logger.info(f"Loading Qwen 7B 4-bit for HyDE pre-generation ({max_hypotheticals} per question)…")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL, quantization_config=bnb, device_map="auto", trust_remote_code=True
    ).eval()

    cache: Dict[str, List[str]] = {}
    # Load existing partial cache if available
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)

    for q in tqdm(questions, desc="HyDE pre-generation"):
        if len(cache.get(q, [])) >= max_hypotheticals:
            continue
        prompt = HYDE_PROMPT.format(question=q)
        if hasattr(tokenizer, "apply_chat_template"):
            msgs = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
        else:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        hypotheticals = []
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    num_return_sequences=max_hypotheticals,
                    pad_token_id=tokenizer.eos_token_id,
                )
            prompt_len = inputs["input_ids"].shape[-1]
            for out in outputs:
                text = tokenizer.decode(out[prompt_len:], skip_special_tokens=True).strip()
                if text:
                    hypotheticals.append(text)
        except Exception as e:
            logger.warning(f"HyDE generation failed for question: {e}")

        cache[q] = hypotheticals

    # Save incrementally
    with open(cache_path, "w") as f:
        json.dump(cache, f)

    # Free Qwen from GPU
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Qwen freed from GPU after HyDE pre-generation.")

    return cache


# ---------------------------------------------------------------------------
# 5. Retrieval functions (Phase 3)
# ---------------------------------------------------------------------------

def _embed_texts(texts: List[str], embed_model) -> np.ndarray:
    return embed_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def _make_chroma_where(doc_filter: Optional[List[str]]) -> Optional[Dict]:
    """Build a ChromaDB where-filter from a list of predicted doc_names."""
    if not doc_filter:
        return None
    if len(doc_filter) == 1:
        return {"doc_name": doc_filter[0]}
    return {"doc_name": {"$in": doc_filter}}


def _chroma_query(collection, embedding: np.ndarray, n_results: int,
                  where: Optional[Dict] = None) -> List[Dict]:
    """Query a ChromaDB collection with a single embedding vector."""
    n = min(n_results, collection.count())
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
        candidates.append({
            "text": text,
            "metadata": meta,
            "_score": float(1.0 - dist),
        })
    return candidates


def _rrf_merge(ranked_lists: List[List[str]], scores_map: Dict[str, Dict],
               k: int = RRF_K, top_n: int = MAIN_K) -> List[Dict]:
    """Reciprocal Rank Fusion over multiple ranked ID lists."""
    rrf_scores: Dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            rrf_scores[doc_id] += 1.0 / (k + rank)
    sorted_ids = sorted(rrf_scores, key=lambda x: -rrf_scores[x])[:top_n]
    return [scores_map[cid] for cid in sorted_ids if cid in scores_map]


def _chunk_id(c: Dict) -> str:
    return f"{c['metadata']['doc_name']}__page_{c['metadata']['page']}__score_{c['_score']:.4f}"


def retrieve_dense(sample: Dict, embed_model, collection, k: int = MAIN_K,
                   doc_filter: Optional[List[str]] = None) -> List[Dict]:
    q_emb = _embed_texts([sample["question"]], embed_model)[0]
    candidates = _chroma_query(collection, q_emb, k,
                               where=_make_chroma_where(doc_filter))
    for i, c in enumerate(candidates):
        c["rank"] = i + 1
    return candidates


def retrieve_dense_query_expansion(sample: Dict, embed_model, collection,
                                   k: int = MAIN_K,
                                   doc_filter: Optional[List[str]] = None) -> List[Dict]:
    """Expand financial terms before dense search."""
    try:
        from src.experiments.query_expansion import process_query_for_experiment
        expanded_q, _ = process_query_for_experiment(sample["question"])
    except Exception:
        expanded_q = sample["question"]
    q_emb = _embed_texts([expanded_q], embed_model)[0]
    candidates = _chroma_query(collection, q_emb, k,
                               where=_make_chroma_where(doc_filter))
    for i, c in enumerate(candidates):
        c["rank"] = i + 1
    return candidates


def retrieve_bm25(sample: Dict, bm25_chunks: List[Dict], bm25_index,
                  k: int = MAIN_K,
                  doc_filter: Optional[List[str]] = None) -> List[Dict]:
    tokens = _finance_tokenize(sample["question"])
    scores = bm25_index.get_scores(tokens)
    if doc_filter:
        allowed = set(doc_filter)
        for i, c in enumerate(bm25_chunks):
            if c["doc_name"] not in allowed:
                scores[i] = 0.0
    top_idxs = np.argsort(scores)[::-1][:k]
    results = []
    for rank, idx in enumerate(top_idxs, start=1):
        if scores[int(idx)] <= 0.0:
            break
        c = bm25_chunks[int(idx)]
        results.append({
            "text": c["text"],
            "metadata": {"doc_name": c["doc_name"], "page": int(c.get("page", -1))},
            "_score": float(scores[idx]),
            "rank": rank,
        })
    return results


def retrieve_splade(sample: Dict, splade_model, splade_tokenizer,
                    splade_chunks: List[Dict], postings: Dict,
                    k: int = MAIN_K,
                    doc_filter: Optional[List[str]] = None) -> List[Dict]:
    """Score all docs against the query's SPLADE vector."""
    import torch
    device = next(splade_model.parameters()).device
    enc = splade_tokenizer(sample["question"], return_tensors="pt",
                           truncation=True, max_length=512).to(device)
    with torch.no_grad():
        out = splade_model(**enc)
        q_vec = torch.log1p(torch.relu(out.logits)).max(dim=1).values
        q_vec = q_vec.cpu().float().numpy()[0]

    allowed = set(doc_filter) if doc_filter else None
    nonzero_terms = np.nonzero(q_vec)[0]
    doc_scores: Dict[int, float] = defaultdict(float)
    for tid in nonzero_terms:
        if tid not in postings:
            continue
        doc_ids, weights = postings[tid]
        for doc_id, w in zip(doc_ids, weights):
            if allowed and splade_chunks[int(doc_id)]["doc_name"] not in allowed:
                continue
            doc_scores[int(doc_id)] += float(q_vec[tid]) * float(w)

    top_idxs = sorted(doc_scores, key=lambda x: -doc_scores[x])[:k]
    results = []
    for rank, idx in enumerate(top_idxs, start=1):
        c = splade_chunks[idx]
        results.append({
            "text": c["text"],
            "metadata": {"doc_name": c["doc_name"], "page": int(c.get("page", -1))},
            "_score": float(doc_scores[idx]),
            "rank": rank,
        })
    return results


def retrieve_hybrid(sample: Dict, embed_model, collection,
                    bm25_chunks: List[Dict], bm25_index,
                    dense_weight: float, sparse_weight: float,
                    k: int = MAIN_K, candidate_k: int = CANDIDATE_K,
                    doc_filter: Optional[List[str]] = None) -> List[Dict]:
    """Dense + BM25 RRF fusion."""
    chroma_where = _make_chroma_where(doc_filter)
    allowed = set(doc_filter) if doc_filter else None

    # Dense retrieval
    q_emb = _embed_texts([sample["question"]], embed_model)[0]
    dense_cands = _chroma_query(collection, q_emb, candidate_k, where=chroma_where)
    dense_ids = [f"{c['metadata']['doc_name']}|{c['metadata']['page']}|{c['text'][:30]}" for c in dense_cands]
    id_to_chunk = {cid: c for cid, c in zip(dense_ids, dense_cands)}

    # BM25 retrieval
    tokens = _finance_tokenize(sample["question"])
    bm25_scores = bm25_index.get_scores(tokens)
    if allowed:
        for i, c in enumerate(bm25_chunks):
            if c["doc_name"] not in allowed:
                bm25_scores[i] = 0.0
    sparse_top = np.argsort(bm25_scores)[::-1][:candidate_k]
    sparse_cands = []
    sparse_ids = []
    for idx in sparse_top:
        if bm25_scores[int(idx)] <= 0.0:
            break
        c = bm25_chunks[int(idx)]
        cid = f"{c['doc_name']}|{c.get('page', -1)}|{c['text'][:30]}"
        sparse_ids.append(cid)
        sparse_cands.append({
            "text": c["text"],
            "metadata": {"doc_name": c["doc_name"], "page": int(c.get("page", -1))},
            "_score": float(bm25_scores[idx]),
        })
        if cid not in id_to_chunk:
            id_to_chunk[cid] = sparse_cands[-1]

    # Weighted RRF
    rrf_scores: Dict[str, float] = defaultdict(float)
    for rank, cid in enumerate(dense_ids, start=1):
        rrf_scores[cid] += dense_weight / (RRF_K + rank)
    for rank, cid in enumerate(sparse_ids, start=1):
        rrf_scores[cid] += sparse_weight / (RRF_K + rank)

    sorted_ids = sorted(rrf_scores, key=lambda x: -rrf_scores[x])[:k]
    results = []
    for rank, cid in enumerate(sorted_ids, start=1):
        c = dict(id_to_chunk[cid])
        c["rank"] = rank
        c["_score"] = float(rrf_scores[cid])
        results.append(c)
    return results


def retrieve_parent_child(sample: Dict, embed_model, child_collection,
                          parent_map: Dict[str, str], k: int = MAIN_K,
                          doc_filter: Optional[List[str]] = None) -> List[Dict]:
    """Retrieve child chunks, then return their parent text."""
    q_emb = _embed_texts([sample["question"]], embed_model)[0]
    child_cands = _chroma_query(child_collection, q_emb, k * 3,
                                where=_make_chroma_where(doc_filter))

    seen_parents: set = set()
    results = []
    for c in child_cands:
        parent_key = c.get("metadata", {}).get("parent_key", "")
        if not parent_key or parent_key in seen_parents:
            continue
        seen_parents.add(parent_key)
        parent_text = parent_map.get(parent_key, c["text"])
        results.append({
            "text":     parent_text,
            "metadata": c["metadata"],
            "_score":   c["_score"],
            "rank":     len(results) + 1,
        })
        if len(results) >= k:
            break
    return results


def retrieve_hyde(sample: Dict, embed_model, collection,
                  hypotheticals: List[str], k: int = MAIN_K,
                  candidate_k: int = CANDIDATE_K,
                  doc_filter: Optional[List[str]] = None) -> List[Dict]:
    """Embed hypotheticals + original query, RRF-merge, return top-k."""
    chroma_where = _make_chroma_where(doc_filter)
    query_texts = [sample["question"]] + (hypotheticals or [])
    all_lists: List[List[str]] = []
    id_to_chunk: Dict[str, Dict] = {}

    for text in query_texts:
        emb = _embed_texts([text], embed_model)[0]
        cands = _chroma_query(collection, emb, candidate_k, where=chroma_where)
        ranked_ids = []
        for c in cands:
            cid = f"{c['metadata']['doc_name']}|{c['metadata']['page']}|{c['text'][:30]}"
            ranked_ids.append(cid)
            if cid not in id_to_chunk:
                id_to_chunk[cid] = c
        all_lists.append(ranked_ids)

    if not all_lists:
        return []

    rrf_scores: Dict[str, float] = defaultdict(float)
    for ranked in all_lists:
        for rank, cid in enumerate(ranked, start=1):
            rrf_scores[cid] += 1.0 / (RRF_K + rank)

    sorted_ids = sorted(rrf_scores, key=lambda x: -rrf_scores[x])[:k]
    return [
        {**id_to_chunk[cid], "rank": i + 1, "_score": float(rrf_scores[cid])}
        for i, cid in enumerate(sorted_ids)
        if cid in id_to_chunk
    ]


def apply_reranker(sample: Dict, candidates: List[Dict],
                   cross_encoder, k: int = MAIN_K) -> List[Dict]:
    """Re-score candidates with a cross-encoder; return top-k."""
    if not candidates or cross_encoder is None:
        return candidates[:k]
    pairs = [(sample["question"], c["text"][:1024]) for c in candidates]
    try:
        scores = cross_encoder.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        results = []
        for rank, (score, c) in enumerate(ranked[:k], start=1):
            c = dict(c)
            c["rank"] = rank
            c["_rerank_score"] = float(score)
            results.append(c)
        return results
    except Exception as e:
        logger.warning(f"Reranking failed: {e}")
        return candidates[:k]


# ---------------------------------------------------------------------------
# 6. Generation (Phase 4) — Qwen alone on GPU
# ---------------------------------------------------------------------------

GEN_PROMPT = (
    "You are a financial analyst answering questions based on SEC filings. "
    "Use ONLY the provided context. If the context does not contain the answer, "
    "say 'I cannot determine this from the provided information.' "
    "Be concise and precise, especially for numerical answers.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)


def generate_answers_for_variant(samples: List[Dict]) -> List[Dict]:
    """
    Load Qwen 7B 4-bit, generate answers for all samples in-place, then free.
    Reads `retrieved_chunks` from each sample; writes `generated_answer`.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    logger.info("Loading Qwen 7B 4-bit for generation…")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL, quantization_config=bnb, device_map="auto", trust_remote_code=True
    ).eval()

    for sample in tqdm(samples, desc="Generating answers"):
        context = "\n\n".join(c["text"] for c in sample.get("retrieved_chunks", [])[:MAIN_K])[:4000]
        prompt = GEN_PROMPT.format(context=context, question=sample["question"])
        if hasattr(tokenizer, "apply_chat_template"):
            msgs = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(formatted, return_tensors="pt", truncation=True,
                               max_length=2048).to(model.device)
        else:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                               max_length=2048).to(model.device)
        try:
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=200,
                    temperature=0.1, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            plen = inputs["input_ids"].shape[-1]
            sample["generated_answer"] = tokenizer.decode(
                out[0][plen:], skip_special_tokens=True
            ).strip()
        except Exception as e:
            logger.warning(f"Generation failed: {e}")
            sample["generated_answer"] = ""

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Qwen freed after generation.")
    return samples


# ---------------------------------------------------------------------------
# 7. Evaluation (Phase 5)
# ---------------------------------------------------------------------------

def compute_retrieval_metrics(samples: List[Dict], k_values: List[int] = K_VALUES) -> Dict:
    """Delegate to the project's shared RetrievalEvaluator."""
    from src.evaluation.retrieval_evaluator import RetrievalEvaluator
    ev = RetrievalEvaluator()
    return ev.compute_metrics(samples, k_values=k_values)


def compute_generative_metrics(samples: List[Dict]) -> Dict:
    """
    Compute ROUGE-L of generated answer vs reference answer.
    Also compute numeric match for metrics-generated questions.
    Returns aggregate dict with per-sample breakdown.
    """
    from rouge_score import rouge_scorer as rs_module
    scorer = rs_module.RougeScorer(["rougeL"], use_stemmer=True)

    rougeL_scores = []
    numeric_matches = []

    for s in samples:
        gen = s.get("generated_answer", "")
        ref = s.get("reference_answer", "")
        if gen and ref:
            score = scorer.score(ref, gen)["rougeL"].fmeasure
        else:
            score = 0.0
        rougeL_scores.append(score)

        if s.get("question_type") == "metrics-generated":
            numeric_matches.append(_numeric_match(gen, ref))

    return {
        "answer_rougeL":  float(np.mean(rougeL_scores)) if rougeL_scores else 0.0,
        "numeric_match":  float(np.mean(numeric_matches)) if numeric_matches else 0.0,
        "n_samples":      len(samples),
        "n_metrics_qs":   len(numeric_matches),
    }


def _numeric_match(pred: str, ref: str, rtol: float = 0.03) -> float:
    """Return 1 if the main numeric value in pred matches ref within rtol, else 0."""
    def extract_main_number(text: str) -> Optional[float]:
        text = re.sub(r"[$,€£%]", "", text)
        matches = re.findall(r"-?\d[\d,]*\.?\d*", text)
        if not matches:
            return None
        try:
            return float(matches[0].replace(",", ""))
        except ValueError:
            return None

    pn, rn = extract_main_number(pred), extract_main_number(ref)
    if pn is None or rn is None:
        return 0.0
    if rn == 0:
        return 1.0 if pn == 0 else 0.0
    return 1.0 if abs(pn - rn) / abs(rn) <= rtol else 0.0


def aggregate_by_group(samples: List[Dict], key_fn, k_values: List[int] = K_VALUES) -> Dict:
    """
    Aggregate metrics by an arbitrary grouping key.
    Returns {group_key: {retrieval_metrics + generative_metrics}}.
    """
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for s in samples:
        groups[key_fn(s)].append(s)

    results = {}
    for group, group_samples in groups.items():
        ret = compute_retrieval_metrics(group_samples, k_values)
        gen = compute_generative_metrics(group_samples)
        results[group] = {**ret, **gen}
    return results


# ---------------------------------------------------------------------------
# 8. Visualization (Phase 6)
# ---------------------------------------------------------------------------

def _save_fig(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    import matplotlib.pyplot as plt
    plt.close(fig)
    logger.info(f"Saved: {path}")


def plot_retrieval_bar_k5(all_results: Dict[str, Dict], plots_dir: str) -> None:
    """4-metric grouped bar chart at k=5 for all methods."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = [v for v in VARIANTS if v["name"] in all_results]
    names   = [VARIANT_LABELS[v["name"]] for v in methods]
    keys    = [v["name"] for v in methods]
    metrics_k = {f"doc_recall@{MAIN_K}": "DocRec@5", f"page_recall@{MAIN_K}": "PageRec@5",
                 f"context_bleu@{MAIN_K}": "BLEU@5", f"context_rougeL@{MAIN_K}": "ROUGE-L@5"}

    x = np.arange(len(keys))
    width = 0.20
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

    fig, ax = plt.subplots(figsize=(max(12, len(keys) * 1.2), 5))
    for i, (metric_key, label) in enumerate(metrics_k.items()):
        vals = [all_results[k].get(metric_key, 0) for k in keys]
        ax.bar(x + (i - 1.5) * width, vals, width, label=label, color=colors[i], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title(f"Retrieval Metrics at k={MAIN_K}")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, os.path.join(plots_dir, "retrieval_bar_k5.pdf"))


def plot_recall_at_k_curves(all_results: Dict[str, Dict], plots_dir: str) -> None:
    """PageRec@k line chart for all methods."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric_prefix, ylabel in [
        (axes[0], "page_recall",  "PageRec@k"),
        (axes[1], "doc_recall",   "DocRec@k"),
    ]:
        colors = cm.tab20(np.linspace(0, 1, len(VARIANTS)))
        for var_cfg, color in zip(VARIANTS, colors):
            name = var_cfg["name"]
            if name not in all_results:
                continue
            vals = [all_results[name].get(f"{metric_prefix}@{k}", 0) for k in K_VALUES]
            ax.plot(K_VALUES, vals, marker="o", label=VARIANT_LABELS[name], color=color, linewidth=1.5)
        ax.set_xlabel("k")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " curves")
        ax.set_xticks(K_VALUES)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    _save_fig(fig, os.path.join(plots_dir, "recall_at_k_curves.pdf"))


def plot_heatmap_question_type(by_type_results: Dict[str, Dict[str, Dict]],
                               plots_dir: str) -> None:
    """Heatmap: methods × question types, coloured by PageRec@5."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    method_names = [v["name"] for v in VARIANTS if v["name"] in by_type_results]
    types = QUESTION_TYPES

    fig, ax = plt.subplots(figsize=(8, max(4, len(method_names) * 0.55)))
    matrix = np.zeros((len(method_names), len(types)))
    for i, name in enumerate(method_names):
        for j, qt in enumerate(types):
            matrix[i, j] = by_type_results[name].get(qt, {}).get(f"page_recall@{MAIN_K}", 0)

    im = ax.imshow(matrix, aspect="auto", cmap="YlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels([t.replace("-", "\n") for t in types], fontsize=9)
    ax.set_yticks(range(len(method_names)))
    ax.set_yticklabels([VARIANT_LABELS[n] for n in method_names], fontsize=8)
    for i in range(len(method_names)):
        for j in range(len(types)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if matrix[i, j] < 0.6 else "white")
    plt.colorbar(im, ax=ax, label="PageRec@5")
    ax.set_title(f"PageRec@{MAIN_K} by Question Type")
    fig.tight_layout()
    _save_fig(fig, os.path.join(plots_dir, "heatmap_by_question_type.pdf"))


def plot_heatmap_doc_type(by_doctype_results: Dict[str, Dict[str, Dict]],
                          plots_dir: str) -> None:
    """Heatmap: methods × doc types, coloured by PageRec@5."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    method_names = [v["name"] for v in VARIANTS if v["name"] in by_doctype_results]
    if not method_names:
        return
    doc_types = sorted({dt for m in by_doctype_results.values() for dt in m})
    if not doc_types:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(doc_types) * 1.5), max(4, len(method_names) * 0.55)))
    matrix = np.zeros((len(method_names), len(doc_types)))
    for i, name in enumerate(method_names):
        for j, dt in enumerate(doc_types):
            matrix[i, j] = by_doctype_results[name].get(dt, {}).get(f"page_recall@{MAIN_K}", 0)

    im = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(doc_types)))
    ax.set_xticklabels([dt.upper() for dt in doc_types], fontsize=9)
    ax.set_yticks(range(len(method_names)))
    ax.set_yticklabels([VARIANT_LABELS[n] for n in method_names], fontsize=8)
    for i in range(len(method_names)):
        for j in range(len(doc_types)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if matrix[i, j] < 0.6 else "white")
    plt.colorbar(im, ax=ax, label="PageRec@5")
    ax.set_title(f"PageRec@{MAIN_K} by Document Type")
    fig.tight_layout()
    _save_fig(fig, os.path.join(plots_dir, "heatmap_by_doc_type.pdf"))


def plot_radar_chart(all_results: Dict[str, Dict], plots_dir: str) -> None:
    """Radar/spider chart comparing top-5 methods across 6 metrics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    radar_metrics = [
        (f"doc_recall@{MAIN_K}",     "DocRec@5"),
        (f"page_recall@{MAIN_K}",    "PageRec@5"),
        (f"context_bleu@{MAIN_K}",   "BLEU@5"),
        (f"context_rougeL@{MAIN_K}", "ROUGE-L@5"),
        ("answer_rougeL",            "AnswerROUGE"),
        ("numeric_match",            "NumericMatch"),
    ]
    metric_keys  = [m[0] for m in radar_metrics]
    metric_labels = [m[1] for m in radar_metrics]
    N = len(metric_keys)

    # Pick best-performing methods by PageRec@5
    ranked = sorted(
        [n for n in all_results if "answer_rougeL" in all_results[n]],
        key=lambda n: -all_results[n].get(f"page_recall@{MAIN_K}", 0)
    )[:6]

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    colors = plt.cm.Set2(np.linspace(0, 1, len(ranked)))

    for name, color in zip(ranked, colors):
        vals = [all_results[name].get(k, 0) for k in metric_keys]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2, color=color, label=VARIANT_LABELS[name])
        ax.fill(angles, vals, alpha=0.1, color=color)

    ax.set_thetagrids(np.degrees(angles[:-1]), metric_labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
    ax.set_title("Method Comparison (Top Methods)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)
    fig.tight_layout()
    _save_fig(fig, os.path.join(plots_dir, "radar_chart.pdf"))


def plot_generative_metrics(all_results: Dict[str, Dict], plots_dir: str) -> None:
    """Bar chart: answer ROUGE-L and numeric match per method."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = [v["name"] for v in VARIANTS if v["name"] in all_results
               and "answer_rougeL" in all_results[v["name"]]]
    if not methods:
        return
    names = [VARIANT_LABELS[m] for m in methods]
    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(methods) * 1.2), 5))
    rougeL_vals  = [all_results[m].get("answer_rougeL",  0) for m in methods]
    numeric_vals = [all_results[m].get("numeric_match",  0) for m in methods]
    ax.bar(x - width / 2, rougeL_vals, width, label="Answer ROUGE-L", color="#3F51B5", alpha=0.85)
    ax.bar(x + width / 2, numeric_vals, width, label="Numeric Match",  color="#009688", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title("Generative Metrics by Method")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, os.path.join(plots_dir, "generative_metrics.pdf"))


def plot_question_type_breakdown(by_type_results: Dict[str, Dict[str, Dict]],
                                 plots_dir: str) -> None:
    """
    Grouped bar chart: for each question type, show PageRec@5 across methods.
    Good for a thesis figure comparing method strengths per question category.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    method_names = [v["name"] for v in VARIANTS if v["name"] in by_type_results]
    fig, axes = plt.subplots(1, len(QUESTION_TYPES), figsize=(16, 5), sharey=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(method_names)))

    for ax, qt in zip(axes, QUESTION_TYPES):
        vals = [by_type_results[n].get(qt, {}).get(f"page_recall@{MAIN_K}", 0)
                for n in method_names]
        bars = ax.barh([VARIANT_LABELS[n] for n in method_names], vals,
                       color=colors, alpha=0.85)
        ax.set_title(qt.replace("-", " ").title(), fontsize=9)
        ax.set_xlim(0, 1)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="x", alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(min(val + 0.01, 0.95), bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", fontsize=6.5)

    axes[0].set_xlabel("PageRec@5")
    fig.suptitle(f"PageRec@{MAIN_K} Breakdown by Question Type", fontsize=11)
    fig.tight_layout()
    _save_fig(fig, os.path.join(plots_dir, "question_type_breakdown.pdf"))


def plot_hybrid_alpha_comparison(all_results: Dict[str, Dict], plots_dir: str) -> None:
    """Side-by-side comparison of hybrid alpha variants."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hybrid_variants = ["hybrid_25_75", "hybrid_50_50", "hybrid_75_25"]
    alpha_labels    = ["25/75 (sparse-heavy)", "50/50", "75/25 (dense-heavy)"]
    metrics = {
        f"doc_recall@{MAIN_K}": "DocRec@5",
        f"page_recall@{MAIN_K}": "PageRec@5",
        f"context_bleu@{MAIN_K}": "BLEU@5",
        f"context_rougeL@{MAIN_K}": "ROUGE-L@5",
    }
    x = np.arange(len(hybrid_variants))
    width = 0.2
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (mkey, mlabel) in enumerate(metrics.items()):
        vals = [all_results.get(v, {}).get(mkey, 0) for v in hybrid_variants]
        ax.bar(x + (i - 1.5) * width, vals, width, label=mlabel, color=colors[i], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(alpha_labels)
    ax.set_ylabel("Score")
    ax.set_title("Hybrid RRF Alpha Sweep")
    ax.legend(loc="upper left")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, os.path.join(plots_dir, "hybrid_alpha_sweep.pdf"))


# ---------------------------------------------------------------------------
# 9. CSV / JSON output helpers
# ---------------------------------------------------------------------------

def save_metrics_table(all_results: Dict[str, Dict], output_dir: str) -> None:
    """Save aggregated results to CSV and LaTeX."""
    import csv

    headers = (
        ["Method"]
        + [f"DocRec@{k}" for k in K_VALUES]
        + [f"PageRec@{k}" for k in K_VALUES]
        + [f"BLEU@{MAIN_K}", f"ROUGE-L@{MAIN_K}"]
        + ["AnswerROUGE-L", "NumericMatch"]
    )

    csv_path = os.path.join(output_dir, "metrics", "baseline_table.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for var in VARIANTS:
            name = var["name"]
            if name not in all_results:
                continue
            r = all_results[name]
            row = [VARIANT_LABELS[name]]
            row += [f"{r.get(f'doc_recall@{k}', float('nan')):.3f}"  for k in K_VALUES]
            row += [f"{r.get(f'page_recall@{k}', float('nan')):.3f}" for k in K_VALUES]
            row += [
                f"{r.get(f'context_bleu@{MAIN_K}', float('nan')):.3f}",
                f"{r.get(f'context_rougeL@{MAIN_K}', float('nan')):.3f}",
                f"{r.get('answer_rougeL', float('nan')):.3f}",
                f"{r.get('numeric_match', float('nan')):.3f}",
            ]
            w.writerow(row)
    logger.info(f"Metrics table saved: {csv_path}")

    # LaTeX snippet for k=5
    tex_path = os.path.join(output_dir, "metrics", "baseline_table_k5.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{tabular}{lcccccc}\n\\toprule\n")
        f.write("\\textbf{Method} & \\textbf{DocRec@5} & \\textbf{PageRec@5} "
                "& \\textbf{BLEU@5} & \\textbf{ROUGE-L@5} "
                "& \\textbf{AnsROUGE-L} & \\textbf{NumMatch} \\\\\n\\midrule\n")
        for var in VARIANTS:
            name = var["name"]
            if name not in all_results:
                continue
            r = all_results[name]
            f.write(
                f"{VARIANT_LABELS[name]} & "
                f"{r.get(f'doc_recall@5', 0):.3f} & "
                f"{r.get(f'page_recall@5', 0):.3f} & "
                f"{r.get(f'context_bleu@5', 0):.3f} & "
                f"{r.get(f'context_rougeL@5', 0):.3f} & "
                f"{r.get('answer_rougeL', 0):.3f} & "
                f"{r.get('numeric_match', 0):.3f} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n")
    logger.info(f"LaTeX table saved: {tex_path}")


def save_by_type_table(by_type_all: Dict[str, Dict[str, Dict]], output_dir: str,
                       label: str = "question_type") -> None:
    import csv
    k = MAIN_K
    path = os.path.join(output_dir, "metrics", f"by_{label}.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Method", label.title(), "N",
                    f"DocRec@{k}", f"PageRec@{k}", f"BLEU@{k}", f"ROUGE-L@{k}",
                    "AnsROUGE-L", "NumericMatch"])
        for var in VARIANTS:
            name = var["name"]
            if name not in by_type_all:
                continue
            for group, metrics in by_type_all[name].items():
                w.writerow([
                    VARIANT_LABELS[name], group, metrics.get("n_samples", 0),
                    f"{metrics.get(f'doc_recall@{k}', float('nan')):.3f}",
                    f"{metrics.get(f'page_recall@{k}', float('nan')):.3f}",
                    f"{metrics.get(f'context_bleu@{k}', float('nan')):.3f}",
                    f"{metrics.get(f'context_rougeL@{k}', float('nan')):.3f}",
                    f"{metrics.get('answer_rougeL', float('nan')):.3f}",
                    f"{metrics.get('numeric_match', float('nan')):.3f}",
                ])
    logger.info(f"By-{label} table saved: {path}")


# ---------------------------------------------------------------------------
# 10. Main orchestration
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="FinanceBench Baseline Experiments")
    p.add_argument("--data-path",     default="data/financebench_open_source.jsonl")
    p.add_argument("--doc-info-path", default="data/financebench_document_information.jsonl")
    p.add_argument("--pdf-dir",       default="pdfs")
    p.add_argument("--vs-dir",        default="vector_stores/baselines")
    p.add_argument("--output-dir",    default="baselines/results")
    p.add_argument("--hyde-cache",    default="baselines/hyde_cache.json")
    p.add_argument("--skip-generation", action="store_true",
                   help="Skip answer generation (retrieval metrics only)")
    p.add_argument("--skip-plots",    action="store_true")
    p.add_argument("--variants",      nargs="*", default=None,
                   help="Run only these variant names (default: all)")
    p.add_argument("--resume",        action="store_true",
                   help="Skip variants whose results JSON already exists")
    p.add_argument("--ner-filter",    action="store_true",
                   help="Add NER-filtered variants alongside each baseline "
                        "(uses entity recognition to predict the target document)")
    p.add_argument("--ner-only",      action="store_true",
                   help="Run only the NER-filtered variants (implies --ner-filter)")
    p.add_argument("--ner-top-k",     type=int, default=3,
                   help="Number of candidate docs returned by NER filter (default 3)")
    return p.parse_args()


def main():
    args = parse_args()
    t_total = time.time()

    # Resolve paths relative to project root
    def abspath(p):
        return str(PROJECT_ROOT / p) if not os.path.isabs(p) else p

    data_path     = abspath(args.data_path)
    doc_info_path = abspath(args.doc_info_path)
    pdf_dir       = abspath(args.pdf_dir)
    vs_dir        = abspath(args.vs_dir)
    output_dir    = abspath(args.output_dir)
    hyde_cache    = abspath(args.hyde_cache)

    for d in [vs_dir, output_dir,
              os.path.join(output_dir, "metrics"),
              os.path.join(output_dir, "plots"),
              os.path.join(output_dir, "predictions")]:
        os.makedirs(d, exist_ok=True)

    use_ner = args.ner_filter or args.ner_only

    base_variants = (
        [v for v in VARIANTS if v["name"] in args.variants]
        if args.variants else VARIANTS
    )

    # Build the full list of (variant_dict, ner_flag) pairs to run
    run_plan: List[tuple] = []       # (variant_dict, use_ner_for_this_run)
    if not args.ner_only:
        for v in base_variants:
            run_plan.append((v, False))
    if use_ner:
        for v in base_variants:
            # Build an NER-variant with suffix _ner
            ner_v = dict(v)
            ner_v["name"] = f"{v['name']}_ner"
            run_plan.append((ner_v, True))

    # All unique variant dicts (for downstream references)
    active_variants = [v for v, _ in run_plan]

    # Extend VARIANT_LABELS with NER variants
    for v, is_ner in run_plan:
        if is_ner and v["name"] not in VARIANT_LABELS:
            base_label = VARIANT_LABELS.get(v["name"].replace("_ner", ""), v["name"])
            VARIANT_LABELS[v["name"]] = f"{base_label} + NER Filter"

    logger.info("=" * 70)
    logger.info("FinanceBench Baseline Experiments")
    logger.info(f"  Variants        : {[v['name'] for v in active_variants]}")
    logger.info(f"  NER doc filter  : {'enabled' if use_ner else 'disabled'}")
    logger.info(f"  Chunking        : RecursiveToken size={CHUNK_SIZE}tok overlap={CHUNK_OVERLAP}tok (BGE-M3 tokenizer)")
    logger.info(f"  Embed model     : {EMBED_MODEL}")
    logger.info(f"  Generator       : {QWEN_MODEL} (4-bit, {'disabled' if args.skip_generation else 'enabled'})")
    logger.info(f"  Output          : {output_dir}")
    logger.info("=" * 70)

    # ----------------------------------------------------------------
    # Load data
    # ----------------------------------------------------------------
    samples_raw, doc_info = load_data(data_path, doc_info_path)
    questions = [s["question"] for s in samples_raw]

    # ── Doc filter (content-based BM25 over first-page text, no oracle) ─────
    ner_filter = None
    if use_ner:
        import sys as _sys
        _sys.path.insert(0, str(PROJECT_ROOT))
        from ner_doc_filter import DocContentFilter
        from src.ingestion.pdf_utils import load_pdf_with_fallback

        logger.info("Building first-page content index for doc filter…")
        # Load first 3 pages of each unique document for the BM25 header index
        unique_docs = {s["doc_name"]: s["doc_link"] for s in samples_raw}
        doc_pages_header: Dict[str, List[Dict]] = {}
        for doc_name, doc_link in unique_docs.items():
            try:
                pages, _ = load_pdf_with_fallback(doc_name, doc_link, pdf_dir)
                doc_pages_header[doc_name] = [
                    {"text": p.page_content}
                    for p in (pages or [])[:3]   # only first 3 pages needed
                ]
            except Exception as e:
                logger.debug(f"Could not load {doc_name} for header index: {e}")
                doc_pages_header[doc_name] = []

        ner_filter = DocContentFilter(
            doc_pages=doc_pages_header,
            doc_info=doc_info,
            n_header_pages=3,
            year_window=1,
        )
        logger.info(f"Doc filter ready (top_k={args.ner_top_k}, BM25 over first-page content)")

    needs_dense  = any(v["index"] in ("dense", "hybrid", "parent_child") for v in active_variants)
    needs_bm25   = any(v["index"] in ("bm25", "hybrid") for v in active_variants)
    needs_splade = any(v["index"] == "splade" for v in active_variants)
    needs_pc     = any(v["parent_child"] for v in active_variants)
    needs_rerank = any(v["rerank"] for v in active_variants)
    needs_hyde   = any(v["hyde_n"] > 0 for v in active_variants)

    # ----------------------------------------------------------------
    # PHASE 1: Build indexes
    # ----------------------------------------------------------------
    logger.info("\n>>> PHASE 1: Building indexes")

    dense_collection = None
    if needs_dense:
        dense_collection = build_dense_index(samples_raw, pdf_dir, vs_dir)

    bm25_chunks, bm25_index = None, None
    if needs_bm25:
        bm25_chunks, bm25_index = build_bm25_index(
            samples_raw, pdf_dir, os.path.join(vs_dir, "bm25_cache"))

    splade_model = splade_tokenizer = splade_chunks = splade_postings = None
    if needs_splade:
        splade_model, splade_tokenizer, splade_chunks, splade_postings = build_splade_index(
            samples_raw, pdf_dir, os.path.join(vs_dir, "splade_cache"))

    pc_collection, parent_map = None, {}
    if needs_pc:
        pc_collection, parent_map = build_parent_child_index(samples_raw, pdf_dir, vs_dir)

    # ----------------------------------------------------------------
    # PHASE 2: HyDE pre-generation (Qwen alone on GPU)
    # ----------------------------------------------------------------
    hyde_cache_data: Dict[str, List[str]] = {}
    if needs_hyde:
        # Temporarily free embedding model from GPU if it's loaded in-process
        # (ChromaDB's SentenceTransformerEF keeps a handle; it will reload on demand)
        import torch
        torch.cuda.empty_cache()
        logger.info("\n>>> PHASE 2: Pre-generating HyDE hypotheticals (Qwen only on GPU)")
        max_hyps = max(v["hyde_n"] for v in active_variants if v["hyde_n"] > 0)
        hyde_cache_data = pre_generate_hypotheticals(questions, max_hyps, hyde_cache)
    else:
        logger.info("\n>>> PHASE 2: Skipped (no HyDE variants)")

    # ----------------------------------------------------------------
    # PHASE 3: Retrieval
    # ----------------------------------------------------------------
    logger.info("\n>>> PHASE 3: Retrieval")
    from sentence_transformers import SentenceTransformer, CrossEncoder

    embed_model = SentenceTransformer(EMBED_MODEL, device="cuda")

    cross_encoder = None
    if needs_rerank:
        logger.info(f"Loading cross-encoder: {RERANKER_MODEL}")
        cross_encoder = CrossEncoder(RERANKER_MODEL, max_length=512)

    # dict: variant_name → List[sample_dict with retrieved_chunks filled in]
    retrieval_results: Dict[str, List[Dict]] = {}

    MAX_K = max(K_VALUES)  # retrieve enough chunks for all @k values

    for var, is_ner_run in tqdm(run_plan, desc="Variants"):
        name = var["name"]

        pred_path = os.path.join(output_dir, "predictions", f"{name}_retrieval.json")
        if args.resume and os.path.exists(pred_path):
            with open(pred_path) as f:
                retrieval_results[name] = json.load(f)
            logger.info(f"  [{name}] Loaded from cache")
            continue

        logger.info(f"\n  Running retrieval: {name} (NER={'on' if is_ner_run else 'off'})")
        samples = copy.deepcopy(samples_raw)
        t0 = time.time()

        for sample in samples:
            q = sample["question"]
            n_hyps = var["hyde_n"]
            hyps = hyde_cache_data.get(q, [])[:n_hyps] if n_hyps > 0 else []

            # NER doc filter: predict target documents from the query alone
            doc_filter: Optional[List[str]] = None
            if is_ner_run and ner_filter is not None:
                doc_filter = ner_filter.predict_target_docs(q, top_k=args.ner_top_k)

            # Retrieve at least MAX_K results so @k=10,20 are meaningful
            candidate_k = max(CANDIDATE_K, MAX_K) if var["rerank"] else MAX_K

            # ---------- Select retrieval strategy ----------
            if var["index"] == "bm25":
                chunks = retrieve_bm25(sample, bm25_chunks, bm25_index,
                                       k=MAX_K, doc_filter=doc_filter)

            elif var["index"] == "splade":
                chunks = retrieve_splade(sample, splade_model, splade_tokenizer,
                                         splade_chunks, splade_postings,
                                         k=MAX_K, doc_filter=doc_filter)

            elif var["index"] == "hybrid":
                alpha = var["alpha"]
                chunks = retrieve_hybrid(
                    sample, embed_model, dense_collection,
                    bm25_chunks, bm25_index,
                    dense_weight=alpha, sparse_weight=(1.0 - alpha),
                    k=MAX_K, candidate_k=candidate_k,
                    doc_filter=doc_filter,
                )

            elif var["parent_child"]:
                chunks = retrieve_parent_child(
                    sample, embed_model, pc_collection, parent_map,
                    k=MAX_K, doc_filter=doc_filter)

            elif var["qexp"]:
                chunks = retrieve_dense_query_expansion(
                    sample, embed_model, dense_collection,
                    k=candidate_k, doc_filter=doc_filter)

            elif n_hyps > 0:
                chunks = retrieve_hyde(
                    sample, embed_model, dense_collection, hyps,
                    k=candidate_k, candidate_k=candidate_k,
                    doc_filter=doc_filter)

            else:  # plain dense
                q_emb = embed_model.encode([q], normalize_embeddings=True)[0]
                chunks = _chroma_query(dense_collection, q_emb, candidate_k,
                                       where=_make_chroma_where(doc_filter))

            # ---------- Optional reranking ----------
            if var["rerank"] and cross_encoder is not None and chunks:
                chunks = apply_reranker(sample, chunks, cross_encoder, k=MAX_K)
            else:
                chunks = chunks[:MAX_K]

            sample["retrieved_chunks"] = chunks

        elapsed = time.time() - t0
        logger.info(f"  [{name}] {len(samples)} questions in {elapsed:.1f}s")

        retrieval_results[name] = samples
        with open(pred_path, "w") as f:
            json.dump(samples, f)

    # Free embedding / reranker before generation
    del embed_model
    if cross_encoder is not None:
        del cross_encoder
    import torch
    torch.cuda.empty_cache()

    # ----------------------------------------------------------------
    # PHASE 4: Generation (Qwen alone on GPU)
    # ----------------------------------------------------------------
    if not args.skip_generation:
        logger.info("\n>>> PHASE 4: Generation")
        for var in active_variants:
            name = var["name"]
            gen_path = os.path.join(output_dir, "predictions", f"{name}_generated.json")
            if args.resume and os.path.exists(gen_path):
                with open(gen_path) as f:
                    retrieval_results[name] = json.load(f)
                logger.info(f"  [{name}] Generated answers loaded from cache")
                continue

            logger.info(f"  Generating answers for: {name}")
            samples = generate_answers_for_variant(retrieval_results[name])
            retrieval_results[name] = samples
            with open(gen_path, "w") as f:
                json.dump(samples, f)
    else:
        logger.info("\n>>> PHASE 4: Skipped (--skip-generation)")

    # ----------------------------------------------------------------
    # PHASE 5: Evaluation
    # ----------------------------------------------------------------
    logger.info("\n>>> PHASE 5: Evaluation")
    all_results: Dict[str, Dict] = {}
    by_question_type: Dict[str, Dict[str, Dict]] = {}
    by_doc_type: Dict[str, Dict[str, Dict]] = {}

    for var in active_variants:
        name = var["name"]
        samples = retrieval_results.get(name, [])
        if not samples:
            continue

        # Attach doc_type to each sample for breakdown
        for s in samples:
            s["doc_type"] = doc_info.get(s["doc_name"], {}).get("doc_type", "unknown")
            s["gics_sector"] = doc_info.get(s["doc_name"], {}).get("gics_sector", "unknown")

        # Overall
        ret_metrics = compute_retrieval_metrics(samples)
        gen_metrics = compute_generative_metrics(samples) if not args.skip_generation else {}
        all_results[name] = {**ret_metrics, **gen_metrics}

        # By question type
        by_qt = aggregate_by_group(samples, lambda s: s.get("question_type", "unknown"))
        by_question_type[name] = by_qt

        # By doc type
        by_dt = aggregate_by_group(samples, lambda s: s.get("doc_type", "unknown"))
        by_doc_type[name] = by_dt

        # By question_type × doc_type
        by_qt_dt = aggregate_by_group(
            samples,
            lambda s: f"{s.get('question_type','?')}|{s.get('doc_type','?')}"
        )

        # Save per-variant metrics
        metrics_path = os.path.join(output_dir, "metrics", f"{name}_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump({
                "overall": all_results[name],
                "by_question_type": by_qt,
                "by_doc_type": by_dt,
                "by_question_type_x_doc_type": by_qt_dt,
            }, f, indent=2)

        # Print headline
        pr5 = all_results[name].get(f"page_recall@{MAIN_K}", 0)
        dr5 = all_results[name].get(f"doc_recall@{MAIN_K}", 0)
        bl5 = all_results[name].get(f"context_bleu@{MAIN_K}", 0)
        rl5 = all_results[name].get(f"context_rougeL@{MAIN_K}", 0)
        ans = all_results[name].get("answer_rougeL", 0)
        num = all_results[name].get("numeric_match", 0)
        logger.info(
            f"  [{VARIANT_LABELS[name]}] "
            f"DocRec@5={dr5:.3f} PageRec@5={pr5:.3f} "
            f"BLEU@5={bl5:.3f} ROUGE-L@5={rl5:.3f} "
            f"AnsROUGE={ans:.3f} NumMatch={num:.3f}"
        )

    # Save combined metrics
    all_metrics_path = os.path.join(output_dir, "metrics", "all_variants_metrics.json")
    with open(all_metrics_path, "w") as f:
        json.dump(all_results, f, indent=2)

    save_metrics_table(all_results, output_dir)
    save_by_type_table(by_question_type, output_dir, label="question_type")
    save_by_type_table(by_doc_type, output_dir, label="doc_type")

    # ----------------------------------------------------------------
    # PHASE 6: Visualizations
    # ----------------------------------------------------------------
    if not args.skip_plots:
        logger.info("\n>>> PHASE 6: Generating plots")
        plots_dir = os.path.join(output_dir, "plots")

        plot_retrieval_bar_k5(all_results, plots_dir)
        plot_recall_at_k_curves(all_results, plots_dir)
        plot_heatmap_question_type(by_question_type, plots_dir)
        plot_heatmap_doc_type(by_doc_type, plots_dir)
        plot_question_type_breakdown(by_question_type, plots_dir)
        plot_hybrid_alpha_comparison(all_results, plots_dir)
        if not args.skip_generation:
            plot_generative_metrics(all_results, plots_dir)
            plot_radar_chart(all_results, plots_dir)

    elapsed_total = (time.time() - t_total) / 60
    logger.info(f"\nTotal wall-clock time: {elapsed_total:.1f} min")
    logger.info(f"Results: {output_dir}")

    # ── Final summary table (all @k values) ──────────────────────────────────
    print("\n" + "=" * 110)
    print(f"{'Method':<42} " + "  ".join(f"PR@{k:>2}" for k in K_VALUES) + "  " +
          "  ".join(f"DR@{k:>2}" for k in K_VALUES))
    print("-" * 110)
    for name in [v["name"] for v in active_variants]:
        if name not in all_results:
            continue
        r   = all_results[name]
        lbl = VARIANT_LABELS.get(name, name)
        pr  = "  ".join(f"{r.get(f'page_recall@{k}', 0):>5.3f}" for k in K_VALUES)
        dr  = "  ".join(f"{r.get(f'doc_recall@{k}',  0):>5.3f}" for k in K_VALUES)
        print(f"{lbl:<42}  {pr}  {dr}")
    print("=" * 110)

    # Save the @k summary to CSV
    summary_csv = os.path.join(output_dir, "metrics", "all_k_summary.csv")
    import csv as _csv
    with open(summary_csv, "w", newline="") as _f:
        w = _csv.writer(_f)
        w.writerow(["Method"] +
                   [f"PageRec@{k}" for k in K_VALUES] +
                   [f"DocRec@{k}"  for k in K_VALUES])
        for name in [v["name"] for v in active_variants]:
            if name not in all_results:
                continue
            r = all_results[name]
            w.writerow(
                [VARIANT_LABELS.get(name, name)] +
                [f"{r.get(f'page_recall@{k}', 0):.4f}" for k in K_VALUES] +
                [f"{r.get(f'doc_recall@{k}',  0):.4f}" for k in K_VALUES]
            )
    logger.info(f"All-k summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
