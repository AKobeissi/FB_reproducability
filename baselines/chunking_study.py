#!/usr/bin/env python3
"""
chunking_study.py
=================
Systematic comparison of chunking strategies on FinanceBench using
Dense BGE-M3 retrieval.  Chunking is the only variable under study.

All chunk sizes are in TOKENS (BGE-M3 tokenizer).

Strategies:
  recursive   – Token-based recursive splitting; size sweep
  semantic    – Sentence-similarity breakpoints; threshold sweep
  contextual  – Section/page context prefix (no LLM)
  structure   – SEC filing section-boundary aware
  late        – BGE-M3 late chunking (full-context token embeddings)
  table_aware – Dual index: tables separate from text

Outputs in <output_dir>/chunking_study/:
  metrics/
    all_strategies.csv        <- DocRec@k + PageRec@k for all configs
    chunk_stats.csv           <- count / avg tokens / std per config
  plots/
    recall_at_k.pdf           <- PageRec@k curves per strategy group
    bar_k5.pdf                <- DocRec@5 / PageRec@5 grouped bar chart
    chunk_size_dist.pdf       <- token count histograms
    semantic_breakpoints/     <- per-doc breakpoint plots (sampled docs)

Usage:
  python baselines/chunking_study.py \\
      --data-path       data/financebench_open_source.jsonl \\
      --doc-info-path   data/financebench_document_information.jsonl \\
      --pdf-dir         pdfs \\
      --vs-dir          vector_stores/chunking_study \\
      --output-dir      baselines/results/chunking_study \\
      --strategies      recursive semantic contextual structure late table_aware \\
      --force-reindex
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in [str(_ROOT), str(_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMBED_MODEL    = "BAAI/bge-m3"
CHROMA_BATCH   = 500
K_VALUES       = [1, 3, 5, 10, 20]
MAIN_K         = 5
N_SEMANTIC_PLOT_DOCS = 5   # how many docs to plot breakpoints for

# Sweep configs ---------------------------------------------------------------
RECURSIVE_CONFIGS = [
    {"name": "recursive_256t_32ov",   "size": 256,  "overlap": 32},
    {"name": "recursive_512t_64ov",   "size": 512,  "overlap": 64},
    {"name": "recursive_1024t_128ov", "size": 1024, "overlap": 128},  # baseline
    {"name": "recursive_2048t_256ov", "size": 2048, "overlap": 256},
]

SEMANTIC_CONFIGS = [
    {"name": "semantic_t03", "threshold": 0.3, "min_sents": 1, "max_sents": 60},
    {"name": "semantic_t05", "threshold": 0.5, "min_sents": 2, "max_sents": 40},  # default
    {"name": "semantic_t07", "threshold": 0.7, "min_sents": 2, "max_sents": 30},
    {"name": "semantic_t09", "threshold": 0.9, "min_sents": 3, "max_sents": 20},
]

CONTEXTUAL_CONFIGS = [
    {"name": "contextual_ctx64",  "size": 1024, "overlap": 128, "ctx_budget": 64},
    {"name": "contextual_ctx128", "size": 1024, "overlap": 128, "ctx_budget": 128},
    {"name": "contextual_ctx256", "size": 1024, "overlap": 128, "ctx_budget": 256},
]

STRUCTURE_CONFIGS = [
    {"name": "structure_1024t", "size": 1024, "overlap": 128},
]

LATE_CONFIGS = [
    {"name": "late_512t_64ov",  "size": 512,  "overlap": 64},
    {"name": "late_1024t_128ov","size": 1024, "overlap": 128},
]

TABLE_AWARE_CONFIGS = [
    {"name": "table_aware_dual_1024t", "size": 1024, "overlap": 128},
]


# ===========================================================================
# PDF extraction
# ===========================================================================

def extract_pdf_pages(pdf_path: str) -> List[Tuple[int, str]]:
    """Return list of (0-indexed page_num, page_text) for a PDF."""
    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = (page.extract_text() or "").strip()
                if text:
                    pages.append((i, text))
        return pages
    except Exception as e:
        logger.warning(f"pdfplumber failed for {pdf_path}: {e}")
        return []


def extract_pdf_with_tables(pdf_path: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Extract text and table blocks separately using pdfplumber.

    Returns:
        text_chunks: [{"page": int, "text": str}]
        table_chunks: [{"page": int, "text": str, "title": str}]
    """
    text_chunks, table_chunks = [], []
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Extract tables with pdfplumber
                tables = page.extract_tables() or []
                table_bboxes = [t.bbox for t in page.find_tables()] if hasattr(page, "find_tables") else []

                # Convert tables to text representation
                for tbl in tables:
                    rows = []
                    for row in tbl:
                        cleaned = [str(cell or "").strip() for cell in row]
                        if any(c for c in cleaned):
                            rows.append(" | ".join(cleaned))
                    if rows:
                        table_text = "\n".join(rows)
                        table_chunks.append({
                            "page": page_num,
                            "text": table_text,
                            "title": "",   # filled below if possible
                        })

                # Get page text; strip out approximate table regions
                page_text = page.extract_text() or ""
                # Simple heuristic: remove lines that look like table rows
                text_lines = []
                for line in page_text.split("\n"):
                    stripped = line.strip()
                    # Skip lines that are mostly digits/pipes (table rows)
                    alnum = sum(c.isalnum() for c in stripped)
                    total = max(len(stripped), 1)
                    if alnum / total > 0.1:
                        text_lines.append(line)
                clean_text = "\n".join(text_lines).strip()
                if clean_text:
                    text_chunks.append({"page": page_num, "text": clean_text})
    except Exception as e:
        logger.warning(f"Table extraction failed for {pdf_path}: {e}. Falling back to text-only.")
        pages = extract_pdf_pages(pdf_path)
        text_chunks = [{"page": p, "text": t} for p, t in pages]

    return text_chunks, table_chunks


# ===========================================================================
# Chunk strategy helpers
# ===========================================================================

def _run_chunker(strategy_name: str, pages: List[Tuple[int, str]],
                 doc_id: str, cfg: Dict) -> "List":
    """Dispatch to the appropriate chunker in chunking_strategies."""
    from src.experiments.chunking_strategies import (
        chunk_recursive, chunk_semantic, chunk_contextual,
        chunk_structure_aware, chunk_late,
    )
    if strategy_name == "recursive":
        return chunk_recursive(pages, doc_id=doc_id,
                               chunk_size=cfg["size"], chunk_overlap=cfg["overlap"])
    elif strategy_name == "semantic":
        return chunk_semantic(pages, doc_id=doc_id,
                              chunk_size=cfg["size"],
                              similarity_threshold=cfg["threshold"],
                              min_sentences=cfg["min_sents"],
                              max_sentences=cfg["max_sents"],
                              embedding_model_name=EMBED_MODEL)
    elif strategy_name == "contextual":
        return chunk_contextual(pages, doc_id=doc_id,
                                chunk_size=cfg["size"], chunk_overlap=cfg["overlap"],
                                context_budget=cfg["ctx_budget"])
    elif strategy_name == "structure":
        return chunk_structure_aware(pages, doc_id=doc_id,
                                     chunk_size=cfg["size"], chunk_overlap=cfg["overlap"])
    elif strategy_name == "late":
        from src.experiments.chunking_strategies import chunk_late
        return chunk_late(pages, doc_id=doc_id,
                          chunk_size=cfg["size"], chunk_overlap=cfg["overlap"])
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


# ===========================================================================
# Late chunking: BGE-M3 late-pooled embeddings
# ===========================================================================

class LateChunkEmbedder:
    """
    Computes BGE-M3 late-pooled embeddings for a set of chunk spans.

    For each document:
      1. Tokenize up to max_seq_len tokens.
      2. Run a single encoder forward pass to get all token embeddings.
      3. Mean-pool the token span [start, end) for each chunk.
      4. For documents > max_seq_len: sliding window with stride, avg overlaps.

    Memory: ~3.5 GB VRAM for BGE-M3 on a single forward pass. Fits on 3090.
    """

    def __init__(self, model_name: str = EMBED_MODEL, max_seq_len: int = 8192,
                 window_stride: int = 4096, device: str = "cuda"):
        import torch
        from transformers import AutoTokenizer, AutoModel
        logger.info(f"Loading late-chunk encoder: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(
            model_name, torch_dtype=torch.float16, trust_remote_code=True
        )
        self.model.to(device).eval()
        self.max_seq_len = max_seq_len
        self.window_stride = window_stride
        self.device = device
        logger.info("Late-chunk encoder loaded.")

    def embed_document(
        self,
        full_text: str,
        chunk_spans: List[Tuple[int, int]],  # (start_tok, end_tok) per chunk
    ) -> np.ndarray:
        """
        Return (n_chunks, hidden_dim) array of late-pooled embeddings.
        chunk_spans are TOKEN indices into full_text's tokenization.
        """
        import torch
        import torch.nn.functional as F

        enc = self.tokenizer(full_text, return_tensors="pt", add_special_tokens=True)
        input_ids = enc["input_ids"][0]
        total_tokens = len(input_ids)

        # Map chunk_spans to actual token positions (account for [CLS])
        # enc offsets: token i in full_text → position i+1 (after [CLS])
        def to_model_pos(tok_idx: int) -> int:
            return min(tok_idx + 1, total_tokens - 1)

        # Collect accumulated embeddings per chunk: {chunk_i: list of (emb, weight)}
        chunk_embs: Dict[int, List[np.ndarray]] = defaultdict(list)

        # Sliding window over the tokenized document
        for win_start in range(0, total_tokens, self.window_stride):
            win_end = min(win_start + self.max_seq_len, total_tokens)
            if win_end - win_start < 2:
                break

            window_ids = input_ids[win_start:win_end].unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self.model(input_ids=window_ids)
            token_embs = out.last_hidden_state[0]  # (win_len, hidden)

            for ci, (cs, ce) in enumerate(chunk_spans):
                # Model positions for this chunk within the full sequence
                ms = to_model_pos(cs)
                me = to_model_pos(ce)
                # Intersect with this window
                local_s = ms - win_start
                local_e = me - win_start
                local_s = max(0, local_s)
                local_e = min(win_end - win_start, local_e)
                if local_e <= local_s:
                    continue
                span_emb = token_embs[local_s:local_e].mean(dim=0)
                span_emb = F.normalize(span_emb, dim=0)
                chunk_embs[ci].append(span_emb.cpu().float().numpy())

            if win_end >= total_tokens:
                break

        # Average across windows for each chunk
        result = np.zeros((len(chunk_spans), self.model.config.hidden_size), dtype=np.float32)
        for ci in range(len(chunk_spans)):
            if chunk_embs[ci]:
                avg = np.mean(chunk_embs[ci], axis=0)
                norm = np.linalg.norm(avg)
                result[ci] = avg / (norm + 1e-10)
        return result

    def free(self):
        import torch
        del self.model
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Late-chunk encoder freed from GPU.")


# ===========================================================================
# ChromaDB index building
# ===========================================================================

def _get_chroma_client(persist_dir: str):
    import chromadb
    from chromadb.config import Settings
    os.makedirs(persist_dir, exist_ok=True)
    return chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )


def _get_or_reset_collection(client, name: str, force: bool):
    existing = [c.name for c in client.list_collections()]
    if name in existing:
        if force:
            client.delete_collection(name)
        else:
            col = client.get_collection(name)
            if col.count() > 0:
                logger.info(f"Reusing existing collection '{name}' ({col.count()} chunks)")
                return col, True   # (collection, already_built)
    return client.create_collection(name, metadata={"hnsw:space": "cosine"}), False


def build_index_from_chunks(
    chunks: List[Dict],          # [{"id", "text", "meta"}]
    collection_name: str,
    vs_dir: str,
    embed_model,                 # SentenceTransformer or None (use precomputed)
    precomputed_embeddings: Optional[np.ndarray] = None,
    force: bool = False,
) -> "chromadb.Collection":
    """
    Build (or reuse) a ChromaDB collection from pre-chunked data.

    chunks: list of {"id": str, "text": str, "meta": dict}
    If precomputed_embeddings is provided, skips re-embedding.
    """
    client = _get_chroma_client(vs_dir)
    collection, already_built = _get_or_reset_collection(client, collection_name, force)
    if already_built:
        return collection

    ids   = [c["id"]   for c in chunks]
    texts = [c["text"] for c in chunks]
    metas = [c["meta"] for c in chunks]

    if precomputed_embeddings is not None:
        embs = precomputed_embeddings.tolist()
    else:
        logger.info(f"Embedding {len(texts)} chunks for '{collection_name}' ...")
        embs_np = embed_model.encode(
            texts, batch_size=64, normalize_embeddings=True,
            show_progress_bar=True,
        )
        embs = embs_np.tolist()

    for start in range(0, len(ids), CHROMA_BATCH):
        end = min(start + CHROMA_BATCH, len(ids))
        collection.add(
            ids=ids[start:end],
            documents=texts[start:end],
            metadatas=metas[start:end],
            embeddings=embs[start:end],
        )

    logger.info(f"Collection '{collection_name}': {collection.count()} chunks indexed.")
    return collection


# ===========================================================================
# Dense BGE-M3 retrieval
# ===========================================================================

def dense_retrieve(
    query: str,
    collection,
    embed_model,
    k: int = 20,
    where: Optional[Dict] = None,
) -> List[Dict]:
    """Query ChromaDB with BGE-M3 embedding, return top-k chunks."""
    emb = embed_model.encode([query], normalize_embeddings=True)[0].tolist()
    kwargs = dict(
        query_embeddings=[emb],
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    if where:
        kwargs["where"] = where
    res = collection.query(**kwargs)
    chunks = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        chunks.append({
            "text": doc,
            "metadata": {
                "doc_name": meta.get("doc_name", ""),
                "page":     meta.get("page", -1),
            },
            "_score": float(1.0 - dist),
        })
    return chunks


def rrf_merge(lists: List[List[Dict]], k: int = 60) -> List[Dict]:
    """Reciprocal Rank Fusion over multiple ranked chunk lists."""
    scores: Dict[str, float] = defaultdict(float)
    id_to_chunk: Dict[str, Dict] = {}
    for ranked in lists:
        for rank, chunk in enumerate(ranked, start=1):
            cid = f"{chunk['metadata']['doc_name']}_{chunk['metadata']['page']}"
            scores[cid] += 1.0 / (k + rank)
            if cid not in id_to_chunk:
                id_to_chunk[cid] = chunk
    merged_ids = sorted(scores, key=lambda x: -scores[x])
    return [id_to_chunk[cid] for cid in merged_ids]


# ===========================================================================
# Evaluation
# ===========================================================================

def compute_metrics(
    samples: List[Dict],
    k_values: List[int] = K_VALUES,
) -> Dict:
    """
    Compute DocRec@k and PageRec@k.

    samples[i]["retrieved_chunks"] = [{"metadata": {"doc_name", "page"}}]
    samples[i]["gold_evidence_segments"] = [{"doc_name", "page"}]
    """
    metrics: Dict[str, float] = {}
    n = len(samples)
    if n == 0:
        return metrics

    for k in k_values:
        doc_hits = 0
        page_hits = 0
        for s in samples:
            retrieved = s["retrieved_chunks"][:k]
            gold_docs  = {ev["doc_name"] for ev in s["gold_evidence_segments"]}
            gold_pages = {(ev["doc_name"], ev["page"]) for ev in s["gold_evidence_segments"]}

            ret_docs  = {c["metadata"]["doc_name"] for c in retrieved}
            ret_pages = {(c["metadata"]["doc_name"], c["metadata"]["page"]) for c in retrieved}

            if gold_docs & ret_docs:
                doc_hits += 1
            if gold_pages & ret_pages:
                page_hits += 1

        metrics[f"doc_recall@{k}"]  = round(doc_hits  / n, 4)
        metrics[f"page_recall@{k}"] = round(page_hits / n, 4)

    return metrics


# ===========================================================================
# Index and evaluate one config
# ===========================================================================

def run_config(
    config_name: str,
    strategy_family: str,
    all_chunks_flat: List[Dict],          # [{"id","text","meta"}]
    samples: List[Dict],
    embed_model,
    vs_dir: str,
    force: bool,
    precomputed_embeddings: Optional[np.ndarray] = None,
) -> Dict:
    """Build index and evaluate one chunking config. Returns metrics dict."""
    logger.info(f"\n{'='*55}")
    logger.info(f"Config: {config_name}  ({len(all_chunks_flat)} chunks)")
    logger.info(f"{'='*55}")

    t0 = time.time()
    collection = build_index_from_chunks(
        all_chunks_flat, config_name, vs_dir, embed_model,
        precomputed_embeddings=precomputed_embeddings, force=force,
    )

    # Retrieve for each sample
    samples_copy = copy.deepcopy(samples)
    for s in samples_copy:
        chunks = dense_retrieve(s["question"], collection, embed_model, k=max(K_VALUES))
        s["retrieved_chunks"] = chunks

    metrics = compute_metrics(samples_copy, K_VALUES)
    metrics["retrieval_time_s"] = round(time.time() - t0, 1)
    metrics["n_chunks"] = len(all_chunks_flat)

    pr5 = metrics.get(f"page_recall@{MAIN_K}", 0)
    dr5 = metrics.get(f"doc_recall@{MAIN_K}",  0)
    logger.info(f"[{config_name}] DocRec@{MAIN_K}={dr5:.3f} | PageRec@{MAIN_K}={pr5:.3f}"
                f" | {metrics['n_chunks']} chunks | {metrics['retrieval_time_s']}s")
    return metrics


def run_table_aware_config(
    config_name: str,
    text_chunks: List[Dict],
    table_chunks: List[Dict],
    samples: List[Dict],
    embed_model,
    vs_dir: str,
    force: bool,
) -> Dict:
    """Build dual indexes (text + table) and merge results with RRF."""
    logger.info(f"\n{'='*55}")
    logger.info(f"Config: {config_name}  (text={len(text_chunks)}, table={len(table_chunks)})")
    logger.info(f"{'='*55}")

    t0 = time.time()
    text_col = build_index_from_chunks(
        text_chunks, f"{config_name}_text", vs_dir, embed_model, force=force
    )
    table_col = build_index_from_chunks(
        table_chunks, f"{config_name}_table", vs_dir, embed_model, force=force
    ) if table_chunks else None

    samples_copy = copy.deepcopy(samples)
    for s in samples_copy:
        text_res  = dense_retrieve(s["question"], text_col,  embed_model, k=max(K_VALUES))
        table_res = (dense_retrieve(s["question"], table_col, embed_model, k=max(K_VALUES))
                     if table_col else [])
        merged = rrf_merge([text_res, table_res]) if table_res else text_res
        s["retrieved_chunks"] = merged

    metrics = compute_metrics(samples_copy, K_VALUES)
    metrics["retrieval_time_s"] = round(time.time() - t0, 1)
    metrics["n_chunks"] = len(text_chunks) + len(table_chunks)
    metrics["n_text_chunks"]  = len(text_chunks)
    metrics["n_table_chunks"] = len(table_chunks)

    pr5 = metrics.get(f"page_recall@{MAIN_K}", 0)
    dr5 = metrics.get(f"doc_recall@{MAIN_K}",  0)
    logger.info(f"[{config_name}] DocRec@{MAIN_K}={dr5:.3f} | PageRec@{MAIN_K}={pr5:.3f}")
    return metrics


# ===========================================================================
# Chunk statistics
# ===========================================================================

def chunk_stats(chunks: List[Dict]) -> Dict:
    """Compute token count statistics over a set of chunks."""
    from src.experiments.chunking_strategies import _count_tokens
    counts = [_count_tokens(c["text"]) for c in chunks]
    if not counts:
        return {}
    return {
        "count":  len(counts),
        "mean":   round(float(np.mean(counts)), 1),
        "median": round(float(np.median(counts)), 1),
        "std":    round(float(np.std(counts)), 1),
        "min":    int(np.min(counts)),
        "max":    int(np.max(counts)),
        "token_counts": counts,     # kept for histogram
    }


# ===========================================================================
# Plots
# ===========================================================================

def _init_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "figure.dpi": 150,
    })
    return plt


def plot_recall_curves(all_results: Dict[str, Dict], out_dir: str):
    """PageRec@k curves grouped by strategy family."""
    plt = _init_matplotlib()

    # Group configs by family prefix
    families: Dict[str, List[str]] = defaultdict(list)
    for cfg_name in all_results:
        family = cfg_name.split("_")[0]
        families[family].append(cfg_name)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric_prefix in zip(axes, ["page_recall", "doc_recall"]):
        label = "PageRec@k" if metric_prefix == "page_recall" else "DocRec@k"
        ax.set_title(label)
        ax.set_xlabel("k")
        ax.set_ylabel(label)

        cmap = plt.cm.get_cmap("tab20")
        all_cfg_names = list(all_results.keys())
        for ci, cfg_name in enumerate(all_cfg_names):
            m = all_results[cfg_name]
            ys = [m.get(f"{metric_prefix}@{k}", float("nan")) for k in K_VALUES]
            ax.plot(K_VALUES, ys, marker="o", label=cfg_name,
                    color=cmap(ci / max(len(all_cfg_names), 1)), linewidth=1.5)

        ax.legend(fontsize=7, ncol=2, loc="lower right")
        ax.set_xticks(K_VALUES)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(out_dir, "recall_at_k.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")


def plot_bar_comparison(all_results: Dict[str, Dict], out_dir: str):
    """Grouped bar chart: DocRec@5 and PageRec@5 for all configs."""
    plt = _init_matplotlib()

    cfg_names = list(all_results.keys())
    doc_vals  = [all_results[n].get(f"doc_recall@{MAIN_K}",  0) for n in cfg_names]
    page_vals = [all_results[n].get(f"page_recall@{MAIN_K}", 0) for n in cfg_names]

    x = np.arange(len(cfg_names))
    w = 0.38

    fig, ax = plt.subplots(figsize=(max(10, len(cfg_names) * 0.7), 5))
    bars1 = ax.bar(x - w / 2, doc_vals,  w, label=f"DocRec@{MAIN_K}",  color="#0072B2")
    bars2 = ax.bar(x + w / 2, page_vals, w, label=f"PageRec@{MAIN_K}", color="#E69F00")

    for bar, val in zip(list(bars1) + list(bars2), doc_vals + page_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(cfg_names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(f"Recall@{MAIN_K}")
    ax.set_title(f"Chunking Strategy Comparison — DocRec & PageRec @ {MAIN_K}")
    ax.legend()
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(out_dir, "bar_k5.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")


def plot_chunk_size_distributions(all_stats: Dict[str, Dict], out_dir: str):
    """Token count histograms for each config (overlaid + individual)."""
    plt = _init_matplotlib()

    configs_with_data = {k: v for k, v in all_stats.items() if "token_counts" in v}
    if not configs_with_data:
        return

    # Overlaid plot
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.cm.get_cmap("tab20")
    for ci, (cfg_name, stats) in enumerate(configs_with_data.items()):
        counts = stats["token_counts"]
        ax.hist(counts, bins=40, alpha=0.4, density=True,
                label=f"{cfg_name} (μ={stats['mean']:.0f})",
                color=cmap(ci / max(len(configs_with_data), 1)))
    ax.set_xlabel("Chunk size (tokens)")
    ax.set_ylabel("Density")
    ax.set_title("Chunk Size Distribution by Strategy")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(out_dir, "chunk_size_dist.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")


def plot_semantic_breakpoints(
    breakpoint_data: List[Dict],   # [{"doc_id", "threshold", "sims", "breakpoints"}]
    out_dir: str,
):
    """
    For each (doc, threshold) pair, plot cosine similarity curve with
    breakpoint positions highlighted.
    """
    plt = _init_matplotlib()
    bp_dir = os.path.join(out_dir, "semantic_breakpoints")
    os.makedirs(bp_dir, exist_ok=True)

    # Group by threshold for multi-panel plots
    by_threshold: Dict[float, List[Dict]] = defaultdict(list)
    for d in breakpoint_data:
        by_threshold[d["threshold"]].append(d)

    for thresh, entries in sorted(by_threshold.items()):
        n_docs = len(entries)
        fig, axes = plt.subplots(n_docs, 1, figsize=(12, 3 * n_docs), squeeze=False)
        fig.suptitle(f"Semantic Breakpoints  (threshold={thresh})", fontsize=13)

        for row_i, entry in enumerate(entries):
            ax = axes[row_i][0]
            sims = entry["sims"]
            bps  = set(entry["breakpoints"])

            xs = list(range(len(sims)))
            ax.plot(xs, sims, color="#0072B2", linewidth=1, alpha=0.8, label="cosine sim")
            ax.axhline(thresh, color="red", linestyle="--", linewidth=1, alpha=0.7,
                       label=f"threshold={thresh}")

            # Breakpoint vertical lines
            for bp in bps:
                if 0 < bp < len(sims):
                    ax.axvline(bp - 0.5, color="orange", linestyle="-", linewidth=0.8, alpha=0.6)

            ax.set_ylabel("Cosine sim")
            ax.set_title(f"{entry['doc_id']}  ({entry['n_chunks']} chunks from {entry['n_sents']} sentences)")
            ax.set_xlim(0, max(len(sims) - 1, 1))
            ax.set_ylim(max(0, min(sims) - 0.05), 1.02)
            ax.grid(True, alpha=0.25)
            if row_i == 0:
                ax.legend(fontsize=8, loc="upper right")

        plt.tight_layout()
        fname = f"breakpoints_t{str(thresh).replace('.', '')}.pdf"
        out = os.path.join(bp_dir, fname)
        plt.savefig(out, bbox_inches="tight")
        plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight")
        plt.close()
        logger.info(f"Saved breakpoint plot: {out}")


def plot_semantic_sweep_summary(all_results: Dict[str, Dict], out_dir: str):
    """Line plot: PageRec@5 vs semantic threshold for semantic configs."""
    plt = _init_matplotlib()
    sem = {k: v for k, v in all_results.items() if k.startswith("semantic_")}
    if not sem:
        return

    # Extract threshold from config name e.g. "semantic_t05" → 0.5
    def _thresh(name: str) -> float:
        try:
            return float(name.split("_t")[1]) / 10
        except Exception:
            return 0.0

    entries = sorted(sem.items(), key=lambda x: _thresh(x[0]))
    thresholds = [_thresh(k) for k, _ in entries]
    page_vals  = [v.get(f"page_recall@{MAIN_K}", 0) for _, v in entries]
    doc_vals   = [v.get(f"doc_recall@{MAIN_K}",  0) for _, v in entries]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(thresholds, page_vals, "o-", color="#E69F00", label=f"PageRec@{MAIN_K}")
    ax.plot(thresholds, doc_vals,  "s--", color="#0072B2", label=f"DocRec@{MAIN_K}")
    ax.set_xlabel("Similarity threshold")
    ax.set_ylabel(f"Recall@{MAIN_K}")
    ax.set_title("Semantic Chunking: Threshold Sweep")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(out_dir, "semantic_threshold_sweep.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")


def plot_chunk_count_vs_recall(all_results: Dict[str, Dict],
                                all_stats: Dict[str, Dict], out_dir: str):
    """Scatter: avg chunk size (tokens) vs PageRec@5, coloured by strategy family."""
    plt = _init_matplotlib()

    xs, ys, labels, colors = [], [], [], []
    family_color = {
        "recursive": "#0072B2", "semantic": "#E69F00",
        "contextual": "#009E73", "structure": "#D55E00",
        "late": "#CC79A7", "table": "#56B4E9",
    }
    for cfg_name, metrics in all_results.items():
        mean_tok = all_stats.get(cfg_name, {}).get("mean", None)
        if mean_tok is None:
            continue
        pr5 = metrics.get(f"page_recall@{MAIN_K}", None)
        if pr5 is None:
            continue
        family = cfg_name.split("_")[0]
        xs.append(mean_tok)
        ys.append(pr5)
        labels.append(cfg_name)
        colors.append(family_color.get(family, "grey"))

    if not xs:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(xs, ys, c=colors, s=80, zorder=3)
    for x, y, lbl in zip(xs, ys, labels):
        ax.annotate(lbl, (x, y), textcoords="offset points",
                    xytext=(5, 3), fontsize=7)

    # Legend patches
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=c, label=f) for f, c in family_color.items()]
    ax.legend(handles=patches, fontsize=8, loc="lower right")

    ax.set_xlabel("Mean chunk size (tokens)")
    ax.set_ylabel(f"PageRec@{MAIN_K}")
    ax.set_title("Chunk Size vs Page Recall")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(out_dir, "chunk_size_vs_recall.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")


# ===========================================================================
# Save results
# ===========================================================================

def save_csv(all_results: Dict[str, Dict], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "all_strategies.csv")
    metrics_keys = (
        [f"page_recall@{k}" for k in K_VALUES]
        + [f"doc_recall@{k}" for k in K_VALUES]
        + ["n_chunks", "retrieval_time_s"]
    )
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config"] + metrics_keys)
        for cfg_name, m in all_results.items():
            w.writerow([cfg_name] + [m.get(k, "") for k in metrics_keys])
    logger.info(f"Saved: {path}")


def save_stats_csv(all_stats: Dict[str, Dict], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "chunk_stats.csv")
    cols = ["config", "count", "mean", "median", "std", "min", "max"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for cfg_name, s in all_stats.items():
            w.writerow([cfg_name] + [s.get(c, "") for c in cols[1:]])
    logger.info(f"Saved: {path}")


# ===========================================================================
# Load FinanceBench
# ===========================================================================

def load_financebench(fb_path: str) -> List[Dict]:
    samples = []
    with open(fb_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            gold = []
            for ev in raw.get("evidence", []):
                gold.append({
                    "doc_name": ev.get("doc_name", raw.get("doc_name", "")),
                    "page":     ev.get("evidence_page_num", -1),
                    "text":     ev.get("evidence_text", ""),
                })
            samples.append({
                "question":              raw.get("question", ""),
                "doc_name":              raw.get("doc_name", ""),
                "doc_link":              raw.get("doc_link", ""),
                "question_type":         raw.get("question_type", "unknown"),
                "gold_evidence_segments": gold,
                "retrieved_chunks":      [],
            })
    logger.info(f"Loaded {len(samples)} FinanceBench samples from {fb_path}")
    return samples


# ===========================================================================
# Main experiment loop
# ===========================================================================

def run_experiment(args):
    os.makedirs(os.path.join(args.output_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "plots"),   exist_ok=True)

    samples = load_financebench(args.data_path)

    # Collect unique PDFs
    unique_docs = {}
    for s in samples:
        if s["doc_name"] not in unique_docs:
            unique_docs[s["doc_name"]] = s.get("doc_link", "")

    pdf_dir = args.pdf_dir
    pdf_files = {Path(p).stem: str(p) for p in Path(pdf_dir).glob("*.pdf")}
    logger.info(f"Found {len(pdf_files)} PDFs in {pdf_dir}")

    # Pre-extract pages (once for all strategies)
    logger.info("Pre-extracting PDF pages ...")
    doc_pages: Dict[str, List[Tuple[int, str]]] = {}
    for doc_name in unique_docs:
        pdf_path = pdf_files.get(doc_name)
        if pdf_path:
            doc_pages[doc_name] = extract_pdf_pages(pdf_path)
        else:
            logger.warning(f"PDF not found: {doc_name}")

    # Load BGE-M3 embedding model
    from sentence_transformers import SentenceTransformer
    logger.info(f"Loading embedding model: {EMBED_MODEL}")
    embed_model = SentenceTransformer(EMBED_MODEL)

    all_results: Dict[str, Dict] = {}
    all_stats:   Dict[str, Dict] = {}
    breakpoint_data: List[Dict]  = []

    strategies_to_run = set(args.strategies)

    # -----------------------------------------------------------------------
    # Recursive sweep
    # -----------------------------------------------------------------------
    if "recursive" in strategies_to_run:
        for cfg in RECURSIVE_CONFIGS:
            logger.info(f"\n--- Recursive: {cfg['name']} ---")
            all_chunks = []
            for doc_name, pages in doc_pages.items():
                from src.experiments.chunking_strategies import chunk_recursive
                chunks = chunk_recursive(pages, doc_id=doc_name,
                                         chunk_size=cfg["size"], chunk_overlap=cfg["overlap"])
                for i, c in enumerate(chunks):
                    all_chunks.append({
                        "id":   f"{doc_name}__{cfg['name']}_{i}",
                        "text": c.text,
                        "meta": {
                            "doc_name": doc_name,
                            "page":     c.page_nums[0] if c.page_nums else -1,
                        },
                    })
            stats = chunk_stats(all_chunks)
            all_stats[cfg["name"]] = stats
            metrics = run_config(cfg["name"], "recursive", all_chunks, samples,
                                 embed_model, args.vs_dir, args.force_reindex)
            all_results[cfg["name"]] = metrics

    # -----------------------------------------------------------------------
    # Semantic sweep
    # -----------------------------------------------------------------------
    if "semantic" in strategies_to_run:
        # Collect breakpoint samples from a few docs
        plot_docs = list(doc_pages.keys())[:N_SEMANTIC_PLOT_DOCS]

        for cfg in SEMANTIC_CONFIGS:
            logger.info(f"\n--- Semantic: {cfg['name']} ---")
            from src.experiments.chunking_strategies import chunk_semantic
            all_chunks = []
            for doc_name, pages in doc_pages.items():
                result = chunk_semantic(
                    pages, doc_id=doc_name,
                    chunk_size=1024,
                    similarity_threshold=cfg["threshold"],
                    min_sentences=cfg["min_sents"],
                    max_sentences=cfg["max_sents"],
                    embedding_model_name=EMBED_MODEL,
                    return_debug=(doc_name in plot_docs),
                )
                if isinstance(result, tuple):
                    chunks, debug = result
                    breakpoint_data.append({
                        "doc_id":     doc_name,
                        "threshold":  cfg["threshold"],
                        "sims":       debug["similarities"],
                        "breakpoints": debug["breakpoint_indices"],
                        "n_chunks":   len(chunks),
                        "n_sents":    debug["num_sentences"],
                    })
                else:
                    chunks = result
                for i, c in enumerate(chunks):
                    all_chunks.append({
                        "id":   f"{doc_name}__{cfg['name']}_{i}",
                        "text": c.text,
                        "meta": {
                            "doc_name": doc_name,
                            "page":     c.page_nums[0] if c.page_nums else -1,
                        },
                    })
            stats = chunk_stats(all_chunks)
            all_stats[cfg["name"]] = stats
            metrics = run_config(cfg["name"], "semantic", all_chunks, samples,
                                 embed_model, args.vs_dir, args.force_reindex)
            all_results[cfg["name"]] = metrics

    # -----------------------------------------------------------------------
    # Contextual sweep
    # -----------------------------------------------------------------------
    if "contextual" in strategies_to_run:
        for cfg in CONTEXTUAL_CONFIGS:
            logger.info(f"\n--- Contextual: {cfg['name']} ---")
            from src.experiments.chunking_strategies import chunk_contextual
            all_chunks = []
            for doc_name, pages in doc_pages.items():
                chunks = chunk_contextual(pages, doc_id=doc_name,
                                          chunk_size=cfg["size"],
                                          chunk_overlap=cfg["overlap"],
                                          context_budget=cfg["ctx_budget"])
                for i, c in enumerate(chunks):
                    all_chunks.append({
                        "id":   f"{doc_name}__{cfg['name']}_{i}",
                        "text": c.text,
                        "meta": {
                            "doc_name": doc_name,
                            "page":     c.page_nums[0] if c.page_nums else -1,
                        },
                    })
            stats = chunk_stats(all_chunks)
            all_stats[cfg["name"]] = stats
            metrics = run_config(cfg["name"], "contextual", all_chunks, samples,
                                 embed_model, args.vs_dir, args.force_reindex)
            all_results[cfg["name"]] = metrics

    # -----------------------------------------------------------------------
    # Structure-aware
    # -----------------------------------------------------------------------
    if "structure" in strategies_to_run:
        for cfg in STRUCTURE_CONFIGS:
            logger.info(f"\n--- Structure-aware: {cfg['name']} ---")
            from src.experiments.chunking_strategies import chunk_structure_aware
            all_chunks = []
            for doc_name, pages in doc_pages.items():
                chunks = chunk_structure_aware(pages, doc_id=doc_name,
                                               chunk_size=cfg["size"],
                                               chunk_overlap=cfg["overlap"])
                for i, c in enumerate(chunks):
                    all_chunks.append({
                        "id":   f"{doc_name}__{cfg['name']}_{i}",
                        "text": c.text,
                        "meta": {
                            "doc_name":       doc_name,
                            "page":           c.page_nums[0] if c.page_nums else -1,
                            "section_header": c.metadata.get("section_header", ""),
                        },
                    })
            stats = chunk_stats(all_chunks)
            all_stats[cfg["name"]] = stats
            metrics = run_config(cfg["name"], "structure", all_chunks, samples,
                                 embed_model, args.vs_dir, args.force_reindex)
            all_results[cfg["name"]] = metrics

    # -----------------------------------------------------------------------
    # Late chunking (BGE-M3 late pooling)
    # -----------------------------------------------------------------------
    if "late" in strategies_to_run:
        from src.experiments.chunking_strategies import chunk_late

        # Load late embedder (only while computing embeddings)
        late_embedder = LateChunkEmbedder(EMBED_MODEL)

        for cfg in LATE_CONFIGS:
            logger.info(f"\n--- Late chunking: {cfg['name']} ---")
            all_chunks = []
            all_embeddings_list = []

            for doc_name, pages in doc_pages.items():
                chunks = chunk_late(pages, doc_id=doc_name,
                                    chunk_size=cfg["size"], chunk_overlap=cfg["overlap"])
                if not chunks:
                    continue

                full_text = "\n".join(t for _, t in pages)
                spans = [c.metadata["token_span"] for c in chunks]
                try:
                    embs = late_embedder.embed_document(full_text, spans)
                except Exception as e:
                    logger.warning(f"Late embedding failed for {doc_name}: {e}. Using fallback.")
                    embs = embed_model.encode(
                        [c.text for c in chunks], normalize_embeddings=True
                    )

                for i, (c, emb) in enumerate(zip(chunks, embs)):
                    all_chunks.append({
                        "id":   f"{doc_name}__{cfg['name']}_{i}",
                        "text": c.text,
                        "meta": {"doc_name": doc_name,
                                 "page": c.page_nums[0] if c.page_nums else -1},
                    })
                    all_embeddings_list.append(emb)

            all_embeddings = np.array(all_embeddings_list) if all_embeddings_list else None

            stats = chunk_stats(all_chunks)
            all_stats[cfg["name"]] = stats
            metrics = run_config(cfg["name"], "late", all_chunks, samples,
                                 embed_model, args.vs_dir, args.force_reindex,
                                 precomputed_embeddings=all_embeddings)
            all_results[cfg["name"]] = metrics

        late_embedder.free()
        del late_embedder

    # -----------------------------------------------------------------------
    # Table-aware (dual index)
    # -----------------------------------------------------------------------
    if "table_aware" in strategies_to_run:
        from src.experiments.chunking_strategies import chunk_recursive
        for cfg in TABLE_AWARE_CONFIGS:
            logger.info(f"\n--- Table-aware: {cfg['name']} ---")
            text_chunks_all, table_chunks_all = [], []

            for doc_name, _ in doc_pages.items():
                pdf_path = pdf_files.get(doc_name)
                if not pdf_path:
                    continue
                text_blocks, table_blocks = extract_pdf_with_tables(pdf_path)

                # Chunk text blocks with recursive splitter
                text_pages = [(b["page"], b["text"]) for b in text_blocks]
                for i, c in enumerate(chunk_recursive(
                    text_pages, doc_id=doc_name,
                    chunk_size=cfg["size"], chunk_overlap=cfg["overlap"]
                )):
                    text_chunks_all.append({
                        "id":   f"{doc_name}__{cfg['name']}_text_{i}",
                        "text": c.text,
                        "meta": {"doc_name": doc_name,
                                 "page": c.page_nums[0] if c.page_nums else -1,
                                 "chunk_type": "text"},
                    })

                # Keep table blocks atomic (one entry per table)
                for i, tbl in enumerate(table_blocks):
                    if tbl["text"].strip():
                        table_chunks_all.append({
                            "id":   f"{doc_name}__{cfg['name']}_table_{i}",
                            "text": tbl["text"],
                            "meta": {"doc_name": doc_name,
                                     "page": tbl["page"],
                                     "chunk_type": "table"},
                        })

            all_stats[cfg["name"]] = chunk_stats(text_chunks_all + table_chunks_all)
            metrics = run_table_aware_config(cfg["name"], text_chunks_all, table_chunks_all,
                                             samples, embed_model, args.vs_dir,
                                             args.force_reindex)
            all_results[cfg["name"]] = metrics

    # -----------------------------------------------------------------------
    # Save + plot
    # -----------------------------------------------------------------------
    metrics_dir = os.path.join(args.output_dir, "metrics")
    plots_dir   = os.path.join(args.output_dir, "plots")

    save_csv(all_results, metrics_dir)
    save_stats_csv(all_stats, metrics_dir)

    with open(os.path.join(metrics_dir, "all_strategies.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    plot_recall_curves(all_results, plots_dir)
    plot_bar_comparison(all_results, plots_dir)
    plot_chunk_size_distributions(all_stats, plots_dir)
    plot_semantic_sweep_summary(all_results, plots_dir)
    plot_chunk_count_vs_recall(all_results, all_stats, plots_dir)

    if breakpoint_data:
        plot_semantic_breakpoints(breakpoint_data, plots_dir)

    # Print summary table
    print(f"\n{'='*70}")
    print(f"{'Config':<30} {'DocRec@5':>10} {'PageRec@5':>10} {'N chunks':>10}")
    print(f"{'='*70}")
    for cfg_name, m in sorted(all_results.items(),
                               key=lambda x: x[1].get(f"page_recall@{MAIN_K}", 0),
                               reverse=True):
        dr5 = m.get(f"doc_recall@{MAIN_K}",  float("nan"))
        pr5 = m.get(f"page_recall@{MAIN_K}", float("nan"))
        nc  = m.get("n_chunks", "?")
        print(f"{cfg_name:<30} {dr5:>10.3f} {pr5:>10.3f} {str(nc):>10}")
    print(f"{'='*70}\n")

    logger.info(f"Results saved to: {args.output_dir}")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="FinanceBench chunking strategy comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-path",     required=True)
    p.add_argument("--doc-info-path", default=None)
    p.add_argument("--pdf-dir",       required=True)
    p.add_argument("--vs-dir",        default="vector_stores/chunking_study")
    p.add_argument("--output-dir",    default="baselines/results/chunking_study")
    p.add_argument(
        "--strategies",
        nargs="+",
        default=["recursive", "semantic", "contextual", "structure", "late", "table_aware"],
        choices=["recursive", "semantic", "contextual", "structure", "late", "table_aware"],
    )
    p.add_argument("--force-reindex", action="store_true",
                   help="Delete and rebuild ChromaDB collections")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
