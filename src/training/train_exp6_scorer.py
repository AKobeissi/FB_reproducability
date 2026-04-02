#!/usr/bin/env python3
"""
src/training/train_exp6_scorer.py
===================================
Experiment 6: BGE-M3 page scorer with temporal hard negatives.

Incorporates all lessons from Experiments 1–5:

  1. Multi-source JSONL  — accepts multiple --jsonl paths (deduplicated by
     qid+doc+page).  Pass both finqa_test_gold_pages.jsonl and the newly
     prepared finqa_train_gold_pages.jsonl to get all ~6,750+ training
     examples without any FinanceBench leakage.

     Two-tier gold text:
       Tier A (_source="pdf"): gold text from PDF extraction — same-doc
         BM25 + temporal negatives available.
       Tier B (_source="json"): gold text from pre_text+post_text in
         train.json (clean, no pypdf noise) — cross-doc random negatives
         used instead (no PDF available for mining).
     The FinQA train split provides ~6,251 examples; ~1,700+ will be Tier A
     (matching PDF found in Final-PDF/), the rest Tier B.

  2. Temporal hard negatives  — pages from the SAME company's OTHER filing
     years replace random cross-doc negatives.  Root-cause: 80 % of doc-level
     retrieval errors in Exp 5 were same-company/different-year confusions.
     Random cross-doc negatives from unrelated companies never trained the
     model to distinguish these cases.

  3. Extended LoRA  — r=32, targets q/k/v + attn-out + FFN-intermediate
     (~6.7 M trainable, 1.17 %).  Exp 3 reached the best training loss with
     extended LoRA; r=32 adds capacity without memory risk.

  4. Hierarchical MNR loss  — page-level + chunk-level MNR.  Chunk negatives
     are the BM25-best 512-token chunk of each page negative (1-to-1), so the
     two terms reinforce rather than conflict.

  5. Loss scale 25  — Exp 5 used scale=50; analysis showed softmax probability
     of the gold ≈ 0.9975 for the mean score gap of 0.12, causing gradient
     vanishing for moderate pairs.  Scale=25 keeps gradients alive across the
     full difficulty spectrum.

  6. Validation split + cosine LR schedule  — 10 % of examples held out.
     Cosine decay avoids the LR spike that caused oscillation in Exp 5.
     load_best_model_at_end=True on eval_loss.

Negative composition (per training row):
  ① same-doc BM25 hard negs    (--hard-negatives, default 3)
  ② temporal BM25 negs         (--temporal-negs,  default 2)
     same company, different year, BM25-mined against the query
  ③ random cross-doc negs      (--cross-doc-negs, default 0, disabled)
     different company — adds diversity but is easy; enable only if needed

No FinanceBench data is referenced.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

try:
    from pypdf import PdfReader
except ImportError:
    logger.error("pypdf not found — pip install pypdf"); sys.exit(1)

try:
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
        losses,
    )
except ImportError:
    logger.error("sentence-transformers not found (v3.0+ required)"); sys.exit(1)

try:
    from datasets import Dataset
except ImportError:
    logger.error("datasets not found — pip install datasets"); sys.exit(1)

try:
    from peft import LoraConfig, get_peft_model, TaskType
except ImportError:
    logger.error("peft not found — pip install peft"); sys.exit(1)

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    logger.warning("rank_bm25 not available — BM25 hard neg mining disabled (pip install rank_bm25)")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

BGE_MAX_SEQ_LENGTH = 2048

# BGE-M3 (XLM-RoBERTa backbone) LoRA target names
_LORA_BASELINE = ["query", "key", "value"]
_LORA_EXTENDED = [
    "query", "key", "value",
    "attention.output.dense",   # attention output projection
    "intermediate.dense",       # FFN up-projection
]

_WS = re.compile(r"\s+")


def normalise(text: str) -> str:
    return _WS.sub(" ", (text or "").strip())


# ─────────────────────────────────────────────────────────────────────────────
# 1.  PDF extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_pdf_pages(pdf_path: Path) -> List[str]:
    """Extract normalised text for each page (0-indexed list)."""
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as e:
        logger.warning(f"  Cannot open {pdf_path.name}: {e}")
        return []
    pages = []
    for page in reader.pages:
        try:
            raw = page.extract_text() or ""
        except Exception:
            raw = ""
        pages.append(normalise(raw))
    return pages


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Token-based chunking (for hierarchical loss)
# ─────────────────────────────────────────────────────────────────────────────

def chunk_text_by_tokens(
    text: str,
    tokenizer,
    chunk_tokens: int = 512,
    overlap_tokens: int = 64,
) -> List[str]:
    if not text.strip():
        return []
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return []
    step = max(1, chunk_tokens - overlap_tokens)
    chunks = []
    for start in range(0, len(ids), step):
        decoded = tokenizer.decode(ids[start:start + chunk_tokens], skip_special_tokens=True)
        if decoded.strip():
            chunks.append(decoded)
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Negative mining
# ─────────────────────────────────────────────────────────────────────────────

def _bm25_rank(
    query: str,
    texts: List[str],
    candidate_indices: List[int],
    n: int,
    cache: Optional[Dict],
    cache_key: Optional[str],
) -> List[str]:
    """Return top-n from texts[candidate_indices] by BM25 score."""
    if not candidate_indices:
        return []
    if len(candidate_indices) <= n:
        return [texts[i] for i in candidate_indices]

    if cache is not None and cache_key and cache_key in cache:
        bm25 = cache[cache_key]
    else:
        tokenized = [texts[i].lower().split() for i in range(len(texts))]
        bm25 = BM25Okapi(tokenized)
        if cache is not None and cache_key:
            cache[cache_key] = bm25

    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(candidate_indices, key=lambda i: -scores[i])
    return [texts[i] for i in ranked[:n]]


def mine_same_doc_negatives(
    query: str,
    page_texts: List[str],
    gold_page: int,
    n: int,
    min_chars: int = 100,
    use_bm25: bool = True,
    bm25_cache: Optional[Dict] = None,
    doc_key: Optional[str] = None,
) -> List[str]:
    """Non-gold pages within the same document, ranked by BM25."""
    valid = [i for i, t in enumerate(page_texts)
             if i != gold_page and len(t) >= min_chars]
    if not valid:
        return []
    if not use_bm25 or not HAS_BM25:
        return [page_texts[i] for i in random.sample(valid, min(n, len(valid)))]
    return _bm25_rank(query, page_texts, valid, n, bm25_cache, doc_key)


def mine_temporal_negatives(
    query: str,
    current_doc: str,
    page_cache: Dict[str, List[str]],
    n: int,
    min_chars: int = 100,
    use_bm25: bool = True,
) -> List[str]:
    """
    Pages from the SAME company's OTHER filing years.

    For AAL_2014_10K, temporal pool = pages from {AAL_2015_10K, AAL_2016_10K, ...}.

    Root cause addressed: in Exp 5, 32 of 40 wrong-doc retrieval errors (80 %)
    were same-company/different-year.  The random cross-doc negatives used in
    Exp 5 sampled from completely unrelated companies (trivially easy) and
    never exposed the model to this exact confusion pattern.

    BM25 is used to select the most query-relevant pages from the temporal pool,
    making these harder than random.
    """
    company = current_doc.split("_")[0].upper()
    temporal_docs = [
        d for d in page_cache
        if d.split("_")[0].upper() == company and d != current_doc
    ]
    if not temporal_docs:
        return []

    # Flatten all pages from temporal docs into one pool
    pool: List[str] = [
        p for doc in temporal_docs
        for p in page_cache[doc]
        if len(p) >= min_chars
    ]
    if not pool:
        return []

    if not use_bm25 or not HAS_BM25:
        return random.sample(pool, min(n, len(pool)))

    tokenized = [p.lower().split() for p in pool]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(range(len(pool)), key=lambda i: -scores[i])
    return [pool[i] for i in ranked[:n]]


def mine_cross_doc_negatives(
    query: str,
    current_doc: str,
    page_cache: Dict[str, List[str]],
    n: int,
    min_chars: int = 100,
) -> List[str]:
    """Random pages from a completely different company (kept for ablation)."""
    company = current_doc.split("_")[0].upper()
    other_docs = [d for d in page_cache if d.split("_")[0].upper() != company]
    if not other_docs:
        return []
    pool = [p for d in other_docs for p in page_cache[d] if len(p) >= min_chars]
    if not pool:
        return []
    return random.sample(pool, min(n, len(pool)))


# ─────────────────────────────────────────────────────────────────────────────
# 4.  BM25 best-chunk extraction (hierarchical loss gold chunk)
# ─────────────────────────────────────────────────────────────────────────────

def extract_best_chunk(
    query: str,
    page_text: str,
    tokenizer,
    chunk_tokens: int = 512,
    overlap_tokens: int = 64,
    min_chars: int = 50,
) -> str:
    """BM25-best 512-token chunk within page_text for hierarchical loss."""
    if not HAS_BM25 or not page_text.strip():
        return page_text
    candidates = [
        c for c in chunk_text_by_tokens(page_text, tokenizer, chunk_tokens, overlap_tokens)
        if len(c) >= min_chars
    ]
    if not candidates:
        return page_text
    if len(candidates) == 1:
        return candidates[0]
    tokenized = [c.lower().split() for c in candidates]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())
    return candidates[int(np.argmax(scores))]


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Multi-source JSONL loader with deduplication
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl_rows(jsonl_paths: List[Path]) -> List[dict]:
    """
    Load rows from one or more JSONL files.
    Deduplicates by (qid, doc_name, page_num) so overlapping files can be
    passed safely.
    """
    seen: set = set()
    rows: List[dict] = []
    for path in jsonl_paths:
        n_before = len(rows)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                ev_list = row.get("evidences_updated", [])
                if not ev_list:
                    continue
                ev = ev_list[0]
                key = (
                    row.get("qid", ""),
                    ev.get("doc_name", ""),
                    ev.get("page_num", -1),
                )
                if key not in seen:
                    seen.add(key)
                    rows.append(row)
        logger.info(
            f"  {path.name}: {len(rows) - n_before} new rows "
            f"({len(rows)} total after dedup)"
        )
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Training dataset builder
# ─────────────────────────────────────────────────────────────────────────────

def build_training_dataset(
    jsonl_paths: List[Path],
    pdf_dir: Path,
    hard_negatives: int = 3,
    temporal_negs: int = 2,
    cross_doc_negs: int = 0,
    use_hierarchical: bool = True,
    tokenizer=None,
    chunk_tokens: int = 512,
    chunk_overlap: int = 64,
    min_page_chars: int = 100,
    seed: int = 42,
) -> Dataset:
    """
    Build a HuggingFace Dataset for SentenceTransformerTrainer.

    Standard column layout (N = hard_negatives + temporal_negs + cross_doc_negs):
      sentence_0           : query
      sentence_1           : gold_page
      sentence_2..{1+N}    : page negatives

    Hierarchical column layout (adds chunk group after page group):
      ...same as above...
      sentence_{2+N}       : gold_chunk (BM25-best 512-tok chunk of gold page)
      sentence_{3+N}..{2+2N}: chunk negatives (BM25-best chunk per page neg)
    """
    if use_hierarchical and tokenizer is None:
        raise ValueError("tokenizer required when use_hierarchical=True")

    random.seed(seed)
    np.random.seed(seed)

    N = hard_negatives + temporal_negs + cross_doc_negs
    n_cols = (2 + 2 * N + 1) if use_hierarchical else (2 + N)

    # ── Pre-load all PDFs into page_cache ────────────────────────────────────
    page_cache: Dict[str, List[str]] = {}
    bm25_cache: Dict[str, "BM25Okapi"] = {}   # cached only for same-doc BM25
    logger.info(f"Pre-loading all PDFs in {pdf_dir} …")
    for pdf_file in sorted(pdf_dir.glob("*.pdf")):
        dk = pdf_file.stem
        if dk not in page_cache:
            page_cache[dk] = extract_pdf_pages(pdf_file)
    logger.info(f"  {len(page_cache)} documents loaded")

    # ── Load JSONL ────────────────────────────────────────────────────────────
    logger.info("Loading training JSONL(s) …")
    rows = load_jsonl_rows(jsonl_paths)
    logger.info(f"Total rows after dedup: {len(rows)}")

    all_texts: List[List[str]] = []
    skipped = {"bad_ev": 0, "no_pdf": 0, "bad_page": 0, "short": 0}
    source_counts = {"pdf": 0, "json": 0}

    for row in rows:
        ev_list   = row.get("evidences_updated", [])
        if not ev_list:
            skipped["bad_ev"] += 1
            continue

        ev        = ev_list[0]
        doc_name  = ev.get("doc_name", "")
        gold_page = ev.get("page_num")
        question  = row.get("question", "").strip()
        source    = ev.get("_source", "pdf")   # "pdf" or "json"

        if not question or gold_page is None or not doc_name:
            skipped["bad_ev"] += 1
            continue

        # ── Resolve gold text ─────────────────────────────────────────────
        if source == "json":
            # Gold text comes from pre_text + post_text in the JSONL — no PDF needed.
            # This enables all ~6,251 train entries (not just those with matching PDFs).
            gold_text = (ev.get("evidence_text") or "").strip()
            if len(gold_text) < min_page_chars:
                skipped["short"] += 1
                continue
            page_texts = None   # no PDF page list available
        else:
            # Tier A: PDF-verified entry — load from page_cache
            if doc_name not in page_cache:
                pdf_path = pdf_dir / f"{doc_name}.pdf"
                if not pdf_path.exists():
                    skipped["no_pdf"] += 1
                    continue
                page_cache[doc_name] = extract_pdf_pages(pdf_path)

            page_texts = page_cache[doc_name]

            if not page_texts or gold_page >= len(page_texts):
                skipped["bad_page"] += 1
                continue

            gold_text = page_texts[gold_page]
            if len(gold_text) < min_page_chars:
                skipped["short"] += 1
                continue

        source_counts[source] += 1

        # ── Mine negatives ────────────────────────────────────────────────
        # For Tier B (json-source) entries, same-doc and temporal negatives are
        # not possible (no PDF in cache).  We substitute cross-doc random negs
        # from the available PDF pool, which still provide a useful training signal.

        if source == "json" or page_texts is None:
            # All negatives come from unrelated companies
            same_negs  = []
            temp_negs  = []
            cross_negs = mine_cross_doc_negatives(
                question, doc_name, page_cache, N,
                min_chars=min_page_chars,
            )
        else:
            # ① same-doc BM25
            same_negs = mine_same_doc_negatives(
                question, page_texts, gold_page, hard_negatives,
                min_chars=min_page_chars,
                use_bm25=HAS_BM25,
                bm25_cache=bm25_cache,
                doc_key=doc_name,
            )

            # ② temporal (same company, different year, BM25-mined)
            temp_negs = mine_temporal_negatives(
                question, doc_name, page_cache, temporal_negs,
                min_chars=min_page_chars,
                use_bm25=HAS_BM25,
            ) if temporal_negs > 0 else []

            # ③ cross-doc random (different company, off by default)
            cross_negs = mine_cross_doc_negatives(
                question, doc_name, page_cache, cross_doc_negs,
                min_chars=min_page_chars,
            ) if cross_doc_negs > 0 else []

        def pad(lst: list, target: int) -> list:
            while len(lst) < target:
                lst.append("")
            return lst[:target]

        if source == "json" or page_texts is None:
            # Tier B: fill all N negative slots with cross-doc randoms
            neg_texts = pad(cross_negs, N)
        else:
            # Tier A: structured negs (same-doc + temporal + cross-doc)
            neg_texts = (
                pad(same_negs,  hard_negatives)
                + pad(temp_negs, temporal_negs)
                + pad(cross_negs, cross_doc_negs)
            )

        # ── Assemble row ──────────────────────────────────────────────────
        texts = [question, gold_text] + neg_texts   # len = 2 + N

        if use_hierarchical:
            gold_chunk = extract_best_chunk(
                question, gold_text, tokenizer, chunk_tokens, chunk_overlap
            )
            chunk_negs = [
                extract_best_chunk(question, pt, tokenizer, chunk_tokens, chunk_overlap)
                if pt else ""
                for pt in neg_texts
            ]
            texts += [gold_chunk] + chunk_negs      # total = 2 + N + 1 + N = 2+2N+1

        while len(texts) < n_cols:
            texts.append("")
        all_texts.append(texts[:n_cols])

    logger.info(f"\nDataset built:")
    logger.info(f"  Total columns     : {n_cols}")
    logger.info(f"  Positive pairs    : {len(all_texts)}")
    logger.info(f"    Tier A (pdf)    : {source_counts['pdf']}  — same-doc + temporal negs")
    logger.info(f"    Tier B (json)   : {source_counts['json']}  — cross-doc random negs (no PDF)")
    logger.info(f"  same-doc negs     : {hard_negatives} (BM25, Tier A only)")
    logger.info(f"  temporal negs     : {temporal_negs} (same company, BM25, Tier A only)")
    logger.info(f"  cross-doc negs    : {cross_doc_negs} (random, Tier A) / {N} (random, Tier B)")
    logger.info(f"  Skipped           : {skipped}")

    if not all_texts:
        logger.error("No training examples built — check JSONL paths and PDF dir.")
        sys.exit(1)

    data_dict = {
        f"sentence_{i}": [r[i] for r in all_texts]
        for i in range(n_cols)
    }
    return Dataset.from_dict(data_dict)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Hierarchical MNR loss
# ─────────────────────────────────────────────────────────────────────────────

class HierarchicalMNRLoss(nn.Module):
    """
    L = alpha * MNR(query, gold_page, neg_pages)
      + (1 - alpha) * MNR(query, gold_chunk, neg_chunks)

    Standard MNR + explicit hard negatives.  The pool for query i is:
      [all positives in batch (size B)] || [N hard negs for query i]
    Shape: [(1+N)*B, D].  Label for query i = i (gold is at column i of block 0).

    Expected column order in sentence_features:
      0          : query
      1          : gold_page
      2..{1+N}   : page negatives
      {2+N}      : gold_chunk
      {3+N}..{2+2N}: chunk negatives
    """

    def __init__(
        self,
        model: SentenceTransformer,
        n_negs: int,
        alpha: float = 0.85,
        scale: float = 25.0,
    ):
        super().__init__()
        self.model  = model
        self.n_negs = n_negs
        self.alpha  = alpha
        self.scale  = scale
        self.xent   = nn.CrossEntropyLoss()

    def _mnr(
        self,
        queries:   torch.Tensor,          # [B, D]
        positives: torch.Tensor,          # [B, D]
        hard_negs: List[torch.Tensor],    # each [B, D]
    ) -> torch.Tensor:
        pool   = torch.cat([positives] + hard_negs, dim=0)  # [(1+K)*B, D]
        q      = F.normalize(queries, dim=-1)
        c      = F.normalize(pool,    dim=-1)
        sim    = torch.mm(q, c.t()) * self.scale             # [B, (1+K)*B]
        labels = torch.arange(queries.size(0), device=queries.device)
        return self.xent(sim, labels)

    def forward(self, sentence_features, labels=None):
        expected = 1 + 1 + self.n_negs + 1 + self.n_negs
        if len(sentence_features) != expected:
            raise ValueError(
                f"Expected {expected} columns, got {len(sentence_features)}. "
                f"Check remove_unused_columns=False."
            )
        embs         = [self.model(f)["sentence_embedding"] for f in sentence_features]
        queries      = embs[0]
        page_pos     = embs[1]
        page_negs    = [embs[i] for i in range(2, 2 + self.n_negs)]
        cs           = 2 + self.n_negs
        chunk_pos    = embs[cs]
        chunk_negs   = [embs[i] for i in range(cs + 1, cs + 1 + self.n_negs)]

        return (
            self.alpha       * self._mnr(queries, page_pos,  page_negs)
            + (1 - self.alpha) * self._mnr(queries, chunk_pos, chunk_negs)
        )

    def get_config_dict(self):
        return {"n_negs": self.n_negs, "alpha": self.alpha, "scale": self.scale}


# ─────────────────────────────────────────────────────────────────────────────
# 8.  LoRA application
# ─────────────────────────────────────────────────────────────────────────────

def apply_lora(
    model: SentenceTransformer,
    r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.1,
    extended: bool = True,
) -> SentenceTransformer:
    target = _LORA_EXTENDED if extended else _LORA_BASELINE
    logger.info(
        f"LoRA: {'EXTENDED (q/k/v + attn-out + ffn-up)' if extended else 'BASELINE (q/k/v)'}"
        f"  r={r}  alpha={lora_alpha}"
    )
    cfg = LoraConfig(
        r=r, lora_alpha=lora_alpha,
        target_modules=target,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    hf_model   = model[0].auto_model
    peft_model = get_peft_model(hf_model, cfg)
    model[0].auto_model = peft_model
    trainable, total = peft_model.get_nb_trainable_parameters()
    logger.info(
        f"LoRA applied — trainable: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)"
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(
    jsonl_paths:        List[Path],
    pdf_dir:            Path,
    output_path:        Path,
    base_model:         str   = "BAAI/bge-m3",
    epochs:             int   = 3,
    batch_size:         int   = 8,
    grad_accumulation:  int   = 4,
    lr:                 float = 1e-5,
    warmup_frac:        float = 0.05,
    lora_r:             int   = 32,
    lora_alpha_lora:    int   = 64,
    lora_dropout:       float = 0.1,
    hard_negatives:     int   = 3,
    temporal_negs:      int   = 2,
    cross_doc_negs:     int   = 0,
    use_hierarchical:   bool  = True,
    extended_lora:      bool  = True,
    hierarchical_alpha: float = 0.85,
    loss_scale:         float = 25.0,
    chunk_tokens:       int   = 512,
    chunk_overlap:      int   = 64,
    val_frac:           float = 0.10,
    seed:               int   = 42,
):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    N = hard_negatives + temporal_negs + cross_doc_negs
    n_cols = (2 + 2 * N + 1) if use_hierarchical else (2 + N)

    logger.info(f"\n{'='*62}")
    logger.info(f"Experiment 6 — Temporal Hard Negatives")
    logger.info(f"{'='*62}")
    logger.info(f"  JSONL sources          : {[p.name for p in jsonl_paths]}")
    logger.info(f"  extended_lora          : {extended_lora}  (r={lora_r}, α={lora_alpha_lora})")
    logger.info(f"  hierarchical_loss      : {use_hierarchical}  (α={hierarchical_alpha})")
    logger.info(f"  loss_scale             : {loss_scale}")
    logger.info(f"  same-doc BM25 negs     : {hard_negatives}")
    logger.info(f"  temporal BM25 negs     : {temporal_negs}  ← key change from Exp 5")
    logger.info(f"  cross-doc random negs  : {cross_doc_negs}")
    logger.info(f"  total negs / row       : {N}")
    logger.info(f"  dataset columns        : {n_cols}")
    logger.info(f"  batch_size             : {batch_size} × grad_accum {grad_accumulation}"
                f" = effective {batch_size * grad_accumulation}")
    logger.info(f"  encoder calls / step   : {batch_size * n_cols}")
    logger.info(f"  lr                     : {lr}  (cosine + warmup {warmup_frac:.0%})")
    logger.info(f"  val_frac               : {val_frac}")
    logger.info(f"  epochs                 : {epochs}  (early stopping on eval_loss)")
    logger.info(f"{'='*62}")

    # ── Load model ────────────────────────────────────────────────────────────
    logger.info(f"\nLoading {base_model} …")
    model = SentenceTransformer(base_model, device=device)
    model.max_seq_length = BGE_MAX_SEQ_LENGTH
    logger.info(f"max_seq_length = {BGE_MAX_SEQ_LENGTH}")

    # ── Apply LoRA ────────────────────────────────────────────────────────────
    model = apply_lora(
        model,
        r=lora_r,
        lora_alpha=lora_alpha_lora,
        lora_dropout=lora_dropout,
        extended=extended_lora,
    )

    # ── Build dataset ─────────────────────────────────────────────────────────
    full_dataset = build_training_dataset(
        jsonl_paths,
        pdf_dir,
        hard_negatives   = hard_negatives,
        temporal_negs    = temporal_negs,
        cross_doc_negs   = cross_doc_negs,
        use_hierarchical = use_hierarchical,
        tokenizer        = model.tokenizer if use_hierarchical else None,
        chunk_tokens     = chunk_tokens,
        chunk_overlap    = chunk_overlap,
        seed             = seed,
    )

    # ── Train / val split ─────────────────────────────────────────────────────
    n_total = len(full_dataset)
    n_val   = max(1, int(n_total * val_frac))
    n_train = n_total - n_val
    split        = full_dataset.train_test_split(test_size=n_val, seed=seed)
    train_dataset = split["train"]
    val_dataset   = split["test"]
    logger.info(f"\nSplit: {n_train} train / {n_val} val")

    # ── Loss ──────────────────────────────────────────────────────────────────
    if use_hierarchical:
        train_loss = HierarchicalMNRLoss(model, n_negs=N, alpha=hierarchical_alpha, scale=loss_scale)
        logger.info(
            f"Loss: HierarchicalMNR "
            f"(α={hierarchical_alpha:.2f}×page + {1-hierarchical_alpha:.2f}×chunk, scale={loss_scale})"
        )
    else:
        train_loss = losses.MultipleNegativesRankingLoss(model, scale=loss_scale)
        logger.info(f"Loss: MNR page-level only  scale={loss_scale}")

    # ── Training args ─────────────────────────────────────────────────────────
    steps_epoch  = max(1, n_train // batch_size)
    total_steps  = steps_epoch * epochs
    warmup_steps = int(total_steps * warmup_frac)
    logger.info(f"Schedule: {steps_epoch} steps/epoch × {epochs} epochs = "
                f"{total_steps} total  ({warmup_steps} warmup, cosine decay)")

    ckpt_dir = output_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    training_args = SentenceTransformerTrainingArguments(
        output_dir                  = str(ckpt_dir),
        num_train_epochs            = epochs,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size  = batch_size,
        gradient_accumulation_steps = grad_accumulation,
        learning_rate               = lr,
        warmup_steps                = warmup_steps,
        lr_scheduler_type           = "cosine",
        fp16                        = (device == "cuda"),
        logging_steps               = 10,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        seed                        = seed,
        dataloader_drop_last        = False,
        remove_unused_columns       = False,
        gradient_checkpointing      = True,
    )

    trainer = SentenceTransformerTrainer(
        model         = model,
        args          = training_args,
        train_dataset = train_dataset,
        eval_dataset  = val_dataset,
        loss          = train_loss,
    )

    logger.info("\nStarting training …")
    trainer.train()

    # ── Save adapter ──────────────────────────────────────────────────────────
    adapter_path = output_path / "adapter"
    adapter_path.mkdir(parents=True, exist_ok=True)
    model[0].auto_model.save_pretrained(str(adapter_path))
    logger.info(f"LoRA adapter saved → {adapter_path}")

    meta = {
        "base_model":            base_model,
        "max_seq_length":        BGE_MAX_SEQ_LENGTH,
        "training_sources":      [str(p) for p in jsonl_paths],
        "epochs":                epochs,
        "batch_size":            batch_size,
        "grad_accumulation":     grad_accumulation,
        "effective_batch":       batch_size * grad_accumulation,
        "lr":                    lr,
        "lr_scheduler":          "cosine",
        "lora_r":                lora_r,
        "lora_alpha":            lora_alpha_lora,
        "extended_lora":         extended_lora,
        "use_hierarchical_loss": use_hierarchical,
        "hierarchical_alpha":    hierarchical_alpha if use_hierarchical else None,
        "loss_scale":            loss_scale,
        "hard_negatives":        hard_negatives,
        "temporal_negs":         temporal_negs,
        "cross_doc_negs":        cross_doc_negs,
        "total_negs":            N,
        "val_frac":              val_frac,
        "n_train":               n_train,
        "n_val":                 n_val,
    }
    with open(output_path / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Metadata → {output_path}/training_meta.json")


# ─────────────────────────────────────────────────────────────────────────────
# 10.  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Exp 6: BGE-M3 page scorer with temporal hard negatives"
    )
    p.add_argument("--jsonl", nargs="+", required=True,
                   help="Training JSONL(s): test + train gold pages (no FB data)")
    p.add_argument("--pdf-dir",       default="Final-PDF")
    p.add_argument("--output",        default="models/exp6_temporal")
    p.add_argument("--base-model",    default="BAAI/bge-m3")
    p.add_argument("--epochs",        type=int,   default=3)
    p.add_argument("--batch-size",    type=int,   default=8)
    p.add_argument("--grad-accum",    type=int,   default=4)
    p.add_argument("--lr",            type=float, default=1e-5)
    p.add_argument("--warmup-frac",   type=float, default=0.05)
    p.add_argument("--lora-r",        type=int,   default=32)
    p.add_argument("--lora-alpha",    type=int,   default=64)
    p.add_argument("--hard-negatives", type=int,  default=3,
                   help="Same-doc BM25 negatives per row")
    p.add_argument("--temporal-negs",  type=int,  default=2,
                   help="Same-company/different-year BM25 negatives per row")
    p.add_argument("--cross-doc-negs", type=int,  default=0,
                   help="Random different-company negatives (off by default)")
    p.add_argument("--loss-scale",    type=float, default=25.0)
    p.add_argument("--hier-alpha",    type=float, default=0.85,
                   help="Weight on page-level MNR term (1-alpha for chunk)")
    p.add_argument("--chunk-tokens",  type=int,   default=512)
    p.add_argument("--no-hierarchical",  action="store_true")
    p.add_argument("--no-extended-lora", action="store_true")
    p.add_argument("--val-frac",      type=float, default=0.10)
    p.add_argument("--seed",          type=int,   default=42)
    args = p.parse_args()

    jsonl_paths = [Path(j) for j in args.jsonl]
    for jp in jsonl_paths:
        if not jp.exists():
            logger.error(f"JSONL not found: {jp}"); sys.exit(1)

    train(
        jsonl_paths        = jsonl_paths,
        pdf_dir            = Path(args.pdf_dir),
        output_path        = Path(args.output),
        base_model         = args.base_model,
        epochs             = args.epochs,
        batch_size         = args.batch_size,
        grad_accumulation  = args.grad_accum,
        lr                 = args.lr,
        warmup_frac        = args.warmup_frac,
        lora_r             = args.lora_r,
        lora_alpha_lora    = args.lora_alpha,
        hard_negatives     = args.hard_negatives,
        temporal_negs      = args.temporal_negs,
        cross_doc_negs     = args.cross_doc_negs,
        use_hierarchical   = not args.no_hierarchical,
        extended_lora      = not args.no_extended_lora,
        hierarchical_alpha = args.hier_alpha,
        loss_scale         = args.loss_scale,
        chunk_tokens       = args.chunk_tokens,
        val_frac           = args.val_frac,
        seed               = args.seed,
    )


if __name__ == "__main__":
    main()
