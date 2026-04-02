#!/usr/bin/env python3
"""
src/training/train_finqa_page_scorer.py
========================================
Fine-tunes BAAI/BGE-M3 as a page-level bi-encoder on FinQA PDF data.

Ablation flags (all default-off to reproduce job-6000 baseline):

  --use-bm25-hard-negs
      Mine hard negatives with BM25 instead of random sampling.
      Requires:  pip install rank_bm25
      Effect:    non-gold pages with highest lexical overlap with the query
                 are selected as negatives, replacing random non-gold pages.

  --use-hierarchical-loss
      Add a chunk-level MNR term alongside the page-level MNR term.
          L = alpha * L_page + (1 - alpha) * L_chunk
      The gold chunk is the BM25-best 512-token chunk within the gold page.
      Chunk negatives are the BM25-best chunks from the page hard-neg pages,
      so they correspond 1-to-1 with the page negatives in the same batch row.

  --extended-lora
      Add the attention output projection and FFN intermediate projection
      to the LoRA target modules, beyond the baseline q/k/v.
          Baseline  : query, key, value                (~2.4M params, 0.41%)
          Extended  : query, key, value,
                      attention.output.dense,
                      intermediate.dense               (~5.1M params, 0.90%)

Ablation matrix (4 runs):
  Exp 1 : --use-hierarchical-loss
  Exp 2 : --use-bm25-hard-negs
  Exp 3 : --extended-lora
  Exp 4 : all three flags combined
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
    logger.error("sentence-transformers not found (need v3.0+)"); sys.exit(1)

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


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

BGE_MAX_SEQ_LENGTH = 2048   # matches job-6000 baseline; covers ~99% of pages

_WS = re.compile(r"\s+")


def normalise(text: str) -> str:
    return _WS.sub(" ", (text or "").strip())


# ─────────────────────────────────────────────────────────────────────────────
# 1.  PDF page extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_pdf_pages(pdf_path: Path) -> List[str]:
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
# 2.  Token-based chunking (used for hierarchical loss data prep)
# ─────────────────────────────────────────────────────────────────────────────

def chunk_text_by_tokens(
    text: str,
    tokenizer,
    chunk_tokens: int = 512,
    overlap_tokens: int = 64,
) -> List[str]:
    """Split text into overlapping chunks of chunk_tokens tokens each."""
    if not text.strip():
        return []
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return []
    step = max(1, chunk_tokens - overlap_tokens)
    chunks = []
    for start in range(0, len(ids), step):
        decoded = tokenizer.decode(
            ids[start : start + chunk_tokens], skip_special_tokens=True
        )
        if decoded.strip():
            chunks.append(decoded)
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# 3.  BM25 hard negative mining
# ─────────────────────────────────────────────────────────────────────────────

def mine_bm25_page_negatives(
    query: str,
    page_texts: List[str],
    gold_page: int,
    n: int,
    min_page_chars: int = 100,
    bm25_cache: Optional[Dict] = None,
    doc_key: Optional[str] = None,
) -> List[str]:
    """
    Return the top-n non-gold pages ranked by BM25 score against the query.

    These are 'hard' negatives: pages with high lexical overlap with the
    query that are not the gold answer page — exactly the failure mode at
    inference time.

    bm25_cache: optional shared dict keyed by doc_key.  Building BM25 over
    a document is O(n_pages) so caching avoids repeating it for every query
    that comes from the same document.
    """
    valid = [
        i for i, t in enumerate(page_texts)
        if i != gold_page and len(t) >= min_page_chars
    ]
    if not valid:
        return []
    if len(valid) <= n:
        return [page_texts[i] for i in valid]

    # Retrieve or build cached BM25 index for this document
    if bm25_cache is not None and doc_key is not None and doc_key in bm25_cache:
        bm25 = bm25_cache[doc_key]
    else:
        tokenized = [page_texts[i].lower().split() for i in range(len(page_texts))]
        bm25 = BM25Okapi(tokenized)
        if bm25_cache is not None and doc_key is not None:
            bm25_cache[doc_key] = bm25

    scores   = bm25.get_scores(query.lower().split())
    ranked   = sorted(valid, key=lambda i: -scores[i])
    return [page_texts[i] for i in ranked[:n]]


# ─────────────────────────────────────────────────────────────────────────────
# 3b.  Cross-document hard negative mining
# ─────────────────────────────────────────────────────────────────────────────

def mine_cross_doc_negatives(
    query: str,
    current_doc: str,
    page_cache: Dict[str, List[str]],
    n: int,
    min_page_chars: int = 100,
    use_bm25: bool = False,
) -> List[str]:
    """
    Sample n pages from OTHER documents as cross-doc hard negatives.

    These simulate the actual test-time scenario: during evaluation the model
    retrieves from a global FAISS index of all 84 FinanceBench docs, so pages
    from different companies/years compete directly.  Training with only
    same-doc negatives means the model never sees this interference pattern.

    use_bm25=False  random pages from random other docs  (cheap, effective)
    use_bm25=True   top-n BM25-ranked pages from the cross-doc pool  (harder)
    """
    other_docs = [d for d in page_cache if d != current_doc]
    if not other_docs:
        return []

    pool = [
        p
        for d in other_docs
        for p in page_cache[d]
        if len(p) >= min_page_chars
    ]
    if not pool:
        return []

    if not use_bm25 or not HAS_BM25:
        return random.sample(pool, min(n, len(pool)))

    tokenized = [p.lower().split() for p in pool]
    bm25      = BM25Okapi(tokenized)
    scores    = bm25.get_scores(query.lower().split())
    ranked    = sorted(range(len(pool)), key=lambda i: -scores[i])
    return [pool[i] for i in ranked[:n]]


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Chunk-level gold/negative extraction (for hierarchical loss)
# ─────────────────────────────────────────────────────────────────────────────

def extract_best_chunk(
    query: str,
    page_text: str,
    tokenizer,
    chunk_tokens: int = 512,
    overlap_tokens: int = 64,
    min_chunk_chars: int = 50,
) -> str:
    """
    Return the chunk of page_text that best matches query by BM25 score.
    Falls back to the full page text if chunking produces nothing usable.
    """
    candidates = [
        c for c in chunk_text_by_tokens(page_text, tokenizer, chunk_tokens, overlap_tokens)
        if len(c) >= min_chunk_chars
    ]
    if not candidates:
        return page_text
    if len(candidates) == 1:
        return candidates[0]

    tokenized = [c.lower().split() for c in candidates]
    bm25      = BM25Okapi(tokenized)
    scores    = bm25.get_scores(query.lower().split())
    return candidates[int(np.argmax(scores))]


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Training dataset builder
# ─────────────────────────────────────────────────────────────────────────────

def build_training_dataset(
    jsonl_path: Path,
    pdf_dir: Path,
    hard_negatives_per_positive: int = 3,
    cross_doc_negs_per_positive: int = 2,
    min_page_chars: int = 100,
    use_bm25: bool = False,
    use_hierarchical: bool = False,
    tokenizer=None,
    chunk_tokens: int = 512,
    chunk_overlap: int = 64,
    seed: int = 42,
    query_prefix: str = "",
) -> Dataset:
    """
    Build a HuggingFace Dataset for SentenceTransformerTrainer.

    Standard mode  — columns per row:
      sentence_0              query
      sentence_1              gold_page
      sentence_2..{1+N}       page hard negatives  (N = hard_negatives_per_positive
                                                     same-doc  +  cross_doc_negs_per_positive
                                                     cross-doc = N_total negatives)

    Hierarchical mode — columns per row (standard + chunk group):
      sentence_0              query
      sentence_1              gold_page
      sentence_2..{1+N}       page hard negatives  (N_total columns)
      sentence_{2+N}          gold_chunk  (BM25-best 512-tok chunk of gold page)
      sentence_{3+N}..{2+2N}  chunk negatives (BM25-best chunk of each page neg)

    Page hard negatives:
      use_bm25=False  random same-doc pages + random cross-doc pages
      use_bm25=True   BM25-mined same-doc pages + BM25-mined cross-doc pages

    cross_doc_negs_per_positive (default 2):
      Pages from OTHER documents added to the negative set.  This closes the
      train/test gap: at inference the model searches a global index of all
      docs, so cross-doc pages are the primary source of false positives.
      Set to 0 to disable and reproduce the original same-doc-only behaviour.

    Chunk negatives always use BM25 within each candidate page, regardless
    of whether use_bm25 is set for the page negatives.
    """
    if use_hierarchical and tokenizer is None:
        raise ValueError("tokenizer must be provided when use_hierarchical=True")
    if use_bm25 and not HAS_BM25:
        logger.error(
            "--use-bm25-hard-negs requires rank_bm25.  "
            "Install with:  pip install rank_bm25"
        )
        sys.exit(1)

    random.seed(seed)
    np.random.seed(seed)

    N_same  = hard_negatives_per_positive
    N_cross = cross_doc_negs_per_positive
    N       = N_same + N_cross          # total negatives per row
    if use_hierarchical:
        n_cols = 2 + 2 * N + 1          # query, page_pos, N page_negs, chunk_pos, N chunk_negs
    else:
        n_cols = 2 + N                  # query, page_pos, N page_negs

    page_cache: Dict[str, List[str]] = {}
    bm25_cache: Dict[str, "BM25Okapi"] = {}  # populated only when use_bm25=True
    all_texts:  List[List[str]] = []
    skipped = {"no_update": 0, "no_pdf": 0, "bad_page": 0}

    # ── Pre-load ALL PDFs in pdf_dir so cross-doc negatives can sample from them
    if N_cross > 0:
        logger.info(f"Pre-loading all PDFs in {pdf_dir} for cross-doc negative pool ...")
        for pdf_file in sorted(pdf_dir.glob("*.pdf")):
            dk = pdf_file.stem
            if dk not in page_cache:
                page_cache[dk] = extract_pdf_pages(pdf_file)
        logger.info(f"  {len(page_cache)} documents pre-loaded")

    logger.info(f"Loading training data from {jsonl_path} ...")
    with open(jsonl_path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    logger.info(f"  {len(rows)} rows loaded")

    for row in rows:
        ev_list  = row.get("evidences_updated", [])
        if not ev_list:
            skipped["no_update"] += 1
            continue

        ev        = ev_list[0]
        doc_name  = ev.get("doc_name", "")
        gold_page = ev.get("page_num")
        question  = row.get("question", "").strip()

        if not question or gold_page is None or not doc_name:
            skipped["no_update"] += 1
            continue

        pdf_path = pdf_dir / f"{doc_name}.pdf"
        if not pdf_path.exists():
            skipped["no_pdf"] += 1
            continue

        if doc_name not in page_cache:
            page_cache[doc_name] = extract_pdf_pages(pdf_path)
        page_texts = page_cache[doc_name]

        if not page_texts or gold_page >= len(page_texts):
            skipped["bad_page"] += 1
            continue

        gold_text = page_texts[gold_page]
        if len(gold_text) < min_page_chars:
            skipped["bad_page"] += 1
            continue

        # ── Sample same-doc page hard negatives ──────────────────────────
        if use_bm25:
            same_doc_negs = mine_bm25_page_negatives(
                question, page_texts, gold_page, N_same,
                min_page_chars=min_page_chars,
                bm25_cache=bm25_cache,
                doc_key=doc_name,
            )
        else:
            neg_indices = [
                i for i, t in enumerate(page_texts)
                if i != gold_page and len(t) >= min_page_chars
            ]
            k = min(N_same, len(neg_indices))
            same_doc_negs = [page_texts[i] for i in random.sample(neg_indices, k)] if k else []

        while len(same_doc_negs) < N_same:
            same_doc_negs.append("")

        # ── Sample cross-doc page hard negatives (new) ────────────────────
        if N_cross > 0:
            cross_negs = mine_cross_doc_negatives(
                question, doc_name, page_cache, N_cross,
                min_page_chars=min_page_chars,
                use_bm25=use_bm25,
            )
        else:
            cross_negs = []

        while len(cross_negs) < N_cross:
            cross_negs.append("")

        neg_page_texts = same_doc_negs + cross_negs   # total length = N

        # Pad to exactly N entries with empty strings (edge-case guard)
        while len(neg_page_texts) < N:
            neg_page_texts.append("")

        # ── Build row ─────────────────────────────────────────────────────
        query_text = f"{query_prefix}{question}" if query_prefix else question
        texts = [query_text, gold_text] + neg_page_texts   # len = 2 + N

        if use_hierarchical:
            # Gold chunk: BM25-best chunk within the gold page
            gold_chunk = extract_best_chunk(
                question, gold_text, tokenizer, chunk_tokens, chunk_overlap
            )
            # Chunk negatives: BM25-best chunk from each page negative
            # An empty page neg produces an empty string (ignored by loss as noise)
            chunk_negs = [
                extract_best_chunk(question, pt, tokenizer, chunk_tokens, chunk_overlap)
                if pt else ""
                for pt in neg_page_texts
            ]
            texts += [gold_chunk] + chunk_negs            # len = 2 + N + 1 + N = 2+2N+1

        # Pad to fixed column width (handles edge cases)
        while len(texts) < n_cols:
            texts.append("")
        all_texts.append(texts[:n_cols])

    logger.info(f"\nTraining set summary:")
    logger.info(f"  Mode                   : {'hierarchical' if use_hierarchical else 'standard'}")
    logger.info(f"  Negatives              : {'BM25-mined' if use_bm25 else 'random'}")
    logger.info(f"  Positive pairs         : {len(all_texts)}")
    logger.info(f"  Same-doc negs / pair   : {N_same}")
    logger.info(f"  Cross-doc negs / pair  : {N_cross}")
    logger.info(f"  Total negs / pair      : {N}")
    logger.info(f"  Total columns          : {n_cols}")
    logger.info(f"  Unique docs cached     : {len(page_cache)}")
    logger.info(f"  Skipped (no update)    : {skipped['no_update']}")
    logger.info(f"  Skipped (no PDF)       : {skipped['no_pdf']}")
    logger.info(f"  Skipped (bad page)     : {skipped['bad_page']}")

    if not all_texts:
        logger.error("No training examples built — check JSONL and PDF paths.")
        sys.exit(1)

    data_dict = {
        f"sentence_{i}": [row[i] for row in all_texts]
        for i in range(n_cols)
    }
    dataset = Dataset.from_dict(data_dict)
    logger.info(f"  Dataset columns        : {dataset.column_names}")
    logger.info(f"  Dataset size           : {len(dataset)}")
    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Hierarchical MNR loss
# ─────────────────────────────────────────────────────────────────────────────

class HierarchicalMNRLoss(nn.Module):
    """
    Combined page-level + chunk-level MultipleNegativesRanking loss.

    Dataset column layout (with N hard negatives per level):
      sentence_0           : query
      sentence_1           : gold_page
      sentence_2..{1+N}    : page hard negatives     (N columns)
      sentence_{2+N}       : gold_chunk
      sentence_{3+N}..{2+2N}: chunk hard negatives   (N columns)

    Loss:
      L = alpha * MNR(query, gold_page, neg_pages)
        + (1-alpha) * MNR(query, gold_chunk, neg_chunks)

    Both MNR terms use in-batch negatives (standard MNR contract) plus the
    N explicit hard negatives for each level.

    Similarity matrix for each term: [B, B*(1+N)]
      Columns 0..B-1      : in-batch positives   (positive for query i = col i)
      Columns B..{B+N*B-1}: explicit hard negatives stacked column-block-wise
    """

    def __init__(
        self,
        model: SentenceTransformer,
        alpha: float = 0.7,
        scale: float = 20.0,
        n_page_negs: int = 3,
        n_chunk_negs: int = 3,
    ):
        super().__init__()
        self.model       = model
        self.alpha       = alpha
        self.scale       = scale
        self.n_page_negs = n_page_negs
        self.n_chunk_negs = n_chunk_negs
        self.cross_entropy = nn.CrossEntropyLoss()

    def _mnr_loss(
        self,
        queries:   torch.Tensor,          # [B, D]
        positives: torch.Tensor,          # [B, D]
        hard_negs: List[torch.Tensor],    # each [B, D], len = n_hard
    ) -> torch.Tensor:
        """
        Standard MNR + explicit hard negatives.

        The candidate pool is [positives || hard_neg_0 || ... || hard_neg_K].
        Its shape is [(1+K)*B, D].  For query i, the gold is at index i in
        the first block — cross-entropy with label=i recovers MNR.
        Hard negatives for query i lie at B+i, 2B+i, ... — treated as
        additional negatives automatically (no special indexing needed).
        """
        candidates = torch.cat([positives] + hard_negs, dim=0)  # [(1+K)*B, D]
        q = F.normalize(queries,    dim=-1)
        c = F.normalize(candidates, dim=-1)
        sim    = torch.mm(q, c.t()) * self.scale                 # [B, (1+K)*B]
        labels = torch.arange(queries.size(0), device=queries.device)
        return self.cross_entropy(sim, labels)

    def forward(self, sentence_features, labels=None):
        # Validate column count on first call
        expected = 1 + 1 + self.n_page_negs + 1 + self.n_chunk_negs
        if len(sentence_features) != expected:
            raise ValueError(
                f"HierarchicalMNRLoss expected {expected} sentence columns "
                f"(1 query + 1 page_pos + {self.n_page_negs} page_negs + "
                f"1 chunk_pos + {self.n_chunk_negs} chunk_negs), "
                f"got {len(sentence_features)}.  "
                f"Check remove_unused_columns=False and dataset column count."
            )

        # Encode each column; each emb is [B, D]
        embs = [self.model(f)["sentence_embedding"] for f in sentence_features]

        queries  = embs[0]
        page_pos = embs[1]
        page_neg_embs = [embs[i] for i in range(2, 2 + self.n_page_negs)]

        chunk_start   = 2 + self.n_page_negs
        chunk_pos     = embs[chunk_start]
        chunk_neg_embs = [
            embs[i] for i in range(chunk_start + 1, chunk_start + 1 + self.n_chunk_negs)
        ]

        page_loss  = self._mnr_loss(queries, page_pos,  page_neg_embs)
        chunk_loss = self._mnr_loss(queries, chunk_pos, chunk_neg_embs)

        total = self.alpha * page_loss + (1.0 - self.alpha) * chunk_loss
        return total

    def get_config_dict(self) -> dict:
        return {
            "alpha":        self.alpha,
            "scale":        self.scale,
            "n_page_negs":  self.n_page_negs,
            "n_chunk_negs": self.n_chunk_negs,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Apply LoRA
# ─────────────────────────────────────────────────────────────────────────────

# PEFT uses suffix matching for target_modules: a module is targeted when
# its full dotted path ends with any of the listed strings.
#
# BGE-M3 (XLM-RoBERTa-based) module names per layer:
#   encoder.layer.{i}.attention.self.query         ← matched by "query"
#   encoder.layer.{i}.attention.self.key            ← matched by "key"
#   encoder.layer.{i}.attention.self.value          ← matched by "value"
#   encoder.layer.{i}.attention.output.dense        ← matched by "attention.output.dense"
#   encoder.layer.{i}.intermediate.dense            ← matched by "intermediate.dense"
#
# "output.dense" (without prefix) would additionally match the FFN down-
# projection encoder.layer.{i}.output.dense.  We deliberately exclude it
# to keep the extended config conservative.

_LORA_BASELINE  = ["query", "key", "value"]
_LORA_EXTENDED  = ["query", "key", "value", "attention.output.dense", "intermediate.dense"]


def apply_lora(
    model: SentenceTransformer,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    extended: bool = False,
) -> SentenceTransformer:
    """
    Apply LoRA adapters to the underlying XLM-RoBERTa model inside BGE-M3.

    Baseline  (extended=False):  q/k/v attention projections  → ~2.4M params (0.41%)
    Extended  (extended=True):   + attention-out + FFN-up     → ~5.1M params (0.90%)
    """
    target_modules = _LORA_EXTENDED if extended else _LORA_BASELINE
    mode_str       = "EXTENDED (q/k/v + attn-out + ffn-intermediate)" if extended \
                     else "BASELINE (q/k/v only)"
    logger.info(f"LoRA mode: {mode_str}")

    lora_config = LoraConfig(
        r              = r,
        lora_alpha     = lora_alpha,
        target_modules = target_modules,
        lora_dropout   = lora_dropout,
        bias           = "none",
        task_type      = TaskType.FEATURE_EXTRACTION,
    )
    hf_model   = model[0].auto_model
    peft_model = get_peft_model(hf_model, lora_config)
    model[0].auto_model = peft_model

    trainable, total = peft_model.get_nb_trainable_parameters()
    logger.info(
        f"LoRA applied — trainable: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)"
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Train
# ─────────────────────────────────────────────────────────────────────────────

def train(
    jsonl_path:           Path,
    pdf_dir:              Path,
    output_path:          Path,
    base_model:           str   = "BAAI/bge-m3",
    epochs:               int   = 10,
    batch_size:           int   = 8,
    grad_accumulation:    int   = 4,
    lr:                   float = 2e-5,
    warmup_frac:          float = 0.1,
    lora_r:               int   = 16,
    lora_alpha:           int   = 32,
    lora_dropout:         float = 0.1,
    hard_negatives:       int   = 3,
    cross_doc_negs:       int   = 2,
    loss_scale:           float = 50.0,
    # ── Ablation flags ────────────────────────────────────────────────────────
    use_bm25_hard_negs:   bool  = False,
    use_hierarchical_loss: bool = False,
    extended_lora:        bool  = False,
    # ── Hierarchical loss hyperparameters ─────────────────────────────────────
    hierarchical_alpha:   float = 0.9,
    chunk_tokens:         int   = 512,
    chunk_overlap:        int   = 64,
    seed:                 int   = 42,
    query_prefix:         str   = "",
):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    effective_batch = batch_size * grad_accumulation
    total_negs = hard_negatives + cross_doc_negs
    n_cols = (2 + 2 * total_negs + 1) if use_hierarchical_loss else (2 + total_negs)

    logger.info(f"\n{'='*60}")
    logger.info(f"Ablation configuration:")
    logger.info(f"  use_bm25_hard_negs      : {use_bm25_hard_negs}")
    logger.info(f"  use_hierarchical_loss   : {use_hierarchical_loss}")
    logger.info(f"  extended_lora           : {extended_lora}")
    logger.info(f"  cross_doc_negs          : {cross_doc_negs}")
    logger.info(f"  loss_scale              : {loss_scale}")
    logger.info(f"  query_prefix            : {repr(query_prefix) if query_prefix else '(none)'}")
    if use_hierarchical_loss:
        logger.info(f"  hierarchical_alpha      : {hierarchical_alpha}")
        logger.info(f"  chunk_tokens            : {chunk_tokens}")
    logger.info(f"\nMemory configuration:")
    logger.info(f"  max_seq_length          : {BGE_MAX_SEQ_LENGTH} tokens")
    logger.info(f"  batch_size              : {batch_size} (per-step micro-batch)")
    logger.info(f"  grad_accumulation       : {grad_accumulation}")
    logger.info(f"  effective_batch         : {effective_batch}")
    logger.info(f"  same_doc_negs           : {hard_negatives}")
    logger.info(f"  cross_doc_negs          : {cross_doc_negs}")
    logger.info(f"  total_negs              : {total_negs}")
    logger.info(f"  dataset_columns         : {n_cols}")
    logger.info(f"  encoder_calls_per_step  : {batch_size * n_cols}")
    logger.info(f"{'='*60}")

    # ── Load base model ──────────────────────────────────────────────────────
    logger.info(f"\nLoading base model: {base_model} ...")
    model = SentenceTransformer(base_model, device=device)
    model.max_seq_length = BGE_MAX_SEQ_LENGTH
    logger.info(f"max_seq_length set to {BGE_MAX_SEQ_LENGTH}")

    # ── Apply LoRA ───────────────────────────────────────────────────────────
    model = apply_lora(
        model,
        r            = lora_r,
        lora_alpha   = lora_alpha,
        lora_dropout = lora_dropout,
        extended     = extended_lora,
    )

    # ── Build training data ──────────────────────────────────────────────────
    # Pass the tokenizer for chunk-based column construction when hierarchical.
    train_dataset = build_training_dataset(
        jsonl_path,
        pdf_dir,
        hard_negatives_per_positive = hard_negatives,
        cross_doc_negs_per_positive = cross_doc_negs,
        use_bm25                    = use_bm25_hard_negs,
        use_hierarchical            = use_hierarchical_loss,
        tokenizer                   = model.tokenizer if use_hierarchical_loss else None,
        chunk_tokens                = chunk_tokens,
        chunk_overlap               = chunk_overlap,
        seed                        = seed,
        query_prefix                = query_prefix,
    )

    # ── Select loss ──────────────────────────────────────────────────────────
    if use_hierarchical_loss:
        train_loss = HierarchicalMNRLoss(
            model,
            alpha        = hierarchical_alpha,
            scale        = loss_scale,
            n_page_negs  = total_negs,
            n_chunk_negs = total_negs,
        )
        logger.info(
            f"\nLoss: HierarchicalMNR  "
            f"(alpha={hierarchical_alpha:.2f} * L_page  +  "
            f"{1 - hierarchical_alpha:.2f} * L_chunk,  scale={loss_scale})"
        )
    else:
        train_loss = losses.MultipleNegativesRankingLoss(model, scale=loss_scale)
        logger.info(f"\nLoss: MultipleNegativesRanking (page-level only, scale={loss_scale})")

    # ── Training args ────────────────────────────────────────────────────────
    n_examples   = len(train_dataset)
    steps_epoch  = max(1, n_examples // batch_size)
    total_steps  = steps_epoch * epochs
    warmup_steps = int(total_steps * warmup_frac)

    logger.info(f"\nTraining configuration:")
    logger.info(f"  Examples             : {n_examples}")
    logger.info(f"  Batch size           : {batch_size} (per-step micro-batch)")
    logger.info(f"  Effective batch      : {effective_batch}")
    logger.info(f"  Grad accumulation    : {grad_accumulation}")
    logger.info(f"  Epochs               : {epochs}")
    logger.info(f"  Steps / epoch        : {steps_epoch}")
    logger.info(f"  Total steps          : {total_steps}")
    logger.info(f"  Warmup steps         : {warmup_steps}")
    logger.info(f"  Learning rate        : {lr}")
    logger.info(f"  max_seq_length       : {BGE_MAX_SEQ_LENGTH}")

    checkpoint_dir = output_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    training_args = SentenceTransformerTrainingArguments(
        output_dir                  = str(checkpoint_dir),
        num_train_epochs            = epochs,
        per_device_train_batch_size = batch_size,
        gradient_accumulation_steps = grad_accumulation,
        learning_rate               = lr,
        warmup_steps                = warmup_steps,
        fp16                        = (device == "cuda"),
        logging_steps               = 10,
        save_strategy               = "no",
        seed                        = seed,
        dataloader_drop_last        = False,
        remove_unused_columns       = False,   # must be False to keep all sentence_i columns
        gradient_checkpointing      = True,
    )

    trainer = SentenceTransformerTrainer(
        model         = model,
        args          = training_args,
        train_dataset = train_dataset,
        loss          = train_loss,
    )

    logger.info("\nStarting training ...")
    trainer.train()
    logger.info("Training complete.")

    # ── Save adapter + tokenizer + metadata ──────────────────────────────────
    output_path.mkdir(parents=True, exist_ok=True)
    adapter_path = output_path / "adapter"
    adapter_path.mkdir(exist_ok=True)

    model[0].auto_model.save_pretrained(str(adapter_path))
    model.tokenizer.save_pretrained(str(output_path / "tokenizer"))

    meta = {
        "base_model":                  base_model,
        "max_seq_length":              BGE_MAX_SEQ_LENGTH,
        "training_data":               str(jsonl_path),
        "epochs":                      epochs,
        "batch_size":                  batch_size,
        "grad_accumulation":           grad_accumulation,
        "effective_batch":             effective_batch,
        "lr":                          lr,
        "lora_r":                      lora_r,
        "lora_alpha":                  lora_alpha,
        "hard_negatives_per_positive":  hard_negatives,
        "cross_doc_negs_per_positive":  cross_doc_negs,
        "total_negs_per_positive":      total_negs,
        "loss_scale":                   loss_scale,
        "n_training_examples":          n_examples,
        "ablation": {
            "use_bm25_hard_negs":    use_bm25_hard_negs,
            "use_hierarchical_loss": use_hierarchical_loss,
            "hierarchical_alpha":    hierarchical_alpha if use_hierarchical_loss else None,
            "extended_lora":         extended_lora,
            "lora_target_modules":   _LORA_EXTENDED if extended_lora else _LORA_BASELINE,
            "chunk_tokens":          chunk_tokens if use_hierarchical_loss else None,
        },
    }
    with open(output_path / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    total_bytes = sum(p.stat().st_size for p in adapter_path.rglob("*") if p.is_file())
    logger.info(f"\n✓ LoRA adapter saved to: {adapter_path}")
    logger.info(f"  Adapter size : {total_bytes / 1e6:.1f} MB")
    logger.info(f"  Metadata     : {output_path / 'training_meta.json'}")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="LoRA fine-tune BGE-M3 page scorer — ablation mode"
    )
    p.add_argument("--jsonl",            default="data/finqa_test_gold_pages.jsonl")
    p.add_argument("--pdf-dir",          default="Final-PDF")
    p.add_argument("--output",           default="models/finqa_page_scorer_lora")
    p.add_argument("--base-model",       default="BAAI/bge-m3")
    p.add_argument("--epochs",           type=int,   default=10)
    p.add_argument("--batch-size",       type=int,   default=8,
                   help="Per-step micro-batch size.  Use 4 for hierarchical loss.")
    p.add_argument("--grad-accum",       type=int,   default=4,
                   help="Gradient accumulation steps.  Use 8 for hierarchical loss.")
    p.add_argument("--lr",               type=float, default=2e-5)
    p.add_argument("--lora-r",           type=int,   default=16)
    p.add_argument("--lora-alpha",       type=int,   default=32)
    p.add_argument("--hard-negatives",   type=int,   default=3,
                   help="Same-doc hard negatives per positive (page level; also chunk level)")
    p.add_argument("--cross-doc-negs",   type=int,   default=2,
                   help="Cross-doc hard negatives per positive.  These pages come from OTHER "
                        "documents and simulate the global-index interference seen at eval time. "
                        "Set 0 to disable (reproduces original same-doc-only behaviour).")
    p.add_argument("--loss-scale",       type=float, default=50.0,
                   help="Temperature scale for MNR cosine logits (default 50.0 vs old 20.0). "
                        "Higher values sharpen the softmax when cosine scores are compressed "
                        "near 1.0, giving the model a clearer gradient signal.")
    p.add_argument("--hierarchical-alpha", type=float, default=0.9,
                   help="Weight for page loss in hierarchical mode (1-alpha = chunk weight). "
                        "Default raised from 0.7 to 0.9 since the task is page retrieval.")
    p.add_argument("--chunk-tokens",     type=int,   default=512,
                   help="Chunk size for gold-chunk extraction in hierarchical mode")
    p.add_argument("--chunk-overlap",    type=int,   default=64)
    p.add_argument("--seed",             type=int,   default=42)
    # ── Ablation flags ────────────────────────────────────────────────────────
    p.add_argument("--use-bm25-hard-negs",    action="store_true",
                   help="Mine hard negatives with BM25 (requires: pip install rank_bm25)")
    p.add_argument("--use-hierarchical-loss", action="store_true",
                   help="Add chunk-level MNR loss alongside page-level MNR")
    p.add_argument("--extended-lora",         action="store_true",
                   help="Extend LoRA to attn-output + FFN-intermediate projections")
    p.add_argument("--query-prefix",      type=str, default="",
                   help="Instruction prefix prepended to every training query before encoding. "
                        "Must match the prefix used at eval time. "
                        "BGE-M3 recommended: 'Represent this sentence for searching relevant passages: '")

    args = p.parse_args()

    for path, label in [(Path(args.jsonl), "--jsonl"), (Path(args.pdf_dir), "--pdf-dir")]:
        if not path.exists():
            logger.error(f"{label} not found: {path}"); sys.exit(1)

    train(
        jsonl_path            = Path(args.jsonl),
        pdf_dir               = Path(args.pdf_dir),
        output_path           = Path(args.output),
        base_model            = args.base_model,
        epochs                = args.epochs,
        batch_size            = args.batch_size,
        grad_accumulation     = args.grad_accum,
        lr                    = args.lr,
        lora_r                = args.lora_r,
        lora_alpha            = args.lora_alpha,
        hard_negatives        = args.hard_negatives,
        cross_doc_negs        = args.cross_doc_negs,
        loss_scale            = args.loss_scale,
        use_bm25_hard_negs    = args.use_bm25_hard_negs,
        use_hierarchical_loss = args.use_hierarchical_loss,
        extended_lora         = args.extended_lora,
        hierarchical_alpha    = args.hierarchical_alpha,
        chunk_tokens          = args.chunk_tokens,
        chunk_overlap         = args.chunk_overlap,
        seed                  = args.seed,
        query_prefix          = args.query_prefix,
    )


if __name__ == "__main__":
    main()