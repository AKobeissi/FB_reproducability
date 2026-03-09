#!/usr/bin/env python3
"""
chunking_strategies.py
======================
Nine chunking strategies for financial RAG, all token-based using the BGE-M3
tokenizer for fair comparison.  Each strategy is a callable that takes page-level
text (a list of (page_num, text) tuples from a single document) and returns a
list of Chunk dataclass instances.

Strategies
----------
1. naive          – Fixed-size sliding window (baseline)
2. recursive      – Recursive splits on structural separators
3. semantic       – Merge adjacent sentences while cosine-sim > threshold
4. adaptive       – Content-density-aware dynamic sizing
5. parent_child   – Hierarchical: index children, retrieve parents
6. table_aware    – Detect tables, keep them atomic
7. late           – Standard chunk boundaries; embeddings from long-context model
8. contextual     – Prepend doc/section/page context to each chunk text
9. metadata       – Fixed chunks enriched with structured metadata fields

All strategies share a common `Chunk` dataclass and use tokens as the unit.
"""

from __future__ import annotations

import re
import math
import logging
from dataclasses import dataclass, field
from typing import (
    Callable, Dict, List, Optional, Sequence, Tuple, Union,
)

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tokenizer singleton (lazy-loaded)
# ---------------------------------------------------------------------------

_TOKENIZER = None
_TOKENIZER_NAME = "BAAI/bge-m3"


def _get_tokenizer(name: str | None = None):
    global _TOKENIZER, _TOKENIZER_NAME
    name = name or _TOKENIZER_NAME
    if _TOKENIZER is None or name != _TOKENIZER_NAME:
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained(name, use_fast=True)
        _TOKENIZER_NAME = name
    return _TOKENIZER


def _tokenize(text: str, tokenizer_name: str | None = None) -> List[int]:
    tok = _get_tokenizer(tokenizer_name)
    return tok.encode(text, add_special_tokens=False)


def _decode(ids: List[int], tokenizer_name: str | None = None) -> str:
    tok = _get_tokenizer(tokenizer_name)
    return tok.decode(ids, skip_special_tokens=True)


def _count_tokens(text: str, tokenizer_name: str | None = None) -> int:
    return len(_tokenize(text, tokenizer_name))


# ---------------------------------------------------------------------------
# Chunk dataclass
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A single chunk produced by any strategy."""
    text: str                          # the chunk content (possibly with context prefix)
    raw_text: str                      # the chunk content WITHOUT any prefix
    doc_id: str = ""                   # document identifier (filename / ticker)
    page_nums: List[int] = field(default_factory=list)  # source page(s)
    token_count: int = 0               # tokens in `text`
    raw_token_count: int = 0           # tokens in `raw_text` only
    strategy: str = ""                 # which chunker produced this
    chunk_index: int = 0               # ordinal within the document
    parent_chunk_index: int = -1       # for parent_child: index of the parent
    # Metadata fields (populated by metadata / contextual strategies)
    metadata: Dict[str, Union[str, int, float, bool]] = field(default_factory=dict)

    def __post_init__(self):
        if self.token_count == 0:
            self.token_count = _count_tokens(self.text)
        if self.raw_token_count == 0:
            self.raw_token_count = _count_tokens(self.raw_text)


# ---------------------------------------------------------------------------
# Helper: sentence splitting
# ---------------------------------------------------------------------------

_SENT_RE = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z])'        # simple sentence boundary
    r'|(?<=\n)\s*(?=\S)'              # newline boundary
)


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences.  Uses regex for speed; falls back to
    nltk.sent_tokenize if available."""
    try:
        import nltk
        nltk.data.find("tokenizers/punkt_tab")
        return nltk.sent_tokenize(text)
    except Exception:
        parts = _SENT_RE.split(text)
        return [s.strip() for s in parts if s.strip()]


# ---------------------------------------------------------------------------
# Helper: table detection heuristic
# ---------------------------------------------------------------------------

_TABLE_PATTERNS = [
    # Lines with multiple columns separated by 2+ spaces or tabs
    re.compile(r"^.{5,}\s{2,}.{5,}\s{2,}.{5,}", re.MULTILINE),
    # Lines with $ signs and numbers aligned
    re.compile(r"^\s*\$?\s*[\d,]+\.?\d*\s{2,}", re.MULTILINE),
    # Markdown-style tables
    re.compile(r"^\|.*\|.*\|", re.MULTILINE),
    # Lines that are mostly digits/punctuation (financial rows)
    re.compile(r"^[\s\d$%,.()\-—–]{20,}$", re.MULTILINE),
]


def _detect_table_regions(text: str) -> List[Tuple[int, int]]:
    """Return (start, end) char offsets of probable table regions."""
    lines = text.split("\n")
    table_lines = set()
    for i, line in enumerate(lines):
        for pat in _TABLE_PATTERNS:
            if pat.search(line):
                table_lines.add(i)
                break

    # Merge consecutive table lines into regions (with 1-line tolerance)
    regions = []
    if not table_lines:
        return regions
    sorted_lines = sorted(table_lines)
    start = sorted_lines[0]
    prev = start
    for ln in sorted_lines[1:]:
        if ln - prev <= 2:  # tolerance for blank lines within tables
            prev = ln
        else:
            regions.append((start, prev))
            start = ln
            prev = ln
    regions.append((start, prev))

    # Convert line numbers to char offsets
    line_offsets = []
    pos = 0
    for line in lines:
        line_offsets.append(pos)
        pos += len(line) + 1  # +1 for \n
    char_regions = []
    for s, e in regions:
        if e - s + 1 >= 3:  # at least 3 lines to count as table
            c_start = line_offsets[s]
            c_end = line_offsets[min(e + 1, len(lines) - 1)] if e + 1 < len(lines) else len(text)
            char_regions.append((c_start, c_end))
    return char_regions


# ---------------------------------------------------------------------------
# Helper: section header detection for financial docs
# ---------------------------------------------------------------------------

_SECTION_HEADER_RE = re.compile(
    r"^(?:ITEM\s+\d+[A-Z]?\.?|PART\s+[IV]+|"
    r"Management.s Discussion|Notes to (?:Consolidated )?Financial Statements|"
    r"Risk Factors|Selected Financial Data|"
    r"Quantitative and Qualitative Disclosures|"
    r"Financial Statements and Supplementary Data|"
    r"Report of Independent Registered|"
    r"Consolidated (?:Balance Sheets?|Statements? of))",
    re.IGNORECASE | re.MULTILINE,
)


def _detect_section_header(text: str) -> str:
    """Return the first section header found in text, or ''."""
    m = _SECTION_HEADER_RE.search(text)
    return m.group(0).strip() if m else ""


# ===================================================================
# STRATEGY 1: Naive / Fixed-size sliding window
# ===================================================================

def chunk_naive(
    pages: List[Tuple[int, str]],
    doc_id: str = "",
    chunk_size: int = 1024,
    chunk_overlap: int = 128,
    tokenizer_name: str | None = None,
) -> List[Chunk]:
    """Fixed-size token window with overlap."""
    full_text = ""
    page_char_ranges: List[Tuple[int, int, int]] = []  # (start, end, page_num)
    for pnum, ptxt in pages:
        start = len(full_text)
        full_text += ptxt + "\n"
        page_char_ranges.append((start, len(full_text), pnum))

    tokens = _tokenize(full_text, tokenizer_name)
    chunks: List[Chunk] = []
    step = max(1, chunk_size - chunk_overlap)
    idx = 0

    for start in range(0, len(tokens), step):
        end = min(start + chunk_size, len(tokens))
        chunk_ids = tokens[start:end]
        text = _decode(chunk_ids, tokenizer_name)

        # Map token span back to page numbers
        # Approximate: decode start/end to find char positions
        prefix_text = _decode(tokens[:start], tokenizer_name)
        chunk_char_start = len(prefix_text)
        chunk_char_end = chunk_char_start + len(text)
        pnums = _pages_for_char_range(chunk_char_start, chunk_char_end, page_char_ranges)

        chunks.append(Chunk(
            text=text, raw_text=text, doc_id=doc_id,
            page_nums=pnums, token_count=len(chunk_ids),
            raw_token_count=len(chunk_ids),
            strategy="naive", chunk_index=idx,
        ))
        idx += 1
        if end >= len(tokens):
            break

    return chunks


def _pages_for_char_range(
    c_start: int, c_end: int, page_ranges: List[Tuple[int, int, int]]
) -> List[int]:
    """Return page numbers that overlap with [c_start, c_end)."""
    pnums = []
    for ps, pe, pn in page_ranges:
        if c_start < pe and c_end > ps:
            pnums.append(pn)
    return pnums if pnums else [page_ranges[0][2]] if page_ranges else [0]


# ===================================================================
# STRATEGY 2: Recursive Chunking (LangChain RCTS style)
# ===================================================================

def chunk_recursive(
    pages: List[Tuple[int, str]],
    doc_id: str = "",
    chunk_size: int = 1024,
    chunk_overlap: int = 128,
    tokenizer_name: str | None = None,
    separators: List[str] | None = None,
) -> List[Chunk]:
    """
    Recursive character/token text splitter.  Tries to split on the most
    structural separator first (\n\n), falls back to less structural ones.
    All sizes in tokens.
    """
    if separators is None:
        separators = ["\n\n\n", "\n\n", "\n", ". ", " "]

    full_text = ""
    page_char_ranges: List[Tuple[int, int, int]] = []
    for pnum, ptxt in pages:
        start = len(full_text)
        full_text += ptxt + "\n"
        page_char_ranges.append((start, len(full_text), pnum))

    raw_splits = _recursive_split(full_text, separators, chunk_size, chunk_overlap, tokenizer_name)

    chunks: List[Chunk] = []
    char_pos = 0
    for idx, text in enumerate(raw_splits):
        loc = full_text.find(text, max(0, char_pos - 200))
        if loc == -1:
            loc = char_pos
        c_end = loc + len(text)
        pnums = _pages_for_char_range(loc, c_end, page_char_ranges)
        n_tok = _count_tokens(text, tokenizer_name)
        chunks.append(Chunk(
            text=text, raw_text=text, doc_id=doc_id,
            page_nums=pnums, token_count=n_tok, raw_token_count=n_tok,
            strategy="recursive", chunk_index=idx,
        ))
        char_pos = c_end
    return chunks


def _recursive_split(
    text: str,
    separators: List[str],
    chunk_size: int,
    chunk_overlap: int,
    tokenizer_name: str | None,
) -> List[str]:
    """Core recursive splitting logic."""
    if not text.strip():
        return []

    # If text fits, return it
    if _count_tokens(text, tokenizer_name) <= chunk_size:
        return [text.strip()] if text.strip() else []

    # Find the best separator that actually splits
    chosen_sep = separators[-1] if separators else " "
    for sep in separators:
        if sep in text:
            chosen_sep = sep
            break

    parts = text.split(chosen_sep)
    remaining_seps = [s for s in separators if s != chosen_sep]

    # Merge parts into chunks respecting the token limit
    merged: List[str] = []
    current = ""
    for part in parts:
        candidate = (current + chosen_sep + part) if current else part
        if _count_tokens(candidate, tokenizer_name) <= chunk_size:
            current = candidate
        else:
            if current.strip():
                # If current itself is too big, recurse with finer separators
                if _count_tokens(current, tokenizer_name) > chunk_size and remaining_seps:
                    merged.extend(_recursive_split(current, remaining_seps, chunk_size, chunk_overlap, tokenizer_name))
                else:
                    merged.append(current.strip())
            current = part

    if current.strip():
        if _count_tokens(current, tokenizer_name) > chunk_size and remaining_seps:
            merged.extend(_recursive_split(current, remaining_seps, chunk_size, chunk_overlap, tokenizer_name))
        else:
            merged.append(current.strip())

    # Add overlap: prepend tail of previous chunk to next chunk
    if chunk_overlap > 0 and len(merged) > 1:
        overlapped = [merged[0]]
        for i in range(1, len(merged)):
            prev_tokens = _tokenize(merged[i - 1], tokenizer_name)
            overlap_ids = prev_tokens[-min(chunk_overlap, len(prev_tokens)):]
            overlap_text = _decode(overlap_ids, tokenizer_name)
            candidate = overlap_text + " " + merged[i]
            if _count_tokens(candidate, tokenizer_name) <= chunk_size + chunk_overlap:
                overlapped.append(candidate)
            else:
                overlapped.append(merged[i])
        merged = overlapped

    return merged


# ===================================================================
# STRATEGY 3: Semantic Chunking
# ===================================================================

def chunk_semantic(
    pages: List[Tuple[int, str]],
    doc_id: str = "",
    chunk_size: int = 1024,
    similarity_threshold: float = 0.5,
    min_sentences: int = 2,
    max_sentences: int = 40,
    tokenizer_name: str | None = None,
    embedding_model_name: str = "BAAI/bge-m3",
) -> List[Chunk]:
    """
    Semantic chunking: split into sentences, embed each, merge adjacent
    sentences while cosine similarity exceeds threshold.  Cap at max tokens.
    """
    full_text = ""
    page_char_ranges: List[Tuple[int, int, int]] = []
    for pnum, ptxt in pages:
        start = len(full_text)
        full_text += ptxt + "\n"
        page_char_ranges.append((start, len(full_text), pnum))

    sentences = _split_sentences(full_text)
    if not sentences:
        return []

    # Embed sentences
    embeddings = _embed_sentences(sentences, embedding_model_name)

    # Compute pairwise cosine similarities between adjacent sentences
    sims = []
    for i in range(len(embeddings) - 1):
        a, b = embeddings[i], embeddings[i + 1]
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
        sims.append(cos)

    # Greedily merge: start a new chunk when similarity drops below threshold
    groups: List[List[int]] = [[0]]  # list of sentence index groups
    for i, sim in enumerate(sims):
        current_group = groups[-1]
        # Check token budget
        candidate_text = " ".join(sentences[j] for j in current_group + [i + 1])
        n_tok = _count_tokens(candidate_text, tokenizer_name)

        if (
            sim >= similarity_threshold
            and len(current_group) < max_sentences
            and n_tok <= chunk_size
        ):
            current_group.append(i + 1)
        else:
            groups.append([i + 1])

    # Enforce min_sentences by merging tiny trailing groups
    final_groups: List[List[int]] = []
    for g in groups:
        if final_groups and len(final_groups[-1]) < min_sentences:
            final_groups[-1].extend(g)
        else:
            final_groups.append(g)

    # Build chunks
    chunks: List[Chunk] = []
    for idx, group in enumerate(final_groups):
        text = " ".join(sentences[i] for i in group)
        # Find char position for page mapping
        first_sent = sentences[group[0]]
        loc = full_text.find(first_sent)
        c_end = loc + len(text) if loc >= 0 else 0
        pnums = _pages_for_char_range(max(loc, 0), c_end, page_char_ranges)
        n_tok = _count_tokens(text, tokenizer_name)
        chunks.append(Chunk(
            text=text, raw_text=text, doc_id=doc_id,
            page_nums=pnums, token_count=n_tok, raw_token_count=n_tok,
            strategy="semantic", chunk_index=idx,
        ))
    return chunks


_SENT_EMBED_MODEL = None
_SENT_EMBED_NAME = None


def _embed_sentences(sentences: List[str], model_name: str) -> np.ndarray:
    """Embed a list of sentences, returning (N, D) array."""
    global _SENT_EMBED_MODEL, _SENT_EMBED_NAME
    if _SENT_EMBED_MODEL is None or _SENT_EMBED_NAME != model_name:
        from sentence_transformers import SentenceTransformer
        _SENT_EMBED_MODEL = SentenceTransformer(model_name)
        _SENT_EMBED_NAME = model_name
    return _SENT_EMBED_MODEL.encode(sentences, show_progress_bar=False, normalize_embeddings=True)


# ===================================================================
# STRATEGY 4: Adaptive Chunking
# ===================================================================

def chunk_adaptive(
    pages: List[Tuple[int, str]],
    doc_id: str = "",
    base_chunk_size: int = 1024,
    chunk_overlap: int = 128,
    min_chunk_size: int = 256,
    max_chunk_size: int = 2048,
    tokenizer_name: str | None = None,
) -> List[Chunk]:
    """
    Adaptive chunking: vary chunk size based on content density.
    Dense content (tables, numbers, named entities) → smaller chunks.
    Narrative prose → larger chunks.

    Density is measured by the fraction of tokens that are digits, currency
    symbols, or in short (≤3 char) words (proxies for tabular / numeric content).
    """
    full_text = ""
    page_char_ranges: List[Tuple[int, int, int]] = []
    for pnum, ptxt in pages:
        start = len(full_text)
        full_text += ptxt + "\n"
        page_char_ranges.append((start, len(full_text), pnum))

    sentences = _split_sentences(full_text)
    if not sentences:
        return []

    # Compute per-sentence density
    densities = []
    for sent in sentences:
        words = sent.split()
        if not words:
            densities.append(0.0)
            continue
        numeric_words = sum(1 for w in words if re.search(r"\d", w))
        short_words = sum(1 for w in words if len(w) <= 3)
        density = (numeric_words + 0.3 * short_words) / len(words)
        densities.append(min(density, 1.0))

    # Group sentences into chunks with adaptive size
    chunks: List[Chunk] = []
    idx = 0
    i = 0
    while i < len(sentences):
        # Compute local density (lookahead window of ~10 sentences)
        window = densities[i : i + 10]
        avg_density = np.mean(window) if window else 0.0

        # Map density to chunk size: high density → smaller chunks
        # Linear interpolation: density=0 → max_chunk_size, density=1 → min_chunk_size
        target_size = int(max_chunk_size - avg_density * (max_chunk_size - min_chunk_size))
        target_size = max(min_chunk_size, min(max_chunk_size, target_size))

        # Accumulate sentences until target size
        group_text = ""
        group_start_i = i
        while i < len(sentences):
            candidate = (group_text + " " + sentences[i]).strip() if group_text else sentences[i]
            n_tok = _count_tokens(candidate, tokenizer_name)
            if n_tok > target_size and group_text:
                break
            group_text = candidate
            i += 1

        if not group_text.strip():
            i += 1
            continue

        loc = full_text.find(sentences[group_start_i])
        c_end = loc + len(group_text) if loc >= 0 else 0
        pnums = _pages_for_char_range(max(loc, 0), c_end, page_char_ranges)
        n_tok = _count_tokens(group_text, tokenizer_name)
        chunks.append(Chunk(
            text=group_text, raw_text=group_text, doc_id=doc_id,
            page_nums=pnums, token_count=n_tok, raw_token_count=n_tok,
            strategy="adaptive", chunk_index=idx,
            metadata={"target_size": target_size, "local_density": float(avg_density)},
        ))
        idx += 1

    return chunks


# ===================================================================
# STRATEGY 5: Parent–Child Chunking
# ===================================================================

def chunk_parent_child(
    pages: List[Tuple[int, str]],
    doc_id: str = "",
    parent_chunk_size: int = 2048,
    parent_overlap: int = 256,
    child_chunk_size: int = 512,
    child_overlap: int = 64,
    tokenizer_name: str | None = None,
) -> List[Chunk]:
    """
    Two-level hierarchy.  Returns BOTH parent and child chunks.
    Children have `parent_chunk_index` set.  At retrieval time, you index the
    children but can optionally return the parent text for generation context.

    Children are what get embedded and indexed.  Parents are stored for
    context expansion.
    """
    # First, create parent chunks using naive strategy
    parents = chunk_naive(
        pages, doc_id=doc_id, chunk_size=parent_chunk_size,
        chunk_overlap=parent_overlap, tokenizer_name=tokenizer_name,
    )
    for p in parents:
        p.strategy = "parent_child_parent"

    # Then subdivide each parent into children
    children: List[Chunk] = []
    child_idx = 0
    for parent in parents:
        parent_tokens = _tokenize(parent.raw_text, tokenizer_name)
        step = max(1, child_chunk_size - child_overlap)
        for start in range(0, len(parent_tokens), step):
            end = min(start + child_chunk_size, len(parent_tokens))
            child_ids = parent_tokens[start:end]
            child_text = _decode(child_ids, tokenizer_name)
            children.append(Chunk(
                text=child_text, raw_text=child_text, doc_id=doc_id,
                page_nums=parent.page_nums.copy(),
                token_count=len(child_ids), raw_token_count=len(child_ids),
                strategy="parent_child_child", chunk_index=child_idx,
                parent_chunk_index=parent.chunk_index,
                metadata={"parent_text": parent.raw_text},
            ))
            child_idx += 1
            if end >= len(parent_tokens):
                break

    # Return both; the caller decides what to index
    return parents + children


# ===================================================================
# STRATEGY 6: Table-Aware Chunking
# ===================================================================

def chunk_table_aware(
    pages: List[Tuple[int, str]],
    doc_id: str = "",
    chunk_size: int = 1024,
    chunk_overlap: int = 128,
    tokenizer_name: str | None = None,
) -> List[Chunk]:
    """
    Detect table regions and keep them as atomic chunks (with column headers
    prepended).  Non-table text is chunked with naive fixed-size strategy.
    If a table exceeds chunk_size tokens, it is kept as-is (oversized) to
    preserve structure.
    """
    full_text = ""
    page_char_ranges: List[Tuple[int, int, int]] = []
    for pnum, ptxt in pages:
        start = len(full_text)
        full_text += ptxt + "\n"
        page_char_ranges.append((start, len(full_text), pnum))

    table_regions = _detect_table_regions(full_text)

    # Split text into table vs non-table segments
    segments: List[Tuple[str, bool, int, int]] = []  # (text, is_table, char_start, char_end)
    prev_end = 0
    for t_start, t_end in table_regions:
        if t_start > prev_end:
            segments.append((full_text[prev_end:t_start], False, prev_end, t_start))
        segments.append((full_text[t_start:t_end], True, t_start, t_end))
        prev_end = t_end
    if prev_end < len(full_text):
        segments.append((full_text[prev_end:], False, prev_end, len(full_text)))

    chunks: List[Chunk] = []
    idx = 0
    for seg_text, is_table, c_start, c_end in segments:
        if not seg_text.strip():
            continue
        pnums = _pages_for_char_range(c_start, c_end, page_char_ranges)

        if is_table:
            # Keep table atomic — prepend a header line if identifiable
            # Look for a title line just before the table
            pre_context_start = max(0, c_start - 200)
            pre_text = full_text[pre_context_start:c_start]
            title_lines = [l.strip() for l in pre_text.split("\n") if l.strip()]
            table_title = title_lines[-1] if title_lines else ""
            final_text = f"[TABLE: {table_title}]\n{seg_text}" if table_title else seg_text
            n_tok = _count_tokens(final_text, tokenizer_name)
            chunks.append(Chunk(
                text=final_text, raw_text=seg_text, doc_id=doc_id,
                page_nums=pnums, token_count=n_tok, raw_token_count=_count_tokens(seg_text, tokenizer_name),
                strategy="table_aware", chunk_index=idx,
                metadata={"is_table": True, "table_title": table_title},
            ))
            idx += 1
        else:
            # Chunk non-table text with naive strategy
            # Build fake page list from this segment
            sub_pages = [(pnums[0] if pnums else 0, seg_text)]
            sub_chunks = chunk_naive(
                sub_pages, doc_id=doc_id, chunk_size=chunk_size,
                chunk_overlap=chunk_overlap, tokenizer_name=tokenizer_name,
            )
            for sc in sub_chunks:
                sc.strategy = "table_aware"
                sc.chunk_index = idx
                sc.page_nums = pnums
                sc.metadata["is_table"] = False
                idx += 1
            chunks.extend(sub_chunks)

    return chunks


# ===================================================================
# STRATEGY 7: Late Chunking  (chunk boundaries only — embeddings
#   are handled externally by a long-context model)
# ===================================================================

def chunk_late(
    pages: List[Tuple[int, str]],
    doc_id: str = "",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    tokenizer_name: str | None = None,
) -> List[Chunk]:
    """
    Late chunking: create fixed-size chunk *boundaries* identically to naive,
    but mark them so the embedding step knows to use a long-context encoder
    to embed the full document first, then mean-pool hidden states per chunk span.

    This function returns Chunk objects with an additional metadata field
    `token_span` = (start_tok, end_tok) that the late-chunking embedder uses.
    The actual late embedding logic lives in the experiment runner / embedder.
    """
    full_text = ""
    page_char_ranges: List[Tuple[int, int, int]] = []
    for pnum, ptxt in pages:
        start = len(full_text)
        full_text += ptxt + "\n"
        page_char_ranges.append((start, len(full_text), pnum))

    tokens = _tokenize(full_text, tokenizer_name)
    chunks: List[Chunk] = []
    step = max(1, chunk_size - chunk_overlap)
    idx = 0

    for start in range(0, len(tokens), step):
        end = min(start + chunk_size, len(tokens))
        chunk_ids = tokens[start:end]
        text = _decode(chunk_ids, tokenizer_name)

        prefix_text = _decode(tokens[:start], tokenizer_name)
        chunk_char_start = len(prefix_text)
        chunk_char_end = chunk_char_start + len(text)
        pnums = _pages_for_char_range(chunk_char_start, chunk_char_end, page_char_ranges)

        chunks.append(Chunk(
            text=text, raw_text=text, doc_id=doc_id,
            page_nums=pnums, token_count=len(chunk_ids),
            raw_token_count=len(chunk_ids),
            strategy="late", chunk_index=idx,
            metadata={"token_span": (start, end), "full_doc_tokens": len(tokens)},
        ))
        idx += 1
        if end >= len(tokens):
            break

    return chunks


# ===================================================================
# STRATEGY 8: Contextual Chunking (no LLM — header/section prepending)
# ===================================================================

def chunk_contextual(
    pages: List[Tuple[int, str]],
    doc_id: str = "",
    chunk_size: int = 1024,
    chunk_overlap: int = 128,
    context_budget: int = 128,
    tokenizer_name: str | None = None,
) -> List[Chunk]:
    """
    Contextual chunking WITHOUT an LLM: prepend a context prefix to each
    chunk containing:
      - Document identifier
      - Page number(s)
      - Detected section header (e.g. "ITEM 7. MD&A")

    The chunk_size budget is split: `context_budget` tokens for the prefix,
    the rest for the raw content.
    """
    content_budget = chunk_size - context_budget

    # First create raw chunks at the reduced size
    raw_chunks = chunk_naive(
        pages, doc_id=doc_id, chunk_size=content_budget,
        chunk_overlap=chunk_overlap, tokenizer_name=tokenizer_name,
    )

    # Build page→text lookup for section detection
    page_texts = {pnum: ptxt for pnum, ptxt in pages}

    chunks: List[Chunk] = []
    for idx, rc in enumerate(raw_chunks):
        # Detect section from surrounding page text
        section = ""
        for pn in rc.page_nums:
            if pn in page_texts:
                section = _detect_section_header(page_texts[pn])
                if section:
                    break

        # Build context prefix
        prefix_parts = [f"Document: {doc_id}"]
        if rc.page_nums:
            prefix_parts.append(f"Page(s): {', '.join(str(p) for p in rc.page_nums)}")
        if section:
            prefix_parts.append(f"Section: {section}")
        prefix = " | ".join(prefix_parts) + "\n\n"

        # Truncate prefix to budget if needed
        prefix_tokens = _tokenize(prefix, tokenizer_name)
        if len(prefix_tokens) > context_budget:
            prefix = _decode(prefix_tokens[:context_budget], tokenizer_name) + "\n"

        full_text = prefix + rc.raw_text
        n_tok = _count_tokens(full_text, tokenizer_name)

        chunks.append(Chunk(
            text=full_text, raw_text=rc.raw_text, doc_id=doc_id,
            page_nums=rc.page_nums, token_count=n_tok,
            raw_token_count=rc.raw_token_count,
            strategy="contextual", chunk_index=idx,
            metadata={"context_prefix": prefix, "section": section},
        ))
    return chunks


# ===================================================================
# STRATEGY 9: Metadata Chunking
# ===================================================================

def chunk_metadata(
    pages: List[Tuple[int, str]],
    doc_id: str = "",
    chunk_size: int = 1024,
    chunk_overlap: int = 128,
    tokenizer_name: str | None = None,
) -> List[Chunk]:
    """
    Metadata-enriched chunking: fixed-size chunks with rich structured metadata.
    The chunk text itself is NOT modified (unlike contextual chunking).
    Metadata fields are stored in chunk.metadata for downstream filtering,
    boosted retrieval, or hybrid scoring.

    Metadata includes:
      - doc_id, page_nums (standard)
      - section_header: detected SEC section
      - has_table: whether the chunk overlaps a detected table region
      - numeric_density: fraction of tokens with digits
      - entity_count: rough count of financial entities ($ amounts, %)
      - position_ratio: chunk position / total chunks (beginning vs end of doc)
    """
    full_text = ""
    page_char_ranges: List[Tuple[int, int, int]] = []
    for pnum, ptxt in pages:
        start = len(full_text)
        full_text += ptxt + "\n"
        page_char_ranges.append((start, len(full_text), pnum))

    table_regions = _detect_table_regions(full_text)

    # Create base chunks with naive strategy
    raw_chunks = chunk_naive(
        pages, doc_id=doc_id, chunk_size=chunk_size,
        chunk_overlap=chunk_overlap, tokenizer_name=tokenizer_name,
    )

    total_chunks = len(raw_chunks)
    page_texts = {pnum: ptxt for pnum, ptxt in pages}

    chunks: List[Chunk] = []
    for idx, rc in enumerate(raw_chunks):
        # Detect section
        section = ""
        for pn in rc.page_nums:
            if pn in page_texts:
                section = _detect_section_header(page_texts[pn])
                if section:
                    break

        # Check table overlap
        rc_loc = full_text.find(rc.raw_text[:80])  # approximate location
        rc_end = rc_loc + len(rc.raw_text) if rc_loc >= 0 else 0
        has_table = any(
            rc_loc < t_end and rc_end > t_start
            for t_start, t_end in table_regions
        ) if rc_loc >= 0 else False

        # Numeric density
        words = rc.raw_text.split()
        numeric_density = sum(1 for w in words if re.search(r"\d", w)) / max(len(words), 1)

        # Entity count ($ amounts, percentages)
        entities = re.findall(r"\$[\d,]+(?:\.\d+)?|\d+(?:\.\d+)?%", rc.raw_text)
        entity_count = len(entities)

        rc.strategy = "metadata"
        rc.chunk_index = idx
        rc.metadata = {
            "section_header": section,
            "has_table": has_table,
            "numeric_density": round(numeric_density, 4),
            "entity_count": entity_count,
            "position_ratio": round(idx / max(total_chunks - 1, 1), 4),
        }
        chunks.append(rc)

    return chunks


# ===================================================================
# Registry / dispatcher
# ===================================================================

STRATEGY_REGISTRY: Dict[str, Callable] = {
    "naive": chunk_naive,
    "recursive": chunk_recursive,
    "semantic": chunk_semantic,
    "adaptive": chunk_adaptive,
    "parent_child": chunk_parent_child,
    "table_aware": chunk_table_aware,
    "late": chunk_late,
    "contextual": chunk_contextual,
    "metadata": chunk_metadata,
}


def get_chunker(name: str) -> Callable:
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy '{name}'. Choose from: {list(STRATEGY_REGISTRY.keys())}")
    return STRATEGY_REGISTRY[name]


def list_strategies() -> List[str]:
    return list(STRATEGY_REGISTRY.keys())