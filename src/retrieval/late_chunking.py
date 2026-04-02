"""
src/retrieval/late_chunking.py
==============================
Late-chunking indexer and retriever for FinanceBench.

Rewrite rationale (vs. the Jina-v3-based predecessor)

This rewrite uses **BGE-M3 as a single model** for both:
  • Document chunk embeddings  — per-token hidden states from the raw
    XLM-RoBERTa backbone, mean-pooled per chunk span (the "late" part).
  • Query embeddings  — ``SentenceTransformer.encode()`` with the model's
    trained CLS pooling + normalisation (the "standard" part).

Both embedding types land in the **same vector space** because:
  - BGE-M3's ``SentenceTransformer`` wrapper applies CLS pooling to the
    same ``last_hidden_state`` tensor that we extract for document chunks.
  - The per-token representations already encode rich semantics; mean-
    pooling a contiguous span produces vectors that are directionally
    aligned with the CLS-pooled query vector (this is the core insight
    of the late-chunking paper by Günther et al., 2024).

Key parameters
--------------
  max_tokens    : int   – size of the sliding window over the document
                          (must match model's max position embeddings;
                          8 190 content tokens for BGE-M3 after CLS/SEP).
  stride        : int   – overlap between consecutive windows in tokens.
                          Set to ``chunk_size`` so every chunk is fully
                          contained in at least one window.
  chunk_size    : int   – number of tokens per retrieval chunk.
  chunk_overlap : int   – token overlap between adjacent chunks.

Compatibility
-------------
  • Python ≥ 3.10  (tested on 3.13.11)
  • transformers ≥ 4.30
  • sentence-transformers ≥ 2.2
  • faiss-cpu or faiss-gpu
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

from src.core.rag_dependencies import Document
from src.retrieval.bm25 import _compute_corpus_fingerprint

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BGE-M3 model name resolution
# ---------------------------------------------------------------------------
_BGE_M3_ALIASES = {
    "bge-m3", "baai/bge-m3", "bge_m3",
}

# Max *content* tokens per window (BGE-M3 has 8192 position embeddings;
# 2 are reserved for [CLS] and [SEP]).
_BGE_M3_MAX_CONTENT_TOKENS = 8190


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _sanitize(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in (name or ""))
    safe = "_".join([p for p in safe.split("_") if p])
    return safe or "unknown"


def _hash_config(config: Dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:8]


def _index_paths(
    experiment, scope: str, fingerprint: str
) -> Tuple[str, str, Dict[str, Any]]:
    config = {
        "scope": scope,
        "fingerprint": fingerprint,
        "chunk_size": getattr(experiment, "chunk_size", None),
        "chunk_overlap": getattr(experiment, "chunk_overlap", None),
        "chunking_strategy": getattr(experiment, "chunking_strategy", None),
        "embedding_model": getattr(experiment, "embedding_model", None),
        "late_model": getattr(experiment, "late_model", None),
        "late_max_tokens": getattr(experiment, "late_max_tokens", None),
        "late_window_stride": getattr(experiment, "late_window_stride", None),
        "late_pooling": getattr(experiment, "late_pooling", None),
        # Cache-bust: v3 adds per-chunk page number tracking via offset_mapping.
        "impl_version": "bge_m3_v3",
    }
    config_hash = _hash_config(config)
    name = f"{_sanitize(scope)}_late_{config_hash}"
    base_dir = Path(getattr(experiment, "vector_store_dir", "."))
    index_dir = base_dir / "late_chunks" / name
    return name, str(index_dir), config


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    norm = np.where(norm == 0.0, 1.0, norm)
    return vec / norm


def _token_windows(
    n_tokens: int, max_tokens: int, stride: int
) -> Iterable[Tuple[int, int]]:
    """
    Yield ``(start, end)`` token-index pairs for a sliding window.

    ``stride`` is the **overlap** between consecutive windows.
    The step is ``max_tokens - stride``.
    """
    if max_tokens <= 0 or n_tokens <= max_tokens:
        yield 0, n_tokens
        return
    step = max(1, max_tokens - stride)
    for start in range(0, n_tokens, step):
        end = min(start + max_tokens, n_tokens)
        yield start, end
        if end >= n_tokens:
            break


# ---------------------------------------------------------------------------
# Document grouping (pages → full documents)
# ---------------------------------------------------------------------------

_GENERIC_SOURCE_VALUES = frozenset({
    "pdf", "local", "none", "url", "web", "unknown", "",
})


def _is_unique_source(value: str) -> bool:
    if not value or value.lower() in _GENERIC_SOURCE_VALUES:
        return False
    return any(c in value for c in ("/", "\\", "."))


def _pages_for_char_range(
    char_start: int,
    char_end: int,
    page_char_ranges: List[Tuple[int, int, int]],
) -> List[int]:
    """Return sorted list of page numbers that overlap [char_start, char_end)."""
    pages: List[int] = []
    for pg_start, pg_end, pg_num in page_char_ranges:
        if pg_start < char_end and pg_end > char_start:
            pages.append(pg_num)
    return sorted(set(pages)) if pages else [0]


# Return type now includes page_char_ranges as the 4th tuple element.
GroupedDoc = Tuple[str, str, dict, List[Tuple[int, int, int]]]


def _group_docs_by_source(docs: List[Any]) -> List[GroupedDoc]:
    """
    Aggregate page-level Document objects into per-document full texts.

    Returns ``[(source_key, full_text, representative_metadata,
                page_char_ranges), ...]``
    where ``page_char_ranges`` is a list of ``(char_start, char_end, page_num)``
    tuples that map character positions in ``full_text`` back to the original
    page numbers.  This is essential for assigning correct page metadata to
    each late-chunked span.

    Grouping key priority: doc_name > source (if path-like) > filename.
    """
    groups: Dict[str, List[Any]] = defaultdict(list)
    for doc in docs:
        meta = getattr(doc, "metadata", {}) or {}
        doc_name = (meta.get("doc_name") or "").strip()
        source = (meta.get("source") or "").strip()
        filename = (meta.get("filename") or "").strip()

        if doc_name and doc_name.lower() not in _GENERIC_SOURCE_VALUES:
            key = doc_name
        elif _is_unique_source(source):
            key = source
        elif filename and filename.lower() not in _GENERIC_SOURCE_VALUES:
            key = filename
        else:
            key = "unknown"

        groups[key].append(doc)

    if len(groups) == 1:
        only_key = next(iter(groups))
        n_pages = len(groups[only_key])
        if only_key.lower() in _GENERIC_SOURCE_VALUES or n_pages > 500:
            logger.warning(
                "_group_docs_by_source: ALL %d pages grouped under '%s'. "
                "This likely means doc_name metadata is missing.",
                n_pages, only_key,
            )

    result: List[GroupedDoc] = []
    for source_key, page_docs in groups.items():
        page_docs.sort(
            key=lambda d: int((getattr(d, "metadata", {}) or {}).get("page", 0))
        )

        # Build full_text by concatenating pages, tracking char ranges
        full_text = ""
        page_char_ranges: List[Tuple[int, int, int]] = []
        for p in page_docs:
            p_meta = getattr(p, "metadata", {}) or {}
            p_num = int(p_meta.get("page", 0))
            p_text = (
                getattr(p, "page_content", None)
                or getattr(p, "content", "")
                or ""
            )
            char_start = len(full_text)
            full_text += p_text + "\n"
            page_char_ranges.append((char_start, len(full_text), p_num))

        rep_meta = dict((getattr(page_docs[0], "metadata", {}) or {}))
        rep_meta["page_count"] = len(page_docs)
        rep_meta["source"] = source_key
        result.append((source_key, full_text, rep_meta, page_char_ranges))
        logger.info(
            "Grouped %d pages -> doc '%s' (%d chars)",
            len(page_docs), source_key, len(full_text),
        )

    return result


# ---------------------------------------------------------------------------
# Core: embed a full document via late chunking
# ---------------------------------------------------------------------------

def _late_embed_document(
    full_text: str,
    base_metadata: dict,
    page_char_ranges: List[Tuple[int, int, int]],
    tokenizer,
    model,
    chunk_size: int,
    chunk_overlap: int,
    max_content_tokens: int,
    stride: int,
    device: str,
) -> Tuple[List[Document], np.ndarray]:
    """
    Embed a full document using late chunking with BGE-M3.

    Algorithm
    ---------
    1. Tokenise the full document *without* special tokens, but *with*
       offset_mapping so we can map token positions → character positions
       → page numbers.
    2. Slide a window of ``max_content_tokens`` over the token sequence.
       For each window:
         a. Wrap with [CLS] + content + [SEP].
         b. Run forward pass → ``last_hidden_state``.
         c. Strip CLS/SEP positions (indices 0 and -1).
         d. Accumulate content hidden states into ``all_hidden``.
       A count tensor tracks coverage for averaging overlapping windows.
    3. Divide ``all_hidden`` by counts → one hidden vector per token.
    4. Slice into fixed-size chunk spans, mean-pool each → one embedding
       per chunk.  Use offset_mapping + page_char_ranges to assign the
       correct page numbers to each chunk.
    5. L2-normalise.
    """
    doc_name = base_metadata.get("doc_name") or base_metadata.get("source", "?")

    # --- Step 1: tokenise with offset mapping ---
    # Use the tokenizer's __call__ to get offset_mapping (token → char range).
    # This lets us map each token span back to character positions, and from
    # there to page numbers.
    try:
        encoded = tokenizer(
            full_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_attention_mask=False,
        )
        token_ids: List[int] = encoded["input_ids"]
        offset_mapping: List[Tuple[int, int]] = encoded["offset_mapping"]
    except Exception:
        # Fallback for tokenizers that don't support offset_mapping
        token_ids = tokenizer.encode(full_text, add_special_tokens=False)
        offset_mapping = None
        logger.warning(
            "Tokenizer does not support offset_mapping — page numbers "
            "will be approximate (computed via prefix decoding)."
        )

    if not token_ids:
        return [], np.empty((0, model.config.hidden_size), dtype=np.float32)

    n_tokens = len(token_ids)
    hidden_size = model.config.hidden_size

    # Hard cap for safety (prevents OOM if all docs accidentally merged)
    MAX_SAFE_TOKENS = 1_000_000
    if n_tokens > MAX_SAFE_TOKENS:
        logger.warning(
            "'%s': %d tokens exceeds safety cap %d — truncating.",
            doc_name, n_tokens, MAX_SAFE_TOKENS,
        )
        token_ids = token_ids[:MAX_SAFE_TOKENS]
        if offset_mapping is not None:
            offset_mapping = offset_mapping[:MAX_SAFE_TOKENS]
        n_tokens = MAX_SAFE_TOKENS

    # Resolve CLS / SEP ids
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    if cls_id is None:
        cls_id = tokenizer.bos_token_id
    if sep_id is None:
        sep_id = tokenizer.eos_token_id
    if cls_id is None or sep_id is None:
        raise RuntimeError(
            f"Tokenizer missing CLS/SEP ids "
            f"(cls={tokenizer.cls_token_id}, sep={tokenizer.sep_token_id}). "
            f"BGE-M3 (XLM-RoBERTa) should always have these."
        )

    # Pre-compute window count for logging
    step = max(1, max_content_tokens - stride)
    n_windows = max(1, (n_tokens - 1) // step + 1) if n_tokens > max_content_tokens else 1
    logger.info(
        "  Embedding '%s': %d tokens, %d window(s) (max_content=%d, stride=%d)",
        doc_name, n_tokens, n_windows, max_content_tokens, stride,
    )

    # --- Step 2: sliding-window forward passes ---
    # Accumulation buffers on CPU to avoid GPU OOM
    all_hidden = torch.zeros(n_tokens, hidden_size, dtype=torch.float32)
    counts = torch.zeros(n_tokens, dtype=torch.float32)

    for win_start, win_end in _token_windows(n_tokens, max_content_tokens, stride):
        content_ids = token_ids[win_start:win_end]
        content_len = len(content_ids)

        # Wrap: [CLS] content_ids [SEP]
        input_ids = torch.tensor(
            [[cls_id] + content_ids + [sep_id]], device=device
        )
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Shape: [1, content_len + 2, hidden_size]
            # Strip CLS (idx 0) and SEP (idx -1)
            hidden = (
                outputs.last_hidden_state
                .squeeze(0)  # [seq_len_with_special, hidden_size]
                [1:-1]       # [content_len, hidden_size]
                .to(dtype=torch.float32)
                .cpu()
            )

        if hidden.shape[0] != content_len:
            logger.warning(
                "  Hidden length mismatch for '%s' window @%d: "
                "expected %d, got %d — skipping.",
                doc_name, win_start, content_len, hidden.shape[0],
            )
            continue

        all_hidden[win_start:win_end] += hidden
        counts[win_start:win_end] += 1.0

        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    # Coverage check
    uncovered = int((counts == 0).sum())
    if uncovered > 0:
        logger.warning(
            "  '%s': %d/%d tokens have zero window coverage.",
            doc_name, uncovered, n_tokens,
        )

    # --- Step 3: average overlapping positions ---
    counts = counts.clamp(min=1.0).unsqueeze(-1)  # [n_tokens, 1]
    all_hidden = all_hidden / counts                # [n_tokens, hidden_size]

    # --- Step 4: slice into chunks and mean-pool ---
    all_chunks: List[Document] = []
    all_vectors: List[np.ndarray] = []

    step_chunk = max(1, chunk_size - chunk_overlap)
    chunk_index = 0

    for span_start in range(0, n_tokens, step_chunk):
        span_end = min(span_start + chunk_size, n_tokens)
        if span_start >= span_end:
            break

        span_tokens = token_ids[span_start:span_end]
        chunk_text = tokenizer.decode(span_tokens, skip_special_tokens=True).strip()
        if not chunk_text:
            chunk_index += 1
            if span_end >= n_tokens:
                break
            continue

        # Mean-pool this span's hidden states
        span_hidden = all_hidden[span_start:span_end]
        vec = span_hidden.mean(dim=0).numpy().astype(np.float32)
        all_vectors.append(vec)

        # --- Determine page numbers for this chunk ---
        if offset_mapping is not None and page_char_ranges:
            # Use offset_mapping to get the char range of this token span
            span_offsets = offset_mapping[span_start:span_end]
            # char_start = first token's start char; char_end = last token's end char
            char_start = span_offsets[0][0]
            char_end = span_offsets[-1][1]
            chunk_pages = _pages_for_char_range(char_start, char_end, page_char_ranges)
        elif page_char_ranges:
            # Fallback: approximate via prefix decoding (slow but correct)
            prefix_text = tokenizer.decode(token_ids[:span_start], skip_special_tokens=True)
            char_start = len(prefix_text)
            char_end = char_start + len(chunk_text)
            chunk_pages = _pages_for_char_range(char_start, char_end, page_char_ranges)
        else:
            chunk_pages = [0]

        chunk_meta = dict(base_metadata)
        chunk_meta.update({
            "late_chunk_start_token": span_start,
            "late_chunk_end_token": span_end,
            "late_chunk_index": chunk_index,
            "token_span": (span_start, span_end),
            "page": chunk_pages[0] if len(chunk_pages) == 1 else chunk_pages[0],
            "page_nums": chunk_pages,
        })
        chunk_id_src = f"{chunk_meta.get('source', 'doc')}_{span_start}_{span_end}"
        chunk_meta["chunk_id"] = hashlib.md5(
            chunk_id_src.encode("utf-8")
        ).hexdigest()

        all_chunks.append(Document(page_content=chunk_text, metadata=chunk_meta))
        chunk_index += 1

        if span_end >= n_tokens:
            break

    if not all_vectors:
        return [], np.empty((0, hidden_size), dtype=np.float32)

    vectors = np.stack(all_vectors, axis=0)

    # NaN guard
    nan_mask = ~np.isfinite(vectors).all(axis=1)
    n_nan = int(nan_mask.sum())
    if n_nan > 0:
        pct = 100.0 * n_nan / len(vectors)
        logger.error(
            "  '%s': %d/%d (%.1f%%) vectors are NaN — dropping.",
            doc_name, n_nan, len(vectors), pct,
        )
        good = ~nan_mask
        vectors = vectors[good]
        all_chunks = [c for c, ok in zip(all_chunks, good.tolist()) if ok]
        if len(vectors) == 0:
            return [], np.empty((0, hidden_size), dtype=np.float32)

    return all_chunks, vectors


# ---------------------------------------------------------------------------
# FAISS index build
# ---------------------------------------------------------------------------

def build_late_chunk_index(
    experiment,
    docs: List[Any],
    scope: str,
    fingerprint: str,
) -> "LateChunkIndex":
    """
    Build (or load from cache) a FAISS inner-product index of late-chunked
    document embeddings.
    """
    if faiss is None:
        raise RuntimeError("faiss-cpu or faiss-gpu is required for late chunking.")

    name, index_dir, config = _index_paths(experiment, scope, fingerprint)
    os.makedirs(index_dir, exist_ok=True)
    index_path = os.path.join(index_dir, "index.faiss")
    chunks_path = os.path.join(index_dir, "chunks.json")
    config_path = os.path.join(index_dir, "config.json")

    # --- Cache check ---
    if all(os.path.exists(p) for p in (index_path, chunks_path, config_path)):
        try:
            with open(config_path, "r") as f:
                stored = json.load(f)
            if stored == config:
                logger.info("Loading cached late-chunk index '%s'.", name)
                index = faiss.read_index(index_path)
                with open(chunks_path, "r") as f:
                    payload = json.load(f)
                chunks = [
                    Document(page_content=c["text"], metadata=c["metadata"])
                    for c in payload
                ]
                st_model, raw_model, raw_tokenizer, device = _load_models(experiment)
                return LateChunkIndex(
                    index=index,
                    chunks=chunks,
                    st_model=st_model,
                    raw_model=raw_model,
                    raw_tokenizer=raw_tokenizer,
                    device=device,
                )
        except Exception as exc:
            logger.warning("Failed to load cache '%s': %s — rebuilding.", name, exc)

    # --- Build index ---
    st_model, raw_model, raw_tokenizer, device = _load_models(experiment)
    chunk_size = int(getattr(experiment, "chunk_size", 512))
    chunk_overlap = int(getattr(experiment, "chunk_overlap", 64))
    max_tokens = int(getattr(experiment, "late_max_tokens", 8192))
    stride = int(getattr(experiment, "late_window_stride", 512))

    # Clamp max_content_tokens to model's actual position limit
    max_content_tokens = min(max_tokens - 2, _BGE_M3_MAX_CONTENT_TOKENS)

    logger.info(
        "Building late-chunk index '%s': chunk_size=%d, overlap=%d, "
        "max_content_tokens=%d, stride=%d",
        name, chunk_size, chunk_overlap, max_content_tokens, stride,
    )

    # Group page-level docs → full documents
    doc_groups = _group_docs_by_source(docs)
    logger.info(
        "Aggregated %d page-level docs into %d full documents.",
        len(docs), len(doc_groups),
    )

    all_chunks: List[Document] = []
    all_vectors: List[np.ndarray] = []

    for doc_idx, (source_key, full_text, rep_meta, page_char_ranges) in enumerate(doc_groups):
        if not full_text.strip():
            logger.warning("Document '%s' is empty — skipping.", source_key)
            continue

        logger.info(
            "Encoding document %d/%d: '%s' (%d chars, %d pages)",
            doc_idx + 1, len(doc_groups), source_key, len(full_text),
            len(page_char_ranges),
        )

        doc_chunks, doc_vectors = _late_embed_document(
            full_text=full_text,
            base_metadata=rep_meta,
            page_char_ranges=page_char_ranges,
            tokenizer=raw_tokenizer,
            model=raw_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_content_tokens=max_content_tokens,
            stride=stride,
            device=device,
        )

        if doc_chunks:
            all_chunks.extend(doc_chunks)
            all_vectors.append(doc_vectors)
            logger.info("  -> %d chunks for '%s'.", len(doc_chunks), source_key)
        else:
            logger.warning("  -> 0 chunks for '%s'.", source_key)

    if not all_chunks:
        raise RuntimeError(
            "Late chunking produced 0 chunks across all documents. "
            "Check that --pdf-dir contains readable PDFs."
        )

    vectors = np.vstack(all_vectors).astype(np.float32)
    vectors = _normalize(vectors)

    # Final NaN guard
    nan_mask = ~np.isfinite(vectors).all(axis=1)
    n_nan = int(nan_mask.sum())
    if n_nan > 0:
        logger.error(
            "%d/%d vectors NaN after normalisation — dropping.", n_nan, len(vectors)
        )
        good = ~nan_mask
        vectors = vectors[good]
        all_chunks = [c for c, ok in zip(all_chunks, good.tolist()) if ok]
        if len(vectors) == 0:
            raise RuntimeError("ALL vectors are NaN. Check model outputs.")
    else:
        logger.info("NaN check passed: all %d vectors finite.", len(vectors))

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    # Persist
    faiss.write_index(index, index_path)
    with open(chunks_path, "w") as f:
        json.dump(
            [{"text": c.page_content, "metadata": c.metadata} for c in all_chunks],
            f,
        )
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(
        "Late-chunk index '%s' built: %d chunks, dim=%d.", name, len(all_chunks), dim,
    )
    return LateChunkIndex(
        index=index,
        chunks=all_chunks,
        st_model=st_model,
        raw_model=raw_model,
        raw_tokenizer=raw_tokenizer,
        device=device,
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _resolve_model_name(experiment, model_name: Optional[str]) -> str:
    """Resolve aliases like 'bge-m3' → 'BAAI/bge-m3'."""
    if not model_name:
        return "BAAI/bge-m3"
    aliases = getattr(experiment, "EMBEDDING_ALIASES", {}) or {}
    if isinstance(aliases, dict):
        resolved = aliases.get(model_name.lower(), model_name)
        if resolved != model_name:
            logger.info("Resolved alias '%s' -> '%s'.", model_name, resolved)
            return resolved
    if model_name.lower().replace("/", "").replace("-", "").replace("_", "") in {
        "bgem3", "baaibgem3",
    }:
        return "BAAI/bge-m3"
    return model_name


def _load_models(experiment) -> Tuple[SentenceTransformer, Any, Any, str]:
    """
    Load two views of the *same* BGE-M3 model:

    1. ``SentenceTransformer`` — used for query encoding with trained
       CLS-pooling + normalisation.
    2. Raw ``AutoModel`` + ``AutoTokenizer`` — used for document encoding
       to extract per-token ``last_hidden_state``.

    Both share the same weights (sentence-transformers loads the same
    checkpoint).  Using a single model for both sides ensures queries and
    document chunks live in the same embedding space.
    """
    # Prefer late_model if set; fall back to embedding_model
    model_name = getattr(experiment, "late_model", None) or getattr(
        experiment, "embedding_model", None
    )
    model_name = _resolve_model_name(experiment, model_name)

    logger.info("Late-chunking model: '%s'", model_name)

    device = getattr(experiment, "device", "cpu")
    if device == "cpu" and torch.cuda.is_available():
        device = "cuda"

    # --- Python 3.13 / transformers compatibility patches ---
    import sys
    import types

    # Patch 1: transformers.onnx removed in transformers ≥ 4.30
    if "transformers.onnx" not in sys.modules:
        stub = types.ModuleType("transformers.onnx")
        class OnnxConfig:
            pass
        stub.OnnxConfig = OnnxConfig
        sys.modules["transformers.onnx"] = stub

    # Patch 2: Python 3.13 mixed-type list comparison in transformers
    try:
        import transformers.core_model_loading as _cml
        _orig = _cml.dot_natural_key
        def _safe_key(key):
            return [(0, p) if isinstance(p, int) else (1, p) for p in _orig(key)]
        _cml.dot_natural_key = _safe_key
    except Exception:
        pass

    # --- Load sentence-transformers model (for query encoding) ---
    logger.info("Loading SentenceTransformer '%s' for query encoding.", model_name)
    st_model = SentenceTransformer(model_name, device=device)

    # --- Load raw transformer model (for document hidden-state extraction) ---
    logger.info("Loading raw AutoModel '%s' for document encoding.", model_name)
    try:
        raw_model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.float32
        )
    except TypeError:
        raw_model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True
        )
    raw_model.to(device)
    raw_model.eval()

    raw_tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )

    # Verify CLS/SEP
    cls_id = raw_tokenizer.cls_token_id or raw_tokenizer.bos_token_id
    sep_id = raw_tokenizer.sep_token_id or raw_tokenizer.eos_token_id
    if cls_id is None or sep_id is None:
        raise RuntimeError(
            f"Tokenizer for '{model_name}' missing CLS/SEP "
            f"(cls={raw_tokenizer.cls_token_id}, sep={raw_tokenizer.sep_token_id})."
        )

    # Sanity check: one forward pass
    logger.info(
        "Raw model loaded. hidden_size=%d, CLS=%d, SEP=%d.",
        raw_model.config.hidden_size, cls_id, sep_id,
    )
    _verify_forward(raw_model, raw_tokenizer, device, cls_id, sep_id)

    return st_model, raw_model, raw_tokenizer, device


def _verify_forward(model, tokenizer, device: str, cls_id: int, sep_id: int):
    """Run a tiny forward pass to verify finite outputs."""
    dummy_content = tokenizer.encode("financial report revenue", add_special_tokens=False)[:10]
    if not dummy_content:
        dummy_content = [100] * 10
    input_ids = torch.tensor(
        [[cls_id] + dummy_content + [sep_id]], dtype=torch.long, device=device
    )
    mask = torch.ones_like(input_ids)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=mask)
    hs = out.last_hidden_state
    if not torch.isfinite(hs).all():
        n_nan = int((~torch.isfinite(hs)).sum())
        raise RuntimeError(
            f"Model forward pass produced {n_nan} NaN/Inf values. "
            "Check CUDA memory and model integrity."
        )
    logger.info("Forward-pass sanity check passed (all finite).")


# ---------------------------------------------------------------------------
# Index + retriever classes
# ---------------------------------------------------------------------------

class LateChunkIndex:
    """
    FAISS inner-product index over late-chunked document embeddings,
    with query encoding via SentenceTransformer.
    """

    def __init__(
        self,
        index,
        chunks: List[Document],
        st_model: SentenceTransformer,
        raw_model,
        raw_tokenizer,
        device: str,
    ):
        self.index = index
        self.chunks = chunks
        self.st_model = st_model
        self.raw_model = raw_model
        self.raw_tokenizer = raw_tokenizer
        self.device = device
        self.dim = getattr(raw_model.config, "hidden_size", None)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a query using SentenceTransformer's standard encode() —
        CLS pooling + L2 normalisation, exactly as BGE-M3 was trained.
        """
        vec = self.st_model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vec = np.asarray(vec, dtype=np.float32)

        if not np.isfinite(vec).all():
            logger.warning("embed_query: NaN in query vector — returning zeros.")
            return np.zeros((self.dim,), dtype=np.float32)

        return vec

    def search(self, query: str, k: int) -> List[Document]:
        query_vec = self.embed_query(query)
        return self.search_by_vector(query_vec, k)

    def search_by_vector(self, query_vec: np.ndarray, k: int) -> List[Document]:
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        query_vec = _normalize(query_vec.astype(np.float32))
        scores, indices = self.index.search(query_vec, k)
        out: List[Document] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            doc = self.chunks[idx]
            doc.metadata = dict(doc.metadata or {})
            doc.metadata["score"] = float(score)
            out.append(doc)
        return out


class LateChunkRetriever:
    """LangChain-compatible retriever wrapper."""

    def __init__(self, index: LateChunkIndex, top_k: int):
        self.index = index
        self.k = top_k

    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.index.search(query, self.k)

    def invoke(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)

    def similarity_search(self, query: str, k: Optional[int] = None) -> List[Document]:
        return self.index.search(query, k or self.k)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_late_index_for_scope(
    experiment,
    docs: List[Any],
    scope: str,
    pdf_dir: Optional[Path] = None,
) -> LateChunkIndex:
    if pdf_dir is None:
        pdf_dir = Path(getattr(experiment, "pdf_local_dir", "."))
    fingerprint = _compute_corpus_fingerprint(pdf_dir)
    return build_late_chunk_index(
        experiment, docs=docs, scope=scope, fingerprint=fingerprint
    )