"""
Chunking utilities for FinanceBench-style RAG experiments.

This module centralizes chunking logic so that new strategies (e.g. hierarchical
10-K section-aware chunking) don't live inside `rag_experiments.py`.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from .rag_dependencies import Document, RecursiveCharacterTextSplitter
except Exception:  # pragma: no cover
    from rag_dependencies import Document, RecursiveCharacterTextSplitter

try:
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover
    AutoTokenizer = None

logger = logging.getLogger(__name__)


DEFAULT_SEPARATORS: List[str] = ["\n\n", "\n", ". ", " ", ""]


@dataclass(frozen=True)
class ChunkingConfig:
    strategy: str
    unit: str
    chunk_size: int
    chunk_overlap: int
    parent_chunk_size: Optional[int] = None
    parent_chunk_overlap: Optional[int] = None
    child_chunk_size: Optional[int] = None
    child_chunk_overlap: Optional[int] = None


def _safe_int(value: Any, default: int) -> int:
    try:
        if value is None:
            return default
        as_int = int(value)
        return as_int if as_int > 0 else default
    except Exception:
        return default


def build_recursive_text_splitter(
    *,
    chunk_size: int,
    chunk_overlap: int,
    unit: str = "chars",
    llm_model_name: Optional[str] = None,
    separators: Optional[Sequence[str]] = None,
    exp_logger: Optional[logging.Logger] = None,
) -> Any:
    """
    Build a RecursiveCharacterTextSplitter, optionally token-based (if supported).
    Mirrors the previous behavior in `rag_experiments.py`.
    """
    exp_logger = exp_logger or logger
    separators = list(separators or DEFAULT_SEPARATORS)

    if RecursiveCharacterTextSplitter is None:
        raise RuntimeError(
            "RecursiveCharacterTextSplitter is not available. "
            "Install langchain / langchain_text_splitters."
        )

    unit = (unit or "chars").lower().strip()

    if unit == "tokens":
        if AutoTokenizer is None:
            exp_logger.warning("transformers not available; falling back to char chunking.")
            unit = "chars"
        elif not llm_model_name:
            exp_logger.warning("No llm_model_name provided; falling back to char chunking.")
            unit = "chars"
        else:
            exp_logger.info(f"Using token-based chunking with model: {llm_model_name}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
                return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                    tokenizer,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=separators,
                )
            except Exception as e:
                exp_logger.warning(
                    f"Failed to load tokenizer for token chunking: {e}. Falling back to chars."
                )
                unit = "chars"

    # Char-based default
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=separators,
    )


def initialize_chunker(experiment) -> Any:
    """
    Initialize and attach the experiment's base text splitter.

    For hierarchical chunking we still attach a child splitter to keep the
    experiment interface stable, but actual hierarchical splitting is handled
    in `chunk_text()`.
    """
    exp_logger = getattr(experiment, "logger", logger)

    strategy = getattr(experiment, "chunking_strategy", "recursive") or "recursive"
    unit = getattr(experiment, "chunking_unit", "chars") or "chars"
    chunk_size = _safe_int(getattr(experiment, "chunk_size", None), 1024)
    chunk_overlap = _safe_int(getattr(experiment, "chunk_overlap", None), 30)

    # For hierarchical: default the "child" size to chunk_size unless explicitly set.
    if strategy == "hierarchical":
        child_size = _safe_int(getattr(experiment, "child_chunk_size", None), chunk_size)
        child_overlap = _safe_int(getattr(experiment, "child_chunk_overlap", None), chunk_overlap)
        splitter = build_recursive_text_splitter(
            chunk_size=child_size,
            chunk_overlap=child_overlap,
            unit=unit,
            llm_model_name=getattr(experiment, "llm_model_name", None),
            exp_logger=exp_logger,
        )
    else:
        splitter = build_recursive_text_splitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            unit=unit,
            llm_model_name=getattr(experiment, "llm_model_name", None),
            exp_logger=exp_logger,
        )

    setattr(experiment, "text_splitter", splitter)
    return splitter


_ITEM_HEADER_RE = re.compile(
    r"(?im)^\s*(?:item|part)\s+"
    r"(?:(?:\d{1,2}(?:\.[0-9]+)?)|[ivx]+)"
    r"[a-z]?\s*"
    r"(?:[\.\-–—:]\s*)?"
    r"(.{0,120})\s*$"
)


def _split_into_sections_10k(text: str) -> List[Tuple[str, str]]:
    """
    Best-effort 10-K section splitter based on "ITEM x" / "PART x" headings.

    Returns a list of (section_title, section_text).
    """
    if not text:
        return []

    lines = text.splitlines()
    sections: List[Tuple[str, List[str]]] = []
    current_title = "Document"
    current_lines: List[str] = []

    def _flush():
        nonlocal current_lines
        joined = "\n".join(current_lines).strip()
        if joined:
            sections.append((current_title, current_lines))
        current_lines = []

    for line in lines:
        match = _ITEM_HEADER_RE.match(line)
        # Heuristic: treat as a section boundary only if the line is short-ish
        # and not obviously a table-of-contents leader line.
        if match and len(line.strip()) <= 160 and not re.search(r"\.{5,}", line):
            # Start new section
            _flush()
            suffix = (match.group(1) or "").strip()
            title = line.strip()
            if suffix and suffix.lower() not in title.lower():
                title = f"{title} {suffix}".strip()
            current_title = title
            continue
        current_lines.append(line)

    _flush()

    out: List[Tuple[str, str]] = []
    for title, buf in sections:
        body = "\n".join(buf).strip()
        if body:
            out.append((title, body))

    if not out:
        return [("Document", text)]
    return out


def _concat_page_documents(pages: Sequence[Any]) -> str:
    parts: List[str] = []
    for doc in pages or []:
        page_text = getattr(doc, "page_content", None) or getattr(doc, "content", None) or ""
        if isinstance(page_text, (bytes, bytearray)):
            page_text = page_text.decode("utf-8", errors="replace")
        page_text = str(page_text).strip()
        if page_text:
            parts.append(page_text)
    return "\n\n".join(parts).strip()


def _fingerprint(text: str, meta: Dict[str, Any], salt: str) -> str:
    base = f"{salt}|{meta.get('doc_name','')}|{meta.get('doc_link','')}|{text[:200]}"
    return hashlib.md5(base.encode("utf-8", errors="ignore")).hexdigest()[:12]


def _get_cached_splitter(
    experiment,
    *,
    cache_attr: str,
    chunk_size: int,
    chunk_overlap: int,
    unit: str,
    llm_model_name: Optional[str],
    exp_logger: logging.Logger,
) -> Any:
    """
    Cache splitters on the experiment instance to avoid re-creating tokenizers /
    splitters repeatedly during large ingests.
    """
    key = (chunk_size, chunk_overlap, (unit or "chars").lower().strip(), llm_model_name or "")
    cached = getattr(experiment, cache_attr, None)
    if isinstance(cached, tuple) and len(cached) == 2 and cached[0] == key:
        return cached[1]

    splitter = build_recursive_text_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        unit=unit,
        llm_model_name=llm_model_name,
        exp_logger=exp_logger,
    )
    setattr(experiment, cache_attr, (key, splitter))
    return splitter


def _to_documents_from_texts(texts: Iterable[str], metadatas: Iterable[Dict[str, Any]]) -> List[Document]:
    docs: List[Document] = []
    for t, m in zip(texts, metadatas):
        docs.append(Document(page_content=t, metadata=m))
    return docs


def chunk_text(experiment, text: Any, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
    """
    Chunk text for the experiment's configured strategy.

    - recursive: just uses experiment.text_splitter (RecursiveCharacterTextSplitter)
    - hierarchical: section-aware (ITEM/PART), then parent chunks, then child chunks
    """
    metadata = dict(metadata or {})
    strategy = getattr(experiment, "chunking_strategy", "recursive") or "recursive"
    exp_logger = getattr(experiment, "logger", logger)

    if getattr(experiment, "text_splitter", None) is None:
        initialize_chunker(experiment)

    # Normalize input to a string or list[Document]
    if isinstance(text, list):
        # Keep legacy behavior for recursive chunking: split_documents over page docs.
        if strategy != "hierarchical":
            splitter = getattr(experiment, "text_splitter", None)
            if splitter is None:
                initialize_chunker(experiment)
                splitter = getattr(experiment, "text_splitter", None)

            documents_input: List[Any] = []
            for doc in text:
                doc_meta = dict(metadata)
                doc_meta.update(getattr(doc, "metadata", None) or {})
                try:
                    doc.metadata = doc_meta
                except Exception:
                    pass
                documents_input.append(doc)
            return splitter.split_documents(documents_input)

        # For hierarchical: merge page docs into a single text blob for section detection
        # (10-K headings often span pages, so per-page splitting hurts).
        pages = text
        merged = _concat_page_documents(pages)
        if not merged:
            return []

        # Add page range metadata if available.
        page_nums: List[int] = []
        for doc in pages:
            meta = getattr(doc, "metadata", None) or {}
            page = meta.get("page")
            try:
                if page is not None:
                    page_nums.append(int(page))
            except Exception:
                continue
        if page_nums:
            metadata = dict(metadata)
            metadata.setdefault("page_start", min(page_nums))
            metadata.setdefault("page_end", max(page_nums))
            metadata.setdefault("num_pages", len(set(page_nums)))

        return chunk_text(experiment, merged, metadata=metadata)

    if isinstance(text, (bytes, bytearray)):
        text = text.decode("utf-8", errors="replace")
    text = str(text or "")
    if not text.strip():
        return []

    if strategy != "hierarchical":
        splitter = getattr(experiment, "text_splitter", None)
        if splitter is None:
            initialize_chunker(experiment)
            splitter = getattr(experiment, "text_splitter", None)
        return splitter.create_documents(texts=[text], metadatas=[metadata])

    # --- Hierarchical chunking (10-K section -> parent -> child) -----------------
    parent_size = _safe_int(getattr(experiment, "parent_chunk_size", None), 4000)
    parent_overlap = _safe_int(getattr(experiment, "parent_chunk_overlap", None), 200)
    child_size = _safe_int(getattr(experiment, "child_chunk_size", None), _safe_int(getattr(experiment, "chunk_size", None), 1024))
    child_overlap = _safe_int(getattr(experiment, "child_chunk_overlap", None), _safe_int(getattr(experiment, "chunk_overlap", None), 30))

    unit = getattr(experiment, "chunking_unit", "chars") or "chars"
    llm_model_name = getattr(experiment, "llm_model_name", None)

    parent_splitter = _get_cached_splitter(
        experiment,
        cache_attr="_hier_parent_splitter",
        chunk_size=parent_size,
        chunk_overlap=parent_overlap,
        unit=unit,
        llm_model_name=llm_model_name,
        exp_logger=exp_logger,
    )
    child_splitter = _get_cached_splitter(
        experiment,
        cache_attr="_hier_child_splitter",
        chunk_size=child_size,
        chunk_overlap=child_overlap,
        unit=unit,
        llm_model_name=llm_model_name,
        exp_logger=exp_logger,
    )

    sections = _split_into_sections_10k(text)
    out_docs: List[Document] = []

    for section_index, (section_title, section_text) in enumerate(sections):
        if not section_text.strip():
            continue

        parent_docs = parent_splitter.create_documents(
            texts=[section_text],
            metadatas=[metadata],
        )

        for parent_index, parent_doc in enumerate(parent_docs):
            parent_text = getattr(parent_doc, "page_content", "") or ""
            if isinstance(parent_text, (bytes, bytearray)):
                parent_text = parent_text.decode("utf-8", errors="replace")
            parent_text = str(parent_text)
            if not parent_text.strip():
                continue

            parent_id = _fingerprint(
                parent_text,
                metadata,
                salt=f"{section_index}:{parent_index}",
            )

            child_docs = child_splitter.create_documents(
                texts=[parent_text],
                metadatas=[metadata],
            )

            for child_index, child_doc in enumerate(child_docs):
                child_text = getattr(child_doc, "page_content", "") or ""
                if isinstance(child_text, (bytes, bytearray)):
                    child_text = child_text.decode("utf-8", errors="replace")
                child_text = str(child_text).strip()
                if not child_text:
                    continue

                # Prepend hierarchical breadcrumbs for better retrieval + grounding
                decorated = f"[{section_title}]\n{child_text}"

                meta = dict(metadata)
                meta.update(
                    {
                        "chunking_strategy": "hierarchical",
                        "chunk_level": "child",
                        "section_title": section_title,
                        "section_index": section_index,
                        "parent_id": parent_id,
                        "parent_index": parent_index,
                        "child_index": child_index,
                    }
                )

                out_docs.append(Document(page_content=decorated, metadata=meta))

    return out_docs

