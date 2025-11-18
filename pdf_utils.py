"""Utilities for resolving and loading PDFs for FinanceBench experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List
import logging
import re
from urllib.parse import urlparse

try:
    from langchain.document_loaders import PyMuPDFLoader
except Exception:  # pragma: no cover - surfaced at runtime with clear error
    PyMuPDFLoader = None

logger = logging.getLogger(__name__)


def _normalize_name(name: str) -> str:
    """Return a simplified version of a filename / doc_name for matching."""
    name = (name or "").lower()
    if "." in name:
        name = name.rsplit(".", 1)[0]
    return re.sub(r"[^a-z0-9]+", "", name)


def _find_local_pdf(doc_name: str, local_dir: Optional[str]) -> Optional[Path]:
    """Try to find a local PDF in `local_dir` that matches `doc_name`."""
    if not local_dir:
        return None

    try:
        pdf_dir = Path(local_dir)
    except Exception:
        return None

    if not pdf_dir.exists() or not pdf_dir.is_dir():
        return None

    norm_target = _normalize_name(doc_name)
    candidates = []

    try:
        for p in pdf_dir.iterdir():
            if not p.is_file() or p.suffix.lower() != ".pdf":
                continue
            stem_norm = _normalize_name(p.stem)
            if stem_norm == norm_target or norm_target in stem_norm:
                candidates.append(p)
    except Exception:
        return None

    if candidates:
        candidates.sort()
        return candidates[0]

    direct = pdf_dir / doc_name
    if direct.exists() and direct.is_file():
        return direct
    if not direct.suffix and direct.with_suffix(".pdf").exists():
        return direct.with_suffix(".pdf")

    return None


def load_pdf_with_fallback(
    doc_name: str,
    doc_link: str,
    local_dir: Optional[str],
) -> Tuple[List, str]:
    """Load a PDF strictly from the local directory using PyMuPDFLoader."""
    if PyMuPDFLoader is None:
        raise RuntimeError(
            "PyMuPDFLoader (langchain.document_loaders) is required but not installed."
        )

    local_path = _find_local_pdf(doc_name, local_dir)
    if local_path is None and doc_link:
        parsed = urlparse(doc_link)
        candidate = Path(parsed.path).stem if parsed.path else Path(doc_link).stem
        logger.debug(
            "Doc '%s' not found by name; attempting link-derived filename '%s'",
            doc_name,
            candidate,
        )
        local_path = _find_local_pdf(candidate, local_dir)

    if local_path is None:
        logger.warning("Could not locate local PDF for '%s'", doc_name)
        return [], "none"

    try:
        loader = PyMuPDFLoader(str(local_path))
        documents = loader.load()
        for doc in documents:
            meta = doc.metadata or {}
            meta.setdefault("doc_name", doc_name)
            meta.setdefault("doc_link", doc_link)
            meta.setdefault("source", "pdf")
            doc.metadata = meta
        return documents, f"local:{local_path.name}"
    except Exception as exc:
        logger.warning("Failed to parse PDF %s: %s", local_path, exc)
        return [], "none"
