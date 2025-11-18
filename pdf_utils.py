"""Utilities for resolving and loading PDFs for FinanceBench experiments.

`load_pdf_with_fallback` is the single public entry point. It takes a
doc_name + doc_link from the dataset, tries to locate a matching PDF
in a local directory (usually `<repo>/pdfs`), and falls back to using
the remote URL via `experiment._load_pdf_text` if necessary.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Callable, Tuple
import logging
import re

logger = logging.getLogger(__name__)


def _normalize_name(name: str) -> str:
    """Return a simplified version of a filename / doc_name for matching."""
    name = name.lower()
    # Remove extension
    if "." in name:
        name = name.rsplit(".", 1)[0]
    # Strip non-alphanumeric characters
    name = re.sub(r"[^a-z0-9]+", "", name)
    return name


def _find_local_pdf(
    doc_name: str,
    local_dir: Optional[str],
) -> Optional[Path]:
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

    # First, try exact stem match across all files in the directory
    try:
        for p in pdf_dir.iterdir():
            if not p.is_file():
                continue
            if p.suffix.lower() != ".pdf":
                continue
            if _normalize_name(p.stem) == norm_target:
                candidates.append(p)
    except Exception:
        # If something goes wrong, just ignore and fall back to URL
        return None

    if candidates:
        # Prefer deterministic ordering
        candidates.sort()
        return candidates[0]

    # As a last resort, try treating doc_name as a direct filename
    direct = pdf_dir / doc_name
    if direct.exists() and direct.is_file():
        return direct
    if not direct.suffix and (direct.with_suffix(".pdf")).exists():
        return direct.with_suffix(".pdf")

    return None


def load_pdf_with_fallback(
    doc_name: str,
    doc_link: str,
    local_dir: Optional[str],
    pdf_loader_func: Callable[[str], Optional[str]],
) -> Tuple[Optional[str], str]:
    """Load PDF text either from local cache or remote URL.

    Parameters
    ----------
    doc_name:
        FinanceBench `doc_name` field for the sample.
    doc_link:
        Original document URL (may be empty in some cases).
    local_dir:
        Local directory containing cached PDFs (e.g. `<repo>/pdfs`).
    pdf_loader_func:
        Function (usually `experiment._load_pdf_text`) that accepts either
        a local filesystem path or a URL, and returns the extracted text.

    Returns
    -------
    (text, source)
        `text` is the extracted PDF text (or None on failure).
        `source` is a string describing where it came from:
            - "local:<filename>" when loaded from local_dir
            - "remote:url"     when loaded from the URL
            - "none"           when nothing could be loaded
    """
    # 1) Try local PDF (preferred)
    try:
        local_path = _find_local_pdf(doc_name, local_dir)
    except Exception:
        local_path = None

    if local_path is not None:
        try:
            logger.info(f"Trying local PDF for '{doc_name}': {local_path}")
            text = pdf_loader_func(str(local_path))
            if text:
                return text, f"local:{local_path.name}"
        except Exception as e:
            logger.warning(f"Failed to load local PDF {local_path}: {e}")

    # 2) Fallback to remote URL if available
    if doc_link:
        try:
            logger.info(f"Falling back to remote URL for '{doc_name}': {doc_link}")
            text = pdf_loader_func(doc_link)
            if text:
                return text, "remote:url"
        except Exception as e:
            logger.warning(f"Failed to load remote PDF {doc_link}: {e}")

    logger.warning(f"Could not load PDF for '{doc_name}' from local or remote sources")
    return None, "none"
