"""
src/ingestion/page_processor.py
Utilities for extracting pages and creating digests for page-level retrieval.
"""
import re
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

NUM_LINE = re.compile(r".*\d.*")

def make_page_digest(text: str, max_chars: int = 6000) -> str:
    """
    Creates a density-focused summary of the page, prioritizing
    lines with numbers, headings, and financial keywords.
    """
    if not text:
        return ""
        
    lines = [re.sub(r"\s+", " ", l).strip() for l in text.splitlines()]
    lines = [l for l in lines if l]

    keep = []
    for l in lines:
        # Keep lines likely useful for FinanceBench-style evidence
        if NUM_LINE.match(l):
            keep.append(l)
            continue
        # Keep headings / section markers / table titles (heuristic)
        if l.isupper() and 6 <= len(l) <= 120:
            keep.append(l)
            continue
        if "Item " in l or "ITEM " in l:
            keep.append(l)
            continue
        # Keep financial context keywords
        if any(u in l.lower() for u in ["in thousands", "in millions", "per share", "%", "fiscal", "year ended"]):
            keep.append(l)
            continue

    digest = "\n".join(keep)
    # Fallback if digest is empty (e.g., purely narrative page), use raw prefix
    if not digest.strip():
        return text[:max_chars]
        
    return digest[:max_chars]

def extract_pages_from_pdf(pdf_path: Path, doc_name: str) -> List[Dict[str, Any]]:
    """
    Extracts pages from a PDF file.
    Returns a list of dicts with:
      - text: Raw text
      - digest: Processed text
      - page: Page number (0-indexed)
      - doc_name: Document name
    """
    try:
        doc = fitz.open(pdf_path)
        pages = []
        for i in range(len(doc)):
            text = doc.load_page(i).get_text("text")
            digest = make_page_digest(text)
            pages.append({
                "text": text,
                "digest": digest,
                "page": i, # 0-indexed
                "doc_name": doc_name,
                "source": str(pdf_path)
            })
        return pages
    except Exception as e:
        logger.error(f"Failed to extract pages from {pdf_path}: {e}")
        return []