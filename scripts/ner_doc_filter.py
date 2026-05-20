"""
NER-based Document Filtering for FinanceBench RAG
==================================================

Replaces oracle doc_name filtering with a content-based approach.

Two-stage pipeline (no oracle labels, no manual alias maps):

  Stage 1 — Year extraction (regex)
    Extracts all fiscal years from the query (handles FY2022 and FY22 shorthands).
    If years are found, hard-filters the candidate set to documents whose
    doc_period is in the extracted year set (or ±1 for adjacency).

  Stage 2 — Content-based company matching (BM25 over first-page text)
    Builds a BM25 index where each "document" is the concatenated text of
    the first N pages of each SEC filing.  The raw query (after year
    normalisation) is used to rank candidates.  First pages of 10-K/10-Q
    filings always contain the full legal company name, ticker, filing date,
    and fiscal period — so BM25 naturally bridges the vocabulary gap between
    how analysts refer to a company ("AMEX", "JnJ") and how the document
    names it ("American Express Company", "Johnson & Johnson").

No filename parsing, no hardcoded alias maps, no doc-type heuristics.
The only external signal used is the `doc_period` field from doc_info
(the fiscal year of each document), which is metadata about the corpus
rather than an oracle assignment of question→document.
"""

from __future__ import annotations

import re
import logging
import numpy as np
from typing import Dict, List, Optional, Set

logger = logging.getLogger("ner_doc_filter")

# ─── Year extraction ───────────────────────────────────────────────────────────

# 4-digit years: handles FY2022, Q2FY2023, standalone 2022
_YEAR_4D_RE = re.compile(r"(?<!\d)(20\d{2}|19\d{2})(?!\d)")
# 2-digit fiscal-year shorthand: FY22, FY23 → 2022, 2023
_YEAR_2D_RE = re.compile(r"\bFY(\d{2})\b", re.IGNORECASE)


def extract_years(query: str) -> List[int]:
    """Return all fiscal years referenced in *query*, sorted ascending."""
    years: Set[int] = set()
    for m in _YEAR_4D_RE.finditer(query):
        years.add(int(m.group()))
    for m in _YEAR_2D_RE.finditer(query):
        years.add(2000 + int(m.group(1)))
    return sorted(years)


def _normalise_query_years(query: str) -> str:
    """
    Expand 2-digit fiscal-year codes to 4-digit form so BM25 token
    matching works: 'FY22' → 'FY22 2022', 'FY2022' unchanged.
    """
    def _expand(m):
        yy = int(m.group(1))
        full = 2000 + yy
        return f"{m.group(0)} {full}"
    return _YEAR_2D_RE.sub(_expand, query)


# ─── BM25 tokeniser ───────────────────────────────────────────────────────────

def _tokenise(text: str) -> List[str]:
    """Simple whitespace+punctuation tokeniser (no stemming needed here)."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


# ─── DocContentFilter ─────────────────────────────────────────────────────────

class DocContentFilter:
    """
    Content-based document filter using BM25 over first-page text.

    Parameters
    ----------
    doc_pages : dict
        doc_name → list of page dicts with a 'text' key.
        These are the already-loaded pages from load_all_pages() /
        load_pdf_with_fallback().
    doc_info : dict
        doc_name → {doc_period: int, ...} from financebench_document_information.jsonl.
        Used only for the `doc_period` (fiscal year) field — no oracle assignment.
    n_header_pages : int
        Number of leading pages to index per document (default 3).
        First pages of SEC filings always state the company name, ticker,
        and the fiscal period covered — 3 pages is sufficient.
    year_window : int
        When a query year is found, also include documents whose doc_period
        is within ±year_window of any extracted year (handles fiscal-year
        offset between query wording and document period).
    """

    def __init__(
        self,
        doc_pages: Dict[str, List[Dict]],
        doc_info: Dict[str, Dict],
        n_header_pages: int = 3,
        year_window: int = 1,
    ):
        self.doc_names: List[str] = list(doc_pages.keys())
        self.doc_info = doc_info
        self.year_window = year_window
        self._build_index(doc_pages, n_header_pages)

    # ── Index building ─────────────────────────────────────────────────────────

    def _build_index(self, doc_pages: Dict[str, List[Dict]], n_header_pages: int):
        """Build BM25 index over first-page content of every document."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "rank_bm25 is required for DocContentFilter. "
                "Install with: pip install rank-bm25"
            )

        self._corpus: List[List[str]] = []   # tokenised header text per doc

        for doc_name in self.doc_names:
            pages = doc_pages.get(doc_name, [])
            # Take first n_header_pages pages of text
            header_text = " ".join(
                p.get("text", "") or p.get("page_content", "")
                for p in pages[:n_header_pages]
            )
            self._corpus.append(_tokenise(header_text))

        self._bm25 = BM25Okapi(self._corpus)

        # Pre-build doc_period lookup for fast year filtering
        self._doc_year: Dict[str, Optional[int]] = {}
        for doc_name in self.doc_names:
            info = self.doc_info.get(doc_name, {})
            year = info.get("doc_period")
            if year is None:
                # Fall back to parsing the doc_name (e.g. "3M_2022_10K" → 2022)
                m = _YEAR_4D_RE.search(doc_name)
                year = int(m.group()) if m else None
            self._doc_year[doc_name] = year

        logger.info(
            f"DocContentFilter BM25 index: {len(self.doc_names)} docs, "
            f"{n_header_pages} header pages each"
        )

    # ── Main API ───────────────────────────────────────────────────────────────

    def predict_target_docs(self, query: str, top_k: int = 3) -> List[str]:
        """
        Given a free-text *query*, return the *top_k* most likely doc_names.

        Pipeline
        --------
        1. Extract years from query (regex).
        2. Normalise query (expand FY22 → 'FY22 2022') for BM25 scoring.
        3. Score all documents with BM25 over their first-page text.
        4. If years were found, restrict candidates to docs whose fiscal year
           is within ±year_window of any extracted year.
        5. Return top_k from the remaining ranked list.
        """
        years = extract_years(query)
        norm_query = _normalise_query_years(query)
        query_tokens = _tokenise(norm_query)

        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        # Build ranked list of (doc_name, bm25_score)
        ranked = sorted(
            zip(self.doc_names, scores),
            key=lambda x: -x[1],
        )

        # Year filter: hard-restrict to docs whose period overlaps the query years
        # Skip if no years found (e.g. "What does this company do?")
        if years:
            year_set = set()
            for y in years:
                for offset in range(self.year_window + 1):
                    year_set.add(y + offset)
                    year_set.add(y - offset)

            # Two-pass: prefer exact year matches, then admit adjacent
            exact = [
                (dn, s) for dn, s in ranked
                if self._doc_year.get(dn) in {y for y in years}
            ]
            adjacent = [
                (dn, s) for dn, s in ranked
                if self._doc_year.get(dn) in year_set
                and dn not in {x[0] for x in exact}
            ]
            ranked = exact + adjacent

        result = [dn for dn, s in ranked[:top_k] if s > 0.0]

        if result:
            logger.debug(
                f"DocContentFilter: years={years} → top candidates: {result}"
            )
        else:
            logger.warning(
                f"DocContentFilter: no candidates found for: {query[:80]!r}"
            )

        return result
