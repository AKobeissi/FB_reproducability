#!/usr/bin/env python3
"""
edgar_pdf_opus.py — EDGAR Filing Downloader with Page-Faithful PDF Conversion
==============================================================================

Downloads SEC filings (10-K, 10-Q, 8-K, 20-F, Earnings Calls) from EDGAR and
produces PDFs where each logical page maps to exactly one PDF page.

Key design:
  1. Native PDFs on EDGAR → download directly (already properly paginated).
  2. HTM-only filings → download HTML, detect logical page boundaries,
     inject CSS page-break rules, render via Playwright with a tall custom
     page size to prevent intra-page overflow, then post-validate.

Page-boundary detection for HTM files:
  - <hr> tags (common EDGAR page separator)
  - Page-number patterns: "Page X of Y", "Page X", standalone numbers,
    "F-XX", "S-XX" (financial-statement and supplemental page numbering)
  - Explicit CSS page-break markers already in the HTML

Post-processing:
  - Detect and remove blank pages (all-white) using pypdf text extraction
  - Validate final page count

EDGAR compliance:
  - Proper User-Agent header (SEC requirement)
  - Rate limiting (0.5s between requests, SEC allows 10 req/s)
  - Retries with exponential backoff

Input:  JSONL files with evidences[].doc_name like "IBM_2024Q2_10Q"
Output: PDF-Opus/{doc_name}.pdf + manifest.jsonl

Usage:
    python scripts/edgar_pdf_opus.py \
        --jsonl data/finqa_test.jsonl data/secqa_test.jsonl \
        --out_dir PDF-Opus

Dependencies:
    pip install requests beautifulsoup4 lxml pypdf playwright
    playwright install chromium
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests

# ---------------------------------------------------------------------------
# SEC / EDGAR constants
# ---------------------------------------------------------------------------
# SEC requires: "FirstName LastName email@domain.com"
EDGAR_USER_AGENT = "Amine Kobeissi amine.kobeissi@umontreal.ca"

HEADERS = {
    "User-Agent": EDGAR_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
}

# SEC allows 10 req/s; 0.5s per request gives comfortable headroom
RATE_LIMIT_SLEEP = 0.5

# Minimum PDF size to consider valid (avoid truncated downloads)
MIN_PDF_BYTES = 5_000

# Page size for HTM→PDF: 8.5in × 22in (extra tall to prevent overflow)
# Standard Letter is 8.5×11; we double the height to accommodate dense pages,
# then rely on CSS page-break injection for proper pagination.
TALL_PAGE_WIDTH  = "8.5in"
TALL_PAGE_HEIGHT = "22in"

# Patterns that indicate page numbers at the bottom of SEC filings
PAGE_NUMBER_PATTERNS = [
    # "Page X of Y", "Page X"
    re.compile(r"^\s*Page\s+\d+\s*(?:of\s+\d+)?\s*$", re.IGNORECASE),
    # "F-XX" (financial statement numbering)
    re.compile(r"^\s*F[\-–]\d+\s*$"),
    # "S-XX" (supplemental numbering)
    re.compile(r"^\s*S[\-–]\d+\s*$"),
    # Standalone numbers (1, 2, ... 200+) — common in 10-K/10-Q
    re.compile(r"^\s*\d{1,3}\s*$"),
    # "- X -" style
    re.compile(r"^\s*[-–—]\s*\d+\s*[-–—]\s*$"),
    # "X of Y" without "Page" prefix
    re.compile(r"^\s*\d+\s+of\s+\d+\s*$", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ParsedDocName:
    doc_name: str
    ticker: str
    year: int
    quarter: Optional[int]  # 1-4 or None for annual / non-quarterly
    form: str               # "10-K", "10-Q", "8-K", "20-F", "EC", etc.


@dataclass
class DownloadResult:
    doc_name: str
    success: bool = False
    pdf_path: str = ""
    pdf_bytes: int = 0
    pdf_pages: int = 0
    method: str = ""        # "native_pdf" | "htm_convert" | "cached" | "failed"
    error: str = ""
    cik: str = ""
    accession: str = ""
    primary_url: str = ""
    filing_date: str = ""
    sources: List[str] = field(default_factory=list)
    pages_referenced: List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def rate_limited_get(url: str, session: requests.Session = None,
                     retries: int = 3, backoff: float = 2.0,
                     **kwargs) -> requests.Response:
    """GET with rate limiting, retries, and exponential backoff."""
    time.sleep(RATE_LIMIT_SLEEP)
    s = session or requests
    for attempt in range(retries):
        try:
            r = s.get(url, headers=HEADERS, timeout=45, **kwargs)
            if r.status_code == 429:
                wait = backoff * (2 ** attempt)
                print(f"    [429] Rate limited — waiting {wait:.0f}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                wait = backoff * (2 ** attempt)
                print(f"    [retry {attempt+1}/{retries}] {e} — waiting {wait:.0f}s")
                time.sleep(wait)
            else:
                raise
    raise requests.exceptions.RetryError(f"Failed after {retries} retries: {url}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Parse doc_name → (ticker, year, quarter, form_type)
# ═══════════════════════════════════════════════════════════════════════════════

_DOC_RE = re.compile(
    r"^(?P<ticker>.+?)_(?P<period>\d{4}(?:Q[1-4])?)_(?P<form>.+)$", re.IGNORECASE
)
_PERIOD_RE = re.compile(r"^(?P<year>\d{4})(?:Q(?P<q>[1-4]))?$", re.IGNORECASE)


def _normalize_form(raw: str) -> str:
    """Normalize form type strings: '10K' → '10-K', '10Q' → '10-Q', etc."""
    t = raw.upper().strip().replace("_", "").replace(" ", "")
    for src, dst in [
        ("10K", "10-K"), ("10Q", "10-Q"), ("8K", "8-K"),
        ("20F", "20-F"), ("40F", "40-F"),
    ]:
        if t == src or t == dst:
            return dst
    # Earnings call variants
    if t in ("EC", "EARNINGSCALL", "EARNINGS"):
        return "EC"
    return t


def parse_doc_name(doc_name: str) -> ParsedDocName:
    """
    Parse doc_name strings:
        'IBM_2024Q2_10Q'   → ('IBM', 2024, 2, '10-Q')
        'AAPL_2020_10K'    → ('AAPL', 2020, None, '10-K')
        'BRK_B_2018_10K'   → ('BRK_B', 2018, None, '10-K')
        'AAL_2013_10K'     → ('AAL', 2013, None, '10-K')
    """
    m = _DOC_RE.match(doc_name.strip())
    if not m:
        raise ValueError(f"Cannot parse doc_name: {doc_name!r}")

    ticker = m.group("ticker").upper()
    period = m.group("period").upper()
    form = _normalize_form(m.group("form"))

    pm = _PERIOD_RE.match(period)
    if not pm:
        raise ValueError(f"Cannot parse period {period!r} in {doc_name!r}")

    year = int(pm.group("year"))
    q = pm.group("q")

    return ParsedDocName(
        doc_name=doc_name,
        ticker=ticker,
        year=year,
        quarter=int(q) if q else None,
        form=form,
    )


def ticker_variants(ticker: str) -> List[str]:
    """Generate plausible ticker spellings for CIK lookup."""
    t = ticker.upper()
    seen, out = set(), []
    for v in [
        t,
        t.replace("_", "-"),
        t.replace("_", "."),
        t.replace("-", "."),
        t.replace(".", "-"),
        t.replace("_", ""),
        t.replace("-", ""),
        t.replace(".", ""),
    ]:
        vu = v.upper()
        if vu not in seen:
            seen.add(vu)
            out.append(vu)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Ticker → CIK resolution
# ═══════════════════════════════════════════════════════════════════════════════

_ticker_to_cik: Dict[str, str] = {}
_ticker_map_loaded = False


def _load_ticker_map():
    """Load SEC's official ticker→CIK mapping (cached across calls)."""
    global _ticker_to_cik, _ticker_map_loaded
    if _ticker_map_loaded:
        return

    print("  Loading SEC ticker→CIK map...")
    try:
        r = rate_limited_get("https://www.sec.gov/files/company_tickers.json")
        for entry in r.json().values():
            tk = entry.get("ticker", "").upper()
            cik = str(entry.get("cik_str", "")).zfill(10)
            if tk:
                _ticker_to_cik[tk] = cik
        print(f"  Loaded {len(_ticker_to_cik)} tickers")
    except Exception as e:
        print(f"  [warn] Failed to load ticker map: {e}")
    _ticker_map_loaded = True


def get_cik(ticker: str) -> Optional[str]:
    """Resolve ticker to 10-digit CIK string."""
    _load_ticker_map()

    for variant in ticker_variants(ticker):
        if variant in _ticker_to_cik:
            return _ticker_to_cik[variant]

    # Fallback: EDGAR full-text search
    try:
        r = rate_limited_get(
            "https://efts.sec.gov/LATEST/search-index",
            params={"q": f'"{ticker}"', "forms": "10-K,10-Q,8-K"},
        )
        hits = r.json().get("hits", {}).get("hits", [])
        for hit in hits:
            src = hit.get("_source", {})
            if src.get("ticker_symbol", "").upper() in ticker_variants(ticker):
                cik = str(src.get("entity_id", "")).zfill(10)
                _ticker_to_cik[ticker.upper()] = cik
                return cik
    except Exception as e:
        print(f"    [warn] EFTS search fallback failed: {e}")

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Find filing on EDGAR (CIK + form + year/quarter → accession number)
# ═══════════════════════════════════════════════════════════════════════════════

QUARTER_END_MONTHS = {1: 3, 2: 6, 3: 9, 4: 12}


def get_accession_number(cik: str, parsed: ParsedDocName) -> Optional[Tuple[str, str, str]]:
    """
    Find the best-matching filing accession number.
    Returns (accession_number, filing_date, primary_doc) or None.
    """
    submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        r = rate_limited_get(submissions_url)
        data = r.json()
    except Exception as e:
        print(f"    [error] Submissions fetch failed: {e}")
        return None

    # Collect all filing records from recent + older filings
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])
    report_dates = recent.get("reportDate", [])

    # Also check older filing files
    older_files = data.get("filings", {}).get("files", [])
    for older in older_files:
        fname = older.get("name", "")
        if not fname:
            continue
        try:
            r2 = rate_limited_get(f"https://data.sec.gov/submissions/{fname}")
            od = r2.json()
            forms.extend(od.get("form", []))
            dates.extend(od.get("filingDate", []))
            accessions.extend(od.get("accessionNumber", []))
            primary_docs.extend(od.get("primaryDocument", []))
            report_dates.extend(od.get("reportDate", []))
        except Exception:
            continue

    target_form = parsed.form
    target_year = parsed.year
    target_quarter = parsed.quarter

    # Build candidates
    best_score = -1
    best_match = None

    for i in range(len(forms)):
        form_i = forms[i].upper().replace(" ", "")
        # Normalize for comparison
        form_normalized = _normalize_form(form_i)

        if form_normalized != target_form:
            # Allow 10-K/A for 10-K, 10-Q/A for 10-Q (amended filings)
            if not (form_normalized.startswith(target_form) and "/A" in form_i.upper()):
                continue

        filing_date = dates[i] if i < len(dates) else ""
        report_date = report_dates[i] if i < len(report_dates) else ""
        acc = accessions[i] if i < len(accessions) else ""
        pdoc = primary_docs[i] if i < len(primary_docs) else ""

        if not acc:
            continue

        # Score this candidate
        score = 0

        # Match on report date year
        rd_year = int(report_date[:4]) if report_date and len(report_date) >= 4 else 0
        fd_year = int(filing_date[:4]) if filing_date and len(filing_date) >= 4 else 0
        rd_month = int(report_date[5:7]) if report_date and len(report_date) >= 7 else 0

        if rd_year == target_year:
            score += 60
        elif fd_year == target_year:
            score += 25
        elif fd_year == target_year + 1 and target_form == "10-K":
            # 10-K for fiscal year X is often filed in year X+1
            score += 20

        # Match quarter if specified
        if target_quarter is not None and rd_month > 0:
            expected_month = QUARTER_END_MONTHS.get(target_quarter)
            if expected_month and rd_month == expected_month:
                score += 40
            elif expected_month and abs(rd_month - expected_month) <= 1:
                score += 20

        # Prefer exact form match over amended
        if form_normalized == target_form and "/A" not in form_i:
            score += 10

        # Prefer filings with a primary doc
        if pdoc:
            score += 5

        if score > best_score:
            best_score = score
            best_match = (acc, filing_date, pdoc)

    return best_match


# ═══════════════════════════════════════════════════════════════════════════════
# 4. List filing documents from the filing index page
# ═══════════════════════════════════════════════════════════════════════════════

def get_filing_documents(cik: str, accession: str) -> List[Dict]:
    """Parse the EDGAR filing index page to list all documents."""
    acc_no_dash = accession.replace("-", "")
    index_url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{acc_no_dash}/"
    base_url = index_url.rstrip("/")

    try:
        r = rate_limited_get(index_url)
        html = r.text
    except Exception as e:
        print(f"    [error] Filing index fetch failed: {e}")
        return []

    # Parse the filing index table
    docs = []
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.IGNORECASE | re.DOTALL)
    for row in rows:
        hrefs = re.findall(r'href="([^"]+)"', row, re.IGNORECASE)
        if not hrefs:
            continue
        filename = hrefs[0].split("/")[-1]
        if not filename or "?" in filename:
            continue
        cells = [
            re.sub(r"<[^>]+>", "", c).strip()
            for c in re.findall(r"<td[^>]*>(.*?)</td>", row, re.IGNORECASE | re.DOTALL)
        ]
        docs.append({
            "sequence": cells[0] if len(cells) > 0 else "",
            "description": cells[1] if len(cells) > 1 else "",
            "document": filename,
            "type": cells[3] if len(cells) > 3 else "",
            "url": f"{base_url}/{filename}",
        })

    return docs


# ═══════════════════════════════════════════════════════════════════════════════
# 5a. Direct PDF download
# ═══════════════════════════════════════════════════════════════════════════════

def download_pdf_direct(url: str, dest: Path) -> bool:
    """Download a native PDF from EDGAR."""
    try:
        r = rate_limited_get(url, stream=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 16):
                f.write(chunk)
        size = dest.stat().st_size
        if size < MIN_PDF_BYTES:
            print(f"    [warn] PDF too small ({size} bytes), may be corrupt")
            dest.unlink(missing_ok=True)
            return False
        print(f"    ✓ Native PDF downloaded ({size // 1024} KB)")
        return True
    except Exception as e:
        print(f"    [error] PDF download failed: {e}")
        dest.unlink(missing_ok=True)
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# 5b. HTM → PDF with page-faithful conversion
# ═══════════════════════════════════════════════════════════════════════════════

def _is_page_number_text(text: str) -> bool:
    """Check if a text string looks like a page number."""
    text = text.strip()
    if not text:
        return False
    return any(pat.match(text) for pat in PAGE_NUMBER_PATTERNS)


def _fix_encoding(html: str) -> str:
    """
    Fix encoding artifacts common in EDGAR HTM filings.

    The "Â" character problem: SEC filings often contain non-breaking spaces
    (U+00A0, \xa0). When the HTML is served as Latin-1 but parsed as UTF-8,
    or vice versa, \xc2\xa0 (the UTF-8 encoding of \xa0) gets misinterpreted,
    producing "Â" followed by a space. This corrupts table cells, making
    numbers like "427,448" appear as "427,448Â Â$".

    Fix strategy:
      1. Ensure a proper UTF-8 charset <meta> tag exists
      2. Replace all \xa0 (non-breaking space) with regular space
      3. Remove orphan "Â" characters that are encoding artifacts
      4. Normalize common HTML entities
    """
    # 1. Ensure UTF-8 charset meta tag is present and first
    #    Remove any existing charset declarations to avoid conflicts
    html = re.sub(
        r'<meta[^>]*charset[^>]*>', '', html, flags=re.IGNORECASE
    )
    charset_tag = '<meta charset="UTF-8">'
    if re.search(r'<head[^>]*>', html, re.IGNORECASE):
        html = re.sub(
            r'(<head[^>]*>)', r'\1' + charset_tag,
            html, count=1, flags=re.IGNORECASE
        )

    # 2. Replace non-breaking spaces (\xa0) with regular spaces
    html = html.replace('\xa0', ' ')

    # 3. Replace &nbsp; entities with regular spaces
    html = re.sub(r'&nbsp;', ' ', html, flags=re.IGNORECASE)

    # 4. Remove orphan "Â" characters that are encoding artifacts
    #    Pattern: "Â" followed or preceded by whitespace, typically adjacent
    #    to numbers, dollar signs, or other data in table cells.
    #    Be careful not to remove "Â" that is part of real text (rare in
    #    English-language SEC filings).
    # Remove "Â" when it appears:
    #   - Before a space/digit/$/( : e.g., "427,448Â $" → "427,448 $"
    #   - After a space/digit/)   : e.g., "Â 427" → " 427"
    #   - Standalone between spaces: "text Â text" → "text  text"
    html = re.sub(r'Â\s', ' ', html)    # "Â " → " "
    html = re.sub(r'\sÂ', ' ', html)    # " Â" → " "
    html = re.sub(r'Â(?=[$(\d])', '', html)  # "Â$" or "Â(" or "Â1" → remove Â
    html = re.sub(r'(?<=[\d)%])\s*Â', '', html)  # "100%Â" → "100%"

    # 5. Normalize Unicode dashes that may cause issues
    html = html.replace('\u2013', '–')  # en-dash
    html = html.replace('\u2014', '—')  # em-dash

    return html


def _resolve_and_inline_images(html: str, base_url: str) -> str:
    """
    Resolve relative image URLs to absolute and attempt to inline as base64.

    Playwright renders from a local temp file, so relative image URLs
    (e.g., src="R123.jpg") won't resolve even with <base href>. This
    function:
      1. Resolves all relative img src attributes to absolute EDGAR URLs
      2. Downloads each image and embeds it as a base64 data URI
      3. Falls back to the absolute URL if download fails (Playwright
         may still be able to fetch it via network)
    """
    try:
        from bs4 import BeautifulSoup
        import base64
    except ImportError:
        return html

    soup = BeautifulSoup(html, "lxml")
    img_tags = soup.find_all("img")

    if not img_tags:
        return html

    # Derive the base directory URL for relative resolution
    # e.g., "https://www.sec.gov/Archives/edgar/data/12345/000001/filing.htm"
    # → "https://www.sec.gov/Archives/edgar/data/12345/000001/"
    if '/' in base_url:
        base_dir = base_url.rsplit('/', 1)[0] + '/'
    else:
        base_dir = base_url + '/'

    inlined = 0
    resolved = 0

    for img in img_tags:
        src = img.get("src", "")
        if not src:
            continue

        # Skip already-inlined data URIs
        if src.startswith("data:"):
            continue

        # Resolve relative to absolute
        if not src.startswith(("http://", "https://")):
            abs_url = base_dir + src.lstrip("./")
            img["src"] = abs_url
            resolved += 1
        else:
            abs_url = src

        # Try to download and inline as base64
        try:
            time.sleep(RATE_LIMIT_SLEEP)
            r = requests.get(abs_url, headers=HEADERS, timeout=20)
            r.raise_for_status()

            # Determine MIME type
            content_type = r.headers.get("Content-Type", "image/png")
            if ";" in content_type:
                content_type = content_type.split(";")[0].strip()
            if "/" not in content_type:
                content_type = "image/png"

            b64 = base64.b64encode(r.content).decode("ascii")
            img["src"] = f"data:{content_type};base64,{b64}"
            inlined += 1
        except Exception:
            # Leave the absolute URL — Playwright may still fetch it
            pass

    if inlined > 0 or resolved > 0:
        print(f"    Images: {inlined} inlined, {resolved} resolved to absolute URLs")

    return str(soup)


def _split_table_at_row(soup, table_tag, hr_tr):
    """
    Split a <table> into two tables at the given <tr>, with a page-break
    <div> between them.

    This is necessary because Chromium ignores CSS page-break properties
    on elements inside table cells. Older EDGAR filings (pre-~2014) wrap
    the entire document in one giant layout <table> with <hr> tags in
    cells as page separators — those <hr> page-break-after styles never
    fire. Splitting the table makes the break div a block-level sibling
    that Chromium WILL honor.

    Returns True if the split was performed.
    """
    from bs4 import NavigableString

    # Find the parent <tbody> if present (rows may be inside one)
    parent_tbody = hr_tr.parent if hr_tr.parent and hr_tr.parent.name == "tbody" else None
    row_container = parent_tbody if parent_tbody else table_tag

    # Collect all sibling rows AFTER hr_tr
    following_rows = []
    sibling = hr_tr.next_sibling
    while sibling is not None:
        next_sib = sibling.next_sibling
        if sibling.name == "tr":
            following_rows.append(sibling.extract())
        elif isinstance(sibling, NavigableString):
            # Skip whitespace text nodes between rows
            sibling.extract()
        elif hasattr(sibling, "name") and sibling.name is not None:
            # Other elements (colgroup, thead remnants, etc.) — keep with next table
            following_rows.append(sibling.extract())
        else:
            if hasattr(sibling, "extract"):
                sibling.extract()
        sibling = next_sib

    if not following_rows:
        # <hr> was in the last row — just add a break after the table
        break_div = soup.new_tag("div")
        break_div["style"] = "page-break-before: always; height: 0; margin: 0; padding: 0;"
        table_tag.insert_after(break_div)
        hr_tr.decompose()
        return True

    # Build the new table with the same attributes (class, style, width, etc.)
    new_table = soup.new_tag("table")
    for attr, val in table_tag.attrs.items():
        if attr.lower() == "id":
            continue  # Don't duplicate HTML IDs
        new_table[attr] = val

    # If original had a <tbody>, wrap rows in one too
    if parent_tbody:
        new_tbody = soup.new_tag("tbody")
        for row in following_rows:
            new_tbody.append(row)
        new_table.append(new_tbody)
    else:
        for row in following_rows:
            new_table.append(row)

    # Create the page-break div
    break_div = soup.new_tag("div")
    break_div["style"] = "page-break-before: always; height: 0; margin: 0; padding: 0;"

    # Insert: [original_table] [break_div] [new_table]
    table_tag.insert_after(new_table)
    table_tag.insert_after(break_div)

    # Remove the <hr> row from the original table
    hr_tr.decompose()

    return True


def _inject_page_breaks(html: str) -> Tuple[str, int]:
    """
    Detect logical page boundaries in SEC filing HTML and inject CSS page breaks.

    Strategy:
      1. <hr> tags OUTSIDE tables → direct CSS page-break-after (simple case)
      2. <hr> tags INSIDE tables  → split the table at the <hr> row, insert
         a page-break <div> between the two halves (Chromium ignores CSS
         page-break inside table cells, so this is required)
      3. Page-number text patterns → CSS page-break-after (fallback for
         filings without <hr> separators)
      4. Existing page-break CSS in the HTML → count but don't modify

    Returns: (modified_html, num_breaks_injected)
    """
    try:
        from bs4 import BeautifulSoup, NavigableString
    except ImportError:
        print("    [error] beautifulsoup4 not installed")
        return html, 0

    soup = BeautifulSoup(html, "lxml")
    breaks_injected = 0
    tables_split = 0

    # --- Strategy 1 & 2: <hr> tags (table-aware) ---
    # Use list() because we modify the tree during iteration.
    # Safety: after a table split, decomposed tags lose their .parent;
    # skip any <hr> that was destroyed by a prior split.
    hr_tags = list(soup.find_all("hr"))
    for hr in hr_tags:
        # Skip if this <hr> was already destroyed by a prior table split
        if hr.parent is None or getattr(hr, "decomposed", False):
            continue

        parent_table = hr.find_parent("table")

        if parent_table is None:
            # OUTSIDE a table — simple CSS page-break works
            hr["style"] = (hr.get("style", "") +
                           "; page-break-after: always !important; "
                           "visibility: hidden !important; "
                           "height: 0 !important; margin: 0 !important;")
            breaks_injected += 1
        else:
            # INSIDE a table — CSS page-break is IGNORED by Chromium.
            # Must split the table at this point.
            hr_tr = hr.find_parent("tr")
            if hr_tr is None:
                # <hr> directly inside <table> but not in a <tr> (unusual)
                # Try to add break after the <hr>'s direct parent
                hr_parent = hr.parent
                if hr_parent and hr_parent != parent_table:
                    hr_parent["style"] = (hr_parent.get("style", "") +
                                          "; page-break-after: always !important;")
                hr["style"] = "display: none !important;"
                breaks_injected += 1
                continue

            if _split_table_at_row(soup, parent_table, hr_tr):
                breaks_injected += 1
                tables_split += 1

    if tables_split > 0:
        print(f"    Split tables at {tables_split} <hr> boundary(ies)")

    # --- Strategy 3: Page-number text elements ---
    if breaks_injected < 5:
        for tag in soup.find_all(["p", "div", "span", "center"]):
            text = tag.get_text(strip=True)
            if _is_page_number_text(text) and len(text) < 20:
                # Check this element is NOT inside a table cell
                # (if it is, the page-break won't fire anyway)
                parent_table = tag.find_parent("table")
                if parent_table is None:
                    existing_style = tag.get("style", "")
                    tag["style"] = existing_style + "; page-break-after: always !important;"
                    breaks_injected += 1

    # --- Strategy 4: Count existing page-break CSS in the document ---
    existing_breaks = 0
    for tag in soup.find_all(["div", "table", "p", "span"]):
        style = tag.get("style", "").lower()
        if "page-break" in style and tag not in hr_tags:
            existing_breaks += 1
    if existing_breaks > 0:
        breaks_injected += existing_breaks

    # --- Inject global CSS ---
    style_tag = soup.new_tag("style")
    style_tag.string = """
    @media print {
        /* NOTE: We deliberately do NOT set page-break-inside: avoid on
           <table> or <tr>. Older EDGAR filings wrap the ENTIRE document
           in one layout table, and that rule would block ALL page breaks.
           Individual data tables may split across pages, but that is the
           lesser evil compared to completely broken pagination. */

        /* Prevent figures/images from breaking across pages */
        img    { page-break-inside: avoid !important; max-width: 100% !important; }
        figure { page-break-inside: avoid !important; }
        /* Prevent orphans/widows */
        p, div { orphans: 3; widows: 3; }
    }

    /* Force all content to fit within page width */
    body {
        max-width: 100% !important;
        overflow-x: hidden !important;
    }
    /* Let tables auto-size columns based on content (NOT fixed) */
    table {
        max-width: 100% !important;
        table-layout: auto !important;
        border-collapse: collapse !important;
    }
    td, th {
        /* Prevent cells from becoming absurdly narrow */
        min-width: 1em !important;
        /* Allow word wrapping but don't force letter-per-line */
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
        /* Preserve horizontal whitespace in financial data */
        white-space: normal !important;
    }
    """
    head = soup.find("head")
    if head:
        head.append(style_tag)
    else:
        soup.insert(0, style_tag)

    return str(soup), breaks_injected


def _remove_blank_pages(pdf_path: Path) -> int:
    """
    Post-process: detect and remove blank pages from a PDF.
    Returns the number of pages removed.
    """
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        return 0

    reader = PdfReader(str(pdf_path))
    total = len(reader.pages)
    if total <= 1:
        return 0

    writer = PdfWriter()
    removed = 0

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text_stripped = text.strip()

        # A page is "blank" if it has essentially no text content
        # Allow very short text (like a single space or newline)
        if len(text_stripped) < 3:
            # Double-check: some pages are intentionally blank
            # (e.g., "This page intentionally left blank")
            # Keep the first and last pages regardless
            if i == 0 or i == total - 1:
                writer.add_page(page)
            else:
                removed += 1
                continue
        else:
            writer.add_page(page)

    if removed > 0:
        with open(pdf_path, "wb") as f:
            writer.write(f)
        print(f"    Removed {removed} blank page(s)")

    return removed


def convert_htm_to_pdf(htm_url: str, dest: Path) -> bool:
    """
    Convert an EDGAR HTM filing to PDF with page-faithful rendering.

    Pipeline:
      1. Download HTML via requests (SEC-compliant User-Agent)
      2. Fix encoding artifacts (Â characters from \xa0 misinterpretation)
      3. Inject <base href> for relative URL resolution
      4. Resolve image URLs and inline as base64 data URIs
      5. Detect and inject CSS page breaks at logical boundaries
      6. Render via Playwright with a tall custom page size
      7. Post-process: remove blank pages
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("    [error] playwright not installed")
        return False

    print(f"    Converting HTM → PDF (page-faithful)...")

    # --- Step 1: Download HTML ---
    try:
        r = rate_limited_get(htm_url)
        if "undeclared automated tool" in r.content[:2000].decode("utf-8", errors="replace").lower():
            print("    [error] SEC bot-block detected — check User-Agent")
            return False

        # ENCODING FIX: SEC filings often use Windows-1252 or Latin-1 but
        # don't always declare it in headers. requests may guess wrong,
        # causing \xa0 (non-breaking space in Win-1252) to become "Â " when
        # decoded as UTF-8. Try multiple strategies:
        raw_bytes = r.content

        # Strategy 1: Check if the HTML declares its own charset
        charset_match = re.search(
            rb'charset[="\s]+([a-zA-Z0-9_-]+)', raw_bytes[:2000]
        )
        if charset_match:
            declared = charset_match.group(1).decode("ascii").lower()
            try:
                raw_html = raw_bytes.decode(declared)
            except (UnicodeDecodeError, LookupError):
                raw_html = raw_bytes.decode("utf-8", errors="replace")
        else:
            # Strategy 2: Try UTF-8 first (most modern filings)
            try:
                raw_html = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                # Strategy 3: Fall back to Windows-1252 (common for older filings)
                try:
                    raw_html = raw_bytes.decode("windows-1252")
                except UnicodeDecodeError:
                    raw_html = raw_bytes.decode("utf-8", errors="replace")

    except Exception as e:
        print(f"    [error] HTM fetch failed: {e}")
        return False

    # --- Step 2: Fix encoding artifacts (Â characters, \xa0, etc.) ---
    raw_html = _fix_encoding(raw_html)

    # --- Step 3: Inject <base href> for relative URL resolution ---
    base_tag = f'<base href="{htm_url}">'
    if re.search(r"<head[^>]*>", raw_html, re.IGNORECASE):
        raw_html = re.sub(
            r"(<head[^>]*>)", r"\1" + base_tag,
            raw_html, count=1, flags=re.IGNORECASE
        )
    else:
        raw_html = base_tag + raw_html

    # --- Step 4: Resolve and inline images as base64 ---
    raw_html = _resolve_and_inline_images(raw_html, htm_url)

    # --- Step 5: Inject page breaks at logical boundaries ---
    modified_html, num_breaks = _inject_page_breaks(raw_html)
    print(f"    Detected {num_breaks} page boundary marker(s)")

    # --- Step 6: Write to temp file and render ---
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            suffix=".html", delete=False, mode="w", encoding="utf-8"
        ) as tmp:
            tmp.write(modified_html)
            tmp_path = tmp.name

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            # Navigate and wait for full load
            page.goto(f"file://{tmp_path}", wait_until="load", timeout=120_000)
            try:
                page.wait_for_load_state("networkidle", timeout=60_000)
            except Exception:
                pass  # Large filings may not reach networkidle

            # Render to PDF with tall page to prevent intra-page overflow
            page.pdf(
                path=str(dest),
                width=TALL_PAGE_WIDTH,
                height=TALL_PAGE_HEIGHT,
                print_background=True,
                prefer_css_page_size=False,  # Use OUR page size, not the doc's
                margin={
                    "top": "0.5in",
                    "bottom": "0.5in",
                    "left": "0.5in",
                    "right": "0.5in",
                },
            )
            browser.close()

        os.unlink(tmp_path)

    except Exception as e:
        print(f"    [error] Playwright conversion failed: {e}")
        traceback.print_exc()
        dest.unlink(missing_ok=True)
        return False

    # --- Step 7: Post-process ---
    if not dest.exists() or dest.stat().st_size < MIN_PDF_BYTES:
        print(f"    [error] Output PDF too small or missing")
        dest.unlink(missing_ok=True)
        return False

    _remove_blank_pages(dest)

    final_size = dest.stat().st_size
    try:
        from pypdf import PdfReader
        num_pages = len(PdfReader(str(dest)).pages)
    except Exception:
        num_pages = -1

    print(f"    ✓ HTM→PDF converted ({final_size // 1024} KB, {num_pages} pages)")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Full pipeline per document
# ═══════════════════════════════════════════════════════════════════════════════

def process_doc(doc_name: str, out_dir: Path, overwrite: bool = False) -> DownloadResult:
    """Process a single doc_name: resolve → download/convert → validate."""
    result = DownloadResult(doc_name=doc_name)
    dest = out_dir / f"{doc_name}.pdf"

    # Skip if already downloaded (unless --overwrite)
    if dest.exists() and dest.stat().st_size > MIN_PDF_BYTES and not overwrite:
        result.success = True
        result.pdf_path = str(dest)
        result.pdf_bytes = dest.stat().st_size
        result.method = "cached"
        try:
            from pypdf import PdfReader
            result.pdf_pages = len(PdfReader(str(dest)).pages)
        except Exception:
            pass
        print(f"  [skip] {doc_name} already exists ({result.pdf_bytes // 1024} KB)")
        return result

    print(f"\n→ {doc_name}")

    # --- Parse doc_name ---
    try:
        parsed = parse_doc_name(doc_name)
    except ValueError as e:
        result.error = str(e)
        print(f"  [error] {e}")
        return result

    print(f"  ticker={parsed.ticker}  year={parsed.year}"
          f"  q={parsed.quarter or 'annual'}  form={parsed.form}")

    # --- Skip earnings calls (no EDGAR filing) ---
    if parsed.form == "EC":
        result.error = "Earnings call transcripts are not available on EDGAR"
        print(f"  [skip] {result.error}")
        return result

    # --- Resolve CIK ---
    cik = get_cik(parsed.ticker)
    if not cik:
        result.error = f"CIK not found for ticker={parsed.ticker}"
        print(f"  [error] {result.error}")
        return result

    result.cik = cik
    print(f"  CIK={cik}")

    # --- Find accession number ---
    match = get_accession_number(cik, parsed)
    if not match:
        result.error = f"No {parsed.form} filing found for {parsed.ticker}/{parsed.year}"
        print(f"  [error] {result.error}")
        return result

    acc, filing_date, primary_doc = match
    result.accession = acc
    result.filing_date = filing_date
    print(f"  accession={acc}  filed={filing_date}")

    # --- List filing documents ---
    docs = get_filing_documents(cik, acc)
    if not docs:
        result.error = "No documents found in filing index"
        print(f"  [error] {result.error}")
        return result

    print(f"  {len(docs)} document(s) in filing index")
    for d in docs[:8]:
        print(f"    seq={d['sequence']:3s}  [{d['type']:15s}]  {d['document']}")
    if len(docs) > 8:
        print(f"    ... and {len(docs) - 8} more")

    # --- Strategy A: Native PDF ---
    pdf_docs = [d for d in docs if d["document"].lower().endswith(".pdf")]
    if pdf_docs:
        # Prefer sequence "1" (primary document)
        pdf_docs.sort(key=lambda d: (d["sequence"] != "1", d["document"]))
        chosen = pdf_docs[0]
        result.primary_url = chosen["url"]
        print(f"  Native PDF found: {chosen['document']}")

        if download_pdf_direct(chosen["url"], dest):
            result.success = True
            result.pdf_path = str(dest)
            result.pdf_bytes = dest.stat().st_size
            result.method = "native_pdf"
            try:
                from pypdf import PdfReader
                result.pdf_pages = len(PdfReader(str(dest)).pages)
            except Exception:
                pass
            return result

    # --- Strategy B: HTM → PDF conversion ---
    htm_docs = [d for d in docs if d["document"].lower().endswith((".htm", ".html"))]
    if not htm_docs:
        result.error = "No PDF or HTM documents found in filing"
        print(f"  [error] {result.error}")
        return result

    # Priority: sequence "1", or filename matching the filing type
    def htm_priority(d):
        name_lower = d["document"].lower()
        ticker_lower = parsed.ticker.lower()
        form_hints = [parsed.form.lower().replace("-", ""), ticker_lower, "annual", "quarterly"]
        has_hint = any(kw in name_lower for kw in form_hints)
        return (d["sequence"] != "1", not has_hint, d["document"])

    htm_docs.sort(key=htm_priority)
    chosen_htm = htm_docs[0]
    result.primary_url = chosen_htm["url"]
    print(f"  No native PDF → converting: {chosen_htm['document']}")

    if convert_htm_to_pdf(chosen_htm["url"], dest):
        result.success = True
        result.pdf_path = str(dest)
        result.pdf_bytes = dest.stat().st_size
        result.method = "htm_convert"
        try:
            from pypdf import PdfReader
            result.pdf_pages = len(PdfReader(str(dest)).pages)
        except Exception:
            pass
    else:
        result.error = "HTM→PDF conversion failed"

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Collect unique doc_names from JSONL files
# ═══════════════════════════════════════════════════════════════════════════════

def collect_doc_names(jsonl_paths: List[str]) -> Dict[str, Dict]:
    """
    Parse JSONL files and collect unique doc_names with metadata.
    Returns {doc_name: {"sources": [jsonl_files], "pages": [referenced_pages]}}.
    """
    doc_info: Dict[str, Dict] = {}

    for path in jsonl_paths:
        p = Path(path)
        if not p.exists():
            print(f"[warn] JSONL file not found: {path}")
            continue

        with open(p) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[warn] {p.name}:{line_num} — JSON parse error: {e}")
                    continue

                for ev in row.get("evidences", []):
                    doc = ev.get("doc_name", "")
                    if not doc:
                        continue
                    if doc not in doc_info:
                        doc_info[doc] = {"sources": [], "pages": []}
                    if str(p) not in doc_info[doc]["sources"]:
                        doc_info[doc]["sources"].append(str(p))
                    page = ev.get("page_num")
                    if page is not None and page not in doc_info[doc]["pages"]:
                        doc_info[doc]["pages"].append(page)

    return doc_info


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Manifest
# ═══════════════════════════════════════════════════════════════════════════════

def append_manifest(manifest_path: Path, result: DownloadResult):
    """Append a result record to the manifest JSONL file."""
    rec = {
        "doc_name": result.doc_name,
        "success": result.success,
        "method": result.method,
        "pdf_path": result.pdf_path,
        "pdf_bytes": result.pdf_bytes,
        "pdf_pages": result.pdf_pages,
        "cik": result.cik,
        "accession": result.accession,
        "primary_url": result.primary_url,
        "filing_date": result.filing_date,
        "sources": result.sources,
        "pages_referenced": result.pages_referenced,
        "error": result.error,
    }
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Download EDGAR filings as page-faithful PDFs"
    )
    parser.add_argument(
        "--jsonl", nargs="+", required=True,
        help="JSONL dataset file(s) (e.g., data/finqa_test.jsonl data/secqa_test.jsonl)"
    )
    parser.add_argument(
        "--out_dir", default="PDF-Opus",
        help="Output directory for PDFs (default: PDF-Opus)"
    )
    parser.add_argument(
        "--only", nargs="*",
        help="Process only these doc_names (for testing)"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-download even if PDF already exists"
    )
    parser.add_argument(
        "--failed-list", default="failed_downloads.txt",
        help="Filename for failed downloads list"
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    # Collect unique doc_names
    doc_info = collect_doc_names(args.jsonl)
    if args.only:
        doc_info = {k: v for k, v in doc_info.items() if k in args.only}

    print(f"\n{'='*60}")
    print(f"EDGAR PDF Opus Downloader")
    print(f"{'='*60}")
    print(f"Input files:  {', '.join(args.jsonl)}")
    print(f"Output dir:   {out_dir}")
    print(f"Unique docs:  {len(doc_info)}")
    print(f"Overwrite:    {args.overwrite}")
    print(f"{'='*60}\n")

    ok, fail = 0, 0
    failed_docs = []

    for doc_name in sorted(doc_info.keys()):
        info = doc_info[doc_name]
        result = process_doc(doc_name, out_dir, overwrite=args.overwrite)

        # Attach source metadata
        result.sources = info["sources"]
        result.pages_referenced = info["pages"]

        # Write to manifest
        append_manifest(manifest_path, result)

        if result.success:
            ok += 1
        else:
            fail += 1
            failed_docs.append(doc_name)

    # Write failed list
    if failed_docs:
        fail_path = out_dir / args.failed_list
        fail_path.write_text("\n".join(failed_docs) + "\n")

    print(f"\n{'='*60}")
    print(f"Done!")
    print(f"  ✓ {ok} downloaded")
    print(f"  ✗ {fail} failed")
    print(f"  Manifest: {manifest_path}")
    if failed_docs:
        print(f"  Failed list: {out_dir / args.failed_list}")
        print(f"\n  Failed filings:")
        for d in failed_docs[:20]:
            print(f"    {d}")
        if len(failed_docs) > 20:
            print(f"    ... and {len(failed_docs) - 20} more")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()