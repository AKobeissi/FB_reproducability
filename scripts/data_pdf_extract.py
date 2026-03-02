#!/usr/bin/env python3
"""
data_pdf_extract_v2.py
======================
Downloads SEC 10-K / 10-Q PDF filings for FinanceBench / SECQA datasets,
ensuring that page numbers in the downloaded PDFs match the `page_num` values
in your evidences — by fetching the *actual* PDF files from EDGAR rather than
rendering HTML to PDF.

Follows the LoFIN / FinanceBench approach:
  - Locate the real PDF artifact in the EDGAR filing index
  - Fall back to the largest HTML → PDF render only when no PDF exists
  - Produce a manifest.jsonl that maps each doc_name to its PDF path and the
    evidence pages that reference it

Usage
-----
pip install requests tqdm playwright pypdf
python -m playwright install chromium  # only needed if HTML fallback is used

python data_pdf_extract_v2.py \\
  --inputs data/finqa_test.jsonl data/secqa_test.jsonl \\
  --out_dir ./pdfs_extended \\
  --user_agent "Your Name your.email@domain.com" \\
  --sleep_s 0.3 \\
  --workers 1

Notes
-----
- SEC fair-access policy: identify yourself with a real User-Agent.
- `--workers` > 1 adds concurrency but risks hitting SEC rate limits. Keep at 1
  unless you add exponential back-off (already included via --retries).
- PDFs are named exactly <doc_name>.pdf so your downstream code can do a simple
  lookup: pdfs_extended/SLG_2011_10K.pdf.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import requests
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_TMPL    = "https://data.sec.gov/submissions/CIK{cik10}.json"
SEC_SUBMISSIONS_EXTRA   = "https://data.sec.gov/submissions/{name}"
SEC_ARCHIVES_BASE       = "https://www.sec.gov/Archives/edgar/data"

# Minimum size in bytes to consider a downloaded PDF valid
MIN_PDF_BYTES = 10_000

# Score weights for filing matching
W_REPORT_YEAR   = 60
W_REPORT_MONTH  = 80
W_FILING_YEAR   = 25
W_FILING_YEAR1  = 20   # 10-K filed the year after the report year
W_HAS_PRIMARY   = 10

QUARTER_END_MONTH = {1: 3, 2: 6, 3: 9, 4: 12}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ParsedDocName:
    doc_name:    str
    ticker_raw:  str            # e.g. "BRK_B" or "AAPL"
    year:        int
    quarter:     Optional[int]  # 1-4 or None for annual
    form:        str            # "10-K" | "10-Q" | "8-K"


@dataclass(frozen=True)
class FilingMatch:
    cik:          int
    accession:    str   # with dashes: 0000950170-22-000796
    primary_doc:  str   # e.g. ebay-20230930.htm
    form:         str
    filing_date:  str   # yyyy-mm-dd
    report_date:  str   # yyyy-mm-dd


@dataclass
class DownloadResult:
    doc_name:    str
    success:     bool
    pdf_path:    str  = ""
    pdf_sha256:  str  = ""
    pdf_bytes:   int  = 0    # size of the downloaded PDF in bytes
    method:      str  = ""   # "direct_pdf" | "html_render" | "failed"
    error:       str  = ""
    cik:         int  = 0
    accession:   str  = ""
    primary_url: str  = ""
    filing_date: str  = ""
    report_date: str  = ""
    sources:     List[str] = field(default_factory=list)
    pages_referenced: List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parsing utilities
# ---------------------------------------------------------------------------
_DOC_RE    = re.compile(r"^(?P<ticker>.+?)_(?P<period>\d{4}(?:Q[1-4])?)_(?P<form>.+)$", re.I)
_PERIOD_RE = re.compile(r"^(?P<year>\d{4})(?:Q(?P<q>[1-4]))?$", re.I)


def parse_doc_name(doc_name: str) -> ParsedDocName:
    """
    Parse strings like:
        SLG_2011_10K
        EBAY_2023Q1_10Q
        BRK_B_2018_10K
        AAPL_2024Q3_10-Q
    """
    m = _DOC_RE.match(doc_name.strip())
    if not m:
        raise ValueError(f"Cannot parse doc_name: {doc_name!r}")

    ticker_raw = m.group("ticker").upper()
    period     = m.group("period").upper()
    form       = _normalize_form(m.group("form"))

    pm = _PERIOD_RE.match(period)
    if not pm:
        raise ValueError(f"Cannot parse period {period!r} in {doc_name!r}")

    year = int(pm.group("year"))
    q    = pm.group("q")
    return ParsedDocName(
        doc_name=doc_name,
        ticker_raw=ticker_raw,
        year=year,
        quarter=int(q) if q else None,
        form=form,
    )


def _normalize_form(raw: str) -> str:
    t = raw.upper().strip().replace("_", "").replace(" ", "")
    # Canonicalize common variants
    for src, dst in [("10K", "10-K"), ("10Q", "10-Q"), ("8K", "8-K")]:
        t = t.replace(src, dst)
    # After replacement, unambiguous
    if "10-K" in t:
        return "10-K"
    if "10-Q" in t:
        return "10-Q"
    if "8-K" in t:
        return "8-K"
    return t


def ticker_variants(ticker_raw: str) -> List[str]:
    """Generate plausible ticker spellings (BRK_B → BRK-B, BRK.B, BRKB …)."""
    t = ticker_raw.upper()
    seen, out = set(), []
    for v in [
        t,
        t.replace("_", "-"),
        t.replace("_", "."),
        t.replace("-", "."),
        t.replace(".", "-"),
        re.sub(r"[_.\-]", "", t),
    ]:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> List[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_size_bytes(directory: Path) -> int:
    """Return total size in bytes of all files under directory."""
    return sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())


def format_bytes(n: int) -> str:
    """Human-readable file size: 1.23 GB, 456 MB, etc."""
    for unit, threshold in [("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)]:
        if n >= threshold:
            return f"{n / threshold:.2f} {unit}"
    return f"{n} B"


def pdf_size_stats(directory: Path) -> dict:
    """
    Return a dict with:
      total_bytes, total_str,
      count, mean_bytes, mean_str,
      min_bytes, min_str, max_bytes, max_str
    for every .pdf file directly in directory.
    """
    sizes = sorted(
        f.stat().st_size
        for f in directory.glob("*.pdf")
        if f.is_file()
    )
    if not sizes:
        return {k: 0 for k in ("total_bytes", "count", "mean_bytes", "min_bytes", "max_bytes")} | \
               {"total_str": "0 B", "mean_str": "0 B", "min_str": "0 B", "max_str": "0 B",
                "smallest_name": "", "largest_name": ""}

    total = sum(sizes)
    # Find names for extremes
    pdfs  = sorted(
        [(f.stat().st_size, f.name) for f in directory.glob("*.pdf") if f.is_file()]
    )
    return {
        "total_bytes":   total,
        "total_str":     format_bytes(total),
        "count":         len(sizes),
        "mean_bytes":    total // len(sizes),
        "mean_str":      format_bytes(total // len(sizes)),
        "min_bytes":     sizes[0],
        "min_str":       format_bytes(sizes[0]),
        "max_bytes":     sizes[-1],
        "max_str":       format_bytes(sizes[-1]),
        "smallest_name": pdfs[0][1],
        "largest_name":  pdfs[-1][1],
    }


# ---------------------------------------------------------------------------
# Dataset extraction
# ---------------------------------------------------------------------------
def extract_doc_info(paths: List[Path]) -> Dict[str, dict]:
    """
    Returns {doc_name: {"sources": [...], "pages": [...pages across all questions]}}
    """
    result: Dict[str, dict] = {}
    for p in paths:
        for item in load_jsonl(p):
            for ev in item.get("evidences", []):
                dn = ev.get("doc_name", "").strip()
                if not dn:
                    continue
                entry = result.setdefault(dn, {"sources": set(), "pages": set()})
                entry["sources"].add(p.name)
                pg = ev.get("page_num")
                if pg is not None:
                    entry["pages"].add(int(pg))
    # Convert sets to sorted lists for JSON serialisation
    return {
        dn: {"sources": sorted(v["sources"]), "pages": sorted(v["pages"])}
        for dn, v in result.items()
    }


# ---------------------------------------------------------------------------
# SEC API helpers
# ---------------------------------------------------------------------------
def _headers(domain: str, user_agent: str) -> Dict[str, str]:
    return {
        "User-Agent":      user_agent,
        "Accept-Encoding": "gzip, deflate",
        "Host":            domain,
    }


def _get_json(
    url: str,
    session: requests.Session,
    user_agent: str,
    sleep_s: float,
    retries: int = 4,
) -> dict:
    domain = url.split("/")[2]
    headers = _headers(domain, user_agent)
    delay = sleep_s
    for attempt in range(retries + 1):
        try:
            time.sleep(delay)
            r = session.get(url, headers=headers, timeout=60)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 10))
                print(f"  [rate-limit] waiting {wait}s …", flush=True)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            if attempt == retries:
                raise
            delay = min(delay * 2, 30)
            print(f"  [retry {attempt+1}] {url} — {exc}", flush=True)
    raise RuntimeError("Unreachable")


def load_ticker_to_cik(cache_path: Path, user_agent: str, sleep_s: float) -> Dict[str, int]:
    if cache_path.exists():
        raw = json.loads(cache_path.read_text())
    else:
        with requests.Session() as s:
            raw = _get_json(SEC_COMPANY_TICKERS_URL, s, user_agent, sleep_s)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(raw))

    mapping: Dict[str, int] = {}
    for rec in raw.values():
        t   = str(rec.get("ticker", "")).upper().strip()
        cik = int(rec.get("cik_str", 0))
        if t and cik:
            mapping[t] = cik
    return mapping


def _iter_filings(
    subs: dict,
    session: requests.Session,
    user_agent: str,
    sleep_s: float,
) -> Iterator[dict]:
    """Yield every filing record from the submissions blob (including older pages)."""
    filings = subs.get("filings", {}) or {}
    recent  = filings.get("recent", {}) or {}

    if isinstance(recent, dict) and "accessionNumber" in recent:
        cols = list(recent.keys())
        n    = len(recent["accessionNumber"])
        for i in range(n):
            yield {c: recent[c][i] for c in cols if i < len(recent[c])}

    for f in filings.get("files", []) or []:
        name = f.get("name")
        if not name:
            continue
        url   = SEC_SUBMISSIONS_EXTRA.format(name=name)
        extra = _get_json(url, session, user_agent, sleep_s)
        er    = (extra.get("filings", {}) or {}).get("recent", {}) or {}
        if isinstance(er, dict) and "accessionNumber" in er:
            cols = list(er.keys())
            n    = len(er["accessionNumber"])
            for i in range(n):
                yield {c: er[c][i] for c in cols if i < len(er[c])}


def find_best_filing(
    parsed:     ParsedDocName,
    cik:        int,
    session:    requests.Session,
    user_agent: str,
    sleep_s:    float,
) -> FilingMatch:
    cik10   = str(cik).zfill(10)
    subs    = _get_json(SEC_SUBMISSIONS_TMPL.format(cik10=cik10), session, user_agent, sleep_s)
    year    = parsed.year
    quarter = parsed.quarter
    form    = parsed.form

    candidates: List[Tuple[int, FilingMatch]] = []

    for rec in _iter_filings(subs, session, user_agent, sleep_s):
        if str(rec.get("form", "")).upper().strip() != form:
            continue

        accession   = str(rec.get("accessionNumber", "")).strip()
        primary_doc = str(rec.get("primaryDocument", "")).strip()
        filing_date = str(rec.get("filingDate", "")).strip()
        report_date = str(rec.get("reportDate", "")).strip()

        if not accession:
            continue

        score = 0
        if report_date.startswith(str(year)):
            score += W_REPORT_YEAR
        if quarter and report_date and len(report_date) >= 7:
            try:
                if int(report_date[5:7]) == QUARTER_END_MONTH[quarter]:
                    score += W_REPORT_MONTH
            except Exception:
                pass
        if filing_date.startswith(str(year)):
            score += W_FILING_YEAR
        if form == "10-K" and filing_date.startswith(str(year + 1)):
            score += W_FILING_YEAR1
        if primary_doc:
            score += W_HAS_PRIMARY

        candidates.append((score, FilingMatch(
            cik=cik,
            accession=accession,
            primary_doc=primary_doc,
            form=form,
            filing_date=filing_date,
            report_date=report_date,
        )))

    if not candidates:
        raise RuntimeError(
            f"No {form} filing found for CIK={cik} year={year} quarter={quarter}"
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    best = candidates[0][1]

    # If primary_doc is missing, resolve it from the index
    if not best.primary_doc:
        best = _resolve_primary_doc(best, session, user_agent, sleep_s)

    return best


def _resolve_primary_doc(
    match:      FilingMatch,
    session:    requests.Session,
    user_agent: str,
    sleep_s:    float,
) -> FilingMatch:
    acc_nodash = match.accession.replace("-", "")
    index_url  = f"{SEC_ARCHIVES_BASE}/{match.cik}/{acc_nodash}/index.json"
    idx        = _get_json(index_url, session, user_agent, sleep_s)
    items      = (idx.get("directory", {}) or {}).get("item", []) or []

    # Prefer PDF, then HTM
    pdfs  = [it["name"] for it in items if it.get("name", "").lower().endswith(".pdf")]
    htmls = [it["name"] for it in items if it.get("name", "").lower().endswith((".htm", ".html"))]

    chosen = (pdfs or htmls or [None])[0]
    if not chosen:
        raise RuntimeError(f"No usable document in index for {match.accession}")

    return FilingMatch(
        cik=match.cik,
        accession=match.accession,
        primary_doc=chosen,
        form=match.form,
        filing_date=match.filing_date,
        report_date=match.report_date,
    )


# ---------------------------------------------------------------------------
# Filing index: find all PDFs
# ---------------------------------------------------------------------------
def list_filing_pdfs(
    match:      FilingMatch,
    session:    requests.Session,
    user_agent: str,
    sleep_s:    float,
) -> List[Tuple[str, int]]:
    """
    Return [(filename, size_bytes)] for every PDF in the filing, largest first.
    Falls back to HTML files when no PDFs exist.
    """
    acc_nodash = match.accession.replace("-", "")
    index_url  = f"{SEC_ARCHIVES_BASE}/{match.cik}/{acc_nodash}/index.json"
    idx        = _get_json(index_url, session, user_agent, sleep_s)
    items      = (idx.get("directory", {}) or {}).get("item", []) or []

    pdfs = [
        (it.get("name", ""), int(it.get("size", 0) or 0))
        for it in items
        if it.get("name", "").lower().endswith(".pdf")
    ]

    if pdfs:
        # Largest PDF first — for SEC filings this is almost always the full document
        pdfs.sort(key=lambda x: x[1], reverse=True)
        return pdfs

    # No PDFs — return HTML files for fallback rendering
    htmls = [
        (it.get("name", ""), int(it.get("size", 0) or 0))
        for it in items
        if it.get("name", "").lower().endswith((".htm", ".html"))
    ]
    htmls.sort(key=lambda x: x[1], reverse=True)
    return htmls


def _pick_best_pdf(
    candidates: List[Tuple[str, int]],
    primary_doc: str,
) -> str:
    """
    Choose the best PDF from the filing index.
    Priority:
      1. The file that matches primary_doc (if it's a PDF)
      2. The largest PDF (most likely to be the full filing)
      3. Any PDF that does NOT look like an exhibit (R*.htm, ex*.pdf, etc.)
    """
    names = [c[0] for c in candidates]

    # 1. primary_doc is itself a PDF
    if primary_doc.lower().endswith(".pdf") and primary_doc in names:
        return primary_doc

    # 2. Filter out obvious exhibits
    non_exhibits = [
        n for n in names
        if not re.search(r"(?:ex|exhibit|_ex\d)", n, re.I)
        and not re.match(r"R\d+\.", n, re.I)
    ]
    if non_exhibits:
        return non_exhibits[0]   # already sorted largest-first

    return names[0]              # fall back to largest


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------
def _download_binary(
    url:        str,
    dest:       Path,
    session:    requests.Session,
    user_agent: str,
    sleep_s:    float,
    retries:    int = 4,
) -> None:
    domain  = url.split("/")[2]
    headers = _headers(domain, user_agent)
    delay   = sleep_s
    for attempt in range(retries + 1):
        try:
            time.sleep(delay)
            with session.get(url, headers=headers, stream=True, timeout=120) as r:
                if r.status_code == 429:
                    wait = int(r.headers.get("Retry-After", 10))
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.open("wb") as f:
                    for chunk in r.iter_content(1 << 16):
                        f.write(chunk)
            return
        except Exception as exc:
            if attempt == retries:
                raise
            delay = min(delay * 2, 30)
            print(f"  [retry {attempt+1}] {url} — {exc}", flush=True)


def _render_html_to_pdf(url: str, dest: Path, user_agent: str) -> None:
    """
    Last-resort: render an SEC HTML filing to PDF via Playwright.
    NOTE: page numbers produced this way will NOT match evidence page_num values.
         A warning is printed so you know which docs fell back to this method.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise RuntimeError(
            "playwright not installed. Run: pip install playwright && "
            "python -m playwright install chromium"
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx     = browser.new_context(
            user_agent=user_agent,
            extra_http_headers={"Accept-Encoding": "gzip, deflate"},
        )
        page = ctx.new_page()
        page.set_default_timeout(120_000)
        page.goto(url, wait_until="domcontentloaded", timeout=120_000)
        page.wait_for_timeout(3_000)
        try:
            page.wait_for_load_state("networkidle", timeout=20_000)
        except Exception:
            pass
        page.pdf(
            path=str(dest),
            format="Letter",
            print_background=True,
            margin={"top": "0.5in", "bottom": "0.5in",
                    "left": "0.5in", "right": "0.5in"},
        )
        browser.close()


# ---------------------------------------------------------------------------
# Core per-document download logic
# ---------------------------------------------------------------------------
def download_one(
    doc_name:   str,
    info:       dict,           # {"sources": [...], "pages": [...]}
    out_dir:    Path,
    ticker_map: Dict[str, int],
    user_agent: str,
    sleep_s:    float,
    retries:    int,
    overwrite:  bool,
) -> DownloadResult:
    out_pdf = out_dir / f"{doc_name}.pdf"

    result = DownloadResult(
        doc_name=doc_name,
        success=False,
        sources=info["sources"],
        pages_referenced=info["pages"],
    )

    if out_pdf.exists() and not overwrite:
        result.success   = True
        result.pdf_path  = str(out_pdf)
        result.pdf_sha256 = sha256_of(out_pdf)
        result.pdf_bytes  = out_pdf.stat().st_size
        result.method    = "cached"
        return result

    # ── 1. Parse doc_name ──────────────────────────────────────────────────
    try:
        parsed = parse_doc_name(doc_name)
    except Exception as e:
        result.error = f"parse_doc_name: {e}"
        return result

    # ── 2. Resolve CIK ────────────────────────────────────────────────────
    cik: Optional[int] = None
    for tv in ticker_variants(parsed.ticker_raw):
        if tv in ticker_map:
            cik = ticker_map[tv]
            break
    if cik is None:
        result.error = f"No CIK for ticker={parsed.ticker_raw}"
        return result
    result.cik = cik

    with requests.Session() as session:
        # ── 3. Find filing ────────────────────────────────────────────────
        try:
            match = find_best_filing(parsed, cik, session, user_agent, sleep_s)
        except Exception as e:
            result.error = f"find_best_filing: {e}"
            return result

        result.accession  = match.accession
        result.filing_date = match.filing_date
        result.report_date = match.report_date

        # ── 4. List files in the filing ───────────────────────────────────
        try:
            candidates = list_filing_pdfs(match, session, user_agent, sleep_s)
        except Exception as e:
            result.error = f"list_filing_pdfs: {e}"
            return result

        if not candidates:
            result.error = "No documents found in filing index"
            return result

        acc_nodash = match.accession.replace("-", "")
        base_url   = f"{SEC_ARCHIVES_BASE}/{cik}/{acc_nodash}"

        # ── 5a. Direct PDF download (preferred — correct page numbers) ────
        pdf_candidates = [c for c in candidates if c[0].lower().endswith(".pdf")]
        if pdf_candidates:
            chosen_name = _pick_best_pdf(pdf_candidates, match.primary_doc)
            url         = f"{base_url}/{chosen_name}"
            result.primary_url = url
            try:
                _download_binary(url, out_pdf, session, user_agent, sleep_s, retries)
                if out_pdf.stat().st_size < MIN_PDF_BYTES:
                    out_pdf.unlink(missing_ok=True)
                    raise RuntimeError(f"Downloaded PDF is suspiciously small ({out_pdf.stat().st_size} bytes)")
                result.success   = True
                result.pdf_path  = str(out_pdf)
                result.pdf_sha256 = sha256_of(out_pdf)
                result.pdf_bytes  = out_pdf.stat().st_size
                result.method    = "direct_pdf"
                return result
            except Exception as e:
                print(f"  [warn] direct PDF download failed for {doc_name}: {e}", flush=True)
                out_pdf.unlink(missing_ok=True)

        # ── 5b. HTML → PDF render (fallback — page numbers may differ) ────
        html_candidates = [c for c in candidates if c[0].lower().endswith((".htm", ".html"))]
        if not html_candidates:
            result.error = "No PDF or HTML found in filing"
            return result

        # Pick the non-exhibit HTML closest to primary_doc
        chosen_html = _pick_best_pdf(html_candidates, match.primary_doc)
        url = f"{base_url}/{chosen_html}"
        result.primary_url = url

        print(
            f"\n  [WARN] {doc_name}: no native PDF — falling back to HTML render. "
            f"Page numbers may NOT match evidence page_num values.",
            flush=True,
        )
        try:
            _render_html_to_pdf(url, out_pdf, user_agent)
            result.success   = True
            result.pdf_path  = str(out_pdf)
            result.pdf_sha256 = sha256_of(out_pdf)
            result.pdf_bytes  = out_pdf.stat().st_size
            result.method    = "html_render"
            return result
        except Exception as e:
            out_pdf.unlink(missing_ok=True)
            result.error = f"html_render: {e}"
            return result


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------
def append_manifest(manifest_path: Path, result: DownloadResult) -> None:
    rec = {
        "doc_name":        result.doc_name,
        "success":         result.success,
        "method":          result.method,
        "pdf_path":        result.pdf_path,
        "pdf_sha256":      result.pdf_sha256,
        "pdf_bytes":       result.pdf_bytes,
        "pdf_size_str":    format_bytes(result.pdf_bytes),
        "cik":             result.cik,
        "accession":       result.accession,
        "primary_url":     result.primary_url,
        "filing_date":     result.filing_date,
        "report_date":     result.report_date,
        "sources":         result.sources,
        "pages_referenced": result.pages_referenced,
        "error":           result.error,
    }
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def load_manifest(manifest_path: Path) -> Dict[str, dict]:
    """Return {doc_name: record} for all successful entries already in the manifest."""
    if not manifest_path.exists():
        return {}
    result = {}
    for line in manifest_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            if rec.get("success"):
                result[rec["doc_name"]] = rec
        except Exception:
            pass
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download SEC filings as PDFs for FinanceBench / SECQA datasets"
    )
    ap.add_argument("--inputs",    nargs="+", required=True,
                    help="JSONL dataset files (finqa_test.jsonl, secqa_test.jsonl …)")
    ap.add_argument("--out_dir",   required=True,
                    help="Directory to write <doc_name>.pdf files")
    ap.add_argument("--user_agent", required=True,
                    help='SEC User-Agent, e.g. "Your Name your.email@domain.com"')
    ap.add_argument("--sleep_s",   type=float, default=0.3,
                    help="Throttle between SEC API requests (default 0.3s)")
    ap.add_argument("--retries",   type=int,   default=4,
                    help="Retries with exponential back-off on network errors")
    ap.add_argument("--workers",   type=int,   default=1,
                    help="Parallel workers (default 1; ≥2 risks SEC rate limits)")
    ap.add_argument("--cache_dir", default=".sec_cache",
                    help="Cache directory for company_tickers.json")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-download PDFs even if they already exist")
    args = ap.parse_args()

    out_dir      = Path(args.out_dir)
    cache_dir    = Path(args.cache_dir)
    manifest_path = out_dir / "manifest.jsonl"

    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect all doc_names ──────────────────────────────────────────────
    input_paths = [Path(p) for p in args.inputs]
    doc_info    = extract_doc_info(input_paths)
    print(f"Found {len(doc_info)} unique doc_name values across {len(input_paths)} file(s).")

    # ── Skip already-successful entries ───────────────────────────────────
    already_done = load_manifest(manifest_path)
    todo = {
        dn: info for dn, info in doc_info.items()
        if args.overwrite or dn not in already_done
    }
    skipped = len(doc_info) - len(todo)
    if skipped:
        print(f"  → {skipped} already in manifest, skipping. Use --overwrite to redo.")
    print(f"  → {len(todo)} to download.\n")

    # ── Load ticker → CIK ─────────────────────────────────────────────────
    print("Loading SEC ticker → CIK map …")
    ticker_map = load_ticker_to_cik(
        cache_dir / "company_tickers.json", args.user_agent, args.sleep_s
    )
    print(f"  Loaded {len(ticker_map):,} tickers.\n")

    # ── Download ──────────────────────────────────────────────────────────
    doc_names   = sorted(todo.keys())
    n_ok        = n_fail = n_html_fallback = 0
    bytes_this_run: int = 0
    failed_docs: List[str] = []

    def _task(dn: str) -> DownloadResult:
        return download_one(
            doc_name=dn,
            info=todo[dn],
            out_dir=out_dir,
            ticker_map=ticker_map,
            user_agent=args.user_agent,
            sleep_s=args.sleep_s,
            retries=args.retries,
            overwrite=args.overwrite,
        )

    if args.workers == 1:
        # Single-threaded — tqdm progress bar works cleanly
        for dn in tqdm(doc_names, desc="Downloading PDFs"):
            result = _task(dn)
            append_manifest(manifest_path, result)
            if result.success:
                n_ok += 1
                if result.method not in ("cached",):
                    bytes_this_run += result.pdf_bytes
                if result.method == "html_render":
                    n_html_fallback += 1
            else:
                n_fail += 1
                failed_docs.append(dn)
                tqdm.write(f"  [FAIL] {dn}: {result.error}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_task, dn): dn for dn in doc_names}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading PDFs"):
                result = fut.result()
                append_manifest(manifest_path, result)
                if result.success:
                    n_ok += 1
                    if result.method not in ("cached",):
                        bytes_this_run += result.pdf_bytes
                    if result.method == "html_render":
                        n_html_fallback += 1
                else:
                    n_fail += 1
                    failed_docs.append(futures[fut])
                    tqdm.write(f"  [FAIL] {futures[fut]}: {result.error}")

    # ── Disk usage summary ────────────────────────────────────────────────
    disk = pdf_size_stats(out_dir)

    print(f"""
══════════════════════════════════════════════════════════
 Download summary
══════════════════════════════════════════════════════════
  Total unique docs      : {len(doc_info)}
  Already cached (skip)  : {skipped}
  Downloaded OK          : {n_ok}
    ↳ direct PDF         : {n_ok - n_html_fallback}  (page numbers match evidence ✓)
    ↳ HTML rendered      : {n_html_fallback}           (page numbers may differ  ⚠)
  Failed                 : {n_fail}

  ── Disk usage ({out_dir}) ──────────────────────────────
  Total PDFs on disk     : {disk["count"]} files
  Total size             : {disk["total_str"]}
  Downloaded this run    : {format_bytes(bytes_this_run)}
  Average per PDF        : {disk["mean_str"]}
  Smallest PDF           : {disk["min_str"]}  ({disk["smallest_name"]})
  Largest PDF            : {disk["max_str"]}  ({disk["largest_name"]})

  PDFs written to        : {out_dir}
  Manifest               : {manifest_path}
══════════════════════════════════════════════════════════""")

    if failed_docs:
        print("\nFailed doc_names:")
        for dn in failed_docs:
            print(f"  {dn}")
        print("\nTip: re-run the same command after fixing errors —")
        print("     already-successful docs will be skipped automatically.")
        sys.exit(1)


if __name__ == "__main__":
    main()