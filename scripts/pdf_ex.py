#!/usr/bin/env python3
"""
EDGAR PDF Downloader for FinanceBench / SEC Filings
----------------------------------------------------
Input:  A .jsonl file where each line has an "evidences" field containing
        doc_name in the format  {TICKER}_{YEAR}_{FORM_TYPE}
        e.g. "SLG_2011_10K"

Output: ./pdfs/{doc_name}.pdf  — one PDF per unique filing

Strategy per filing:
  1. Find the filing on EDGAR (CIK → accession number)
  2. Parse the filing index to list all documents
  3. If a .pdf exists → download it directly
  4. If only .htm  → download and convert to PDF via playwright (headless Chromium)

Install dependencies:
    pip install requests playwright
    playwright install chromium

Usage:
    python download_edgar_pdfs.py --jsonl financebench.jsonl
    python download_edgar_pdfs.py --jsonl financebench.jsonl --out_dir pdfs/
    python download_edgar_pdfs.py --jsonl financebench.jsonl --only SLG_2011_10K AAPL_2020_10K
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

# SEC requires this exact format: "First Last email@domain.com"
# See: https://www.sec.gov/os/accessing-edgar-data
EDGAR_USER_AGENT = "Amine Kobeissi amine.kobeissi@umontreal.ca"

HEADERS = {
    "User-Agent": EDGAR_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
}
SLEEP = 0.5   # SEC allows 10 req/s; 0.5s gives comfortable headroom


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def sleep_get(url: str, **kwargs) -> requests.Response:
    time.sleep(SLEEP)
    return requests.get(url, headers=HEADERS, timeout=30, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Parse doc_name  →  (ticker, year, form_type)
# ══════════════════════════════════════════════════════════════════════════════

def parse_doc_name(doc_name: str):
    """
    'SLG_2011_10K'  → ('SLG', 2011, '10-K')
    'AAPL_2020_10K' → ('AAPL', 2020, '10-K')
    'MFC_2012_20F'  → ('MFC', 2012, '20-F')
    """
    parts = doc_name.rsplit("_", 2)
    if len(parts) != 3:
        raise ValueError(f"Cannot parse doc_name: {doc_name!r}  (expected TICKER_YEAR_FORMTYPE)")
    ticker, year_str, form_raw = parts
    year = int(year_str)
    form_type = re.sub(r"^(\d+)([A-Z].*)$", r"\1-\2", form_raw)
    return ticker.upper(), year, form_type


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Ticker → CIK
# ══════════════════════════════════════════════════════════════════════════════

_cik_cache: dict = {}

def get_cik(ticker: str) -> str | None:
    if ticker in _cik_cache:
        return _cik_cache[ticker]

    # Primary: EDGAR's official ticker→CIK map
    try:
        r = sleep_get("https://www.sec.gov/files/company_tickers.json")
        r.raise_for_status()
        for entry in r.json().values():
            if entry.get("ticker", "").upper() == ticker:
                cik = str(entry["cik_str"]).zfill(10)
                _cik_cache[ticker] = cik
                print(f"  CIK={cik}")
                return cik
    except Exception as e:
        print(f"  [warn] ticker map failed: {e}")

    # Fallback: EDGAR company search
    try:
        r = sleep_get(
            "https://efts.sec.gov/LATEST/search-index",
            params={"q": f'"{ticker}"', "forms": "10-K"},
        )
        r.raise_for_status()
        hits = r.json().get("hits", {}).get("hits", [])
        for hit in hits:
            src = hit.get("_source", {})
            if src.get("ticker_symbol", "").upper() == ticker:
                entity_id = str(src.get("entity_id", "")).zfill(10)
                _cik_cache[ticker] = entity_id
                print(f"  CIK={entity_id}  (via search fallback)")
                return entity_id
    except Exception as e:
        print(f"  [warn] EDGAR search fallback failed: {e}")

    return None


# ══════════════════════════════════════════════════════════════════════════════
# 3.  CIK + year + form_type  →  accession number
# ══════════════════════════════════════════════════════════════════════════════

def get_accession_number(cik: str, year: int, form_type: str) -> str | None:
    def _search_block(block: dict) -> str | None:
        forms = block.get("form", [])
        dates = block.get("filingDate", [])
        accs  = block.get("accessionNumber", [])
        candidates = []
        for form, date, acc in zip(forms, dates, accs):
            if form.upper() != form_type.upper():
                continue
            fy = int(date[:4])
            if fy == year or fy == year + 1:
                candidates.append((fy, date, acc))
        if not candidates:
            return None
        # prefer filings in year+1 (most common), then year itself; earliest date first
        candidates.sort(key=lambda x: (abs(x[0] - (year + 1)), x[1]))
        return candidates[0][2]

    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    print(f"  [debug] fetching {url!r}  (cik bytes: {[hex(ord(c)) for c in cik]})")
    try:
        r = sleep_get(url)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  [error] submissions fetch failed: {e}")
        return None

    acc = _search_block(data.get("filings", {}).get("recent", {}))
    if acc:
        return acc

    for file_info in data.get("filings", {}).get("files", []):
        try:
            r2 = sleep_get(f"https://data.sec.gov/submissions/{file_info['name']}")
            r2.raise_for_status()
            acc = _search_block(r2.json())
            if acc:
                return acc
        except Exception:
            continue

    return None


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Accession number → list of documents
# ══════════════════════════════════════════════════════════════════════════════

def get_filing_documents(cik: str, accession_number: str) -> list[dict]:
    """
    Returns list of dicts: {sequence, description, document, type, url}
    Parses the EDGAR filing index HTML.
    """
    cik_int   = int(cik)
    acc_clean = accession_number.replace("-", "")
    base_url  = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_clean}"

    # Try both index URL patterns
    html = ""
    for idx_url in [
        f"{base_url}/{accession_number}-index.htm",
        f"{base_url}/{accession_number}-index.html",
        f"{base_url}/",
    ]:
        try:
            r = sleep_get(idx_url)
            if r.ok and len(r.text) > 200:
                html = r.text
                print(f"  Index: {idx_url}")
                break
        except Exception:
            continue

    if not html:
        print(f"  [error] Could not fetch filing index for {accession_number}")
        return []

    docs = []
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.IGNORECASE | re.DOTALL)
    for row in rows:
        hrefs = re.findall(r'href="([^"]+)"', row, re.IGNORECASE)
        if not hrefs:
            continue
        filename = hrefs[0].split("/")[-1]
        if not filename or "?" in filename:
            continue
        # strip HTML tags from cells
        cells = [re.sub(r"<[^>]+>", "", c).strip()
                 for c in re.findall(r"<td[^>]*>(.*?)</td>", row, re.IGNORECASE | re.DOTALL)]
        docs.append({
            "sequence":    cells[0] if len(cells) > 0 else "",
            "description": cells[1] if len(cells) > 1 else "",
            "document":    filename,
            "type":        cells[3] if len(cells) > 3 else "",
            "url":         f"{base_url}/{filename}",
        })

    return docs


# ══════════════════════════════════════════════════════════════════════════════
# 5a.  Direct PDF download
# ══════════════════════════════════════════════════════════════════════════════

def download_pdf_direct(url: str, dest: Path) -> bool:
    try:
        r = sleep_get(url, stream=True)
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 16):
                f.write(chunk)
        print(f"  ✓ PDF downloaded  ({dest.stat().st_size // 1024} KB)")
        return True
    except Exception as e:
        print(f"  [error] PDF download failed: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# 5b.  HTM → PDF via Playwright (headless Chromium)
# ══════════════════════════════════════════════════════════════════════════════

def convert_htm_to_pdf(htm_url: str, dest: Path) -> bool:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("  [error] playwright not installed. Run:")
        print("            pip install playwright && playwright install chromium")
        return False

    print(f"  Converting HTM to PDF via headless Chromium…")

    # Download HTML first via requests (uses our SEC-compliant User-Agent),
    # then convert from a local temp file so playwright never touches SEC directly.
    import tempfile, os
    try:
        r = sleep_get(htm_url)
        r.raise_for_status()
        # Check we got actual HTML, not a block page
        if "undeclared automated tool" in r.text.lower() or "reference id:" in r.text.lower():
            print("  [error] SEC returned a bot-block page — check your User-Agent")
            return False
    except Exception as e:
        print(f"  [error] HTM fetch failed: {e}")
        return False

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Inject <base href> so relative URLs (images, CSS, fonts) resolve
        # back to the original SEC server rather than the temp file path.
        html = r.text
        base_tag = f'<base href="{htm_url}">'
        html = re.sub(r"(<head[^>]*>)", r"\1" + base_tag, html, count=1, flags=re.IGNORECASE)
        if base_tag not in html:
            html = base_tag + html

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as tmp:
            tmp.write(html)
            tmp_path = tmp.name

        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            # "load" waits for CSS and images; "domcontentloaded" does not
            page.goto(f"file://{tmp_path}", wait_until="load", timeout=60_000)
            # Wait for deferred resources (lazy-loaded images, web fonts) to settle
            try:
                page.wait_for_load_state("networkidle", timeout=60_000)
            except Exception:
                # Very large filings may never reach true networkidle;
                # "load" already guarantees CSS/images are ready
                pass
            page.pdf(
                path=str(dest),
                format="Letter",
                print_background=True,      # render background colors and images
                prefer_css_page_size=True,  # respect the document's own @page CSS rules
                margin={
                    "top": "15mm", "bottom": "15mm",
                    "left": "15mm", "right": "15mm",
                },
            )
            browser.close()

        os.unlink(tmp_path)
        print(f"  ✓ HTM→PDF  ({dest.stat().st_size // 1024} KB)")
        return True
    except Exception as e:
        print(f"  [error] playwright conversion failed: {e}")
        if dest.exists() and dest.stat().st_size < 1024:
            dest.unlink()
        return False


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Full pipeline per doc
# ══════════════════════════════════════════════════════════════════════════════

def process_doc(doc_name: str, out_dir: Path, no_convert: bool = False) -> bool:
    dest = out_dir / f"{doc_name}.pdf"
    if dest.exists() and dest.stat().st_size > 1024:
        print(f"  [skip] {doc_name} already downloaded")
        return True

    print(f"\n→ {doc_name}")
    try:
        ticker, year, form_type = parse_doc_name(doc_name)
    except ValueError as e:
        print(f"  [error] {e}")
        return False
    print(f"  ticker={ticker}  year={year}  form={form_type}")

    # Step 1: CIK
    cik = get_cik(ticker)
    if not cik:
        print(f"  [error] CIK not found for {ticker!r}")
        return False

    # Step 2: Accession number
    acc = get_accession_number(cik, year, form_type)
    if not acc:
        print(f"  [error] No {form_type} found for {ticker} / {year}")
        return False
    print(f"  accession={acc}")

    # Step 3: Filing documents
    docs = get_filing_documents(cik, acc)
    if not docs:
        return False

    print(f"  {len(docs)} document(s) in filing:")
    for d in docs[:10]:
        print(f"    seq={d['sequence']:3s}  [{d['type']:15s}]  {d['document']}")

    # Step 4a: native PDF?
    pdf_docs = [d for d in docs if d["document"].lower().endswith(".pdf")]
    if pdf_docs:
        # sequence "1" is the primary document
        pdf_docs.sort(key=lambda d: (d["sequence"] != "1", d["document"]))
        url = pdf_docs[0]["url"]
        print(f"  Native PDF: {pdf_docs[0]['document']}")
        return download_pdf_direct(url, dest)

    # Step 4b: HTM fallback
    htm_docs = [d for d in docs if d["document"].lower().endswith((".htm", ".html"))]
    if not htm_docs:
        print(f"  [error] No PDF or HTM documents found")
        return False

    if no_convert:
        print(f"  [skip] HTM-only filing — skipping playwright conversion (--no-convert)")
        return False

    # prefer sequence "1" or filename hints
    def htm_priority(d):
        return (d["sequence"] != "1",
                not any(kw in d["document"].lower() for kw in ["10k", "10-k", "annual", ticker.lower()]),
                d["document"])

    htm_docs.sort(key=htm_priority)
    primary = htm_docs[0]
    print(f"  No native PDF → converting: {primary['document']}")
    return convert_htm_to_pdf(primary["url"], dest)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def collect_doc_names(jsonl_path: str) -> list[str]:
    seen = set()
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for ev in row.get("evidences", []):
                doc = ev.get("doc_name", "")
                if doc:
                    seen.add(doc)
    return sorted(seen)


def main():
    parser = argparse.ArgumentParser(
        description="Download EDGAR PDFs for a FinanceBench-style JSONL dataset"
    )
    parser.add_argument("--jsonl",      required=True,  help="Path to the .jsonl dataset file")
    parser.add_argument("--out_dir",    default="pdfs", help="Output directory (default: ./pdfs)")
    parser.add_argument("--only",       nargs="*",      help="Only process these doc_names")
    parser.add_argument("--no-convert", action="store_true",
                        help="Skip playwright HTM→PDF conversion (report failure instead)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    doc_names = collect_doc_names(args.jsonl)
    if args.only:
        in_file = set(doc_names)
        doc_names = args.only   # allow names not in the JSONL too (for testing)

    print(f"Found {len(doc_names)} unique filings to download → {out_dir}/")

    ok, fail = 0, 0
    failed_docs = []

    for doc_name in doc_names:
        success = process_doc(doc_name, out_dir, no_convert=args.no_convert)
        if success:
            ok += 1
        else:
            fail += 1
            failed_docs.append(doc_name)

    print(f"\n{'='*55}")
    print(f"Done.  ✓ {ok} downloaded   ✗ {fail} failed")

    if failed_docs:
        print("\nFailed filings:")
        for d in failed_docs:
            print(f"  {d}")
        retry_path = out_dir / "failed_downloads.txt"
        retry_path.write_text("\n".join(failed_docs))
        print(f"\nFailed list saved to: {retry_path}")
    else:
        print("All filings downloaded successfully!")


if __name__ == "__main__":
    main()