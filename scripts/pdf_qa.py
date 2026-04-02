#!/usr/bin/env python3
"""
pdf_qa.py — PDF Quality Assurance for EDGAR financial filings
==============================================================

Evaluates PDFs produced by pdf_ex.py (page-break-injection pipeline) against
four quality criteria:

  1. PAGE_OVERFLOW   — a logical source page bleeds across 2+ PDF pages,
                       producing a short/orphaned page with just a header,
                       footer, page number, or stray line.

  2. RENDER_FAILURE  — a page is blank or nearly blank (figures / characters
                       failed to render), OR garbled Unicode replacement chars
                       indicate a broken character-map.

  3. CONTENT_CUT     — meaningful text is sliced at a page boundary:
                         • mid-word hyphenation (word- \n word)
                         • sentence continuation (no terminal punctuation +
                           next page starts lower-case)
                         • orphaned section heading at the bottom of a page

  4. TOO_SHORT       — fewer than 5 pages total (likely only the cover page
                       was captured; full filing body was missed).

Usage
-----
  # Evaluate one file
  python pdf_qa.py --pdf path/to/AAPL_2022_10K.pdf

  # Evaluate a whole directory
  python pdf_qa.py --pdf-dir pdfs-extended/ --out-report qa_report.json --summary qa_summary.csv

  # Only print files that have at least one issue
  python pdf_qa.py --pdf-dir pdfs-extended/ --flagged-only

  # Re-generate a list of files to re-render (exit-code 1 if any failures)
  python pdf_qa.py --pdf-dir pdfs-extended/ --regen-list regen.txt --strict

Install
-------
  pip install pdfplumber pypdf           # pdfplumber is strongly preferred
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# ── Library availability ──────────────────────────────────────────────────────
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

if not HAS_PDFPLUMBER and not HAS_PYPDF:
    sys.exit(
        "[pdf_qa] Neither pdfplumber nor pypdf found.\n"
        "Install with:  pip install pdfplumber pypdf"
    )

# ── Constants / thresholds ────────────────────────────────────────────────────

# A page is "near-blank" if it has fewer characters than this.
# Financial pages are dense; legitimate sparse pages (cover, ToC, signatures)
# are caught by the SPARSE_LEGITIMATE whitelist below.
NEAR_BLANK_CHARS = 80

# Overflow orphan: a page is a suspected overflow artifact if its char count
# is below this fraction of the median page length.
OVERFLOW_MEDIAN_RATIO = 0.08

# Minimum absolute chars for a page to trigger overflow suspicion
# (avoids false positives on genuinely short exhibit pages).
OVERFLOW_ABS_MAX = 200

# Pages whose text matches these patterns are "legitimately sparse"
# and should NOT be flagged as overflow artifacts.
_LEGIT_SPARSE_RE = re.compile(
    r"(?i)(exhibit|signature|power\s+of\s+attorney|"
    r"incorporated\s+by\s+reference|certif|consent|"
    r"table\s+of\s+contents|index\s+to\s+financial|"
    r"selected\s+financial|this\s+page\s+intentionally\s+left\s+blank)",
)

# Minimum page count to be considered a full filing.
MIN_PAGES = 5

# Fraction of characters that are Unicode replacement (U+FFFD) or other
# C0/C1 control chars (exc. tab, LF, CR) to trigger garbled-text warning.
GARBLE_RATIO_THRESHOLD = 0.04

# Orphaned heading: if the last N visible chars of a page are a short line
# that looks like a section header, flag as potential orphan.
ORPHAN_HEADING_MAX_LEN = 80
_HEADING_RE = re.compile(
    r"(?i)^(item\s+\d|note\s+\d|part\s+[ivx]+|section\s+\d|"
    r"financial\s+statements?|management.{0,20}discussion|"
    r"selected\s+(financial|consolidated))",
)

# ── Issue severity ────────────────────────────────────────────────────────────
SEVERITY_ERROR   = "ERROR"
SEVERITY_WARNING = "WARNING"

# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Issue:
    severity: str
    code: str
    page: Optional[int]   # 1-indexed; None = file-level
    detail: str

@dataclass
class PageInfo:
    page_num: int          # 1-indexed
    char_count: int
    has_images: bool
    first_100: str         # first 100 printable chars (for diagnostics)
    last_100: str          # last 100 printable chars

@dataclass
class PDFReport:
    pdf_path: str
    num_pages: int
    issues: list[Issue] = field(default_factory=list)
    page_infos: list[PageInfo] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(i.severity == SEVERITY_ERROR for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == SEVERITY_WARNING for i in self.issues)

    @property
    def is_clean(self) -> bool:
        return not self.issues

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == SEVERITY_ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == SEVERITY_WARNING)

    def to_dict(self) -> dict:
        return {
            "pdf": self.pdf_path,
            "num_pages": self.num_pages,
            "status": "CLEAN" if self.is_clean else ("ERROR" if self.has_errors else "WARNING"),
            "errors": self.error_count,
            "warnings": self.warning_count,
            "issues": [asdict(i) for i in self.issues],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Text extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_pages_pdfplumber(path: Path) -> list[PageInfo]:
    """Extract per-page text and image presence using pdfplumber."""
    infos: list[PageInfo] = []
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            # pdfplumber returns None for totally blank pages
            text_clean = text.strip()
            has_images = bool(page.images)
            first_100 = text_clean[:100].replace("\n", " ")
            last_100  = text_clean[-100:].replace("\n", " ") if text_clean else ""
            infos.append(PageInfo(
                page_num   = i,
                char_count = len(text_clean),
                has_images = has_images,
                first_100  = first_100,
                last_100   = last_100,
            ))
    return infos


def _extract_pages_pypdf(path: Path) -> list[PageInfo]:
    """Fallback extractor using pypdf (less spatial accuracy)."""
    infos: list[PageInfo] = []
    reader = PdfReader(str(path))
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text_clean = text.strip()
        # pypdf doesn't give image presence cheaply; approximate via /Resources
        try:
            res = page.get("/Resources", {})
            xobj = res.get("/XObject", {})
            has_images = any(
                xobj[k].get("/Subtype") == "/Image"
                for k in xobj
            ) if xobj else False
        except Exception:
            has_images = False
        first_100 = text_clean[:100].replace("\n", " ")
        last_100  = text_clean[-100:].replace("\n", " ") if text_clean else ""
        infos.append(PageInfo(
            page_num   = i,
            char_count = len(text_clean),
            has_images = has_images,
            first_100  = first_100,
            last_100   = last_100,
        ))
    return infos


def extract_pages(path: Path) -> list[PageInfo]:
    if HAS_PDFPLUMBER:
        try:
            return _extract_pages_pdfplumber(path)
        except Exception as e:
            print(f"  [pdfplumber error] {e} — falling back to pypdf", file=sys.stderr)
    if HAS_PYPDF:
        return _extract_pages_pypdf(path)
    raise RuntimeError("No PDF extraction library available.")


# ─────────────────────────────────────────────────────────────────────────────
# Individual checks
# ─────────────────────────────────────────────────────────────────────────────

def check_page_count(pages: list[PageInfo]) -> list[Issue]:
    """CHECK 4 — file has fewer than MIN_PAGES pages."""
    if len(pages) < MIN_PAGES:
        return [Issue(
            severity = SEVERITY_ERROR,
            code     = "TOO_SHORT",
            page     = None,
            detail   = (
                f"Only {len(pages)} page(s). "
                f"Expected ≥{MIN_PAGES}. "
                "Likely only the cover/intro page was captured — "
                "the 'single' page-detection fallback may have fired."
            ),
        )]
    return []


def check_blank_and_render_failures(pages: list[PageInfo]) -> list[Issue]:
    """CHECK 2 — blank pages and garbled-character rendering failures."""
    issues: list[Issue] = []
    for p in pages:
        if p.char_count == 0 and not p.has_images:
            issues.append(Issue(
                severity = SEVERITY_ERROR,
                code     = "BLANK_PAGE",
                page     = p.page_num,
                detail   = (
                    "Page is completely empty (no extractable text, no images). "
                    "Chromium height-limit truncation or a failed render."
                ),
            ))
        elif p.char_count < NEAR_BLANK_CHARS and not p.has_images:
            # Don't flag if this looks like a legitimate sparse page
            combined = (p.first_100 + " " + p.last_100).strip()
            if not _LEGIT_SPARSE_RE.search(combined):
                issues.append(Issue(
                    severity = SEVERITY_WARNING,
                    code     = "NEAR_BLANK_PAGE",
                    page     = p.page_num,
                    detail   = (
                        f"Only {p.char_count} char(s), no images. "
                        f"Content: {combined!r}"
                    ),
                ))

    # Garbled text: high ratio of replacement / control characters
    for p in pages:
        if p.char_count < 20:
            continue
        # Combine first and last samples for a quick garble check
        sample = p.first_100 + p.last_100
        garble_chars = sum(
            1 for c in sample
            if c == "\ufffd"                            # Unicode replacement char
            or (ord(c) < 32 and c not in "\t\n\r")     # C0 control (exc. whitespace)
            or (0x80 <= ord(c) <= 0x9F)                 # C1 control block
        )
        ratio = garble_chars / max(len(sample), 1)
        if ratio >= GARBLE_RATIO_THRESHOLD:
            issues.append(Issue(
                severity = SEVERITY_WARNING,
                code     = "GARBLED_TEXT",
                page     = p.page_num,
                detail   = (
                    f"{garble_chars}/{len(sample)} sampled chars are "
                    f"replacement/control chars ({ratio:.1%}). "
                    "Font embedding or character-map failure."
                ),
            ))
    return issues


def check_page_overflow(pages: list[PageInfo]) -> list[Issue]:
    """
    CHECK 1 — a logical source page bleeds onto a separate PDF page,
    leaving an orphaned short page.

    Heuristic: flag a page as a suspected overflow artifact when ALL of:
      (a) its char_count < OVERFLOW_MEDIAN_RATIO × median_char_count
      (b) its char_count < OVERFLOW_ABS_MAX
      (c) it is NOT the first or last page (those can legitimately be short)
      (d) the text doesn't look like a legitimate sparse page
      (e) the PREVIOUS page also doesn't end with a clean terminal structure
          (its last_100 doesn't end with a paragraph-terminal char)
    """
    issues: list[Issue] = []
    if len(pages) < 3:
        return issues

    # Compute median over mid-document pages (exclude first and last 2)
    mid_counts = [p.char_count for p in pages[2:-2]] if len(pages) > 6 else \
                 [p.char_count for p in pages[1:-1]]
    if not mid_counts:
        return issues
    med = statistics.median(mid_counts) or 1

    for idx, p in enumerate(pages):
        # Skip first and last page
        if idx == 0 or idx == len(pages) - 1:
            continue

        if p.char_count >= OVERFLOW_ABS_MAX:
            continue
        if p.char_count / med >= OVERFLOW_MEDIAN_RATIO:
            continue

        combined = (p.first_100 + " " + p.last_100).strip()
        if _LEGIT_SPARSE_RE.search(combined):
            continue

        # Additional signal: previous page ended abruptly (no terminal punct)
        prev = pages[idx - 1]
        prev_tail = prev.last_100.rstrip()
        prev_ended_cleanly = prev_tail and prev_tail[-1] in ".!?:;\"'"
        overflow_hint = "" if prev_ended_cleanly else \
            f" (prev page ends mid-sentence: {prev_tail[-40:]!r})"

        issues.append(Issue(
            severity = SEVERITY_WARNING,
            code     = "PAGE_OVERFLOW",
            page     = p.page_num,
            detail   = (
                f"Suspected overflow artifact: {p.char_count} chars "
                f"({p.char_count/med:.1%} of median {int(med)}).{overflow_hint} "
                f"Content: {combined!r}"
            ),
        ))

    return issues


def check_content_continuity(pages: list[PageInfo]) -> list[Issue]:
    """
    CHECK 3 — text is cut at a page boundary without proper closure.

    Three sub-checks per page boundary:
      (a) HYPHEN_CUT    — last word ends with a hyphen (mid-word break)
      (b) SENTENCE_CUT  — last line lacks terminal punctuation AND next page
                          starts with a lower-case letter (continuation)
      (c) ORPHAN_HEAD   — last line looks like a section heading with nothing
                          after it (the section body starts on the next page)
    """
    issues: list[Issue] = []

    for idx in range(len(pages) - 1):
        curr = pages[idx]
        nxt  = pages[idx + 1]

        if not curr.last_100:
            continue

        tail = curr.last_100.rstrip()
        head = nxt.first_100.lstrip() if nxt.first_100 else ""

        # ── (a) Hyphen cut ──────────────────────────────────────────────────
        # Match a word ending with a single hyphen (not an em-dash or ----)
        if re.search(r"\b[A-Za-z]{2,}-$", tail):
            issues.append(Issue(
                severity = SEVERITY_WARNING,
                code     = "CONTENT_CUT_HYPHEN",
                page     = curr.page_num,
                detail   = (
                    f"Page ends with hyphenated word fragment: {tail[-40:]!r} "
                    f"→ next page starts: {head[:40]!r}"
                ),
            ))

        # ── (b) Sentence cut ────────────────────────────────────────────────
        elif (
            tail
            and tail[-1] not in ".!?:;\"'"   # no terminal punctuation
            and head
            and head[0].islower()             # continuation in lower case
            and curr.char_count > NEAR_BLANK_CHARS  # page has real content
        ):
            issues.append(Issue(
                severity = SEVERITY_WARNING,
                code     = "CONTENT_CUT_SENTENCE",
                page     = curr.page_num,
                detail   = (
                    f"Sentence appears to continue across page boundary. "
                    f"End: {tail[-50:]!r} → Start: {head[:50]!r}"
                ),
            ))

        # ── (c) Orphaned section heading ────────────────────────────────────
        # Last non-empty block of text on the page is a short heading line
        last_line = tail.split("\n")[-1].strip() if "\n" in curr.last_100 else tail
        if (
            last_line
            and len(last_line) <= ORPHAN_HEADING_MAX_LEN
            and _HEADING_RE.match(last_line)
            and curr.char_count > NEAR_BLANK_CHARS
        ):
            issues.append(Issue(
                severity = SEVERITY_WARNING,
                code     = "ORPHAN_HEADING",
                page     = curr.page_num,
                detail   = (
                    f"Section heading at bottom of page with no body following: "
                    f"{last_line!r}. Body likely starts on page {nxt.page_num}."
                ),
            ))

    return issues


# ─────────────────────────────────────────────────────────────────────────────
# Master checker
# ─────────────────────────────────────────────────────────────────────────────

def check_pdf(path: Path) -> PDFReport:
    """Run all quality checks on a single PDF and return a PDFReport."""
    try:
        pages = extract_pages(path)
    except Exception as e:
        report = PDFReport(pdf_path=str(path), num_pages=0)
        report.issues.append(Issue(
            severity = SEVERITY_ERROR,
            code     = "UNREADABLE",
            page     = None,
            detail   = f"Could not open / parse PDF: {e}",
        ))
        return report

    report = PDFReport(pdf_path=str(path), num_pages=len(pages), page_infos=pages)

    report.issues.extend(check_page_count(pages))
    report.issues.extend(check_blank_and_render_failures(pages))
    report.issues.extend(check_page_overflow(pages))
    report.issues.extend(check_content_continuity(pages))

    # Sort issues by page number (file-level issues first, then ascending)
    report.issues.sort(key=lambda i: (i.page is None, i.page or 0))

    return report


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

_ANSI_RED    = "\033[31m"
_ANSI_YELLOW = "\033[33m"
_ANSI_GREEN  = "\033[32m"
_ANSI_BOLD   = "\033[1m"
_ANSI_RESET  = "\033[0m"


def _color(text: str, code: str, use_color: bool) -> str:
    return f"{code}{text}{_ANSI_RESET}" if use_color else text


def print_report(report: PDFReport, verbose: bool = False, use_color: bool = True) -> None:
    name = Path(report.pdf_path).name
    if report.is_clean:
        status = _color("✓ CLEAN", _ANSI_GREEN, use_color)
        print(f"  {status}  {name}  ({report.num_pages} pages)")
    else:
        severity = "ERROR" if report.has_errors else "WARNING"
        color    = _ANSI_RED if report.has_errors else _ANSI_YELLOW
        tag      = _color(severity, color, use_color)
        counts   = f"{report.error_count}E / {report.warning_count}W"
        print(f"  {tag}  {name}  ({report.num_pages} pages)  [{counts}]")
        if verbose or report.has_errors:
            for iss in report.issues:
                sev_tag = _color(iss.severity[:4], _ANSI_RED if iss.severity == SEVERITY_ERROR else _ANSI_YELLOW, use_color)
                page_tag = f"p{iss.page}" if iss.page else "file"
                print(f"       [{sev_tag}] {iss.code} ({page_tag}): {iss.detail}")


def write_json_report(reports: list[PDFReport], out_path: Path) -> None:
    data = {
        "summary": {
            "total":    len(reports),
            "clean":    sum(1 for r in reports if r.is_clean),
            "errors":   sum(1 for r in reports if r.has_errors),
            "warnings": sum(1 for r in reports if r.has_warnings and not r.has_errors),
        },
        "files": [r.to_dict() for r in reports],
    }
    out_path.write_text(json.dumps(data, indent=2))
    print(f"\nJSON report written → {out_path}")


def write_csv_summary(reports: list[PDFReport], out_path: Path) -> None:
    fields = ["pdf", "num_pages", "status", "errors", "warnings", "issue_codes"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in reports:
            codes = ";".join(sorted({i.code for i in r.issues}))
            writer.writerow({
                "pdf":        Path(r.pdf_path).name,
                "num_pages":  r.num_pages,
                "status":     "CLEAN" if r.is_clean else ("ERROR" if r.has_errors else "WARNING"),
                "errors":     r.error_count,
                "warnings":   r.warning_count,
                "issue_codes": codes,
            })
    print(f"CSV summary written  → {out_path}")


def write_regen_list(reports: list[PDFReport], out_path: Path) -> None:
    """Write doc_names (stem of PDF filenames) that need re-rendering."""
    bad = [
        Path(r.pdf_path).stem
        for r in reports
        if not r.is_clean
    ]
    if bad:
        out_path.write_text("\n".join(bad) + "\n")
        print(f"Re-render list ({len(bad)} files) → {out_path}")
    else:
        print("No files need re-rendering.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate quality of EDGAR PDFs produced by pdf_ex.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage")[0],
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdf",     metavar="FILE",
                     help="Evaluate a single PDF file")
    src.add_argument("--pdf-dir", metavar="DIR",
                     help="Evaluate all *.pdf files in a directory")

    parser.add_argument("--out-report", metavar="FILE",
                        help="Write full JSON report to this file")
    parser.add_argument("--summary",    metavar="FILE",
                        help="Write one-row-per-PDF CSV summary to this file")
    parser.add_argument("--regen-list", metavar="FILE",
                        help="Write doc_names of flagged files (for --only re-run)")
    parser.add_argument("--flagged-only", action="store_true",
                        help="Only print files that have at least one issue")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print all issue details (not just errors)")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable ANSI colour output")
    parser.add_argument("--strict", action="store_true",
                        help="Exit with code 1 if any ERROR-level issues found")

    args = parser.parse_args()
    use_color = not args.no_color and sys.stdout.isatty()

    # ── Collect PDF paths ─────────────────────────────────────────────────────
    if args.pdf:
        paths = [Path(args.pdf)]
    else:
        paths = sorted(Path(args.pdf_dir).glob("*.pdf"))

    if not paths:
        print("[pdf_qa] No PDF files found.", file=sys.stderr)
        return 1

    print(f"\n{'='*60}")
    print(f"  pdf_qa — evaluating {len(paths)} PDF(s)")
    print(f"{'='*60}\n")

    # ── Run checks ────────────────────────────────────────────────────────────
    reports: list[PDFReport] = []
    for path in paths:
        report = check_pdf(path)
        reports.append(report)
        if not args.flagged_only or not report.is_clean:
            print_report(report, verbose=args.verbose, use_color=use_color)

    # ── Aggregate summary ─────────────────────────────────────────────────────
    total    = len(reports)
    n_clean  = sum(1 for r in reports if r.is_clean)
    n_errors = sum(1 for r in reports if r.has_errors)
    n_warn   = sum(1 for r in reports if r.has_warnings and not r.has_errors)

    # Breakdown by issue code
    code_counts: dict[str, int] = {}
    for r in reports:
        for iss in r.issues:
            code_counts[iss.code] = code_counts.get(iss.code, 0) + 1

    print(f"\n{'='*60}")
    print(f"  Results: {total} total  |  {n_clean} clean  |  {n_errors} with errors  |  {n_warn} warnings-only")
    if code_counts:
        print(f"\n  Issue breakdown:")
        for code, cnt in sorted(code_counts.items(), key=lambda x: -x[1]):
            print(f"    {code:<30} {cnt} occurrence(s)")
    print(f"{'='*60}\n")

    # ── Write outputs ─────────────────────────────────────────────────────────
    if args.out_report:
        write_json_report(reports, Path(args.out_report))
    if args.summary:
        write_csv_summary(reports, Path(args.summary))
    if args.regen_list:
        write_regen_list(reports, Path(args.regen_list))

    if args.strict and n_errors > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())