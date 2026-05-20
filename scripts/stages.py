#!/usr/bin/env python3
"""
stage_pdfs.py — Filter PDF-Opus manifest and copy qualifying PDFs to Final-PDF.

Filters applied (ALL must pass):
  1. success == true
  2. pdf_pages >= 5
  3. filing_date >= 2009-01-01  (excludes 2008 and earlier)

Usage:
    python stage_pdfs.py --manifest PDF-Opus/manifest.jsonl --src PDF-Opus --dst Final-PDF
"""

import argparse
import json
import shutil
from pathlib import Path
from datetime import date

# ── Filter thresholds ─────────────────────────────────────────────────────────
MIN_PAGES       = 5
MIN_DATE        = date(2009, 1, 1)   # strictly >= 2009-01-01 (excludes 2008)


def parse_date(s: str) -> date | None:
    """Parse ISO date string, return None on failure."""
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])   # handles "2021-10-29" or datetime strings
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="PDF-Opus/manifest.jsonl")
    parser.add_argument("--src",      default="PDF-Opus",   help="Source PDF directory")
    parser.add_argument("--dst",      default="Final-PDF",  help="Destination directory")
    parser.add_argument("--dry-run",  action="store_true",  help="Print actions without copying")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    src_dir       = Path(args.src)
    dst_dir       = Path(args.dst)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    if not args.dry_run:
        dst_dir.mkdir(parents=True, exist_ok=True)

    # ── Counters ──────────────────────────────────────────────────────────────
    total          = 0
    n_success      = 0   # success == true (before other filters)
    n_failed       = 0   # success == false
    n_too_short    = 0
    n_too_old      = 0
    n_no_date      = 0
    n_missing_pdf  = 0
    n_copied       = 0

    rejected = []        # (doc_name, [reasons])
    accepted = []        # doc_names that passed all filters

    with manifest_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[warn] JSON parse error: {e}")
                continue

            total += 1
            doc_name   = row.get("doc_name", f"row_{total}")
            success    = row.get("success", False)
            pdf_pages  = row.get("pdf_pages", 0)
            filing_date_str = row.get("filing_date", "")
            # pdf_path in manifest may point to a different machine; resolve locally
            pdf_filename = doc_name + ".pdf"
            pdf_src = src_dir / pdf_filename

            if success:
                n_success += 1
            else:
                n_failed += 1

            # ── Apply filters ─────────────────────────────────────────────────
            reasons = []

            if not success:
                reasons.append("success=false")

            if pdf_pages < MIN_PAGES:
                reasons.append(f"pdf_pages={pdf_pages} < {MIN_PAGES}")
                n_too_short += 1

            filing_date = parse_date(filing_date_str)
            if filing_date is None:
                reasons.append(f"no_date ('{filing_date_str}')")
                n_no_date += 1
            elif filing_date < MIN_DATE:
                reasons.append(f"filing_date={filing_date_str} (before 2009)")
                n_too_old += 1

            if not pdf_src.exists():
                reasons.append(f"pdf_missing ({pdf_filename})")
                n_missing_pdf += 1

            # ── Decision ──────────────────────────────────────────────────────
            if reasons:
                rejected.append((doc_name, reasons))
                continue

            # Passed all filters
            accepted.append(doc_name)
            dst_path = dst_dir / pdf_filename

            if args.dry_run:
                print(f"  [dry-run] MOVE {pdf_src} → {dst_path}  (pages={pdf_pages}, date={filing_date_str})")
            else:
                shutil.move(str(pdf_src), dst_path)
                n_copied += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"MANIFEST SUMMARY")
    print(f"{sep}")
    print(f"  Total rows in manifest : {total}")
    print(f"  success=true           : {n_success}")
    print(f"  success=false          : {n_failed}")
    print(sep)
    print(f"FILTER BREAKDOWN (of {total} total)")
    print(f"  Excluded: success=false        : {n_failed}")
    print(f"  Excluded: pdf_pages < {MIN_PAGES}       : {n_too_short}")
    print(f"  Excluded: filing_date < 2009   : {n_too_old}")
    print(f"  Excluded: date missing         : {n_no_date}")
    print(f"  Excluded: PDF file missing     : {n_missing_pdf}")
    print(f"  (note: one doc may hit multiple reasons)")
    print(sep)
    print(f"RESULT")
    print(f"  Passed all filters : {len(accepted)}")
    if args.dry_run:
        print(f"  [dry-run — no files copied]")
    else:
        print(f"  Moved to {dst_dir}  : {n_copied}")
    print(sep)

    # ── Write rejected log ────────────────────────────────────────────────────
    rejected_path = dst_dir / "rejected_docs.jsonl" if not args.dry_run else Path("rejected_docs_dryrun.jsonl")
    rejected_path.parent.mkdir(parents=True, exist_ok=True)
    with rejected_path.open("w") as f:
        for doc_name, reasons in rejected:
            f.write(json.dumps({"doc_name": doc_name, "reasons": reasons}) + "\n")

    accepted_path = dst_dir / "accepted_docs.txt" if not args.dry_run else Path("accepted_docs_dryrun.txt")
    accepted_path.parent.mkdir(parents=True, exist_ok=True)
    accepted_path.write_text("\n".join(accepted) + "\n")

    print(f"  Logs written:")
    print(f"    {rejected_path}")
    print(f"    {accepted_path}")
    print(sep)


if __name__ == "__main__":
    main()