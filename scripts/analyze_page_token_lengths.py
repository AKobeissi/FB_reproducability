#!/usr/bin/env python3
"""
analyze_page_token_lengths.py
==============================
Estimates BGE-M3 token lengths per page using chars/4 approximation.
BGE-M3 uses SentencePiece; financial English averages ~4 chars/token.
This runs in seconds with no model download required.

Usage:  python analyze_page_token_lengths.py --pdf-dir Final-PDF
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List

try:
    from pypdf import PdfReader
except ImportError:
    print("pypdf not found — pip install pypdf"); sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("numpy not found — pip install numpy"); sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs): return it

_WS = re.compile(r"\s+")

CHARS_PER_TOKEN = 4.0   # BGE-M3 SentencePiece, English financial text


def normalise(text: str) -> str:
    return _WS.sub(" ", (text or "").strip())


def chars_to_tokens(text: str) -> int:
    return max(1, round(len(text) / CHARS_PER_TOKEN))


def extract_pdf_pages(pdf_path: Path) -> List[str]:
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as e:
        print(f"  [WARN] {pdf_path.name}: {e}")
        return []
    pages = []
    for page in reader.pages:
        try:
            raw = page.extract_text() or ""
        except Exception:
            raw = ""
        pages.append(normalise(raw))
    return pages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir",   default="Final-PDF")
    parser.add_argument("--min-chars", type=int, default=50,
                        help="Skip pages shorter than this (blank/header pages)")
    parser.add_argument("--output",    default="analyze_page_token_lengths.json")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.exists():
        print(f"[ERROR] Not found: {pdf_dir}"); sys.exit(1)

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"[ERROR] No PDFs in {pdf_dir}"); sys.exit(1)

    print(f"Analyzing {len(pdfs)} PDFs in {pdf_dir}  (chars/4 token estimate) ...")

    all_lengths  = []
    blank_pages  = 0
    total_pages  = 0
    per_doc      = {}

    for pdf_path in tqdm(pdfs, desc="PDFs"):
        pages      = extract_pdf_pages(pdf_path)
        doc_lens   = []
        for text in pages:
            total_pages += 1
            if len(text) < args.min_chars:
                blank_pages += 1
                continue
            n = chars_to_tokens(text)
            doc_lens.append(n)
            all_lengths.append(n)

        if doc_lens:
            per_doc[pdf_path.stem] = {
                "n_pages":   len(pages),
                "n_content": len(doc_lens),
                "mean":      round(float(np.mean(doc_lens)),  1),
                "median":    round(float(np.median(doc_lens)),1),
                "p95":       round(float(np.percentile(doc_lens, 95)), 0),
                "p99":       round(float(np.percentile(doc_lens, 99)), 0),
                "max":       int(np.max(doc_lens)),
            }

    a = np.array(all_lengths)

    pcts      = [50, 75, 90, 95, 99, 99.5, 100]
    pct_vals  = {p: float(np.percentile(a, p)) for p in pcts}
    thresholds = [512, 1024, 1536, 2048, 3072, 4096]
    coverage  = {t: float(np.mean(a <= t) * 100) for t in thresholds}

    print(f"\n{'='*60}")
    print(f"  PAGE TOKEN LENGTH ANALYSIS  (chars/{CHARS_PER_TOKEN:.0f} estimate)")
    print(f"{'='*60}")
    print(f"  PDFs           : {len(pdfs)}")
    print(f"  Total pages    : {total_pages}")
    print(f"  Blank/skipped  : {blank_pages}  (< {args.min_chars} chars)")
    print(f"  Content pages  : {len(all_lengths)}")
    print(f"\n  Distribution:")
    print(f"  {'Stat':<12} {'Tokens':>7}")
    print(f"  {'-'*21}")
    print(f"  {'mean':<12} {np.mean(a):>7.0f}")
    print(f"  {'std':<12} {np.std(a):>7.0f}")
    for p in pcts:
        print(f"  {'p'+str(p):<12} {pct_vals[p]:>7.0f}")
    print(f"\n  Coverage at max_seq_length cutoffs:")
    print(f"  {'Threshold':<12} {'Coverage':>10}")
    print(f"  {'-'*24}")
    for t, cov in coverage.items():
        marker = "  ← current training setting" if t == 2048 else ""
        print(f"  {t:<12} {cov:>9.2f}%{marker}")

    # Recommendation
    rec = next((t for t, cov in coverage.items() if cov >= 99.0), 4096)
    p99 = pct_vals[99]
    print(f"\n  p99 token length = {p99:.0f}")
    print(f"  Smallest threshold covering 99% of pages = {rec}")
    if rec <= 2048:
        print(f"  ✓ max_seq_length=2048 is correct — covers 99%+ of all pages.")
    else:
        print(f"  ✗ max_seq_length=2048 covers only {coverage[2048]:.1f}% of pages.")
        print(f"    Recommended: max_seq_length={rec}")

    # Docs with long pages
    long_docs = [(d, s) for d, s in per_doc.items() if s["max"] > 2048]
    if long_docs:
        long_docs.sort(key=lambda x: x[1]["max"], reverse=True)
        print(f"\n  Docs with pages > 2048 tokens ({len(long_docs)} docs):")
        for doc, s in long_docs[:10]:
            print(f"    {doc[:50]:<50}  max={s['max']:>5}  p99={s['p99']:>5.0f}")
        if len(long_docs) > 10:
            print(f"    ... and {len(long_docs)-10} more")

    out = {
        "estimation": f"chars/{CHARS_PER_TOKEN}",
        "n_pdfs": len(pdfs),
        "total_pages": total_pages,
        "blank_pages": blank_pages,
        "content_pages": len(all_lengths),
        "global_stats": {
            "mean": float(np.mean(a)), "std": float(np.std(a)),
            "min": int(np.min(a)), "max": int(np.max(a)),
            **{f"p{p}": round(v, 1) for p, v in pct_vals.items()},
        },
        "coverage_pct": {str(t): round(cov, 2) for t, cov in coverage.items()},
        "per_doc": per_doc,
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Full stats → {args.output}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()