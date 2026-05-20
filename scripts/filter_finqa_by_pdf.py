#!/usr/bin/env python3
"""
filter_finqa_by_pdf.py  —  Keep only rows where the PDF exists in Final-PDF/
=============================================================================

Reads data/finqa_test.jsonl and writes a new JSONL containing only the rows
whose doc_name has a corresponding PDF in the Final-PDF/ directory AND is not
on the blocklist of known-bad PDFs.

Usage
-----
  python filter_finqa_by_pdf.py \
      --lofin   data/finqa_test.jsonl \
      --pdf-dir Final-PDF \
      --output  data/finqa_test_pdf_only.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Blocklist — PDFs with known rendering / extraction problems.
# Excluded even if the file exists in Final-PDF/.
# All suffixes are normalised to uppercase (10K, 10Q) before comparison.
# ─────────────────────────────────────────────────────────────────────────────
BLOCKLIST: set = {
    "YUM_2024_10K",
    "YUM_2023_10K",
    "XEL_2023_10K",
    "V_2014_10K",
    "UNP_2018_10K",
    "UPS_2012_10K",
    "UNP_2015_10K",
    "T_2024Q2_10Q",
    "UAA_2024_10K",
    "T_2020_10K",
    "T_2021_10K",
    "T_2022_10K",
    "T_2023_10K",
    "STT_2014_10K",
    "STT_2023_10K",
    "TFX_2015_10K",
    "SBUX_2022_10K",
    "RSG_2012_10K",
    "PPG_2023_10K",
    "PPG_2024_10K",
    "PPG_2008_10K",
    "PM_2017_10K",
    "MS_2022_10K",
    "MO_2016_10K",
    "MRK_2023_10K",
    "MMM_2013_10K",
    "MMM_2015_10K",
    "MO_2014_10K",
    "LLY_2024Q2_10Q",
    "JPM_2023_10K",
    "KHC_2018_10K",
    "JPM_2022_10K",
    "JNJ_2018_10K",
    "IT_2024_10K",
    "INTC_2013_10K",
    "INTC_2015_10K",
    "HUM_2023_10K",
    "HMW_2017_10K",
    "HMW_2018_10K",
    "HUM_2014_10K",
    "GPN_2013_10K",
    "GPN_2014_10K",
    "GOOGL_2023_10K",
    "GPC_2023_10K",
    "GPC_2024_10K",
    "GM_2023_10K",
    "GE_2022_10K",
    "FIS_2010_10K",
    "FIS_2012_10K",
    "FIS_2024_10K",
    "ETR_2016_10K",
    "ETR_2017_10K",
    "ETR_2013_10K",
    "ETR_2015_10K",
    "ECL_2016_10K",
    "ECL_2017_10K",
    "C_2015_10K",
    "CE_2013_10K",
    "CDW_2015_10K",
    "CB_2008_10K",
    "CDNS_2015_10K",
    "BKR_2017_10K",
    "BA_2024_10K",
    "BA_2024Q1_10Q",
    "BA_2024Q3_10Q",
    "BA_2023_10K",
    "BAC_2022_10K",
    "BALL_2011_10K",
    "BALL_2012_10K",
    "BAC_2021_10K",
    "AON_2018_10K",
    "ANET_2015_10K",
    "AON_2015_10K",
    "AMGN_2021_10K",
    "ALLE_2015_10K",
    "AAL_2013_10K",
}


def normalise_doc_name(doc_name: str) -> str:
    """Normalise suffix capitalisation for blocklist comparison: 10k->10K, 10q->10Q."""
    parts = doc_name.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0] + "_" + parts[1].upper()
    return doc_name.upper()


_STEM_RE = re.compile(
    r"^(?P<ticker>.+?)_(?P<year>\d{4})(?:Q\d)?_(?P<form>.+)$", re.IGNORECASE
)


def build_pdf_set(pdf_dir: Path) -> set:
    """
    Returns a set of doc_name strings (PDF stems) found in pdf_dir.
    e.g. {"ZBH_2008_10K", "AAPL_2020_10K", ...}
    """
    stems = set()
    for p in pdf_dir.glob("*.pdf"):
        if _STEM_RE.match(p.stem):
            stems.add(p.stem)
    logger.info(f"Found {len(stems)} PDFs in {pdf_dir}")
    return stems


def main():
    parser = argparse.ArgumentParser(
        description="Filter finqa_test.jsonl to rows with available, non-blocklisted PDFs"
    )
    parser.add_argument(
        "--lofin", default="data/finqa_test.jsonl",
        help="Input LoFIN JSONL (default: data/finqa_test.jsonl)"
    )
    parser.add_argument(
        "--pdf-dir", default="Final-PDF",
        help="PDF directory (default: Final-PDF)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSONL (default: <lofin_stem>_pdf_only.jsonl)"
    )
    args = parser.parse_args()

    lofin_path = Path(args.lofin)
    pdf_dir    = Path(args.pdf_dir)
    out_path   = Path(args.output) if args.output \
                 else lofin_path.parent / (lofin_path.stem + "_pdf_only.jsonl")

    for p, label in [(lofin_path, "--lofin"), (pdf_dir, "--pdf-dir")]:
        if not p.exists():
            logger.error(f"{label} not found: {p}")
            sys.exit(1)

    pdf_set = build_pdf_set(pdf_dir)

    kept              = 0
    dropped_no_pdf    = 0
    dropped_blocklist = 0
    missing_docs: set = set()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(lofin_path, encoding="utf-8") as fin, \
         open(out_path,   "w", encoding="utf-8") as fout:

        for lineno, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"  line {lineno}: JSON error -- {e}")
                continue

            evidences = row.get("evidences", [])
            doc_name  = evidences[0].get("doc_name", "") if evidences else ""
            norm_name = normalise_doc_name(doc_name)

            # Blocklist check first
            if norm_name in BLOCKLIST:
                dropped_blocklist += 1
                continue

            # PDF existence check
            if doc_name in pdf_set:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
            else:
                dropped_no_pdf += 1
                missing_docs.add(doc_name)

    total = kept + dropped_no_pdf + dropped_blocklist
    print(f"\n{'='*56}")
    print(f"  Filter complete")
    print(f"{'='*56}")
    print(f"  Input rows              : {total}")
    print(f"  Kept                    : {kept}")
    print(f"  Dropped (no PDF)        : {dropped_no_pdf}")
    print(f"  Dropped (blocklist)     : {dropped_blocklist}  ({len(BLOCKLIST)} entries)")
    print(f"  Output                  : {out_path}")
    print(f"{'='*56}")
    if missing_docs:
        print(f"\n  Missing PDFs ({len(missing_docs)} unique docs):")
        for d in sorted(missing_docs)[:20]:
            print(f"    {d}")
        if len(missing_docs) > 20:
            print(f"    ... and {len(missing_docs) - 20} more")
    print()


if __name__ == "__main__":
    main()