#!/usr/bin/env python3
"""
scripts/prepare_finqa_train_gold_pages.py
==========================================
Converts finqa/train.json  →  data/finqa_train_gold_pages.jsonl

The FinQA train split encodes the gold page directly in the filename:
    "AAL/2018/page_13.pdf"  →  company=AAL, year=2018, page_num=12  (0-indexed)

Convention confirmed against test split: html_page_num - 1 == pdf_page_0idx.

Two-tier strategy (all ~6,251 entries are kept):
  Tier A — PDF available and not blocklisted:
    1. Parse (company, year, html_page) from filename.
    2. Locate the matching PDF in Final-PDF/.
    3. Validate by token-overlap (pre_text/post_text vs extracted page).
    4. Refine page index with ±search_window scan if needed.
    Output: _source="pdf", evidence_text from PDF extraction.

  Tier B — no PDF, or blocklisted, or overlap < threshold:
    Use pre_text + post_text from train.json directly as evidence_text.
    This is actually cleaner than PDF extraction (no pypdf noise).
    Output: _source="json", page_num from filename (html_page - 1).

Gold text from pre_text + post_text is the same evidence the model needs to
rank at inference time — the PDF page is only needed for negative mining.
Entries without a PDF will use cross-doc random negatives during training.

Output schema per row (matches test JSONL):
  {
    "qid":              "AAL/2018/page_13.pdf-0",
    "question":         "...",
    "answer":           "...",
    "evidences":        [{"page_num": 12, "doc_name": "AAL_2018_10K"}],
    "evidences_updated":[{"page_num": 12, "doc_name": "AAL_2018_10K",
                          "pre_text": "...", "post_text": "...",
                          "evidence_text": "...",
                          "_match_score": 0.82,   # 0.0 for Tier B
                          "_source": "pdf"}]       # or "json"
  }

Usage
-----
  python scripts/prepare_finqa_train_gold_pages.py \\
      --finqa-train  finqa/train.json \\
      --pdf-dir      Final-PDF \\
      --output       data/finqa_train_gold_pages.jsonl \\
      --min-overlap  0.05

No FinanceBench data is touched.  The output is safe to use as training data.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

try:
    from pypdf import PdfReader
except ImportError:
    logger.error("pypdf not found — pip install pypdf")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):  # type: ignore
        return it

# ─────────────────────────────────────────────────────────────────────────────
# Blocklist of PDFs with known-bad text extraction (from filter_finqa_by_pdf.py)
# ─────────────────────────────────────────────────────────────────────────────
BLOCKLIST: Set[str] = {
    "AAL_2013_10K", "ALLE_2015_10K", "AMGN_2021_10K", "AON_2015_10K",
    "AON_2018_10K", "ANET_2015_10K", "BA_2023_10K", "BA_2024_10K",
    "BAC_2021_10K", "BAC_2022_10K", "BALL_2011_10K", "BALL_2012_10K",
    "BKR_2017_10K", "C_2015_10K", "CB_2008_10K", "CDNS_2015_10K",
    "CDW_2015_10K", "CE_2013_10K", "ECL_2016_10K", "ECL_2017_10K",
    "ETR_2013_10K", "ETR_2015_10K", "ETR_2016_10K", "ETR_2017_10K",
    "FIS_2010_10K", "FIS_2012_10K", "FIS_2024_10K", "GE_2022_10K",
    "GM_2023_10K", "GOOGL_2023_10K", "GPC_2023_10K", "GPC_2024_10K",
    "GPN_2013_10K", "GPN_2014_10K", "HMW_2017_10K", "HMW_2018_10K",
    "HUM_2014_10K", "HUM_2023_10K", "INTC_2013_10K", "INTC_2015_10K",
    "IT_2024_10K", "JNJ_2018_10K", "JPM_2022_10K", "JPM_2023_10K",
    "KHC_2018_10K", "LLY_2024Q2_10Q", "MO_2014_10K", "MO_2016_10K",
    "MMM_2013_10K", "MMM_2015_10K", "MRK_2023_10K", "MS_2022_10K",
    "PM_2017_10K", "PPG_2008_10K", "PPG_2023_10K", "PPG_2024_10K",
    "RSG_2012_10K", "SBUX_2022_10K", "STT_2014_10K", "STT_2023_10K",
    "T_2020_10K", "T_2021_10K", "T_2022_10K", "T_2023_10K",
    "T_2024Q2_10Q", "TFX_2015_10K", "UAA_2024_10K", "UNP_2015_10K",
    "UNP_2018_10K", "UPS_2012_10K", "V_2014_10K", "XEL_2023_10K",
    "YUM_2023_10K", "YUM_2024_10K",
}

_WS = re.compile(r"\s+")
_STEM_RE = re.compile(
    r"^(?P<ticker>.+?)_(?P<year>\d{4})(?:Q\d)?_(?P<form>.+)$", re.IGNORECASE
)
_FN_RE = re.compile(
    r"^(?P<ticker>[^/]+)/(?P<year>\d{4})/page_(?P<page>\d+)\.pdf(?:-\d+)?$",
    re.IGNORECASE,
)


def normalise(text: str) -> str:
    return _WS.sub(" ", (text or "").strip().lower())


def flatten(field) -> str:
    if isinstance(field, list):
        return " ".join(str(s) for s in field if s)
    return str(field) if field else ""


# ─────────────────────────────────────────────────────────────────────────────
# Build (TICKER_UPPER, YEAR) → (pdf_path, doc_name) index
# ─────────────────────────────────────────────────────────────────────────────

def build_pdf_index(pdf_dir: Path) -> Dict[Tuple[str, str], Tuple[Path, str]]:
    index: Dict[Tuple[str, str], Tuple[Path, str]] = {}
    for p in sorted(pdf_dir.glob("*.pdf")):
        m = _STEM_RE.match(p.stem)
        if not m:
            continue
        key = (m.group("ticker").upper(), m.group("year"))
        if key not in index:
            index[key] = (p, p.stem)
    logger.info(f"PDF index: {len(index)} (ticker, year) entries in {pdf_dir}")
    return index


# ─────────────────────────────────────────────────────────────────────────────
# PDF extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_pdf_pages(pdf_path: Path) -> List[str]:
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as e:
        logger.warning(f"  Cannot open {pdf_path.name}: {e}")
        return []
    pages = []
    for page in reader.pages:
        try:
            raw = page.extract_text() or ""
        except Exception:
            raw = ""
        pages.append(normalise(raw))
    return pages


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight token-overlap sanity check (no heavy dependency)
# ─────────────────────────────────────────────────────────────────────────────

def token_overlap(anchor: str, page_text: str, min_tokens: int = 5) -> float:
    """
    Jaccard overlap of unigrams between anchor and page_text.
    Uses lower-cased whitespace tokens; punctuation stays attached.
    Returns a float in [0, 1].  Fast: O(|anchor| + |page|) with sets.
    """
    a_toks = set(anchor.lower().split())
    p_toks = set(page_text.lower().split())
    if len(a_toks) < min_tokens or not p_toks:
        return 0.0
    return len(a_toks & p_toks) / len(a_toks | p_toks)


def find_best_page_by_overlap(
    anchor: str,
    page_texts: List[str],
    prior_page: int,
    search_window: int = 5,
) -> Tuple[int, float]:
    """
    Search ±search_window pages around prior_page for the best token overlap.
    Falls back to prior_page if nothing is better.
    """
    n = len(page_texts)
    lo = max(0, prior_page - search_window)
    hi = min(n - 1, prior_page + search_window)

    best_page = prior_page
    best_score = token_overlap(anchor, page_texts[prior_page]) if prior_page < n else 0.0

    for i in range(lo, hi + 1):
        if i == prior_page:
            continue
        score = token_overlap(anchor, page_texts[i])
        if score > best_score:
            best_score = score
            best_page = i

    return best_page, best_score


# ─────────────────────────────────────────────────────────────────────────────
# Main conversion
# ─────────────────────────────────────────────────────────────────────────────

def convert(
    train_json:   Path,
    pdf_dir:      Path,
    out_path:     Path,
    reject_path:  Path,
    min_overlap:  float,
    search_window: int,
    max_rows:     Optional[int],
):
    pdf_index = build_pdf_index(pdf_dir)
    page_cache: Dict[str, List[str]] = {}

    with open(train_json, encoding="utf-8") as f:
        entries = json.load(f)

    if max_rows:
        entries = entries[:max_rows]

    logger.info(f"Processing {len(entries)} FinQA train entries …")

    counters = {
        "tier_a_pdf":   0,   # PDF found, overlap OK → gold text from PDF
        "tier_b_json":  0,   # no PDF / blocklist / low-overlap → gold text from JSON
        "missing_qa":   0,   # no question, skip entirely
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as fout, \
         open(reject_path, "w", encoding="utf-8") as frej:

        for entry_idx, entry in enumerate(tqdm(entries, desc="Converting", unit="row")):
            fn = entry.get("filename", "")
            m  = _FN_RE.match(fn.strip())
            if not m:
                counters["missing_qa"] += 1
                frej.write(json.dumps({"idx": entry_idx, "fn": fn, "reason": "bad_filename"}) + "\n")
                continue

            ticker   = m.group("ticker").upper()
            year     = m.group("year")
            html_pg  = int(m.group("page"))
            page_0   = html_pg - 1   # html is 1-indexed; confirmed against test split

            qa = entry.get("qa", {})
            question = (qa.get("question") or "").strip()
            answer   = (qa.get("answer")   or "").strip()
            entry_id = entry.get("id", fn)

            if not question:
                counters["missing_qa"] += 1
                frej.write(json.dumps({"idx": entry_idx, "fn": fn, "reason": "no_question"}) + "\n")
                continue

            pre_text  = flatten(entry.get("pre_text", ""))
            post_text = flatten(entry.get("post_text", ""))
            json_evidence = normalise(f"{pre_text} {post_text}".strip())

            # ── Try to upgrade to Tier A (PDF-verified) ────────────────────
            pdf_key = (ticker, year)
            tier_a = False
            doc_name = f"{ticker}_{year}_10K"   # fallback canonical name
            refined_page = page_0
            overlap_score = 0.0

            if pdf_key in pdf_index:
                pdf_path, pdf_doc_name = pdf_index[pdf_key]
                if pdf_doc_name not in BLOCKLIST:
                    if pdf_doc_name not in page_cache:
                        page_cache[pdf_doc_name] = extract_pdf_pages(pdf_path)
                    pages = page_cache[pdf_doc_name]

                    if pages and page_0 < len(pages):
                        anchor = normalise(f"{pre_text[-400:]} {post_text[:400]}")
                        refined_page, overlap_score = find_best_page_by_overlap(
                            anchor, pages, page_0, search_window
                        )
                        if overlap_score >= min_overlap:
                            doc_name  = pdf_doc_name
                            tier_a    = True

            # ── Assemble output row ────────────────────────────────────────
            source = "pdf" if tier_a else "json"
            row = {
                "qid": entry_id,
                "question": question,
                "answer": answer,
                "evidences": [{"page_num": refined_page, "doc_name": doc_name}],
                "evidences_updated": [{
                    "page_num":      refined_page,
                    "doc_name":      doc_name,
                    "pre_text":      pre_text,
                    "post_text":     post_text,
                    "evidence_text": json_evidence,   # always from JSON — clean
                    "_match_score":  round(overlap_score, 4),
                    "_source":       source,
                }],
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            if tier_a:
                counters["tier_a_pdf"] += 1
            else:
                counters["tier_b_json"] += 1

    total = sum(counters.values())
    kept  = counters["tier_a_pdf"] + counters["tier_b_json"]
    print(f"\n{'='*60}")
    print(f"  FinQA train → JSONL  complete")
    print(f"{'='*60}")
    print(f"  Entries processed      : {total}")
    print(f"  ✓  Tier A (PDF verify) : {counters['tier_a_pdf']}  — PDF found + overlap OK")
    print(f"  ✓  Tier B (JSON text)  : {counters['tier_b_json']}  — pre/post text from train.json")
    print(f"  ✓  Total kept          : {kept}  ({100*kept/max(total,1):.1f}%)")
    print(f"  ✗  missing QA (skipped): {counters['missing_qa']}")
    print(f"  Output                 : {out_path}")
    print(f"  Reject log             : {reject_path}")
    print(f"{'='*60}\n")
    print("  NOTE: Tier B entries use pre_text+post_text as gold evidence.")
    print("  During training, Tier B rows receive cross-doc random negatives")
    print("  (no same-doc/temporal mining without a PDF).\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Convert finqa/train.json to JSONL with 0-indexed PDF gold pages"
    )
    p.add_argument("--finqa-train", default="finqa/train.json")
    p.add_argument("--pdf-dir",     default="Final-PDF")
    p.add_argument("--output",      default="data/finqa_train_gold_pages.jsonl")
    p.add_argument("--reject-log",  default="data/finqa_train_gold_pages.rejects.jsonl")
    p.add_argument("--min-overlap", type=float, default=0.05,
                   help="Min token-overlap ratio to accept a page (default 0.05). "
                        "Financial table pages score 0.05–0.15 even on the correct page "
                        "due to pypdf mangling column structure into flat tokens.")
    p.add_argument("--search-window", type=int, default=5,
                   help="Pages around html_page-1 to search for best overlap (default 5)")
    p.add_argument("--max-rows",    type=int, default=None,
                   help="Truncate input (for quick testing)")
    args = p.parse_args()

    for path, label in [(Path(args.finqa_train), "--finqa-train"),
                        (Path(args.pdf_dir), "--pdf-dir")]:
        if not path.exists():
            logger.error(f"{label} not found: {path}")
            sys.exit(1)

    convert(
        train_json    = Path(args.finqa_train),
        pdf_dir       = Path(args.pdf_dir),
        out_path      = Path(args.output),
        reject_path   = Path(args.reject_log),
        min_overlap   = args.min_overlap,
        search_window = args.search_window,
        max_rows      = args.max_rows,
    )


if __name__ == "__main__":
    main()
