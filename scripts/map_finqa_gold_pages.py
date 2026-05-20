#!/usr/bin/env python3
"""
map_finqa_gold_pages.py  —  Augment LoFIN finqa_test.jsonl with PDF gold pages
===============================================================================

Takes data/finqa_test.jsonl (LoFIN base — has qid, question, answer, evidences)
and enriches each row with a new `evidences_updated` key by:

  1. Looking up the matching entry in finqa/test.json by normalising both
     the JSONL `qid` and the JSON `id` to the same string before comparing.
     (JSON escapes \/ and / are equivalent after parsing, but we normalise
     explicitly to be safe.)

  2. Using the matched entry's pre_text / post_text to fuzzy-match against
     the extracted text of each page in the corresponding Final-PDF/{doc}.pdf.

  3. Writing the matched 0-indexed PDF page number back into evidences_updated
     on the correct row of finqa_test.jsonl.

The qid↔id match is the critical safety gate: if no match is found, the row
is written through unchanged (no evidences_updated added) and logged as a
warning so you can investigate.

Output format per row
----------------------
All original fields from finqa_test.jsonl are preserved unchanged, plus:

  "evidences_updated": [
    {
      "page_num":      44,           <- 0-indexed PDF page (FinanceBench convention)
      "doc_name":      "ZBH_2008_10K",
      "pre_text":      "...",
      "post_text":     "...",
      "evidence_text": "...",        <- pre_text + post_text, no table
      "_match_score":  0.83
    }
  ]

Rows where the PDF page cannot be confidently matched (score < threshold) are
written to the output WITHOUT evidences_updated and logged to the reject file.

Usage
-----
  python map_finqa_gold_pages.py \\
      --lofin    data/finqa_test.jsonl \\
      --finqa    finqa/test.json \\
      --pdf-dir  Final-PDF \\
      --output   data/finqa_test_gold_pages.jsonl \\
      --threshold 0.65

Dependencies
------------
  pip install pypdf rapidfuzz tqdm
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Dependency checks
# ─────────────────────────────────────────────────────────────────────────────
try:
    from pypdf import PdfReader
except ImportError:
    logger.error("pypdf not found — install with: pip install pypdf")
    sys.exit(1)

try:
    from rapidfuzz import fuzz as rfuzz
except ImportError:
    logger.error("rapidfuzz not found — install with: pip install rapidfuzz")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):  # type: ignore
        return it


# ─────────────────────────────────────────────────────────────────────────────
# 1.  ID normalisation
#     Both sides must pass through this before any comparison.
#     Handles:   "ZBH\/2008\/page_57.pdf-4"  →  "ZBH/2008/page_57.pdf-4"
#     (Python's json.loads already unescapes \/, but we strip+lower just in
#     case of any trailing whitespace or casing drift.)
# ─────────────────────────────────────────────────────────────────────────────

def norm_id(raw: str) -> str:
    """Normalise a FinQA id for safe comparison across both file formats."""
    # json.loads already converts \/ → /; this handles any remaining edge cases
    return raw.replace("\\/", "/").strip()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Load finqa/test.json and build a lookup index keyed by normalised id
# ─────────────────────────────────────────────────────────────────────────────

def load_finqa_index(finqa_path: Path) -> Dict[str, dict]:
    """
    Returns {norm_id(entry["id"]): entry} for every entry in finqa/test.json.
    Warns on duplicate ids (should not happen in practice).
    """
    logger.info(f"Loading FinQA source: {finqa_path} …")
    with open(finqa_path, encoding="utf-8") as f:
        entries = json.load(f)

    if not isinstance(entries, list):
        logger.error(f"Expected a JSON array in {finqa_path}, got {type(entries).__name__}")
        sys.exit(1)

    index: Dict[str, dict] = {}
    for entry in entries:
        raw_id = entry.get("id") or entry.get("qid") or ""
        key = norm_id(raw_id)
        if not key:
            continue
        if key in index:
            logger.warning(f"  Duplicate id in FinQA source: {key!r} — keeping first")
        else:
            index[key] = entry

    logger.info(f"  {len(index)} entries indexed from {finqa_path.name}")

    # Diagnostic: show a few normalised keys so you can eyeball the format
    sample = list(index.keys())[:3]
    logger.info(f"  Sample normalised ids: {sample}")

    return index


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Build ticker+year → (pdf_path, doc_name) index from Final-PDF/
# ─────────────────────────────────────────────────────────────────────────────

_STEM_RE = re.compile(
    r"^(?P<ticker>.+?)_(?P<year>\d{4})(?:Q\d)?_(?P<form>.+)$", re.IGNORECASE
)

# FinQA id format: "ZBH/2008/page_57.pdf-4"
_ID_RE = re.compile(
    r"^(?P<ticker>[^/]+)/(?P<year>\d{4})/page_(?P<html_page>\d+)\.pdf(?:-\d+)?$",
    re.IGNORECASE,
)


def build_pdf_index(pdf_dir: Path) -> Dict[Tuple[str, str], Tuple[Path, str]]:
    """
    Returns {(TICKER_UPPER, YEAR_STR): (pdf_path, doc_name)}.
    doc_name = PDF stem, e.g. "ZBH_2008_10K".
    """
    index: Dict[Tuple[str, str], Tuple[Path, str]] = {}
    for p in sorted(pdf_dir.glob("*.pdf")):
        m = _STEM_RE.match(p.stem)
        if not m:
            logger.debug(f"  Skipping unrecognised PDF: {p.name}")
            continue
        key = (m.group("ticker").upper(), m.group("year"))
        if key in index:
            logger.warning(
                f"  Duplicate (ticker, year) {key} — "
                f"keeping {index[key][0].name}, ignoring {p.name}"
            )
        else:
            index[key] = (p, p.stem)

    logger.info(f"PDF index: {len(index)} unique (ticker, year) entries in {pdf_dir}")
    return index


def parse_finqa_id(entry_id: str) -> Tuple[str, str, int]:
    """
    Parse normalised FinQA id → (ticker_upper, year_str, html_page).
    html_page is kept for the original `evidences` field only.
    """
    m = _ID_RE.match(entry_id.strip())
    if not m:
        raise ValueError(f"Cannot parse FinQA id: {entry_id!r}")
    return (
        m.group("ticker").upper(),
        m.group("year"),
        int(m.group("html_page")),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Text helpers
# ─────────────────────────────────────────────────────────────────────────────

_WS = re.compile(r"\s+")


def normalise_text(text: str) -> str:
    return _WS.sub(" ", text.lower()).strip()


def flatten(field) -> str:
    """FinQA pre_text/post_text are lists of strings — join them."""
    if isinstance(field, list):
        return " ".join(str(s) for s in field if s)
    return str(field) if field else ""


def build_anchor(pre_text: str, post_text: str, n_sentences: int) -> str:
    """
    Tail of pre_text + head of post_text as the fuzzy-match anchor.
    These sentences flank the table and are most likely to appear together
    on the same PDF page as the evidence.
    """
    def split_sent(text: str) -> List[str]:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    pre_sents  = split_sent(pre_text)
    post_sents = split_sent(post_text)

    pre_anchor  = " ".join(pre_sents[-n_sentences:])  if pre_sents  else pre_text[:400]
    post_anchor = " ".join(post_sents[:n_sentences])  if post_sents else post_text[:400]

    return normalise_text(f"{pre_anchor} {post_anchor}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PDF page extraction  (cached per doc_name)
# ─────────────────────────────────────────────────────────────────────────────

def extract_pdf_pages(pdf_path: Path) -> List[str]:
    """0-indexed list of normalised page texts. Empty string for blank pages."""
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
        pages.append(normalise_text(raw))
    return pages


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Fuzzy page matching
# ─────────────────────────────────────────────────────────────────────────────

def find_best_page(
    anchor: str,
    page_texts: List[str],
    threshold: float,
) -> Tuple[Optional[int], float]:
    """
    token_set_ratio handles partial inclusion (short anchor inside long page)
    and minor OCR/whitespace noise.

    threshold in [0, 1]; rapidfuzz scores in [0, 100].
    Returns (best_page_0indexed, score) or (None, score) if below threshold.
    """
    best_idx: Optional[int] = None
    best_score = 0.0

    for idx, page_text in enumerate(page_texts):
        if not page_text:
            continue
        score = rfuzz.token_set_ratio(anchor, page_text)
        if score > best_score:
            best_score = score
            best_idx = idx

    normalised = best_score / 100.0
    return (best_idx, normalised) if normalised >= threshold else (None, normalised)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Main loop
# ─────────────────────────────────────────────────────────────────────────────

def process(
    lofin_path:  Path,
    finqa_path:  Path,
    pdf_dir:     Path,
    out_path:    Path,
    reject_path: Path,
    threshold:   float,
    n_sentences: int,
    verbose:     bool,
):
    # ── Load resources ───────────────────────────────────────────────────────
    finqa_index = load_finqa_index(finqa_path)     # norm_id → finqa entry
    pdf_index   = build_pdf_index(pdf_dir)         # (ticker, year) → (path, doc_name)

    # ── Read LoFIN JSONL ─────────────────────────────────────────────────────
    lofin_rows = []
    with open(lofin_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                lofin_rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"  {lofin_path.name}:{lineno} — JSON error: {e}")
    logger.info(f"Loaded {len(lofin_rows)} rows from {lofin_path.name}")

    # Diagnostic: show sample qids from LoFIN so you can verify normalisation
    sample_qids = [norm_id(r.get("qid","")) for r in lofin_rows[:3]]
    logger.info(f"Sample normalised qids from LoFIN: {sample_qids}")

    # ── Sanity check: how many LoFIN qids have a match in finqa_index? ───────
    n_matched_ids = sum(
        1 for r in lofin_rows if norm_id(r.get("qid", "")) in finqa_index
    )
    logger.info(
        f"ID match check: {n_matched_ids}/{len(lofin_rows)} LoFIN rows found "
        f"in finqa/test.json  "
        f"({'OK' if n_matched_ids == len(lofin_rows) else 'WARNING: some rows will be skipped'})"
    )
    if n_matched_ids == 0:
        logger.error(
            "Zero qid matches between LoFIN and FinQA source! "
            "Check that both files are for the same split (test/train) and "
            "that the id format matches after normalisation."
        )
        sys.exit(1)

    # ── Page text cache ──────────────────────────────────────────────────────
    page_cache: Dict[str, List[str]] = {}

    counters = {
        "ok":              0,
        "drop_threshold":  0,
        "drop_no_pdf":     0,
        "drop_no_pages":   0,
        "skip_no_finqa":   0,   # LoFIN row with no matching entry in finqa_index
        "skip_no_text":    0,   # matched but pre+post both empty
    }
    rejects = []

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as fout:
        for row in tqdm(lofin_rows, desc="Augmenting rows", unit="q"):

            # ── Normalise qid from LoFIN ──────────────────────────────────
            raw_qid  = row.get("qid") or row.get("id") or ""
            norm_qid = norm_id(raw_qid)

            # ── Look up matching entry in finqa/test.json ─────────────────
            finqa_entry = finqa_index.get(norm_qid)

            if finqa_entry is None:
                # THIS SHOULD NOT HAPPEN — log loudly
                logger.warning(
                    f"  [NO MATCH] qid {norm_qid!r} not found in finqa/test.json — "
                    f"row written through unchanged"
                )
                counters["skip_no_finqa"] += 1
                rejects.append({"qid": raw_qid, "reason": "skip_no_finqa"})
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            # ── Extract pre_text / post_text from the matched finqa entry ─
            pre_raw  = flatten(finqa_entry.get("pre_text",  ""))
            post_raw = flatten(finqa_entry.get("post_text", ""))

            if not pre_raw and not post_raw:
                counters["skip_no_text"] += 1
                rejects.append({"qid": raw_qid, "reason": "skip_no_text"})
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            # ── Identify the PDF via (ticker, year) from the normalised id ─
            try:
                ticker, year, _html_page = parse_finqa_id(norm_qid)
            except ValueError as e:
                logger.warning(f"  Could not parse id {norm_qid!r}: {e}")
                counters["drop_no_pdf"] += 1
                rejects.append({"qid": raw_qid, "reason": "bad_id_format"})
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            pdf_key = (ticker, year)
            if pdf_key not in pdf_index:
                counters["drop_no_pdf"] += 1
                rejects.append({
                    "qid": raw_qid,
                    "ticker": ticker, "year": year,
                    "reason": "drop_no_pdf",
                })
                if verbose:
                    logger.info(f"  ✗ no PDF for ({ticker}, {year})")
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            pdf_path, doc_name = pdf_index[pdf_key]

            # ── Extract pages (cached per document) ───────────────────────
            if doc_name not in page_cache:
                page_cache[doc_name] = extract_pdf_pages(pdf_path)
            page_texts = page_cache[doc_name]

            if not page_texts:
                counters["drop_no_pages"] += 1
                rejects.append({"qid": raw_qid, "doc_name": doc_name,
                                 "reason": "drop_no_pages"})
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            # ── Fuzzy match ───────────────────────────────────────────────
            anchor    = build_anchor(pre_raw, post_raw, n_sentences)
            best_page, best_score = find_best_page(anchor, page_texts, threshold)

            if best_page is None:
                counters["drop_threshold"] += 1
                rejects.append({
                    "qid": raw_qid, "doc_name": doc_name,
                    "reason": "drop_threshold",
                    "best_score": round(best_score, 4),
                })
                if verbose:
                    logger.info(
                        f"  ✗ {norm_qid[:55]}  "
                        f"score={best_score:.3f} < {threshold}"
                    )
                # Write row through WITHOUT evidences_updated
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            # ── Augment the row with evidences_updated ────────────────────
            evidence_text = f"{pre_raw} {post_raw}".strip()

            augmented = dict(row)   # shallow copy — preserves all original fields
            augmented["evidences_updated"] = [
                {
                    "page_num":      best_page,       # 0-indexed, FinanceBench convention
                    "doc_name":      doc_name,
                    "pre_text":      pre_raw,
                    "post_text":     post_raw,
                    "evidence_text": evidence_text,
                    "_match_score":  round(best_score, 4),
                }
            ]

            fout.write(json.dumps(augmented, ensure_ascii=False) + "\n")
            counters["ok"] += 1

            if verbose:
                logger.info(
                    f"  ✓ {norm_qid[:52]}  "
                    f"pdf_p={best_page}  score={best_score:.3f}  doc={doc_name}"
                )

    # ── Write reject log ──────────────────────────────────────────────────
    with open(reject_path, "w", encoding="utf-8") as frej:
        for r in rejects:
            frej.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── Summary ───────────────────────────────────────────────────────────
    total    = len(lofin_rows)
    retained = counters["ok"]

    print(f"\n{'='*64}")
    print(f"  Augmentation complete")
    print(f"{'='*64}")
    print(f"  LoFIN rows in              : {total}")
    print(f"  ✓  evidences_updated added : {retained}  ({100*retained/total:.1f}%)")
    print(f"  Rows written through (no evidences_updated):")
    print(f"       below threshold       : {counters['drop_threshold']}  "
          f"(threshold={threshold})")
    print(f"       PDF not found         : {counters['drop_no_pdf']}")
    print(f"       no extractable pages  : {counters['drop_no_pages']}")
    print(f"       no pre/post text      : {counters['skip_no_text']}")
    print(f"       no finqa id match     : {counters['skip_no_finqa']}")
    print(f"  Output JSONL               : {out_path}")
    print(f"  Reject log                 : {reject_path}")
    print(f"{'='*64}\n")

    if counters["skip_no_finqa"] > 0:
        print(
            f"  [WARNING] {counters['skip_no_finqa']} rows had no qid match in "
            f"finqa/test.json.\n"
            f"  Check the reject log — this likely means a split mismatch "
            f"(e.g. LoFIN uses train data but you passed test.json).\n"
        )
    if counters["drop_threshold"] > retained * 0.3:
        print(
            f"  [hint] >30% dropped at threshold={threshold}. "
            f"Inspect the reject log _match_score values and try --threshold 0.55.\n"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 8.  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Augment LoFIN finqa_test.jsonl with PDF gold pages via fuzzy matching"
        )
    )
    parser.add_argument(
        "--lofin", default="data/finqa_test.jsonl",
        help="LoFIN base JSONL (default: data/finqa_test.jsonl)"
    )
    parser.add_argument(
        "--finqa", default="finqa/test.json",
        help="Raw FinQA JSON with pre_text/post_text (default: finqa/test.json)"
    )
    parser.add_argument(
        "--pdf-dir", default="Final-PDF",
        help="Directory of {doc_name}.pdf files (default: Final-PDF)"
    )
    parser.add_argument(
        "--output", default=None,
        help=(
            "Output JSONL path. "
            "Default: data/finqa_test_gold_pages.jsonl"
        )
    )
    parser.add_argument(
        "--threshold", type=float, default=0.65,
        help="Min fuzzy match score [0–1] to accept a page (default: 0.65)"
    )
    parser.add_argument(
        "--context-sentences", type=int, default=3,
        help=(
            "Sentences from tail of pre_text + head of post_text "
            "for the matching anchor (default: 3)"
        )
    )
    parser.add_argument(
        "--reject-log", default=None,
        help="Path for the reject log (default: next to output, .rejects.jsonl)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Log per-question scores and decisions"
    )
    args = parser.parse_args()

    lofin_path = Path(args.lofin)
    finqa_path = Path(args.finqa)
    pdf_dir    = Path(args.pdf_dir)

    for p, label in [(lofin_path, "--lofin"), (finqa_path, "--finqa"), (pdf_dir, "--pdf-dir")]:
        if not p.exists():
            logger.error(f"{label} path not found: {p}")
            sys.exit(1)

    out_path = (
        Path(args.output) if args.output
        else lofin_path.parent / (lofin_path.stem + "_gold_pages.jsonl")
    )
    reject_path = (
        Path(args.reject_log) if args.reject_log
        else out_path.with_name(out_path.stem + ".rejects.jsonl")
    )

    process(
        lofin_path   = lofin_path,
        finqa_path   = finqa_path,
        pdf_dir      = pdf_dir,
        out_path     = out_path,
        reject_path  = reject_path,
        threshold    = args.threshold,
        n_sentences  = args.context_sentences,
        verbose      = args.verbose,
    )


if __name__ == "__main__":
    main()