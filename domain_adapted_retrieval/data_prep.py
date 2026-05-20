"""
Training data preparation for the domain-adapted bi-encoder.

Source: finqa_test_gold_pages.jsonl — 530 FinQA Q&A entries, each mapped to
        a specific page in one of the Final-PDF PDFs.

Strategy for improving page-level recall:
------------------------------------------
The critical bottleneck in the existing system is page-level retrieval:
documents are found (DocRec@5 ≈ 0.93) but the wrong PAGE is returned
(PageRec@5 ≈ 0.46).  A general-purpose bi-encoder embeds all pages of a
10-K similarly because it has not seen financial Q&A paired with specific
financial-statement pages.

We address this with two complementary signals:

1. POSITIVE pairs: (question, full_page_text)
   Full page text is extracted from the PDF (not just the evidence snippet)
   so the embedding space at train time matches inference time.

2. INTRA-DOCUMENT HARD NEGATIVES: pages from the SAME filing, different page.
   Specifically, we treat pages within ±hard_neg_window of the gold page as
   "hard" negatives and random pages from the same document as "medium"
   negatives.  This forces the model to learn the difference between, e.g.,
   the income statement page and the cash-flow statement page — exactly the
   discrimination needed for page-level recall.

Batch negatives (via MultipleNegativesRankingLoss) provide easy negatives
from unrelated documents for free.
"""

import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pdfplumber
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PDF page extraction
# ---------------------------------------------------------------------------

def _parse_pdf_name(doc_name: str):
    """Parse company/year/doc_type from a PDF filename stem (TICKER_YEAR_DOCTYPE)."""
    parts = doc_name.split("_")
    company  = parts[0] if len(parts) >= 1 else doc_name
    year     = parts[1] if len(parts) >= 2 else ""
    doc_type = parts[2] if len(parts) >= 3 else ""
    return company, year, doc_type


def _extract_all_pages(pdf_path: str, doc_name: str, max_chars: int = 2000) -> Dict[int, str]:
    """
    Extract text for every page in a PDF.

    Returns a dict mapping 0-indexed page number → embed_text (truncated,
    prefixed with PDF-filename-derived company/year/doc_type context).
    This matches the format used at inference time in build_index.py.
    Pages with no extractable text are omitted.
    """
    company, year, doc_type = _parse_pdf_name(doc_name)
    doc_context = " ".join(filter(None, [company, year, doc_type]))

    pages: Dict[int, str] = {}
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                text = text.strip()
                if text:
                    truncated = text[:max_chars]
                    embed_text = f"{doc_context}: {truncated}" if doc_context else truncated
                    pages[page_idx] = embed_text
    except Exception as e:
        logger.warning(f"Failed to extract {pdf_path}: {e}")
    return pages


def _cache_pdf_pages(
    pdf_dir: str,
    doc_names: List[str],
    max_chars: int = 2000,
) -> Dict[str, Dict[int, str]]:
    """
    Load and cache page embed_texts for a list of documents.

    Returns {doc_name: {page_num: embed_text}}.
    embed_text is prefixed with PDF-filename-derived context to match
    the format used by build_index.py at inference time.
    PDFs not found in pdf_dir are skipped with a warning.
    """
    cache: Dict[str, Dict[int, str]] = {}
    missing = 0

    for doc_name in tqdm(doc_names, desc="Extracting training PDFs"):
        pdf_path = os.path.join(pdf_dir, f"{doc_name}.pdf")
        if not os.path.exists(pdf_path):
            logger.debug(f"PDF not found (skipping): {pdf_path}")
            missing += 1
            continue
        cache[doc_name] = _extract_all_pages(pdf_path, doc_name, max_chars)

    if missing:
        logger.warning(f"{missing}/{len(doc_names)} PDFs not found in {pdf_dir}")

    return cache


# ---------------------------------------------------------------------------
# Training pair construction
# ---------------------------------------------------------------------------

def load_finqa_gold_pages(path: str) -> List[Dict]:
    """Load finqa_test_gold_pages.jsonl, returning a list of dicts."""
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    logger.info(f"Loaded {len(data)} entries from {path}")
    return data


def build_training_pairs(
    config,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Build (question, positive_page_text, hard_negatives) training triplets.

    Pipeline:
      1. Load finqa_test_gold_pages.jsonl.
      2. Collect the unique (doc_name, page_num) pairs referenced.
      3. Extract full page texts from Final-PDF PDFs (cached to avoid repeated IO).
      4. For each (question, page_num) pair, harvest hard negatives:
           • Nearby pages (within ±hard_neg_window) from the SAME document.
           • Random pages from the same document for medium-difficulty negatives.
      5. Split 90/10 into train / validation.

    Returns:
        all_pairs, train_pairs, val_pairs
        Each pair is a dict with keys:
            question, positive_page_text, doc_name, page_num, hard_negatives
    """
    dc = config.data
    tc = config.training

    raw_data = load_finqa_gold_pages(dc.finqa_gold_pages_path)

    # Collect unique doc names so we only open each PDF once
    doc_names_needed: List[str] = []
    seen_docs = set()
    for entry in raw_data:
        for ev in entry.get("evidences_updated", []):
            doc_name = ev.get("doc_name", "")
            if doc_name and doc_name not in seen_docs:
                doc_names_needed.append(doc_name)
                seen_docs.add(doc_name)

    # Extract all page texts (cached in memory)
    doc_page_cache = _cache_pdf_pages(dc.finqa_pdf_dir, doc_names_needed, tc.max_page_chars)

    # Build raw pairs
    all_pairs: List[Dict] = []
    skipped = 0

    for entry in raw_data:
        question = entry["question"]

        for ev in entry.get("evidences_updated", []):
            doc_name = ev.get("doc_name", "")
            page_num = ev.get("page_num", -1)

            if doc_name not in doc_page_cache:
                skipped += 1
                continue

            doc_pages = doc_page_cache[doc_name]   # {page_num: text}

            if page_num not in doc_pages:
                # Try ±1 tolerance (PDF libraries can be off-by-one)
                alt = doc_pages.get(page_num - 1) or doc_pages.get(page_num + 1)
                if alt:
                    positive_text = alt
                else:
                    skipped += 1
                    continue
            else:
                positive_text = doc_pages[page_num]

            all_pairs.append({
                "question": question,
                "positive_page_text": positive_text,
                "doc_name": doc_name,
                "page_num": page_num,
                "hard_negatives": [],  # filled below
            })

    logger.info(f"Pairs created: {len(all_pairs)} (skipped {skipped} due to missing PDFs/pages)")

    # Attach hard negatives
    _attach_hard_negatives(all_pairs, doc_page_cache, tc)

    # Shuffle and split
    random.seed(config.seed)
    random.shuffle(all_pairs)

    n_val = max(1, int(len(all_pairs) * tc.eval_split))
    val_pairs = all_pairs[:n_val]
    train_pairs = all_pairs[n_val:]

    # Stats
    with_negs = sum(1 for p in all_pairs if p["hard_negatives"])
    avg_negs = (
        sum(len(p["hard_negatives"]) for p in all_pairs) / len(all_pairs)
        if all_pairs else 0
    )
    logger.info(
        f"Train: {len(train_pairs)} | Val: {len(val_pairs)} | "
        f"Pairs with hard negatives: {with_negs}/{len(all_pairs)} | "
        f"Avg hard negs per pair: {avg_negs:.1f}"
    )

    return all_pairs, train_pairs, val_pairs


def _attach_hard_negatives(
    pairs: List[Dict],
    doc_page_cache: Dict[str, Dict[int, str]],
    tc,
) -> None:
    """
    Mutate each pair dict in-place, adding a 'hard_negatives' list.

    Hard negatives are selected as follows (in priority order):
      1. Nearby pages (|delta_page| <= hard_neg_window) from the same document.
         These are the hardest negatives — often the same section of the filing.
      2. Random pages from the same document (medium difficulty).
    """
    window = tc.hard_neg_window
    max_negs = tc.num_hard_negatives_per_example

    for pair in pairs:
        doc_name = pair["doc_name"]
        gold_page = pair["page_num"]
        doc_pages = doc_page_cache.get(doc_name, {})

        other_pages = [(pn, pt) for pn, pt in doc_pages.items() if pn != gold_page]

        if not other_pages:
            pair["hard_negatives"] = []
            continue

        # Split into nearby (hard) and far (medium)
        nearby = [(pn, pt) for pn, pt in other_pages if abs(pn - gold_page) <= window]
        far = [(pn, pt) for pn, pt in other_pages if abs(pn - gold_page) > window]

        random.shuffle(nearby)
        random.shuffle(far)

        # Prioritize nearby pages, then fill with far pages
        selected = (nearby + far)[:max_negs]
        pair["hard_negatives"] = [pt for _, pt in selected]


# ---------------------------------------------------------------------------
# Sentence-Transformers format conversion
# ---------------------------------------------------------------------------

def to_sentence_transformer_examples(pairs: List[Dict]):
    """
    Convert pair dicts to sentence_transformers.InputExample objects.

    For each pair we produce:
      • One (anchor, positive) example — leverages in-batch negatives.
      • One (anchor, positive, hard_neg) triplet per hard negative —
        explicitly teaches the model to push hard negatives away.

    Both are processed by MultipleNegativesRankingLoss.  The triplet form
    adds the hard negative as the first entry in the negatives list for that
    anchor, which increases training signal beyond in-batch negatives alone.
    """
    from sentence_transformers import InputExample

    examples = []
    for pair in pairs:
        q = pair["question"]
        pos = pair["positive_page_text"]

        # Basic pair (always added)
        examples.append(InputExample(texts=[q, pos]))

        # Hard negative triplets
        for neg in pair["hard_negatives"]:
            if neg and neg.strip() and neg != pos:
                examples.append(InputExample(texts=[q, pos, neg]))

    logger.info(
        f"Converted {len(pairs)} pairs → {len(examples)} InputExample objects "
        f"({len(examples) - len(pairs)} hard-negative triplets added)"
    )
    return examples


# ---------------------------------------------------------------------------
# Quick sanity check (run as __main__)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from domain_adapted_retrieval.config import ExperimentConfig

    cfg = ExperimentConfig()
    all_pairs, train_pairs, val_pairs = build_training_pairs(cfg)

    print(f"\n=== Sample training pair ===")
    if train_pairs:
        p = train_pairs[0]
        print(f"Question: {p['question'][:120]}")
        print(f"Positive (first 200 chars): {p['positive_page_text'][:200]}")
        print(f"Hard negatives: {len(p['hard_negatives'])}")
        if p["hard_negatives"]:
            print(f"Hard neg 0 (first 150 chars): {p['hard_negatives'][0][:150]}")
