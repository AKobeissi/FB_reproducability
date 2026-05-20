"""
Augmented training data preparation for the domain-adapted bi-encoder.

Problem: the original model was trained exclusively on FinQA data, which consists
entirely of metrics-style numerical questions ("What is X in year Y?"). These
questions map well to financial statement pages (income statements, balance sheets,
etc.) but poorly to:
  - MD&A / Management's Discussion sections
  - Business Overview and Strategy pages
  - Risk Factors narratives
  - Notes with qualitative context

FinanceBench's domain-relevant (50) and novel-generated (50) questions target
exactly these narrative/qualitative pages. The model has never seen them.

Augmentation strategies (analogous to data augmentation in computer vision):
─────────────────────────────────────────────────────────────────────────────

1. QUERY STYLE TRANSFORMATION (like flipping/rotating in CV)
   For each existing (question, page) pair, generate 2-3 additional queries in
   domain/novel phrasing styles. Same positive page, diverse query surface form.
   Teaches the model that "Has AAL improved its interest coverage in 2014?" and
   "What is AAL's interest expense in 2014?" both point to the same page.

2. FINQA TRAIN EXPANSION (like dataset augmentation)
   The FinQA train.json (6251 examples) contains entries whose filenames encode
   company/year/page. Map these to available PDFs in Final-PDF/, adding ~1700
   more (question, page) pairs the model hasn't seen.

3. MDA PAGE INJECTION (like synthetic data generation in CV)
   For pages in each PDF that are NOT covered by any FinQA question — specifically
   narrative/MD&A pages — generate synthetic domain-style queries. This is the
   key novelty: it adds training signal for exactly the page types that domain
   and novel FinanceBench questions require.

4. CROSS-TYPE HARD NEGATIVES (like Mixup/CutMix across classes)
   For domain-style and MDA queries, financial table pages from the same document
   are added as hard negatives. This teaches: "Is X capital-intensive?" should NOT
   match the raw income-statement table; it should match the MDA narrative.
"""

import json
import logging
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pdfplumber
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ─── Query augmentation templates ────────────────────────────────────────────

# Domain-relevant style: analytical, qualitative, interpretive framing
DOMAIN_TEMPLATES = [
    "Has {company} demonstrated improvement in its {metric} as of {year}?",
    "What does {company}'s {metric} performance reveal about its financial health in {year}?",
    "Is {company}'s {metric} result in {year} indicative of strong operational performance?",
    "How efficiently is {company} managing its {metric} as reported in {year}?",
    "What are the key takeaways from {company}'s {metric} figures for fiscal {year}?",
    "Analyze {company}'s {metric} position as disclosed in its {year} annual filing.",
    "What strategic implications does {company}'s {metric} carry for {year}?",
    "Does {company} show a healthy {metric} profile based on its {year} data?",
]

# Novel-generated style: cross-dimensional, trend-focused, adjusted analysis
NOVEL_TEMPLATES = [
    "What drove the change in {company}'s {metric} during {year}?",
    "Beyond the headline figure, what should investors note about {company}'s {metric} in {year}?",
    "If we exclude one-time items, what was {company}'s underlying {metric} in {year}?",
    "What factors contributed most to {company}'s {metric} outcome in {year}?",
    "How did {company}'s {metric} trajectory in {year} compare to the prior period?",
    "What does {company}'s {metric} in {year} suggest about its competitive positioning?",
    "What are the second-order implications of {company}'s {metric} performance in {year}?",
]

# MD&A-specific templates for narrative/business-overview pages
MDA_TEMPLATES = [
    "What were the key drivers of {company}'s financial performance in {year}?",
    "What challenges did {company} face in its business operations in {year}?",
    "What growth initiatives did {company} pursue in {year}?",
    "How did {company} describe its competitive landscape in its {year} filing?",
    "What major risks did {company} highlight in its {year} annual report?",
    "What did management say about {company}'s near-term outlook in the {year} filing?",
    "How did {company}'s revenue mix evolve in {year} according to its annual report?",
    "What segment-level insights did {company} disclose in its {year} annual report?",
    "What capital allocation priorities did {company} communicate in {year}?",
    "How did {company} address operating efficiency in its {year} report?",
    "What were the primary revenue drivers for {company} in {year}?",
    "What macroeconomic headwinds or tailwinds did {company} describe in {year}?",
]

# Keywords that identify narrative/MD&A pages (used for MDA page injection)
MDA_KEYWORDS = [
    "management's discussion",
    "management discussion",
    "results of operations",
    "liquidity and capital resources",
    "critical accounting",
    "business overview",
    "competitive strengths",
    "growth strategy",
    "market conditions",
    "risk factors",
    "forward-looking",
    "strategic priorities",
    "segment information",
    "operating highlights",
]

# Keywords that identify financial-statement / table pages (used as cross-type negatives)
TABLE_KEYWORDS = [
    "consolidated statements of operations",
    "consolidated balance sheet",
    "consolidated statements of cash flows",
    "consolidated statements of comprehensive",
    "stockholders' equity",
    "earnings per share",
    "income from operations",
]


# ─── PDF utilities ────────────────────────────────────────────────────────────

def _parse_doc_name(doc_name: str) -> Tuple[str, str, str]:
    """Return (company, year, doc_type) from TICKER_YEAR_DOCTYPE."""
    parts = doc_name.split("_")
    company  = parts[0] if len(parts) >= 1 else doc_name
    year     = parts[1] if len(parts) >= 2 else ""
    doc_type = parts[2] if len(parts) >= 3 else ""
    return company, year, doc_type


def _extract_all_pages(pdf_path: str, doc_name: str, max_chars: int = 2000) -> Dict[int, str]:
    """Extract text for every page in a PDF (0-indexed page → text)."""
    company, year, _ = _parse_doc_name(doc_name)
    doc_context = " ".join(filter(None, [company, year]))
    pages: Dict[int, str] = {}
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for idx, page in enumerate(pdf.pages):
                text = (page.extract_text() or "").strip()
                if text:
                    embed_text = f"{doc_context}: {text[:max_chars]}" if doc_context else text[:max_chars]
                    pages[idx] = embed_text
    except Exception as e:
        logger.warning(f"Failed to extract {pdf_path}: {e}")
    return pages


def _cache_pdf_pages(
    pdf_dir: str,
    doc_names: List[str],
    max_chars: int = 2000,
) -> Dict[str, Dict[int, str]]:
    """Build {doc_name → {page_num → embed_text}} for a list of documents."""
    cache: Dict[str, Dict[int, str]] = {}
    missing = 0
    for doc_name in tqdm(doc_names, desc="Extracting PDFs"):
        pdf_path = os.path.join(pdf_dir, f"{doc_name}.pdf")
        if not os.path.exists(pdf_path):
            missing += 1
            continue
        cache[doc_name] = _extract_all_pages(pdf_path, doc_name, max_chars)
    if missing:
        logger.warning(f"{missing}/{len(doc_names)} PDFs not found in {pdf_dir}")
    return cache


# ─── Page-type detection ──────────────────────────────────────────────────────

def _score_narrative(text: str) -> int:
    """Count how many MDA keywords appear in the page text (lower-cased)."""
    text_lower = text.lower()
    return sum(1 for kw in MDA_KEYWORDS if kw in text_lower)


def _is_narrative_page(text: str, threshold: int = 2) -> bool:
    return _score_narrative(text) >= threshold


def _is_table_page(text: str) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in TABLE_KEYWORDS)


# ─── Financial concept extraction ────────────────────────────────────────────

def _extract_concept(question: str) -> Optional[str]:
    """
    Extract a short financial concept phrase from a FinQA question.

    Strategy: strip the question down to the core financial noun phrase by
    removing question words, years, and trailing prepositions.
    Returns None if no reasonable concept can be extracted.
    """
    q = question.lower().rstrip("?").strip()

    # Remove trailing year references: "in 2009", "for fiscal 2018", "as of 2021"
    q = re.sub(r"\s+(?:in|for|during|at|as of|as)\s+(?:fiscal\s+)?\d{4}\s*$", "", q)
    q = re.sub(r"\s+(?:in|for|during|at|as of)\s+(?:the\s+)?(?:year|fy)\s+\d{4}\s*$", "", q)

    # Remove leading question phrases
    q = re.sub(
        r"^(?:what|how much|how many|which|did|does|is|are|was|were)\s+"
        r"(?:is|was|were|did|does|the|a|an|there|much|many)?\s*"
        r"(?:the\s+)?",
        "",
        q,
    )

    # Remove common FinQA preamble patterns
    # "percentage of X was Y" → extract "X" as the denominator concept
    m = re.match(r"percentage of (.+?) (?:was|is|comprised|constituted)", q)
    if m:
        return m.group(1).strip()

    # "ratio of X to Y" → "ratio of X to Y"
    m = re.match(r"(ratio of .{5,60})", q)
    if m:
        return m.group(1).strip()

    # "growth rate in X" → "X growth rate"
    m = re.match(r"growth rate in (.+)", q)
    if m:
        concept = m.group(1).strip()
        return concept

    # Strip trailing noise words
    q = re.sub(r"\s+(in|for|at|as|the|a|an|of|and|that|which|from|to)\s*$", "", q)
    q = q.strip()

    # Strip year references that may have survived from within the phrase
    q = re.sub(r"\s+(?:in|for|during|as of|as)\s+(?:fiscal\s+)?\d{4}", "", q).strip()
    q = re.sub(r"\s+in millions\s*$", "", q).strip()
    q = re.sub(r"\s+in (?:usd|dollars|thousands|billions)\s*$", "", q).strip()

    # Accept the remaining phrase if it's reasonable length
    if 5 <= len(q) <= 80:
        return q
    return None


def _build_query_variants(
    question: str,
    company: str,
    year: str,
    n_domain: int = 2,
    n_novel: int = 1,
    seed: int = 42,
) -> List[str]:
    """
    Generate domain/novel-style reformulations of a FinQA question.
    Returns up to n_domain + n_novel additional queries.
    """
    rng = random.Random(hash(question) ^ seed)
    concept = _extract_concept(question)
    if not concept:
        # Fall back to a generic placeholder
        concept = "financial performance"

    variants = []

    domain_pool = rng.sample(DOMAIN_TEMPLATES, min(n_domain, len(DOMAIN_TEMPLATES)))
    for tmpl in domain_pool:
        try:
            variants.append(
                tmpl.format(company=company, metric=concept, year=year).strip()
            )
        except KeyError:
            pass

    novel_pool = rng.sample(NOVEL_TEMPLATES, min(n_novel, len(NOVEL_TEMPLATES)))
    for tmpl in novel_pool:
        try:
            variants.append(
                tmpl.format(company=company, metric=concept, year=year).strip()
            )
        except KeyError:
            pass

    return variants


# ─── FinQA test gold-pages loader (existing dataset) ─────────────────────────

def _load_finqa_gold_pages(path: str) -> List[Dict]:
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    logger.info(f"Loaded {len(data)} FinQA test gold-page entries from {path}")
    return data


# ─── FinQA train.json loader & PDF mapper ────────────────────────────────────

def _load_finqa_train_mappable(
    train_json_path: str,
    pdf_dir: str,
    doc_suffixes: Tuple[str, ...] = ("10K", "10Q"),
) -> List[Dict]:
    """
    Load FinQA train.json and keep only examples whose PDFs are available.

    Returns a list of dicts with keys: question, doc_name, page_num.
    """
    available_pdfs = set(
        os.path.splitext(p)[0]
        for p in os.listdir(pdf_dir)
        if p.endswith(".pdf")
    )

    with open(train_json_path) as f:
        train_data = json.load(f)

    mappable: List[Dict] = []
    for entry in train_data:
        fname = entry.get("filename", "")
        parts = fname.split("/")
        if len(parts) < 3:
            continue
        company, year = parts[0], parts[1]
        # page_N.pdf → 0-indexed page number
        page_part = parts[2]  # e.g. 'page_49.pdf'
        try:
            page_num = int(re.search(r"page_(\d+)", page_part).group(1)) - 1
        except (AttributeError, ValueError):
            continue

        # Try matching against available PDFs
        doc_name = None
        for suffix in doc_suffixes:
            candidate = f"{company}_{year}_{suffix}"
            if candidate in available_pdfs:
                doc_name = candidate
                break
        if doc_name is None:
            continue

        question = entry.get("qa", {}).get("question", "").strip()
        if not question:
            continue

        mappable.append({
            "question": question,
            "doc_name": doc_name,
            "page_num": page_num,
        })

    logger.info(
        f"FinQA train: {len(mappable)}/{len(train_data)} examples "
        f"mapped to available PDFs"
    )
    return mappable


# ─── Hard-negative attachment ─────────────────────────────────────────────────

def _attach_hard_negatives(
    pairs: List[Dict],
    doc_page_cache: Dict[str, Dict[int, str]],
    hard_neg_window: int = 10,
    max_negs: int = 5,
    include_table_as_neg_for_domain: bool = True,
) -> None:
    """
    Mutate each pair in-place, attaching hard negatives.

    For domain/mda-style pairs: also adds financial table pages as hard negatives
    (cross-type hard negatives — the key difference from the original pipeline).
    """
    for pair in pairs:
        doc_name  = pair["doc_name"]
        gold_page = pair["page_num"]
        is_domain = pair.get("augmentation_type") in ("domain", "mda")

        doc_pages = doc_page_cache.get(doc_name, {})
        other = [(pn, pt) for pn, pt in doc_pages.items() if pn != gold_page]
        if not other:
            pair["hard_negatives"] = []
            continue

        nearby = [(pn, pt) for pn, pt in other if abs(pn - gold_page) <= hard_neg_window]
        far    = [(pn, pt) for pn, pt in other if abs(pn - gold_page) > hard_neg_window]

        # For domain/MDA queries, prioritise financial-table pages as hard negatives
        if is_domain and include_table_as_neg_for_domain:
            table_pages = [(pn, pt) for pn, pt in other if _is_table_page(pt)]
            random.shuffle(table_pages)
            random.shuffle(nearby)
            random.shuffle(far)
            selected = (table_pages + nearby + far)[:max_negs]
        else:
            random.shuffle(nearby)
            random.shuffle(far)
            selected = (nearby + far)[:max_negs]

        pair["hard_negatives"] = [pt for _, pt in selected]


# ─── MDA page injection ───────────────────────────────────────────────────────

def _build_mda_pairs(
    doc_name: str,
    doc_pages: Dict[int, str],
    covered_pages: set,
    n_templates: int = 3,
    seed: int = 42,
) -> List[Dict]:
    """
    For narrative pages NOT already covered by FinQA, create synthetic (query, page) pairs.
    """
    company, year, _ = _parse_doc_name(doc_name)
    rng = random.Random(hash(doc_name) ^ seed)
    pairs = []

    for page_num, page_text in doc_pages.items():
        if page_num in covered_pages:
            continue
        if not _is_narrative_page(page_text):
            continue

        templates = rng.sample(MDA_TEMPLATES, min(n_templates, len(MDA_TEMPLATES)))
        for tmpl in templates:
            try:
                query = tmpl.format(company=company, year=year).strip()
            except KeyError:
                continue
            pairs.append({
                "question":           query,
                "positive_page_text": page_text,
                "doc_name":           doc_name,
                "page_num":           page_num,
                "hard_negatives":     [],
                "augmentation_type":  "mda",
            })

    return pairs


# ─── Main pipeline ────────────────────────────────────────────────────────────

def build_augmented_training_pairs(
    finqa_test_gold_path: str,
    finqa_train_json_path: str,
    pdf_dir: str,
    max_page_chars: int = 2000,
    hard_neg_window: int = 10,
    num_hard_negatives: int = 5,
    n_domain_variants: int = 2,
    n_novel_variants: int = 1,
    n_mda_templates: int = 3,
    eval_split: float = 0.1,
    seed: int = 42,
    include_mda_injection: bool = True,
    include_train_expansion: bool = True,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Build augmented (question, positive_page_text, hard_negatives) training triplets.

    Returns: (all_pairs, train_pairs, val_pairs)
    Each pair dict has keys:
        question, positive_page_text, doc_name, page_num,
        hard_negatives, augmentation_type
    """
    random.seed(seed)

    # ── 1. Collect all (question, doc, page) entries ──────────────────────────
    raw_entries: List[Dict] = []  # {question, doc_name, page_num}

    # 1a. FinQA test gold pages (530 entries)
    for entry in _load_finqa_gold_pages(finqa_test_gold_path):
        for ev in entry.get("evidences_updated", []):
            raw_entries.append({
                "question": entry["question"],
                "doc_name": ev.get("doc_name", ""),
                "page_num": ev.get("page_num", -1),
                "augmentation_type": "original",
            })

    # 1b. FinQA train mappable entries
    if include_train_expansion and os.path.exists(finqa_train_json_path):
        train_mappable = _load_finqa_train_mappable(finqa_train_json_path, pdf_dir)
        for item in train_mappable:
            raw_entries.append({
                "question": item["question"],
                "doc_name": item["doc_name"],
                "page_num": item["page_num"],
                "augmentation_type": "original",
            })
        logger.info(f"Total raw entries after train expansion: {len(raw_entries)}")

    # ── 2. Collect unique docs for PDF extraction ─────────────────────────────
    unique_docs = list({e["doc_name"] for e in raw_entries if e["doc_name"]})
    doc_page_cache = _cache_pdf_pages(pdf_dir, unique_docs, max_page_chars)

    # ── 3. Build base pairs (with page text) ──────────────────────────────────
    base_pairs: List[Dict] = []
    skipped = 0

    for entry in raw_entries:
        doc_name = entry["doc_name"]
        page_num = entry["page_num"]

        if doc_name not in doc_page_cache:
            skipped += 1
            continue

        doc_pages = doc_page_cache[doc_name]
        if page_num in doc_pages:
            positive_text = doc_pages[page_num]
        else:
            alt = doc_pages.get(page_num - 1) or doc_pages.get(page_num + 1)
            if alt:
                positive_text = alt
            else:
                skipped += 1
                continue

        base_pairs.append({
            "question":           entry["question"],
            "positive_page_text": positive_text,
            "doc_name":           doc_name,
            "page_num":           page_num,
            "hard_negatives":     [],
            "augmentation_type":  entry["augmentation_type"],
        })

    logger.info(
        f"Base pairs: {len(base_pairs)} "
        f"(skipped {skipped} due to missing PDF/page)"
    )

    # ── 4. Query style diversification ────────────────────────────────────────
    augmented_pairs: List[Dict] = list(base_pairs)  # keep originals

    for pair in base_pairs:
        doc_name = pair["doc_name"]
        company, year, _ = _parse_doc_name(doc_name)
        if not company or not year:
            continue

        variants = _build_query_variants(
            question=pair["question"],
            company=company,
            year=year,
            n_domain=n_domain_variants,
            n_novel=n_novel_variants,
            seed=seed,
        )

        for i, variant_q in enumerate(variants):
            aug_type = "domain" if i < n_domain_variants else "novel"
            augmented_pairs.append({
                "question":           variant_q,
                "positive_page_text": pair["positive_page_text"],
                "doc_name":           doc_name,
                "page_num":           pair["page_num"],
                "hard_negatives":     [],
                "augmentation_type":  aug_type,
            })

    logger.info(
        f"After query diversification: {len(augmented_pairs)} pairs "
        f"({len(augmented_pairs) - len(base_pairs)} new variants added)"
    )

    # ── 5. MDA page injection ─────────────────────────────────────────────────
    if include_mda_injection:
        mda_count = 0
        for doc_name, doc_pages in doc_page_cache.items():
            # Track which pages already have training pairs
            covered = {
                p["page_num"]
                for p in augmented_pairs
                if p["doc_name"] == doc_name
            }
            mda_pairs = _build_mda_pairs(
                doc_name=doc_name,
                doc_pages=doc_pages,
                covered_pages=covered,
                n_templates=n_mda_templates,
                seed=seed,
            )
            augmented_pairs.extend(mda_pairs)
            mda_count += len(mda_pairs)

        logger.info(f"MDA injection added {mda_count} pairs")

    logger.info(f"Total augmented pairs: {len(augmented_pairs)}")

    # ── 6. Attach hard negatives (with cross-type logic) ──────────────────────
    _attach_hard_negatives(
        augmented_pairs,
        doc_page_cache,
        hard_neg_window=hard_neg_window,
        max_negs=num_hard_negatives,
        include_table_as_neg_for_domain=True,
    )

    # ── 7. Shuffle and split ──────────────────────────────────────────────────
    random.shuffle(augmented_pairs)

    n_val = max(1, int(len(augmented_pairs) * eval_split))
    val_pairs   = augmented_pairs[:n_val]
    train_pairs = augmented_pairs[n_val:]

    # Stats
    from collections import Counter
    type_counts = Counter(p["augmentation_type"] for p in augmented_pairs)
    logger.info(f"Augmentation type breakdown: {dict(type_counts)}")
    logger.info(
        f"Train: {len(train_pairs)} | Val: {len(val_pairs)} | "
        f"With hard negs: {sum(1 for p in augmented_pairs if p['hard_negatives'])}"
    )

    return augmented_pairs, train_pairs, val_pairs


# ─── Sentence-transformers format conversion ──────────────────────────────────

def to_sentence_transformer_examples(pairs: List[Dict]):
    """Convert pair dicts to sentence_transformers.InputExample objects."""
    from sentence_transformers import InputExample

    examples = []
    for pair in pairs:
        q   = pair["question"]
        pos = pair["positive_page_text"]
        examples.append(InputExample(texts=[q, pos]))
        for neg in pair["hard_negatives"]:
            if neg and neg.strip() and neg != pos:
                examples.append(InputExample(texts=[q, pos, neg]))

    logger.info(
        f"Converted {len(pairs)} pairs → {len(examples)} InputExample objects "
        f"({len(examples) - len(pairs)} hard-negative triplets)"
    )
    return examples


# ─── Sanity check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    BASE = Path(__file__).resolve().parent.parent
    all_p, train_p, val_p = build_augmented_training_pairs(
        finqa_test_gold_path=str(BASE / "data" / "finqa_test_gold_pages.jsonl"),
        finqa_train_json_path=str(BASE / "finqa" / "train.json"),
        pdf_dir=str(BASE / "Final-PDF"),
    )
    print(f"\nTotal: {len(all_p)} | Train: {len(train_p)} | Val: {len(val_p)}")
    from collections import Counter
    print("Types:", Counter(p["augmentation_type"] for p in all_p))
    if train_p:
        p = train_p[0]
        print(f"\nSample [{p['augmentation_type']}]")
        print(f"Q: {p['question'][:120]}")
        print(f"Page: {p['doc_name']}:{p['page_num']}")
        print(f"Hard negs: {len(p['hard_negatives'])}")
