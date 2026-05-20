#!/usr/bin/env python
"""
FinanceBench Metadata-Filtering Baselines
=========================================
Tests how pre-filtering the retrieval corpus by metadata inferred from the
query text (company name and/or year) affects retrieval performance.

Five filter variants are applied BEFORE retrieval, using only the query text
(no FinanceBench oracle metadata):

  company_only       — keep only docs belonging to the company named in query
  year_exact         — keep only docs from the exact year(s) mentioned
  year_window        — keep docs within [min_year-2 , max_year+2]
  company_year_exact — company AND exact year
  company_year_window— company AND year ± 2 window

Each filter variant is applied to two retrieval pipelines:
  1. Base BGE-M3 dense retrieval   (baseline)
  2. MultiHyDE + FT-ReRanker       (best retrieval pipeline)

Company / year extraction is done entirely from query text, deriving
company aliases automatically from the collection's doc_name inventory.

Outputs → outputs/metadata_filter/
  summary.json / summary.csv
  metrics_by_variant.json
  plots/recall_comparison.pdf  (and .png)
  plots/filter_impact.pdf
  predictions/{variant_name}_retrieval.json
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "baselines"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("meta_filter")

EMBED_MODEL    = "BAAI/bge-m3"
FT_RERANKER    = "checkpoints/ft_cross_encoder"               # fine-tuned CE
BASE_RERANKER  = "BAAI/bge-reranker-v2-m3"                   # fallback

K_VALUES   = [1, 3, 5, 10, 20]
MAIN_K     = 5
CANDIDATE_K = 20
RRF_K       = 60

QUESTION_TYPES = ["metrics-generated", "domain-relevant", "novel-generated"]

# ---------------------------------------------------------------------------
# 1. Doc-name parsing  (derives metadata from corpus inventory only)
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"(\d{4})")


def parse_doc_name(doc_name: str) -> Tuple[str, Optional[int], str]:
    """
    Parse a doc_name into (company_prefix, year, doc_type).

    Examples
    --------
    3M_2018_10K              → ("3M",              2018, "10K")
    AMCOR_2023Q2_10Q         → ("AMCOR",           2023, "10Q")
    JOHNSON_JOHNSON_2022_10K → ("JOHNSON_JOHNSON",  2022, "10K")
    AMCOR_2022_8K_dated-...  → ("AMCOR",           2022, "8K")
    """
    parts = doc_name.split("_")
    year = None
    year_idx = None
    for i, p in enumerate(parts):
        m = _YEAR_RE.match(p)          # matches 2018 or 2023Q2
        if m and len(m.group(1)) == 4:
            year = int(m.group(1))
            year_idx = i
            break
    if year_idx is None:
        return doc_name, None, "UNKNOWN"
    company = "_".join(parts[:year_idx])
    doc_type = "_".join(parts[year_idx + 1:]) if year_idx + 1 < len(parts) else "UNKNOWN"
    # Normalise: strip trailing dated-... suffix from doc_type
    doc_type = doc_type.split("_dated")[0]
    return company, year, doc_type


def build_corpus_inventory(all_doc_names: List[str]) -> Dict:
    """
    Build a rich inventory of the corpus from its doc_names alone.

    Returns a dict with:
      'doc_meta'      : {doc_name → {company, year, doc_type}}
      'companies'     : {company_prefix → set(doc_names)}
      'years'         : {year → set(doc_names)}
      'alias_map'     : {lowercase_alias → company_prefix}
    """
    doc_meta: Dict[str, Dict] = {}
    companies: Dict[str, Set[str]] = defaultdict(set)
    years: Dict[int, Set[str]] = defaultdict(set)

    for dn in all_doc_names:
        company, year, doc_type = parse_doc_name(dn)
        doc_meta[dn] = {"company": company, "year": year, "doc_type": doc_type}
        companies[company].add(dn)
        if year:
            years[year].add(dn)

    # Build alias map: multiple lowercase representations of each company prefix
    alias_map: Dict[str, str] = {}
    for company in companies:
        # 1) Full lowercase (e.g. "activisionblizzard")
        alias_map[company.lower()] = company
        # 2) Space-separated by _ (e.g. "johnson johnson")
        alias_map[company.lower().replace("_", " ")] = company
        # 3) CamelCase split: ACTIVISIONBLIZZARD → ["activision", "blizzard"]
        words = re.findall(r"[A-Z][a-z]*|[A-Z]+(?=[A-Z])|[a-z]+|[0-9]+", company)
        if len(words) > 1:
            alias_map[" ".join(w.lower() for w in words)] = company

    return {
        "doc_meta":  doc_meta,
        "companies": companies,
        "years":     years,
        "alias_map": alias_map,
    }


# ---------------------------------------------------------------------------
# 2. Query-text parsers
# ---------------------------------------------------------------------------

# Matches: FY2018, FY2023, FY 2018, fiscal 2022, 2022
_QUERY_YEAR_RE = re.compile(r"\bFY\s?(\d{4})\b|\bfiscal\s(\d{4})\b|\bQ[1-4]\s+of\s+FY\s?(\d{4})\b", re.IGNORECASE)
# Also bare 4-digit years (less reliable, used as fallback)
_BARE_YEAR_RE = re.compile(r"\b(20\d{2})\b")


def extract_years_from_query(question: str) -> List[int]:
    """
    Extract fiscal years mentioned in the query.
    Prefers explicit FY/Q-notation; falls back to bare 4-digit years.
    """
    years: Set[int] = set()
    for m in _QUERY_YEAR_RE.finditer(question):
        for g in m.groups():
            if g:
                years.add(int(g))
    if not years:
        for m in _BARE_YEAR_RE.finditer(question):
            years.add(int(m.group(1)))
    return sorted(years)


def extract_company_from_query(question: str, alias_map: Dict[str, str]) -> Optional[str]:
    """
    Attempt to find the company name in the query by matching against known
    corpus aliases.  Uses longest-match to avoid false positives.

    Returns the company_prefix (as it appears in doc_names) or None.
    """
    q_lower = question.lower()
    # Sort aliases by descending length so we pick the most specific match
    best: Optional[str] = None
    best_len = 0
    for alias, company in alias_map.items():
        if alias in q_lower and len(alias) > best_len:
            best = company
            best_len = len(alias)
    return best


# ---------------------------------------------------------------------------
# 3. Filter-list builder
# ---------------------------------------------------------------------------

def build_doc_filter(
    company: Optional[str],
    years: List[int],
    year_window: int,
    inventory: Dict,
    mode: str,   # "company_only" | "year_exact" | "year_window" |
                 # "company_year_exact" | "company_year_window"
) -> Optional[List[str]]:
    """
    Build the list of allowed doc_names for a given filter mode.

    Returns None (no filter) when the mode cannot be satisfied (e.g. company
    not extracted), so retrieval falls back to full-corpus search.
    """
    all_doc_names = list(inventory["doc_meta"].keys())

    if mode == "company_only":
        if not company:
            return None
        return sorted(inventory["companies"].get(company, []))

    if mode == "year_exact":
        if not years:
            return None
        allowed: Set[str] = set()
        for y in years:
            allowed |= inventory["years"].get(y, set())
        return sorted(allowed) or None

    if mode == "year_window":
        if not years:
            return None
        lo, hi = min(years) - year_window, max(years) + year_window
        allowed = {dn for dn, meta in inventory["doc_meta"].items()
                   if meta["year"] and lo <= meta["year"] <= hi}
        return sorted(allowed) or None

    if mode == "company_year_exact":
        if not company or not years:
            return None
        allowed = {dn for dn in inventory["companies"].get(company, [])
                   if inventory["doc_meta"][dn]["year"] in years}
        return sorted(allowed) or None

    if mode == "company_year_window":
        if not company or not years:
            return None
        lo, hi = min(years) - year_window, max(years) + year_window
        allowed = {dn for dn in inventory["companies"].get(company, [])
                   if inventory["doc_meta"][dn]["year"] is not None
                   and lo <= inventory["doc_meta"][dn]["year"] <= hi}
        return sorted(allowed) or None

    return None   # unknown mode → no filter


# ---------------------------------------------------------------------------
# 4. Retrieval helpers (imported from run_baselines.py)
# ---------------------------------------------------------------------------

def _import_baselines():
    """Import key functions from run_baselines.py (same directory)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_baselines",
        str(PROJECT_ROOT / "baselines" / "run_baselines.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------------

def page_recall_at_k(sample: Dict, k: int) -> float:
    """Fraction of gold pages found in top-k retrieved chunks (matches RetrievalEvaluator)."""
    gold_pages = {
        (seg["doc_name"], str(seg["page"]).strip())
        for seg in sample.get("gold_evidence_segments", [])
        if seg.get("page") is not None
    }
    if not gold_pages:
        return 0.0
    retrieved = {
        (c["metadata"].get("doc_name", ""), str(c["metadata"].get("page", "")).strip())
        for c in sample.get("retrieved_chunks", [])[:k]
    }
    return len(retrieved & gold_pages) / len(gold_pages)


def doc_recall_at_k(sample: Dict, k: int) -> float:
    """1 if any retrieved chunk (top-k) comes from the gold document, else 0."""
    gold_docs = {seg["doc_name"] for seg in sample.get("gold_evidence_segments", [])}
    if not gold_docs:
        return 0.0
    chunks = sample.get("retrieved_chunks", [])[:k]
    for c in chunks:
        if c.get("metadata", {}).get("doc_name") in gold_docs:
            return 1.0
    return 0.0


def compute_metrics(samples: List[Dict]) -> Dict:
    metrics: Dict = {}
    for k in K_VALUES:
        pr = np.mean([page_recall_at_k(s, k) for s in samples])
        dr = np.mean([doc_recall_at_k(s, k) for s in samples])
        metrics[f"PageRec@{k}"] = float(pr)
        metrics[f"DocRec@{k}"]  = float(dr)
    return metrics


def compute_metrics_by_type(samples: List[Dict]) -> Dict[str, Dict]:
    by_type: Dict[str, List[Dict]] = defaultdict(list)
    for s in samples:
        by_type[s.get("question_type", "unknown")].append(s)
    result = {}
    for qt, subs in by_type.items():
        result[qt] = compute_metrics(subs)
        result[qt]["n"] = len(subs)
    return result


def compute_filter_stats(samples_meta: List[Dict]) -> Dict:
    """Compute how often metadata was extractable and filter set size."""
    n = len(samples_meta)
    company_found = sum(1 for m in samples_meta if m.get("company"))
    years_found   = sum(1 for m in samples_meta if m.get("years"))
    both_found    = sum(1 for m in samples_meta if m.get("company") and m.get("years"))
    filter_sizes  = [m.get("filter_size", 0) for m in samples_meta if m.get("filter_size") is not None]
    return {
        "total": n,
        "company_extraction_rate": company_found / n,
        "year_extraction_rate":    years_found / n,
        "both_extraction_rate":    both_found / n,
        "avg_filter_size": float(np.mean(filter_sizes)) if filter_sizes else 0.0,
        "min_filter_size": int(min(filter_sizes)) if filter_sizes else 0,
        "max_filter_size": int(max(filter_sizes)) if filter_sizes else 0,
    }


# ---------------------------------------------------------------------------
# 6. Plotting
# ---------------------------------------------------------------------------

def make_recall_comparison_plot(all_results: Dict[str, Dict], output_dir: str):
    """Bar chart: PageRec@5 for every variant side by side."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        logger.warning("matplotlib not available — skipping plots")
        return

    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

    # Order: no-filter baselines first, then metadata variants
    variant_order = [
        "dense_no_filter",
        "multihyde_ft_no_filter",
        "dense_company_only",
        "dense_year_exact",
        "dense_year_window",
        "dense_company_year_exact",
        "dense_company_year_window",
        "multihyde_ft_company_only",
        "multihyde_ft_year_exact",
        "multihyde_ft_year_window",
        "multihyde_ft_company_year_exact",
        "multihyde_ft_company_year_window",
    ]
    labels = {
        "dense_no_filter":                   "BGE-M3\n(no filter)",
        "multihyde_ft_no_filter":            "MultiHyDE+FT\n(no filter)",
        "dense_company_only":                "BGE-M3\nCompany",
        "dense_year_exact":                  "BGE-M3\nYear (exact)",
        "dense_year_window":                 "BGE-M3\nYear ±2",
        "dense_company_year_exact":          "BGE-M3\nCo+Year",
        "dense_company_year_window":         "BGE-M3\nCo+Year ±2",
        "multihyde_ft_company_only":         "MultiHyDE+FT\nCompany",
        "multihyde_ft_year_exact":           "MultiHyDE+FT\nYear (exact)",
        "multihyde_ft_year_window":          "MultiHyDE+FT\nYear ±2",
        "multihyde_ft_company_year_exact":   "MultiHyDE+FT\nCo+Year",
        "multihyde_ft_company_year_window":  "MultiHyDE+FT\nCo+Year ±2",
    }

    existing = [v for v in variant_order if v in all_results]
    x        = np.arange(len(existing))
    y        = [all_results[v]["metrics"].get("PageRec@5", 0) for v in existing]
    colors   = ["#2563eb" if "dense_" in v else "#dc2626" for v in existing]
    # No-filter variants get lighter shade
    colors   = [
        "#93c5fd" if v in ("dense_no_filter", "multihyde_ft_no_filter") else c
        for v, c in zip(existing, colors)
    ]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(x, y, color=colors, edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, y):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(v, v) for v in existing], fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Page Recall @ 5")
    ax.set_title("Metadata Pre-Filtering Impact on Retrieval (Page Recall @ 5)")
    ax.axvline(x=1.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    dense_patch = mpatches.Patch(color="#2563eb", label="BGE-M3 Dense")
    ft_patch    = mpatches.Patch(color="#dc2626", label="MultiHyDE + FT-ReRanker")
    ax.legend(handles=[dense_patch, ft_patch], fontsize=9)

    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = os.path.join(output_dir, "plots", f"recall_comparison.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved {path}")
    plt.close(fig)


def make_recall_at_k_plot(all_results: Dict[str, Dict], output_dir: str):
    """Line plot: PageRec@k curves for selected variants."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    key_variants = [
        "dense_no_filter",
        "dense_company_year_window",
        "multihyde_ft_no_filter",
        "multihyde_ft_company_year_window",
    ]
    style = {
        "dense_no_filter":               ("#2563eb", "--", "BGE-M3 (no filter)"),
        "dense_company_year_window":     ("#2563eb", "-",  "BGE-M3 + Co+Year±2"),
        "multihyde_ft_no_filter":        ("#dc2626", "--", "MultiHyDE+FT (no filter)"),
        "multihyde_ft_company_year_window": ("#dc2626", "-", "MultiHyDE+FT + Co+Year±2"),
    }

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for v in key_variants:
        if v not in all_results:
            continue
        m = all_results[v]["metrics"]
        ys = [m.get(f"PageRec@{k}", 0) for k in K_VALUES]
        color, ls, label = style[v]
        ax.plot(K_VALUES, ys, color=color, linestyle=ls, marker="o",
                markersize=4, linewidth=1.8, label=label)

    ax.set_xlabel("k")
    ax.set_ylabel("Page Recall @ k")
    ax.set_title("Page Recall @ k — Metadata Filter vs. No Filter")
    ax.set_xticks(K_VALUES)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = os.path.join(output_dir, "plots", f"recall_at_k.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved {path}")
    plt.close(fig)


def make_filter_impact_plot(all_results: Dict[str, Dict], output_dir: str):
    """Horizontal bar chart showing delta PageRec@5 vs no-filter baseline."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    dense_base = all_results.get("dense_no_filter", {}).get("metrics", {}).get("PageRec@5", 0)
    ft_base    = all_results.get("multihyde_ft_no_filter", {}).get("metrics", {}).get("PageRec@5", 0)

    filter_variants = [
        "company_only", "year_exact", "year_window",
        "company_year_exact", "company_year_window",
    ]
    labels = {
        "company_only":        "Company only",
        "year_exact":          "Year (exact)",
        "year_window":         "Year ± 2",
        "company_year_exact":  "Company + Year (exact)",
        "company_year_window": "Company + Year ± 2",
    }

    fig, ax = plt.subplots(figsize=(8, 4))
    y_pos = np.arange(len(filter_variants))

    dense_deltas = []
    ft_deltas    = []
    for fv in filter_variants:
        dv = f"dense_{fv}"
        tv = f"multihyde_ft_{fv}"
        dense_deltas.append(all_results.get(dv, {}).get("metrics", {}).get("PageRec@5", dense_base) - dense_base)
        ft_deltas.append(all_results.get(tv, {}).get("metrics", {}).get("PageRec@5", ft_base) - ft_base)

    height = 0.35
    ax.barh(y_pos + height / 2, dense_deltas, height,
            color=["#2563eb" if d >= 0 else "#93c5fd" for d in dense_deltas],
            label="BGE-M3 Dense")
    ax.barh(y_pos - height / 2, ft_deltas, height,
            color=["#dc2626" if d >= 0 else "#fca5a5" for d in ft_deltas],
            label="MultiHyDE + FT-ReRanker")

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([labels[fv] for fv in filter_variants])
    ax.set_xlabel("ΔPageRec@5 vs. No-Filter Baseline")
    ax.set_title("Metadata Filter Impact on Page Recall @ 5")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = os.path.join(output_dir, "plots", f"filter_impact.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",          default="data/financebench_open_source.jsonl")
    ap.add_argument("--doc-info",      default="data/financebench_document_information.jsonl")
    ap.add_argument("--pdf-dir",       default="pdfs")
    ap.add_argument("--vs-dir",        default="vector_stores/baselines")
    ap.add_argument("--hyde-cache",    default="baselines/hyde_cache.json")
    ap.add_argument("--ft-reranker",   default=FT_RERANKER)
    ap.add_argument("--output-dir",    default="outputs/metadata_filter")
    ap.add_argument("--year-window",   type=int, default=2,
                    help="±N years for year_window / company_year_window filters")
    ap.add_argument("--resume",        action="store_true",
                    help="Skip variants whose prediction file already exists")
    ap.add_argument("--dense-only",    action="store_true",
                    help="Run only BGE-M3 dense variants (skip MultiHyDE+FT)")
    ap.add_argument("--skip-multihyde-ft", action="store_true",
                    help="Skip MultiHyDE + FT-ReRanker variants")
    args = ap.parse_args()

    # Resolve paths relative to project root
    root = PROJECT_ROOT
    def abspath(p):
        return str((root / p).resolve()) if not os.path.isabs(p) else p

    data_path    = abspath(args.data)
    doc_info_path= abspath(args.doc_info)
    pdf_dir      = abspath(args.pdf_dir)
    vs_dir       = abspath(args.vs_dir)
    hyde_cache_p = abspath(args.hyde_cache)
    output_dir   = abspath(args.output_dir)

    for d in [output_dir, os.path.join(output_dir, "predictions"),
              os.path.join(output_dir, "plots")]:
        os.makedirs(d, exist_ok=True)

    # ----------------------------------------------------------------
    # Import baselines module for shared utilities
    # ----------------------------------------------------------------
    logger.info("Importing baselines module…")
    rb = _import_baselines()

    # ----------------------------------------------------------------
    # Load data
    # ----------------------------------------------------------------
    logger.info("Loading dataset…")
    samples_raw, doc_info = rb.load_data(data_path, doc_info_path)
    logger.info(f"  {len(samples_raw)} samples loaded")

    # ----------------------------------------------------------------
    # Build corpus inventory from doc_names (no oracle metadata)
    # ----------------------------------------------------------------
    all_doc_names = list({s["doc_name"] for s in samples_raw})
    inventory = build_corpus_inventory(all_doc_names)
    logger.info(f"Corpus inventory: {len(all_doc_names)} documents, "
                f"{len(inventory['companies'])} companies")

    # ----------------------------------------------------------------
    # Pre-extract metadata from every query
    # ----------------------------------------------------------------
    logger.info("Extracting company/year from queries…")
    query_meta: List[Dict] = []
    for s in samples_raw:
        company = extract_company_from_query(s["question"], inventory["alias_map"])
        years   = extract_years_from_query(s["question"])
        query_meta.append({
            "financebench_id": s["financebench_id"],
            "question":        s["question"],
            "doc_name":        s["doc_name"],   # ground-truth, NOT used for filtering
            "company":         company,
            "years":           years,
        })

    # Extraction quality check (informational only)
    n = len(query_meta)
    co_found = sum(1 for m in query_meta if m["company"])
    yr_found = sum(1 for m in query_meta if m["years"])
    both     = sum(1 for m in query_meta if m["company"] and m["years"])
    logger.info(f"  Company found: {co_found}/{n} ({100*co_found/n:.1f}%)")
    logger.info(f"  Year found:    {yr_found}/{n} ({100*yr_found/n:.1f}%)")
    logger.info(f"  Both found:    {both}/{n} ({100*both/n:.1f}%)")

    # Save extraction results for transparency
    with open(os.path.join(output_dir, "query_metadata_extraction.json"), "w") as f:
        json.dump(query_meta, f, indent=2)

    # ----------------------------------------------------------------
    # Load HyDE cache (for MultiHyDE variants)
    # ----------------------------------------------------------------
    hyde_cache: Dict[str, List[str]] = {}
    if os.path.exists(hyde_cache_p) and not (args.dense_only or args.skip_multihyde_ft):
        with open(hyde_cache_p) as f:
            hyde_cache = json.load(f)
        logger.info(f"HyDE cache loaded: {len(hyde_cache)} entries")

    # ----------------------------------------------------------------
    # Build dense index
    # ----------------------------------------------------------------
    logger.info("Building / loading dense vector index…")
    dense_collection = rb.build_dense_index(samples_raw, pdf_dir, vs_dir)

    # ----------------------------------------------------------------
    # Load models
    # ----------------------------------------------------------------
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import torch

    # Fix reranker non-determinism so filter comparisons are apples-to-apples
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f"Loading embedding model: {EMBED_MODEL}")
    embed_model = SentenceTransformer(EMBED_MODEL, device="cuda")

    ft_reranker_path = abspath(args.ft_reranker)
    if not os.path.exists(ft_reranker_path):
        ft_reranker_path = BASE_RERANKER
        logger.warning(f"FT reranker not found at {abspath(args.ft_reranker)}, "
                       f"using base: {BASE_RERANKER}")
    logger.info(f"Loading reranker: {ft_reranker_path}")
    cross_encoder = CrossEncoder(ft_reranker_path, max_length=512,
                                 device="cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------------------------------------------
    # Define filter variants
    # ----------------------------------------------------------------
    FILTER_MODES = [
        "company_only",
        "year_exact",
        "year_window",
        "company_year_exact",
        "company_year_window",
    ]
    PIPELINE_PREFIXES = ["dense"]
    if not args.dense_only and not args.skip_multihyde_ft:
        PIPELINE_PREFIXES.append("multihyde_ft")

    # Also run no-filter baselines for comparison
    ALL_VARIANTS = [f"{pipe}_no_filter" for pipe in PIPELINE_PREFIXES] + [
        f"{pipe}_{mode}"
        for pipe in PIPELINE_PREFIXES
        for mode in FILTER_MODES
    ]

    # ----------------------------------------------------------------
    # Run all variants
    # ----------------------------------------------------------------
    all_results: Dict[str, Dict] = {}

    for variant_name in tqdm(ALL_VARIANTS, desc="Filter Variants"):
        pred_path = os.path.join(output_dir, "predictions", f"{variant_name}_retrieval.json")
        if args.resume and os.path.exists(pred_path):
            with open(pred_path) as f:
                saved = json.load(f)
            all_results[variant_name] = {
                "metrics":        saved["metrics"],
                "by_question_type": saved.get("by_question_type", {}),
                "filter_stats":   saved.get("filter_stats", {}),
            }
            logger.info(f"  [{variant_name}] Loaded from cache — "
                        f"PageRec@5={saved['metrics'].get('PageRec@5', 0):.3f}")
            continue

        # Parse variant name
        pipeline = "dense" if variant_name.startswith("dense") else "multihyde_ft"
        mode = variant_name[len(pipeline) + 1:]   # strip "dense_" or "multihyde_ft_"
        use_filter = (mode != "no_filter")

        import copy
        samples = copy.deepcopy(samples_raw)
        samples_meta_out: List[Dict] = []
        t0 = time.time()

        for i, sample in enumerate(samples):
            qm = query_meta[i]

            # Build doc_filter
            doc_filter: Optional[List[str]] = None
            filter_size = len(all_doc_names)   # default = full corpus

            if use_filter:
                doc_filter = build_doc_filter(
                    company=qm["company"],
                    years=qm["years"],
                    year_window=args.year_window,
                    inventory=inventory,
                    mode=mode,
                )
                filter_size = len(doc_filter) if doc_filter else len(all_doc_names)

            samples_meta_out.append({
                "company":      qm["company"],
                "years":        qm["years"],
                "filter_size":  filter_size,
                "doc_filter":   doc_filter,
            })

            # ── Retrieval ────────────────────────────────────────────
            if pipeline == "dense":
                # Plain BGE-M3 dense
                chunks = rb.retrieve_dense(
                    sample, embed_model, dense_collection,
                    k=max(rb.K_VALUES),
                    doc_filter=doc_filter,
                )
            else:
                # MultiHyDE dense retrieval
                hyps = hyde_cache.get(sample["question"], [])[:3]
                chunks = rb.retrieve_hyde(
                    sample, embed_model, dense_collection, hyps,
                    k=CANDIDATE_K,
                    candidate_k=CANDIDATE_K,
                    doc_filter=doc_filter,
                )
                # FT-ReRanker
                if chunks:
                    chunks = rb.apply_reranker(
                        sample, chunks, cross_encoder, k=max(rb.K_VALUES)
                    )

            sample["retrieved_chunks"] = chunks

        elapsed = time.time() - t0
        logger.info(f"  [{variant_name}] {len(samples)} samples in {elapsed:.1f}s")

        # ── Evaluate ─────────────────────────────────────────────────
        metrics      = compute_metrics(samples)
        by_qtype     = compute_metrics_by_type(samples)
        filter_stats = compute_filter_stats(samples_meta_out)

        logger.info(f"  [{variant_name}] PageRec@5={metrics['PageRec@5']:.3f}  "
                    f"DocRec@5={metrics['DocRec@5']:.3f}  "
                    f"avg_filter_size={filter_stats['avg_filter_size']:.1f}")

        # Save per-variant predictions + metrics
        to_save = {
            "variant":          variant_name,
            "metrics":          metrics,
            "by_question_type": by_qtype,
            "filter_stats":     filter_stats,
            "samples":          samples,
        }
        with open(pred_path, "w") as f:
            json.dump(to_save, f)

        all_results[variant_name] = {
            "metrics":          metrics,
            "by_question_type": by_qtype,
            "filter_stats":     filter_stats,
        }

    # ----------------------------------------------------------------
    # Free GPU memory before plotting
    # ----------------------------------------------------------------
    del embed_model, cross_encoder
    import torch
    torch.cuda.empty_cache()

    # ----------------------------------------------------------------
    # Summary output
    # ----------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info(f"{'Variant':<40}  {'PageRec@1':>9} {'PageRec@5':>9} {'DocRec@5':>9}  avg_filter")
    logger.info("-" * 80)
    for vn in ALL_VARIANTS:
        if vn not in all_results:
            continue
        m  = all_results[vn]["metrics"]
        fs = all_results[vn]["filter_stats"]
        logger.info(
            f"{vn:<40}  {m.get('PageRec@1', 0):>9.3f} {m.get('PageRec@5', 0):>9.3f} "
            f"{m.get('DocRec@5', 0):>9.3f}  {fs.get('avg_filter_size', 0):>8.1f}"
        )

    # Save consolidated summary
    summary = {
        "variants": {
            vn: {
                "metrics":          all_results[vn]["metrics"],
                "by_question_type": all_results[vn].get("by_question_type", {}),
                "filter_stats":     all_results[vn].get("filter_stats", {}),
            }
            for vn in ALL_VARIANTS if vn in all_results
        }
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # CSV summary
    import csv
    csv_path = os.path.join(output_dir, "summary.csv")
    fieldnames = ["variant", "PageRec@1", "PageRec@3", "PageRec@5", "PageRec@10",
                  "PageRec@20", "DocRec@5",
                  "company_extraction_rate", "year_extraction_rate",
                  "avg_filter_size"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for vn in ALL_VARIANTS:
            if vn not in all_results:
                continue
            m  = all_results[vn]["metrics"]
            fs = all_results[vn]["filter_stats"]
            w.writerow({
                "variant":                   vn,
                "PageRec@1":                 round(m.get("PageRec@1", 0), 4),
                "PageRec@3":                 round(m.get("PageRec@3", 0), 4),
                "PageRec@5":                 round(m.get("PageRec@5", 0), 4),
                "PageRec@10":                round(m.get("PageRec@10", 0), 4),
                "PageRec@20":                round(m.get("PageRec@20", 0), 4),
                "DocRec@5":                  round(m.get("DocRec@5", 0), 4),
                "company_extraction_rate":   round(fs.get("company_extraction_rate", 0), 4),
                "year_extraction_rate":      round(fs.get("year_extraction_rate", 0), 4),
                "avg_filter_size":           round(fs.get("avg_filter_size", 0), 2),
            })
    logger.info(f"\nSummary CSV saved to {csv_path}")

    # ----------------------------------------------------------------
    # Plots
    # ----------------------------------------------------------------
    logger.info("\n>>> Generating plots")
    make_recall_comparison_plot(all_results, output_dir)
    make_recall_at_k_plot(all_results, output_dir)
    make_filter_impact_plot(all_results, output_dir)

    logger.info("\nDone! Outputs in: " + output_dir)


if __name__ == "__main__":
    main()
