#!/usr/bin/env python3
"""
add_precision_to_metrics.py
============================
Fast script: reads prediction files, computes ONLY precision@k metrics and
fixes the Earnings doc_type ('unknown' → 'Earnings'), then patches the
existing *_metrics.json files without recomputing BLEU/ROUGE.

Run from repo root:
    python3 baselines/add_precision_to_metrics.py
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT    = Path(__file__).resolve().parent.parent
PREDICTIONS_DIR = PROJECT_ROOT / "baselines/results/predictions"
METRICS_DIR     = PROJECT_ROOT / "baselines/results/metrics"
DOC_INFO_PATH   = PROJECT_ROOT / "data/financebench_document_information.jsonl"
K_VALUES        = [1, 3, 5, 10, 20]

VARIANTS = [
    "bm25", "splade", "hybrid_25_75", "hybrid_50_50", "hybrid_75_25",
    "dense_bge_m3", "parent_child", "query_expansion", "hyde", "multi_hyde",
    "bge_reranker", "multi_hyde_reranker", "dense_bge_m3_ft_reranker",
    "multi_hyde_ft_reranker",
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_doc_info() -> dict:
    doc_info = {}
    if DOC_INFO_PATH.exists():
        with open(DOC_INFO_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                doc_info[d["doc_name"]] = d
    return doc_info


def normalize_doc_name(name: str) -> str:
    if not name:
        return ""
    name = str(name).lower().strip()
    if name.endswith(".pdf"):
        name = name[:-4]
    return name


def check_match(retrieved_text: str, gold_text: str, threshold: float = 0.7) -> bool:
    if not retrieved_text or not gold_text:
        return False
    norm_ret  = " ".join(retrieved_text.lower().split())
    norm_gold = " ".join(gold_text.lower().split())
    if norm_gold in norm_ret:
        return True
    ret_toks  = set(norm_ret.split())
    gold_toks = set(norm_gold.split())
    if not gold_toks:
        return False
    return len(ret_toks & gold_toks) / len(gold_toks) >= threshold


# ---------------------------------------------------------------------------
# Compute precision + re-derive doc_type breakdown
# ---------------------------------------------------------------------------

def compute_precision_for_samples(samples: list) -> dict:
    """
    Returns a flat dict: doc_precision@k, page_precision@k, chunk_precision@k
    for each k in K_VALUES, averaged over all samples.
    """
    prec = {
        "doc_precision":   {k: [] for k in K_VALUES},
        "page_precision":  {k: [] for k in K_VALUES},
        "chunk_precision": {k: [] for k in K_VALUES},
    }

    for sample in samples:
        retrieved    = sample.get("retrieved_chunks", [])
        gold_segs    = sample.get("gold_evidence_segments", [])

        if isinstance(gold_segs, dict):
            gold_segs = [gold_segs]
        elif not isinstance(gold_segs, list):
            gold_segs = []

        gold_docs  = set()
        gold_pages = set()
        gold_texts = []

        for seg in gold_segs:
            doc_name = seg.get("doc_name") or seg.get("document")
            if doc_name:
                gold_docs.add(normalize_doc_name(doc_name))
            if doc_name and seg.get("page") is not None:
                gold_pages.add((normalize_doc_name(doc_name), str(seg["page"]).strip()))
            text = seg.get("text") or seg.get("evidence_text")
            if text:
                gold_texts.append(text)

        if not gold_segs and not sample.get("reference_answer", ""):
            continue

        for k in K_VALUES:
            k_chunks = retrieved[:k]
            actual_k = len(k_chunks)
            if actual_k == 0:
                prec["doc_precision"][k].append(0.0)
                prec["page_precision"][k].append(0.0)
                prec["chunk_precision"][k].append(0.0)
                continue

            doc_hits   = 0
            page_hits  = 0
            chunk_hits = 0

            for chunk in k_chunks:
                meta = chunk.get("metadata", {})
                text = chunk.get("text", "")

                raw_doc = meta.get("doc_name") or meta.get("source")
                doc     = normalize_doc_name(raw_doc)
                page    = meta.get("page")
                if page is not None:
                    page = str(page).strip()

                if doc in gold_docs:
                    doc_hits += 1
                if (doc, page) in gold_pages:
                    page_hits += 1

                for gt in gold_texts:
                    if check_match(text, gt):
                        chunk_hits += 1
                        break

            # precision = relevant retrieved / k (not actual_k, so decreases with k as expected)
            prec["doc_precision"][k].append(doc_hits   / k)
            prec["page_precision"][k].append(page_hits  / k)
            prec["chunk_precision"][k].append(chunk_hits / k)

    aggregated = {}
    for metric, k_dict in prec.items():
        for k, vals in k_dict.items():
            aggregated[f"{metric}@{k}"] = float(np.mean(vals)) if vals else 0.0
    return aggregated


def recompute_doc_type_breakdown(samples: list) -> dict:
    """
    Re-aggregate by doc_type using doc_type that is already attached to samples.
    Returns {doc_type: {metric: value}} — only precision metrics (merged into existing).
    """
    groups = defaultdict(list)
    for s in samples:
        groups[s.get("doc_type", "unknown")].append(s)

    result = {}
    for dt, grp in groups.items():
        result[dt] = compute_precision_for_samples(grp)
    return result


def recompute_qt_breakdown(samples: list) -> dict:
    groups = defaultdict(list)
    for s in samples:
        groups[s.get("question_type", "unknown")].append(s)

    result = {}
    for qt, grp in groups.items():
        result[qt] = compute_precision_for_samples(grp)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading doc_info …")
    doc_info = load_doc_info()

    for variant in VARIANTS:
        pred_path    = PREDICTIONS_DIR / f"{variant}_retrieval.json"
        metrics_path = METRICS_DIR     / f"{variant}_metrics.json"

        if not pred_path.exists():
            print(f"  [SKIP] {pred_path.name} not found")
            continue
        if not metrics_path.exists():
            print(f"  [SKIP] {metrics_path.name} not found (run run_baselines.py first)")
            continue

        print(f"\nPatching: {variant}")
        data    = json.loads(pred_path.read_text())
        samples = data if isinstance(data, list) else data.get("results", [])

        # Attach doc_type from doc_info (fixes Earnings/unknown)
        for s in samples:
            doc_name   = s.get("doc_name", "")
            info       = doc_info.get(doc_name, {})
            s["doc_type"] = info.get("doc_type", "unknown")

        # Compute precision for overall
        overall_prec = compute_precision_for_samples(samples)

        # Compute precision by doc_type and question_type
        dt_prec = recompute_doc_type_breakdown(samples)
        qt_prec = recompute_qt_breakdown(samples)

        # Load existing metrics and patch them
        existing = json.loads(metrics_path.read_text())

        # Patch overall
        existing["overall"].update(overall_prec)

        # Patch by_doc_type: rename 'unknown' → 'Earnings' if present and add precision
        old_dt = existing.get("by_doc_type", {})
        # Detect if 'unknown' is the Earnings bucket (by checking our newly computed dt_prec)
        if "Earnings" in dt_prec and "unknown" in old_dt and "Earnings" not in old_dt:
            old_dt["Earnings"] = old_dt.pop("unknown")
        for dt, prec_vals in dt_prec.items():
            if dt not in old_dt:
                old_dt[dt] = {}
            old_dt[dt].update(prec_vals)
        existing["by_doc_type"] = old_dt

        # Patch by_question_type
        old_qt = existing.get("by_question_type", {})
        for qt, prec_vals in qt_prec.items():
            if qt not in old_qt:
                old_qt[qt] = {}
            old_qt[qt].update(prec_vals)
        existing["by_question_type"] = old_qt

        # Patch by_question_type_x_doc_type precision too
        qt_dt_grps = defaultdict(list)
        for s in samples:
            qt_dt_grps[f"{s.get('question_type','?')}|{s.get('doc_type','?')}"].append(s)
        old_qt_dt = existing.get("by_question_type_x_doc_type", {})
        for key, grp in qt_dt_grps.items():
            prec_vals = compute_precision_for_samples(grp)
            if key not in old_qt_dt:
                old_qt_dt[key] = {}
            old_qt_dt[key].update(prec_vals)
        existing["by_question_type_x_doc_type"] = old_qt_dt

        metrics_path.write_text(json.dumps(existing, indent=2))

        pr5  = existing["overall"].get("page_recall@5",    0)
        pp5  = existing["overall"].get("page_precision@5", 0)
        dr5  = existing["overall"].get("doc_recall@5",     0)
        dp5  = existing["overall"].get("doc_precision@5",  0)
        print(f"  PageRec@5={pr5:.3f}  PagePrec@5={pp5:.3f}  "
              f"DocRec@5={dr5:.3f}  DocPrec@5={dp5:.3f}  → patched {metrics_path.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
