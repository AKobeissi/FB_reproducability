"""
Evaluation harness for the domain-adapted financial RAG pipeline.

Ablation study over six retrieval variants — all use GLOBAL search
(no oracle document filtering):

  1. baseline_global        — BGE-M3 baseline, global search
  2. ft_global              — FT BGE-M3, global search (shows FT effect alone)
  3. ft_global_hyde         — FT BGE-M3 + Multi-HyDE
  4. ft_global_rerank       — FT BGE-M3 + cross-encoder reranking
  5. ft_global_hier         — FT BGE-M3 + hierarchical chunk retrieval + reranking
  6. ft_global_hyde_rerank  — FULL: FT + HyDE + reranking (no oracle filtering)

Metrics computed:
  • doc_recall@k    — fraction of questions where correct document was retrieved
  • page_recall@k   — fraction of questions where correct page was retrieved
  • context_bleu@k  — max BLEU-4 of retrieved text vs. gold evidence
  • context_rougeL@k— max ROUGE-L of retrieved text vs. gold evidence
  • numeric_match   — fraction of metrics-type questions answered correctly (approx)

All metrics are also broken down by question type:
  • metrics-generated  (50 questions)
  • domain-relevant    (50 questions)
  • novel-generated    (50 questions)
"""

import copy
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# FinanceBench question types
QUESTION_TYPES = ["metrics-generated", "domain-relevant", "novel-generated"]


# ---------------------------------------------------------------------------
# FinanceBench data loading
# ---------------------------------------------------------------------------

def load_financebench(path: str) -> List[Dict]:
    """
    Load financebench_open_source.jsonl.

    Normalises evidence into gold_evidence_segments with 0-indexed page numbers
    matching the PDF extraction convention.
    """
    samples = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)

            gold_segments = []
            for ev in raw.get("evidence", []):
                seg = {
                    "text": ev.get("evidence_text", ""),
                    "evidence_text": ev.get("evidence_text", ""),
                    "doc_name": ev.get("doc_name", raw.get("doc_name", "")),
                    "page": ev.get("evidence_page_num", -1),  # 0-indexed
                }
                gold_segments.append(seg)

            samples.append({
                "financebench_id": raw.get("financebench_id", ""),
                "question": raw.get("question", ""),
                "reference_answer": raw.get("answer", ""),
                "question_type": raw.get("question_type", "unknown"),
                "doc_name": raw.get("doc_name", ""),
                "gold_evidence_segments": gold_segments,
                "retrieved_chunks": [],
                "generated_answer": "",
            })

    type_counts = ", ".join(
        f"{qt}: {sum(1 for s in samples if s['question_type'] == qt)}"
        for qt in QUESTION_TYPES
    )
    logger.info(f"Loaded {len(samples)} FinanceBench samples from {path} ({type_counts})")
    return samples


# ---------------------------------------------------------------------------
# Retrieval for a single variant
# ---------------------------------------------------------------------------

def run_retrieval_for_variant(
    samples: List[Dict],
    pipeline,
    enable_hyde: bool,
    enable_rerank: bool,
    k_values: List[int],
    use_doc_filter: bool = True,
    use_hierarchical: bool = False,
    generator=None,
) -> List[Dict]:
    """
    Run retrieval (and optionally generation) for all samples.

    Passes `doc_name` to the pipeline when doc_filter is enabled, so the
    pipeline can restrict its ChromaDB search to the target document.
    """
    max_k = max(k_values)

    for sample in tqdm(samples, desc="Retrieving"):
        question = sample["question"]
        # Pass doc_name for doc-filtered retrieval
        doc_name = sample["doc_name"] if use_doc_filter else None

        try:
            chunks = pipeline.retrieve_chunks(
                question,
                doc_name=doc_name,
                enable_hyde=enable_hyde,
                enable_rerank=enable_rerank,
                k=max_k,
            )
        except Exception as e:
            logger.warning(f"Retrieval failed for '{question[:60]}…': {e}")
            chunks = []

        sample["retrieved_chunks"] = chunks

        if generator is not None:
            try:
                answer = generator.generate(question, chunks[:5])
            except Exception as e:
                logger.warning(f"Generation failed: {e}")
                answer = ""
            sample["generated_answer"] = answer

    return samples


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(samples: List[Dict], k_values: List[int]) -> Dict:
    """Use the project's shared RetrievalEvaluator."""
    fb_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if fb_root not in sys.path:
        sys.path.insert(0, fb_root)
    from src.evaluation.retrieval_evaluator import RetrievalEvaluator

    evaluator = RetrievalEvaluator()
    return evaluator.compute_metrics(samples, k_values=k_values)


def compute_metrics_by_type(
    samples: List[Dict],
    k_values: List[int],
) -> Dict[str, Dict]:
    """
    Compute retrieval metrics separately for each question_type.

    Returns {question_type: metrics_dict}.
    """
    by_type: Dict[str, List[Dict]] = defaultdict(list)
    for s in samples:
        by_type[s.get("question_type", "unknown")].append(s)

    type_metrics: Dict[str, Dict] = {}
    for qt, type_samples in by_type.items():
        if not type_samples:
            continue
        try:
            m = compute_metrics(type_samples, k_values)
            m["num_samples"] = len(type_samples)
            type_metrics[qt] = m
        except Exception as e:
            logger.warning(f"Could not compute metrics for type '{qt}': {e}")
            type_metrics[qt] = {"num_samples": len(type_samples)}

    return type_metrics


# ---------------------------------------------------------------------------
# Ablation study
# ---------------------------------------------------------------------------

def run_ablation(
    config,
    ft_embed_model,
    baseline_embed_model,
    ft_page_collection,
    baseline_page_collection,
    ft_chunk_collection=None,
    baseline_chunk_collection=None,
    hyde_generator=None,
    reranker=None,
    generator=None,
    run_generation: bool = False,
) -> Dict[str, Dict]:
    """
    Run the full ablation study across all variants.

    Returns {variant_name: metrics_dict}.
    Each metrics_dict also includes 'by_question_type': {type: metrics}.
    """
    from domain_adapted_retrieval.pipeline import FinancialRetrievalPipeline

    samples_raw = load_financebench(config.data.financebench_data_path)
    k_values = config.k_eval_values
    out_dir = config.output_dir
    os.makedirs(os.path.join(out_dir, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "metrics"), exist_ok=True)

    all_results: Dict[str, Dict] = {}

    # -----------------------------------------------------------------------
    # Variant definitions
    # (name, embed_model, page_coll, chunk_coll, doc_filter, hyde, rerank, hier)
    # All variants use doc_filter=False (global search, no oracle document access).
    # -----------------------------------------------------------------------
    variants = [
        # Baseline: vanilla BGE-M3, global search
        ("baseline_global",
         baseline_embed_model, baseline_page_collection, None,
         False, False, False, False),

        # FT BGE-M3, global search — shows FT effect alone
        ("ft_global",
         ft_embed_model, ft_page_collection, None,
         False, False, False, False),

        # FT BGE-M3 + Multi-HyDE, global
        ("ft_global_hyde",
         ft_embed_model, ft_page_collection, None,
         False, True, False, False),

        # FT BGE-M3 + cross-encoder reranking, global
        ("ft_global_rerank",
         ft_embed_model, ft_page_collection, None,
         False, False, True, False),

        # FT BGE-M3 + hierarchical chunk retrieval + reranking, global
        ("ft_global_hier",
         ft_embed_model, ft_page_collection, ft_chunk_collection,
         False, False, True, True),

        # FULL: FT + HyDE + reranking, global (no oracle filtering)
        ("ft_global_hyde_rerank",
         ft_embed_model, ft_page_collection, None,
         False, True, True, False),
    ]

    for (variant_name, emb_model, page_coll, chunk_coll,
         use_docfilter, use_hyde, use_rerank, use_hier) in variants:

        logger.info(f"\n{'='*60}")
        logger.info(f"Running variant: {variant_name}")
        logger.info(
            f"  DocFilter={use_docfilter} | HyDE={use_hyde} | "
            f"Rerank={use_rerank} | Hierarchical={use_hier}"
        )
        logger.info(f"{'='*60}")

        # Skip HyDE variants if no generator loaded
        if use_hyde and hyde_generator is None:
            logger.warning(f"Skipping {variant_name}: HyDE generator not loaded.")
            continue

        # Skip hierarchical if chunk collection not available
        if use_hier and chunk_coll is None:
            logger.warning(
                f"Skipping {variant_name}: chunk collection not built. "
                "Run with --build-chunk-index."
            )
            continue

        # Build pipeline for this variant
        pipe = FinancialRetrievalPipeline(
            config=config,
            embed_model=emb_model,
            page_collection=page_coll,
            chunk_collection=chunk_coll if use_hier else None,
            hyde_generator=hyde_generator if use_hyde else None,
            reranker=reranker if use_rerank else None,
            use_doc_filter=use_docfilter,
            use_hierarchical=use_hier,
        )

        samples = copy.deepcopy(samples_raw)

        t0 = time.time()
        samples = run_retrieval_for_variant(
            samples,
            pipe,
            enable_hyde=use_hyde,
            enable_rerank=use_rerank,
            k_values=k_values,
            use_doc_filter=use_docfilter,
            use_hierarchical=use_hier,
            generator=generator if run_generation else None,
        )
        elapsed = time.time() - t0
        logger.info(f"Retrieval took {elapsed:.1f}s for {len(samples)} questions")

        # Overall metrics
        metrics = compute_metrics(samples, k_values)
        metrics["retrieval_time_s"] = round(elapsed, 1)
        metrics["num_samples"] = len(samples)

        # Per-question-type metrics
        metrics["by_question_type"] = compute_metrics_by_type(samples, k_values)

        all_results[variant_name] = metrics

        # Log headline numbers
        pr5 = metrics.get("page_recall@5", 0)
        dr5 = metrics.get("doc_recall@5", 0)
        bleu5 = metrics.get("context_bleu@5", 0)
        rouge5 = metrics.get("context_rougeL@5", 0)
        logger.info(
            f"[{variant_name}] DocRec@5={dr5:.3f} | PageRec@5={pr5:.3f} | "
            f"BLEU@5={bleu5:.3f} | ROUGE-L@5={rouge5:.3f}"
        )

        # Log per-type breakdown
        for qt, tm in metrics.get("by_question_type", {}).items():
            logger.info(
                f"  [{qt}] n={tm.get('num_samples',0)} | "
                f"DocRec@5={tm.get('doc_recall@5', 0):.3f} | "
                f"PageRec@5={tm.get('page_recall@5', 0):.3f}"
            )

        # Save predictions
        pred_path = os.path.join(out_dir, "predictions", f"{variant_name}_predictions.json")
        with open(pred_path, "w") as f:
            json.dump(
                {"variant": variant_name, "results": _serialise_samples(samples)},
                f, indent=2
            )

        # Save per-variant metrics
        metrics_path = os.path.join(out_dir, "metrics", f"{variant_name}_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

    # Save aggregated metrics
    agg_path = os.path.join(out_dir, "metrics", "all_variants_metrics.json")
    with open(agg_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Save clean tables
    _save_metrics_tables(all_results, config, out_dir)

    return all_results


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _serialise_samples(samples: List[Dict]) -> List[Dict]:
    out = []
    for s in samples:
        out.append({
            "financebench_id": s.get("financebench_id", ""),
            "question": s.get("question", ""),
            "reference_answer": s.get("reference_answer", ""),
            "question_type": s.get("question_type", ""),
            "doc_name": s.get("doc_name", ""),
            "generated_answer": s.get("generated_answer", ""),
            "gold_evidence_segments": s.get("gold_evidence_segments", []),
            "retrieved_chunks": [
                {"text": c["text"][:500], "metadata": c.get("metadata", {})}
                for c in s.get("retrieved_chunks", [])
            ],
        })
    return out


# ---------------------------------------------------------------------------
# Table printing and saving
# ---------------------------------------------------------------------------

_VARIANT_LABELS = {
    "baseline_global":          "BGE-M3 Baseline (global)",
    "ft_global":                "FT BGE-M3 (global)",
    "ft_global_hyde":           "FT BGE-M3 + HyDE (global)",
    "ft_global_rerank":         "FT BGE-M3 + ReRank (global)",
    "ft_global_hier":           "FT BGE-M3 + Hier + ReRank (global)",
    "ft_global_hyde_rerank":    "FT BGE-M3 + HyDE + ReRank (global, Ours)",
}


def _save_metrics_tables(all_results: Dict, config, out_dir: str) -> None:
    """Save overall ablation CSV + per-question-type CSV, then pretty-print."""
    import csv

    k = config.main_k
    overall_metrics = [
        f"doc_recall@{k}", f"page_recall@{k}",
        f"context_bleu@{k}", f"context_rougeL@{k}",
        "numeric_match", "mrr",
    ]

    # ----- Overall ablation table -----
    csv_path = os.path.join(out_dir, "metrics", "ablation_table.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method"] + overall_metrics)

        # Prior baselines
        for method, prior in config.prior_results.items():
            row = [method] + [
                f"{prior.get(m, float('nan')):.3f}" for m in overall_metrics
            ]
            writer.writerow(row)

        writer.writerow(["---"] * (len(overall_metrics) + 1))

        # New variants
        for vname, metrics in all_results.items():
            label = _VARIANT_LABELS.get(vname, vname)
            row = [label] + [
                f"{metrics.get(m, float('nan')):.3f}" for m in overall_metrics
            ]
            writer.writerow(row)

    logger.info(f"Ablation table saved to: {csv_path}")

    # ----- Per-question-type table -----
    type_csv_path = os.path.join(out_dir, "metrics", "ablation_by_type.csv")
    type_metrics = [f"doc_recall@{k}", f"page_recall@{k}",
                    f"context_bleu@{k}", f"context_rougeL@{k}"]
    with open(type_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Method", "QuestionType", "N"] + type_metrics
        writer.writerow(header)
        for vname, metrics in all_results.items():
            label = _VARIANT_LABELS.get(vname, vname)
            for qt in QUESTION_TYPES:
                tm = metrics.get("by_question_type", {}).get(qt, {})
                n = tm.get("num_samples", 0)
                row = [label, qt, n] + [
                    f"{tm.get(m, float('nan')):.3f}" for m in type_metrics
                ]
                writer.writerow(row)
    logger.info(f"Per-type table saved to: {type_csv_path}")

    # ----- Pretty-print to stdout -----
    _print_ablation_table(all_results, config)
    _print_by_type_table(all_results, config)


def _print_ablation_table(all_results: Dict, config) -> None:
    k = config.main_k
    header_metrics = [f"DocRec@{k}", f"PageRec@{k}", f"BLEU@{k}", f"ROUGEL@{k}"]
    metric_keys = [
        f"doc_recall@{k}", f"page_recall@{k}",
        f"context_bleu@{k}", f"context_rougeL@{k}",
    ]

    col_w = 12
    name_w = 42
    sep = "-" * (name_w + col_w * len(header_metrics))

    print(f"\n{'Ablation Results at k=' + str(k):^{name_w + col_w * len(header_metrics)}}")
    print(sep)
    print(f"{'Method':<{name_w}}" + "".join(f"{m:>{col_w}}" for m in header_metrics))
    print(sep)

    # Prior results from config
    for method, prior in config.prior_results.items():
        vals = [prior.get(mk, float("nan")) for mk in metric_keys]
        print(f"{method[:name_w-1]:<{name_w}}" + "".join(f"{v:>{col_w}.3f}" for v in vals))

    print(sep)

    for vname, metrics in all_results.items():
        label = _VARIANT_LABELS.get(vname, vname)[:name_w - 1]
        vals = [metrics.get(mk, float("nan")) for mk in metric_keys]
        print(f"{label:<{name_w}}" + "".join(f"{v:>{col_w}.3f}" for v in vals))

    print(sep + "\n")


def _print_by_type_table(all_results: Dict, config) -> None:
    k = config.main_k
    header_metrics = [f"DocRec@{k}", f"PageRec@{k}", f"BLEU@{k}", f"ROUGEL@{k}"]
    metric_keys = [
        f"doc_recall@{k}", f"page_recall@{k}",
        f"context_bleu@{k}", f"context_rougeL@{k}",
    ]

    col_w = 12
    name_w = 42
    type_w = 20
    sep = "-" * (name_w + type_w + col_w * len(header_metrics))

    print(f"\n{'Results by Question Type (k=' + str(k) + ')':^{name_w + type_w + col_w * len(header_metrics)}}")
    print(sep)
    print(
        f"{'Method':<{name_w}}"
        f"{'QuestionType':<{type_w}}"
        + "".join(f"{m:>{col_w}}" for m in header_metrics)
    )
    print(sep)

    for vname, metrics in all_results.items():
        label = _VARIANT_LABELS.get(vname, vname)[:name_w - 1]
        by_type = metrics.get("by_question_type", {})
        for qt in QUESTION_TYPES:
            tm = by_type.get(qt, {})
            n = tm.get("num_samples", 0)
            vals = [tm.get(mk, float("nan")) for mk in metric_keys]
            qt_label = f"{qt} (n={n})"
            print(
                f"{label:<{name_w}}"
                f"{qt_label:<{type_w}}"
                + "".join(f"{v:>{col_w}.3f}" for v in vals)
            )
        print()  # blank line between variants

    print(sep + "\n")
