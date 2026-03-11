#!/usr/bin/env python3
"""
chunk_property_analysis.py
==========================
Analyse the properties of chunks produced by each strategy and correlate
them with downstream retrieval quality (page_recall@k, chunk_recall@k)
and generation quality (context BLEU / ROUGE between retrieved chunk
and gold evidence).

Produces:
  1. Per-chunk property CSV (one row per chunk across all strategies)
  2. Per-strategy aggregate property CSV
  3. Property–quality correlation CSV
  4. Visualisation-ready JSON for plotting

Usage
-----
    python chunk_property_analysis.py \
        --chunk-dir ./chunking_results/chunk_data \
        --results-dir ./chunking_results \
        --financebench-json /path/to/financebench.json \
        --output-dir ./chunking_results/analysis

Dependencies: numpy, pandas, rouge_score, nltk, sentence-transformers
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------

def _import_pandas():
    import pandas as pd
    return pd

def _import_rouge():
    from rouge_score import rouge_scorer
    return rouge_scorer

def _import_nltk_bleu():
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    return sentence_bleu, SmoothingFunction


# ===================================================================
# 1. Load chunk data
# ===================================================================

def load_chunk_files(chunk_dir: str) -> Dict[str, Dict[str, List[dict]]]:
    """
    Load all chunks_*.json files from chunk_dir.
    Returns {strategy_name: {doc_id: [chunk_dict, ...]}}.
    """
    data = {}
    for fp in sorted(Path(chunk_dir).glob("chunks_*.json")):
        strategy_name = fp.stem.replace("chunks_", "")
        with open(fp) as f:
            doc_chunks = json.load(f)
        data[strategy_name] = doc_chunks
        total = sum(len(v) for v in doc_chunks.values())
        logger.info(f"  Loaded {strategy_name}: {len(doc_chunks)} docs, {total} chunks")
    return data


# ===================================================================
# 2. Load FinanceBench ground truth
# ===================================================================

def load_financebench(fb_path: str) -> List[Dict[str, Any]]:
    """Load FinanceBench samples with evidence info."""
    with open(fb_path) as f:
        raw = [json.loads(line) for line in f if line.strip()]

    # Handle both list and dict formats
    if isinstance(raw, dict):
        samples = list(raw.values()) if not isinstance(list(raw.values())[0], str) else [raw]
    else:
        samples = raw

    parsed = []
    for s in samples:
        if not isinstance(s, dict):
            continue
        parsed.append({
            "question": s.get("question", ""),
            "answer": s.get("answer", ""),
            "doc_id": s.get("doc_name", s.get("doc_id", s.get("ticker", ""))),
            "evidence_text": s.get("evidence", s.get("evidence_text", "")),
            "evidence_page": s.get("evidence_page_num",
                                   s.get("page_num",
                                          s.get("evidence_page", None))),
        })
    return parsed


# ===================================================================
# 3. Load scored experiment results
# ===================================================================

def load_scored_results(results_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Find all *_scored.json files under results_dir and load them.
    Returns {strategy_name: [result_dict_per_sample, ...]}.
    """
    results = {}
    for fp in Path(results_dir).rglob("*_scored.json"):
        # Try to infer strategy from directory name
        parts = fp.parts
        strategy_name = None
        for part in parts:
            for s in ["naive", "recursive", "semantic", "adaptive",
                       "parent_child", "table_aware", "late",
                       "contextual", "metadata"]:
                if s in part:
                    strategy_name = part
                    break
            if strategy_name:
                break
        if not strategy_name:
            strategy_name = fp.stem.replace("_scored", "")

        try:
            with open(fp) as f:
                data = json.load(f)
            if isinstance(data, list):
                results[strategy_name] = data
            elif isinstance(data, dict):
                results[strategy_name] = data.get("results", data.get("samples", []))
        except Exception as e:
            logger.warning(f"Failed to load {fp}: {e}")

    return results


# ===================================================================
# 4. Compute per-chunk properties
# ===================================================================

def compute_chunk_properties(
    chunk_dict: dict,
    embedding_model=None,
) -> Dict[str, Any]:
    """
    Compute structural, lexical, and financial properties of a single chunk.
    """
    text = chunk_dict.get("raw_text", chunk_dict.get("text", ""))
    token_count = chunk_dict.get("raw_token_count", chunk_dict.get("token_count", 0))
    page_nums = chunk_dict.get("page_nums", [])
    metadata = chunk_dict.get("metadata", {})

    words = text.split()
    n_words = len(words)
    sentences = _simple_sent_split(text)
    n_sentences = len(sentences)

    props: Dict[str, Any] = {
        # --- Structural ---
        "token_count": token_count,
        "word_count": n_words,
        "sentence_count": n_sentences,
        "page_span": len(set(page_nums)),          # how many pages this chunk spans
        "char_count": len(text),
        "avg_sentence_len_words": round(n_words / max(n_sentences, 1), 2),

        # --- Lexical ---
        "type_token_ratio": _type_token_ratio(words),
        "avg_word_length": round(sum(len(w) for w in words) / max(n_words, 1), 2),

        # --- Financial / numeric density ---
        "numeric_density": _numeric_density(words),
        "dollar_amount_count": len(re.findall(r"\$[\d,]+(?:\.\d+)?", text)),
        "percentage_count": len(re.findall(r"\d+(?:\.\d+)?%", text)),
        "entity_count": metadata.get("entity_count",
                                      len(re.findall(r"\$[\d,]+(?:\.\d+)?|\d+(?:\.\d+)?%", text))),

        # --- Table indicators ---
        "has_table": bool(metadata.get("has_table", _has_table_heuristic(text))),
        "table_row_density": _table_row_density(text),

        # --- Boundary quality ---
        "starts_mid_sentence": _starts_mid_sentence(text),
        "ends_mid_sentence": _ends_mid_sentence(text),
        "boundary_quality": 1.0 - 0.5 * _starts_mid_sentence(text) - 0.5 * _ends_mid_sentence(text),

        # --- Metadata from strategy ---
        "section_header": metadata.get("section_header", ""),
        "position_ratio": metadata.get("position_ratio", -1),
        "local_density": metadata.get("local_density", -1),
        "target_size": metadata.get("target_size", -1),
    }

    return props


def _simple_sent_split(text: str) -> List[str]:
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]


def _type_token_ratio(words: List[str]) -> float:
    if not words:
        return 0.0
    unique = set(w.lower() for w in words)
    return round(len(unique) / len(words), 4)


def _numeric_density(words: List[str]) -> float:
    if not words:
        return 0.0
    count = sum(1 for w in words if re.search(r"\d", w))
    return round(count / len(words), 4)


def _has_table_heuristic(text: str) -> bool:
    lines = text.split("\n")
    multi_space_lines = sum(1 for l in lines if re.search(r"\S\s{2,}\S", l))
    return multi_space_lines >= 3


def _table_row_density(text: str) -> float:
    """Fraction of lines that look like table rows."""
    lines = text.split("\n")
    if not lines:
        return 0.0
    table_lines = sum(1 for l in lines
                      if re.search(r"\S\s{2,}\S", l) or re.search(r"^\s*[\d$%,.()\-]{10,}", l))
    return round(table_lines / len(lines), 4)


def _starts_mid_sentence(text: str) -> float:
    """1.0 if chunk starts mid-sentence (lowercase letter), 0.0 otherwise."""
    stripped = text.lstrip()
    if not stripped:
        return 0.0
    return 1.0 if stripped[0].islower() else 0.0


def _ends_mid_sentence(text: str) -> float:
    """1.0 if chunk ends without sentence-terminal punctuation."""
    stripped = text.rstrip()
    if not stripped:
        return 0.0
    return 0.0 if stripped[-1] in ".!?\"'" else 1.0


# ===================================================================
# 5. Compute chunk–evidence overlap (BLEU / ROUGE)
# ===================================================================

def compute_context_overlap(
    chunk_text: str,
    evidence_text: str,
) -> Dict[str, float]:
    """
    Compute BLEU and ROUGE-L between a retrieved chunk and the gold
    evidence text.  This measures how well the chunk captures the evidence.
    """
    result = {"ctx_bleu": 0.0, "ctx_rouge_1": 0.0, "ctx_rouge_l": 0.0}
    if not chunk_text.strip() or not evidence_text.strip():
        return result

    # BLEU
    try:
        sentence_bleu, SmoothingFunction = _import_nltk_bleu()
        ref_tokens = evidence_text.lower().split()
        hyp_tokens = chunk_text.lower().split()
        if ref_tokens and hyp_tokens:
            result["ctx_bleu"] = round(sentence_bleu(
                [ref_tokens], hyp_tokens,
                smoothing_function=SmoothingFunction().method1,
            ), 4)
    except Exception:
        pass

    # ROUGE
    try:
        rouge_scorer = _import_rouge()
        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        scores = scorer.score(evidence_text, chunk_text)
        result["ctx_rouge_1"] = round(scores["rouge1"].fmeasure, 4)
        result["ctx_rouge_l"] = round(scores["rougeL"].fmeasure, 4)
    except Exception:
        pass

    return result


def compute_evidence_coverage(
    chunks: List[dict],
    evidence_text: str,
    evidence_page: Optional[int],
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Given all chunks of a document, find the best chunk(s) that cover
    the evidence, simulating an oracle retriever.
    """
    if not evidence_text or not chunks:
        return {"best_ctx_bleu": 0, "best_ctx_rouge_l": 0, "evidence_page_covered": False}

    best_bleu = 0.0
    best_rouge = 0.0
    page_covered = False

    for c in chunks:
        text = c.get("raw_text", c.get("text", ""))
        overlap = compute_context_overlap(text, evidence_text)
        best_bleu = max(best_bleu, overlap["ctx_bleu"])
        best_rouge = max(best_rouge, overlap["ctx_rouge_l"])

        if evidence_page and evidence_page in c.get("page_nums", []):
            page_covered = True

    return {
        "best_ctx_bleu": round(best_bleu, 4),
        "best_ctx_rouge_l": round(best_rouge, 4),
        "evidence_page_covered": page_covered,
    }


# ===================================================================
# 6. Compute intra-chunk coherence (requires embeddings)
# ===================================================================

_COHERENCE_MODEL = None


def compute_intra_chunk_coherence(text: str, model_name: str = "BAAI/bge-m3") -> float:
    """
    Average pairwise cosine similarity between sentence embeddings within
    a chunk.  High = topically coherent.
    """
    global _COHERENCE_MODEL
    sentences = _simple_sent_split(text)
    if len(sentences) < 2:
        return 1.0

    try:
        if _COHERENCE_MODEL is None:
            from sentence_transformers import SentenceTransformer
            _COHERENCE_MODEL = SentenceTransformer(model_name)
        embs = _COHERENCE_MODEL.encode(sentences, normalize_embeddings=True, show_progress_bar=False)
        # Pairwise cosine (embeddings are normalised so dot product = cosine)
        sim_matrix = embs @ embs.T
        n = len(sentences)
        # Mean of upper triangle (excluding diagonal)
        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += sim_matrix[i, j]
                count += 1
        return round(total / max(count, 1), 4)
    except Exception:
        return -1.0


# ===================================================================
# 7. Full analysis pipeline
# ===================================================================

def run_analysis(
    chunk_data: Dict[str, Dict[str, List[dict]]],
    fb_samples: List[Dict[str, Any]],
    scored_results: Dict[str, List[Dict[str, Any]]],
    output_dir: str,
    compute_coherence: bool = False,
    embedding_model_name: str = "BAAI/bge-m3",
):
    """
    Main analysis loop.  Produces per-chunk properties, aggregates,
    and correlations.
    """
    pd = _import_pandas()
    os.makedirs(output_dir, exist_ok=True)

    # Build evidence lookup: doc_id → [sample, ...]
    evidence_by_doc: Dict[str, List[dict]] = defaultdict(list)
    for s in fb_samples:
        doc_id = s.get("doc_id", "")
        if doc_id:
            evidence_by_doc[doc_id].append(s)

    # ---------------------------------------------------------------
    # Per-chunk property rows
    # ---------------------------------------------------------------
    all_chunk_rows: List[Dict[str, Any]] = []
    strategy_agg: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for strategy_name, doc_chunks in chunk_data.items():
        logger.info(f"  Analysing strategy: {strategy_name}")
        for doc_id, chunks in doc_chunks.items():
            # Filter out parent chunks (only index children for parent_child)
            indexable = [c for c in chunks if c.get("strategy") != "parent_child_parent"]

            for c in indexable:
                props = compute_chunk_properties(c)
                props["strategy"] = strategy_name
                props["doc_id"] = doc_id
                props["chunk_index"] = c.get("chunk_index", -1)

                # Compute coherence if requested (expensive)
                if compute_coherence:
                    text = c.get("raw_text", c.get("text", ""))
                    props["intra_chunk_coherence"] = compute_intra_chunk_coherence(
                        text, embedding_model_name
                    )

                # Evidence overlap for this doc
                if doc_id in evidence_by_doc:
                    for sample in evidence_by_doc[doc_id]:
                        evidence_text = sample.get("evidence_text", "")
                        if evidence_text:
                            text = c.get("raw_text", c.get("text", ""))
                            overlap = compute_context_overlap(text, evidence_text)
                            props[f"ctx_bleu_q{fb_samples.index(sample)}"] = overlap["ctx_bleu"]
                            props[f"ctx_rouge_l_q{fb_samples.index(sample)}"] = overlap["ctx_rouge_l"]

                all_chunk_rows.append(props)

                # Aggregate
                for key in ["token_count", "word_count", "sentence_count", "page_span",
                            "type_token_ratio", "numeric_density", "dollar_amount_count",
                            "percentage_count", "has_table", "table_row_density",
                            "boundary_quality", "starts_mid_sentence", "ends_mid_sentence"]:
                    if key in props:
                        strategy_agg[strategy_name][key].append(props[key])

    # ---------------------------------------------------------------
    # Save per-chunk properties
    # ---------------------------------------------------------------
    chunk_df = pd.DataFrame(all_chunk_rows)
    chunk_csv = os.path.join(output_dir, "per_chunk_properties.csv")
    chunk_df.to_csv(chunk_csv, index=False)
    logger.info(f"  Per-chunk properties: {chunk_csv} ({len(chunk_df)} rows)")

    # ---------------------------------------------------------------
    # Per-strategy aggregates
    # ---------------------------------------------------------------
    agg_rows = []
    for strategy_name, metrics in strategy_agg.items():
        row = {"strategy": strategy_name}
        for key, values in metrics.items():
            arr = np.array(values, dtype=float)
            row[f"{key}_mean"] = round(float(np.mean(arr)), 4)
            row[f"{key}_std"] = round(float(np.std(arr)), 4)
            row[f"{key}_median"] = round(float(np.median(arr)), 4)
        agg_rows.append(row)

    agg_df = pd.DataFrame(agg_rows)
    agg_csv = os.path.join(output_dir, "strategy_aggregates.csv")
    agg_df.to_csv(agg_csv, index=False)
    logger.info(f"  Strategy aggregates: {agg_csv}")

    # ---------------------------------------------------------------
    # Evidence coverage per strategy
    # ---------------------------------------------------------------
    coverage_rows = []
    for strategy_name, doc_chunks in chunk_data.items():
        for sample in fb_samples:
            doc_id = sample.get("doc_id", "")
            if doc_id not in doc_chunks:
                continue
            chunks = [c for c in doc_chunks[doc_id]
                      if c.get("strategy") != "parent_child_parent"]
            evidence_text = sample.get("evidence_text", "")
            evidence_page = sample.get("evidence_page")

            cov = compute_evidence_coverage(chunks, evidence_text, evidence_page)
            coverage_rows.append({
                "strategy": strategy_name,
                "doc_id": doc_id,
                "question": sample.get("question", "")[:100],
                **cov,
            })

    cov_df = pd.DataFrame(coverage_rows)
    cov_csv = os.path.join(output_dir, "evidence_coverage.csv")
    cov_df.to_csv(cov_csv, index=False)
    logger.info(f"  Evidence coverage: {cov_csv}")

    # Per-strategy mean coverage
    if not cov_df.empty:
        cov_summary = cov_df.groupby("strategy").agg({
            "best_ctx_bleu": ["mean", "std"],
            "best_ctx_rouge_l": ["mean", "std"],
            "evidence_page_covered": "mean",
        }).round(4)
        cov_summary_csv = os.path.join(output_dir, "evidence_coverage_summary.csv")
        cov_summary.to_csv(cov_summary_csv)
        logger.info(f"  Evidence coverage summary: {cov_summary_csv}")

    # ---------------------------------------------------------------
    # Merge with retrieval metrics (from scored results)
    # ---------------------------------------------------------------
    retrieval_rows = []
    for strategy_name, results in scored_results.items():
        for r in results:
            if not isinstance(r, dict):
                continue
            retrieval_rows.append({
                "strategy": strategy_name,
                "doc_id": r.get("doc_id", r.get("doc_name", "")),
                "question": r.get("question", "")[:100],
                "page_recall": r.get("page_recall", r.get("page_recall@k", None)),
                "chunk_recall": r.get("chunk_recall", r.get("chunk_recall@k", None)),
                "doc_recall": r.get("doc_recall", r.get("doc_recall@k", None)),
                "bleu": r.get("bleu", None),
                "rouge_l": r.get("rouge_l", r.get("rougeL", None)),
                "accuracy": r.get("accuracy", r.get("correct", None)),
            })

    if retrieval_rows:
        ret_df = pd.DataFrame(retrieval_rows)
        ret_csv = os.path.join(output_dir, "retrieval_metrics_by_strategy.csv")
        ret_df.to_csv(ret_csv, index=False)
        logger.info(f"  Retrieval metrics: {ret_csv}")

        # Strategy-level aggregates
        numeric_cols = ["page_recall", "chunk_recall", "doc_recall", "bleu", "rouge_l", "accuracy"]
        ret_agg = ret_df.groupby("strategy")[numeric_cols].agg(["mean", "std"]).round(4)
        ret_agg_csv = os.path.join(output_dir, "retrieval_metrics_summary.csv")
        ret_agg.to_csv(ret_agg_csv)
        logger.info(f"  Retrieval summary: {ret_agg_csv}")

    # ---------------------------------------------------------------
    # Property → quality correlation
    # ---------------------------------------------------------------
    if retrieval_rows and not cov_df.empty:
        _compute_correlations(agg_rows, retrieval_rows, cov_df, output_dir)

    logger.info(f"\n  Analysis complete. All outputs in: {output_dir}")


def _compute_correlations(
    agg_rows: List[dict],
    retrieval_rows: List[dict],
    cov_df,
    output_dir: str,
):
    """Compute Spearman correlations between aggregate chunk properties
    and aggregate retrieval metrics per strategy."""
    pd = _import_pandas()
    from scipy import stats

    # Build strategy-level property table
    prop_df = pd.DataFrame(agg_rows).set_index("strategy")

    # Build strategy-level retrieval table
    ret_df = pd.DataFrame(retrieval_rows)
    ret_agg = ret_df.groupby("strategy").agg({
        "page_recall": "mean",
        "chunk_recall": "mean",
        "doc_recall": "mean",
    }).rename(columns={
        "page_recall": "page_recall_mean",
        "chunk_recall": "chunk_recall_mean",
        "doc_recall": "doc_recall_mean",
    })

    # Build strategy-level coverage table
    cov_agg = cov_df.groupby("strategy").agg({
        "best_ctx_bleu": "mean",
        "best_ctx_rouge_l": "mean",
    }).rename(columns={
        "best_ctx_bleu": "ctx_bleu_mean",
        "best_ctx_rouge_l": "ctx_rouge_l_mean",
    })

    # Merge
    merged = prop_df.join(ret_agg, how="inner").join(cov_agg, how="inner")
    if merged.empty or len(merged) < 3:
        logger.warning("  Not enough strategies with both properties and metrics for correlation")
        return

    # Compute Spearman for each (property, metric) pair
    property_cols = [c for c in prop_df.columns if c.endswith("_mean")]
    metric_cols = [c for c in merged.columns if c in [
        "page_recall_mean", "chunk_recall_mean", "doc_recall_mean",
        "ctx_bleu_mean", "ctx_rouge_l_mean",
    ]]

    corr_rows = []
    for pc in property_cols:
        for mc in metric_cols:
            vals_p = merged[pc].dropna()
            vals_m = merged[mc].dropna()
            common = vals_p.index.intersection(vals_m.index)
            if len(common) < 3:
                continue
            rho, pval = stats.spearmanr(merged.loc[common, pc], merged.loc[common, mc])
            corr_rows.append({
                "property": pc,
                "metric": mc,
                "spearman_rho": round(float(rho), 4),
                "p_value": round(float(pval), 4),
                "n_strategies": len(common),
            })

    if corr_rows:
        corr_df = pd.DataFrame(corr_rows)
        corr_csv = os.path.join(output_dir, "property_quality_correlations.csv")
        corr_df.to_csv(corr_csv, index=False)
        logger.info(f"  Property–quality correlations: {corr_csv}")

        # Print top correlations
        corr_df_sorted = corr_df.reindex(
            corr_df["spearman_rho"].abs().sort_values(ascending=False).index
        )
        logger.info("\n  Top property–quality correlations:")
        for _, row in corr_df_sorted.head(15).iterrows():
            sig = "***" if row["p_value"] < 0.01 else "**" if row["p_value"] < 0.05 else "*" if row["p_value"] < 0.1 else ""
            logger.info(f"    {row['property']:35s} ↔ {row['metric']:25s}  "
                         f"ρ={row['spearman_rho']:+.3f}  p={row['p_value']:.3f} {sig}")


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyse chunk properties and correlate with retrieval quality."
    )
    parser.add_argument("--chunk-dir", required=True,
                        help="Directory containing chunks_*.json files")
    parser.add_argument("--results-dir", default=None,
                        help="Root dir containing *_scored.json files from RAG experiments")
    parser.add_argument("--financebench-json", default=None,
                        help="Path to FinanceBench dataset JSON")
    parser.add_argument("--output-dir", default="./analysis_output")
    parser.add_argument("--compute-coherence", action="store_true",
                        help="Compute intra-chunk coherence (slow, requires GPU)")
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")

    args = parser.parse_args()

    # Load data
    logger.info("Loading chunk data ...")
    chunk_data = load_chunk_files(args.chunk_dir)
    if not chunk_data:
        logger.error(f"No chunk files found in {args.chunk_dir}")
        sys.exit(1)

    fb_samples = []
    if args.financebench_json:
        logger.info("Loading FinanceBench samples ...")
        fb_samples = load_financebench(args.financebench_json)
        logger.info(f"  {len(fb_samples)} samples loaded")

    scored_results = {}
    if args.results_dir:
        logger.info("Loading scored results ...")
        scored_results = load_scored_results(args.results_dir)
        logger.info(f"  {len(scored_results)} strategy result sets loaded")

    # Run analysis
    run_analysis(
        chunk_data=chunk_data,
        fb_samples=fb_samples,
        scored_results=scored_results,
        output_dir=args.output_dir,
        compute_coherence=args.compute_coherence,
        embedding_model_name=args.embedding_model,
    )


if __name__ == "__main__":
    main()