#!/usr/bin/env python3
"""
run_chunking_experiments.py
===========================
Sweep all 9 chunking strategies through the FinanceBench RAG pipeline.

Each strategy uses the **same BGE-M3 dense retriever** (except late chunking,
which additionally needs a long-context encoder for the embedding step).
All chunk sizes are in tokens for fair comparison.

Integration
-----------
This script creates a RAGExperiment per strategy, overriding the chunking
logic via a hook.  It writes per-strategy JSON results that the companion
`chunk_property_analysis.py` script consumes for property–quality correlation.

Usage
-----
    # Run all strategies on FinanceBench (unified pipeline, dense retrieval)
    python run_chunking_experiments.py \
        --pdf-dir /path/to/financebench_pdfs \
        --output-root ./chunking_results \
        --embedding-model bge-m3 \
        --top-k 5

    # Run a subset
    python run_chunking_experiments.py \
        --strategies naive recursive semantic \
        --pdf-dir /path/to/financebench_pdfs

    # With late chunking model
    python run_chunking_experiments.py \
        --strategies late \
        --late-model jinaai/jina-embeddings-v2-base-en \
        --pdf-dir /path/to/financebench_pdfs
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import glob
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so we can import src.core.rag_experiments
# Adjust this if your repo layout differs.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent  # adjust if needed
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from chunking_strategies import (
    Chunk,
    STRATEGY_REGISTRY,
    chunk_naive,
    chunk_recursive,
    chunk_semantic,
    chunk_adaptive,
    chunk_parent_child,
    chunk_table_aware,
    chunk_late,
    chunk_contextual,
    chunk_metadata,
    _count_tokens,
    _get_tokenizer,
)

# Try to import RAGExperiment — if unavailable, we run in standalone mode
try:
    from src.core.rag_experiments import RAGExperiment
    HAS_RAG_EXPERIMENT = True
except ImportError:
    HAS_RAG_EXPERIMENT = False
    print("[WARN] Could not import RAGExperiment. Running in standalone chunking-only mode.")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ===================================================================
# Strategy configurations
# ===================================================================

def build_strategy_configs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """
    Build a list of experiment configs, one per strategy.  Each config
    contains all parameters the chunker and RAGExperiment need.
    """
    configs = []

    # Baseline: 1024 tokens + 128 overlap (your existing baseline)
    BASELINE_SIZE = args.chunk_size        # default 1024
    BASELINE_OVERLAP = args.chunk_overlap  # default 128

    # ---------------------------------------------------------------
    # 1. Naive (fixed) chunking
    # ---------------------------------------------------------------
    if "naive" in args.strategies:
        configs.append({
            "name": f"naive_tok{BASELINE_SIZE}_ov{BASELINE_OVERLAP}",
            "strategy": "naive",
            "chunking_strategy": "fixed",   # maps to RAGExperiment
            "chunking_unit": "tokens",
            "chunk_size": BASELINE_SIZE,
            "chunk_overlap": BASELINE_OVERLAP,
            "chunker_kwargs": {
                "chunk_size": BASELINE_SIZE,
                "chunk_overlap": BASELINE_OVERLAP,
            },
        })

    # ---------------------------------------------------------------
    # 2. Recursive
    # ---------------------------------------------------------------
    if "recursive" in args.strategies:
        configs.append({
            "name": f"recursive_tok{BASELINE_SIZE}_ov{BASELINE_OVERLAP}",
            "strategy": "recursive",
            "chunking_strategy": "recursive",
            "chunking_unit": "tokens",
            "chunk_size": BASELINE_SIZE,
            "chunk_overlap": BASELINE_OVERLAP,
            "chunker_kwargs": {
                "chunk_size": BASELINE_SIZE,
                "chunk_overlap": BASELINE_OVERLAP,
                "separators": ["\n\n\n", "\n\n", "\n", ". ", " "],
            },
        })

    # ---------------------------------------------------------------
    # 3. Semantic
    # ---------------------------------------------------------------
    if "semantic" in args.strategies:
        configs.append({
            "name": f"semantic_tok{BASELINE_SIZE}_t{args.semantic_threshold}",
            "strategy": "semantic",
            "chunking_strategy": "semantic",
            "chunking_unit": "tokens",
            "chunk_size": BASELINE_SIZE,
            "chunk_overlap": 0,  # semantic doesn't use overlap
            "chunker_kwargs": {
                "chunk_size": BASELINE_SIZE,
                "similarity_threshold": args.semantic_threshold,
                "min_sentences": args.semantic_min_sentences,
                "max_sentences": args.semantic_max_sentences,
                "embedding_model_name": args.embedding_model,
            },
        })

    # ---------------------------------------------------------------
    # 4. Adaptive
    # ---------------------------------------------------------------
    if "adaptive" in args.strategies:
        configs.append({
            "name": f"adaptive_base{BASELINE_SIZE}_min{args.adaptive_min}_max{args.adaptive_max}",
            "strategy": "adaptive",
            "chunking_strategy": "adaptive",
            "chunking_unit": "tokens",
            "chunk_size": BASELINE_SIZE,
            "chunk_overlap": BASELINE_OVERLAP,
            "chunker_kwargs": {
                "base_chunk_size": BASELINE_SIZE,
                "chunk_overlap": BASELINE_OVERLAP,
                "min_chunk_size": args.adaptive_min,
                "max_chunk_size": args.adaptive_max,
            },
        })

    # ---------------------------------------------------------------
    # 5. Parent-Child
    # ---------------------------------------------------------------
    if "parent_child" in args.strategies:
        configs.append({
            "name": f"parent_child_p{args.parent_size}_c{args.child_size}",
            "strategy": "parent_child",
            "chunking_strategy": "parent_child",
            "chunking_unit": "tokens",
            "chunk_size": args.child_size,  # indexing unit = child
            "chunk_overlap": args.child_overlap,
            "chunker_kwargs": {
                "parent_chunk_size": args.parent_size,
                "parent_overlap": args.parent_overlap,
                "child_chunk_size": args.child_size,
                "child_overlap": args.child_overlap,
            },
        })

    # ---------------------------------------------------------------
    # 6. Table-Aware
    # ---------------------------------------------------------------
    if "table_aware" in args.strategies:
        configs.append({
            "name": f"table_aware_tok{BASELINE_SIZE}_ov{BASELINE_OVERLAP}",
            "strategy": "table_aware",
            "chunking_strategy": "table_aware",
            "chunking_unit": "tokens",
            "chunk_size": BASELINE_SIZE,
            "chunk_overlap": BASELINE_OVERLAP,
            "chunker_kwargs": {
                "chunk_size": BASELINE_SIZE,
                "chunk_overlap": BASELINE_OVERLAP,
            },
        })

    # ---------------------------------------------------------------
    # 7. Late Chunking
    # ---------------------------------------------------------------
    if "late" in args.strategies:
        configs.append({
            "name": f"late_tok{args.late_chunk_size}_ov{args.late_chunk_overlap}",
            "strategy": "late",
            "chunking_strategy": "late",
            "chunking_unit": "tokens",
            "chunk_size": args.late_chunk_size,
            "chunk_overlap": args.late_chunk_overlap,
            "late_model": args.late_model,
            "late_max_tokens": args.late_max_tokens,
            "late_window_stride": args.late_window_stride,
            "late_pooling": args.late_pooling,
            "chunker_kwargs": {
                "chunk_size": args.late_chunk_size,
                "chunk_overlap": args.late_chunk_overlap,
            },
        })

    # ---------------------------------------------------------------
    # 8. Contextual (no LLM)
    # ---------------------------------------------------------------
    if "contextual" in args.strategies:
        configs.append({
            "name": f"contextual_tok{BASELINE_SIZE}_ctx{args.context_budget}",
            "strategy": "contextual",
            "chunking_strategy": "contextual",
            "chunking_unit": "tokens",
            "chunk_size": BASELINE_SIZE,
            "chunk_overlap": BASELINE_OVERLAP,
            "chunker_kwargs": {
                "chunk_size": BASELINE_SIZE,
                "chunk_overlap": BASELINE_OVERLAP,
                "context_budget": args.context_budget,
            },
        })

    # ---------------------------------------------------------------
    # 9. Metadata
    # ---------------------------------------------------------------
    if "metadata" in args.strategies:
        configs.append({
            "name": f"metadata_tok{BASELINE_SIZE}_ov{BASELINE_OVERLAP}",
            "strategy": "metadata",
            "chunking_strategy": "metadata",
            "chunking_unit": "tokens",
            "chunk_size": BASELINE_SIZE,
            "chunk_overlap": BASELINE_OVERLAP,
            "chunker_kwargs": {
                "chunk_size": BASELINE_SIZE,
                "chunk_overlap": BASELINE_OVERLAP,
            },
        })

    return configs


# ===================================================================
# Standalone chunking (when RAGExperiment is not available or
# for pre-chunking + analysis)
# ===================================================================

def extract_pages_from_pdf(pdf_path: str) -> List[Tuple[int, str]]:
    """Extract (page_num, text) tuples from a PDF."""
    try:
        import pymupdf  # PyMuPDF
        doc = pymupdf.open(pdf_path)
    except ImportError:
        import fitz  # older PyMuPDF import name
        doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            pages.append((i + 1, text))
    doc.close()
    return pages


def run_standalone_chunking(
    config: Dict[str, Any],
    pdf_dir: str,
    output_dir: str,
    doc_ids: Optional[List[str]] = None,
) -> Dict[str, List[Chunk]]:
    """
    Run a single chunking strategy across all PDFs.  Returns
    {doc_id: [Chunk, ...]} and saves to JSON.
    """
    strategy_name = config["strategy"]
    chunker = STRATEGY_REGISTRY[strategy_name]
    kwargs = config.get("chunker_kwargs", {})

    pdf_files = sorted(Path(pdf_dir).glob("*.pdf"))
    if doc_ids:
        pdf_files = [p for p in pdf_files if p.stem in doc_ids]

    logger.info(f"[{strategy_name}] Chunking {len(pdf_files)} PDFs ...")
    all_chunks: Dict[str, List[Chunk]] = {}
    total_chunks = 0

    for pdf_path in pdf_files:
        doc_id = pdf_path.stem
        pages = extract_pages_from_pdf(str(pdf_path))
        if not pages:
            logger.warning(f"  Skipping {doc_id}: no text extracted")
            continue

        chunks = chunker(pages=pages, doc_id=doc_id, **kwargs)
        all_chunks[doc_id] = chunks
        total_chunks += len(chunks)

    logger.info(f"[{strategy_name}] Produced {total_chunks} chunks across {len(all_chunks)} docs")

    # Save chunk data for analysis
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"chunks_{config['name']}.json")
    serializable = {}
    for doc_id, chunks in all_chunks.items():
        serializable[doc_id] = [
            {
                "text": c.text,
                "raw_text": c.raw_text,
                "doc_id": c.doc_id,
                "page_nums": c.page_nums,
                "token_count": c.token_count,
                "raw_token_count": c.raw_token_count,
                "strategy": c.strategy,
                "chunk_index": c.chunk_index,
                "parent_chunk_index": c.parent_chunk_index,
                "metadata": {
                    k: v for k, v in c.metadata.items()
                    if k != "parent_text"  # don't serialize full parent text
                    and k != "full_doc_tokens"
                },
            }
            for c in chunks
        ]
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"[{strategy_name}] Saved chunks to {out_path}")

    return all_chunks


# ===================================================================
# RAGExperiment integration
# ===================================================================

def run_rag_experiment(
    config: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Run one chunking strategy through RAGExperiment's unified pipeline.
    Returns (output_file, scored_file) paths.
    """
    if not HAS_RAG_EXPERIMENT:
        logger.error("RAGExperiment not available.  Run in standalone mode.")
        return None, None

    cfg_root = os.path.join(args.output_root, config["name"])
    strategy = config["strategy"]

    # Map our strategy names to what RAGExperiment expects
    rag_chunking_strategy = config.get("chunking_strategy", "fixed")

    # --- Build RAGExperiment ---
    exp_kwargs = dict(
        experiment_type=RAGExperiment.UNIFIED,
        llm_model=args.llm_model,
        chunk_size=config["chunk_size"],
        chunk_overlap=config.get("chunk_overlap", 128),
        top_k=args.top_k,
        embedding_model=args.embedding_model,
        output_dir=cfg_root,
        vector_store_dir=args.vector_store_dir,
        pdf_local_dir=args.pdf_dir,
        load_in_8bit=not args.no_8bit,
        use_api=args.use_api,
        api_base_url=args.api_base_url,
        api_key_env=args.api_key_env,
        use_all_pdfs=args.use_all_pdfs,
        eval_type=None if args.eval_type == "none" else args.eval_type,
        eval_mode=args.eval_mode,
        chunking_strategy=rag_chunking_strategy,
        chunking_unit="tokens",
        chunk_tokenizer_name=args.chunk_tokenizer_name,
        use_faiss_chunking=args.use_faiss_chunking,
    )

    # Strategy-specific parameters
    if strategy == "semantic":
        exp_kwargs.update(
            semantic_similarity_threshold=config["chunker_kwargs"].get("similarity_threshold", 0.5),
            semantic_min_sentences=config["chunker_kwargs"].get("min_sentences", 2),
            semantic_max_sentences=config["chunker_kwargs"].get("max_sentences", 40),
        )
    elif strategy == "parent_child":
        exp_kwargs.update(
            parent_chunk_size=config["chunker_kwargs"]["parent_chunk_size"],
            parent_chunk_overlap=config["chunker_kwargs"]["parent_overlap"],
            child_chunk_size=config["chunker_kwargs"]["child_chunk_size"],
            child_chunk_overlap=config["chunker_kwargs"]["child_overlap"],
        )
    elif strategy == "late":
        exp_kwargs.update(
            late_model=config.get("late_model"),
            late_max_tokens=config.get("late_max_tokens", 8192),
            late_window_stride=config.get("late_window_stride", 512),
            late_pooling=config.get("late_pooling", "mean"),
        )

    try:
        exp = RAGExperiment(**exp_kwargs)
    except TypeError as e:
        # If RAGExperiment doesn't accept some kwargs, strip them and retry
        logger.warning(f"RAGExperiment init failed ({e}), retrying with reduced kwargs")
        for key in ["semantic_similarity_threshold", "semantic_min_sentences",
                     "semantic_max_sentences", "parent_chunk_size", "parent_chunk_overlap",
                     "child_chunk_size", "child_chunk_overlap", "late_model",
                     "late_max_tokens", "late_window_stride", "late_pooling"]:
            exp_kwargs.pop(key, None)
        exp = RAGExperiment(**exp_kwargs)

    # Set unified pipeline knobs
    exp.unified_use_hyde = args.unified_hyde
    exp.unified_hyde_k = args.unified_hyde_k
    exp.unified_retrieval = args.unified_retrieval
    exp.unified_use_rerank = args.unified_rerank
    exp.unified_reranker_style = args.unified_reranker_style

    logger.info(f"[{strategy}] Running RAGExperiment (unified pipeline) ...")
    t0 = time.time()
    exp.run_experiment(num_samples=args.num_samples)
    elapsed = time.time() - t0
    logger.info(f"[{strategy}] Completed in {elapsed:.1f}s")

    # Find output files
    output_file = _latest_file(
        os.path.join(exp.output_dir, f"{RAGExperiment.UNIFIED}_*.json")
    )
    scored_file = _latest_file(
        os.path.join(getattr(exp, "results_dir", exp.output_dir),
                     f"{RAGExperiment.UNIFIED}_*_scored.json")
    )
    return output_file, scored_file


def _latest_file(path_glob: str) -> Optional[str]:
    candidates = glob.glob(path_glob, recursive=True)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


# ===================================================================
# Summary table
# ===================================================================

def collect_summary(
    config: Dict[str, Any],
    output_file: Optional[str],
    scored_file: Optional[str],
    chunk_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Collect a summary row for the results table."""
    row = {
        "strategy": config["strategy"],
        "config_name": config["name"],
        "chunk_size": config["chunk_size"],
        "chunk_overlap": config.get("chunk_overlap", 0),
    }

    # Parse scored results if available
    if scored_file and os.path.exists(scored_file):
        try:
            with open(scored_file) as f:
                scored = json.load(f)
            # Extract aggregate metrics
            if isinstance(scored, list):
                results = scored
            elif isinstance(scored, dict):
                results = scored.get("results", scored.get("samples", []))
            else:
                results = []

            if results:
                # page_recall@k
                page_recalls = [r.get("page_recall", r.get("page_recall@k", 0)) for r in results if isinstance(r, dict)]
                if page_recalls:
                    row["page_recall@k_mean"] = round(np.mean(page_recalls), 4)
                    row["page_recall@k_std"] = round(np.std(page_recalls), 4)

                # chunk_recall@k
                chunk_recalls = [r.get("chunk_recall", r.get("chunk_recall@k", 0)) for r in results if isinstance(r, dict)]
                if chunk_recalls:
                    row["chunk_recall@k_mean"] = round(np.mean(chunk_recalls), 4)

                # doc_recall@k
                doc_recalls = [r.get("doc_recall", r.get("doc_recall@k", 0)) for r in results if isinstance(r, dict)]
                if doc_recalls:
                    row["doc_recall@k_mean"] = round(np.mean(doc_recalls), 4)

                # BLEU / ROUGE (if available from generation eval)
                bleus = [r.get("bleu", 0) for r in results if isinstance(r, dict) and "bleu" in r]
                if bleus:
                    row["bleu_mean"] = round(np.mean(bleus), 4)
                rouges = [r.get("rouge_l", r.get("rougeL", 0)) for r in results if isinstance(r, dict) and ("rouge_l" in r or "rougeL" in r)]
                if rouges:
                    row["rouge_l_mean"] = round(np.mean(rouges), 4)

                # Accuracy (exact match or LLM-judge)
                accs = [r.get("accuracy", r.get("correct", 0)) for r in results if isinstance(r, dict) and ("accuracy" in r or "correct" in r)]
                if accs:
                    row["accuracy_mean"] = round(np.mean(accs), 4)

        except Exception as e:
            logger.warning(f"Could not parse scored file {scored_file}: {e}")

    if chunk_stats:
        row.update(chunk_stats)

    return row


def write_summary_csv(rows: List[Dict[str, Any]], output_path: str):
    """Write summary table to CSV."""
    if not rows:
        return
    # Collect all keys
    all_keys = []
    for r in rows:
        for k in r:
            if k not in all_keys:
                all_keys.append(k)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Summary written to {output_path}")


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run chunking strategy sweep on FinanceBench."
    )
    # --- General ---
    parser.add_argument("--pdf-dir", required=True, help="Directory with FinanceBench PDFs")
    parser.add_argument("--output-root", default="./chunking_results",
                        help="Root output directory")
    parser.add_argument("--strategies", nargs="+",
                        default=["naive", "recursive", "semantic", "adaptive",
                                 "parent_child", "table_aware", "late",
                                 "contextual", "metadata"],
                        choices=list(STRATEGY_REGISTRY.keys()),
                        help="Which strategies to run")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Limit number of FinanceBench samples (None=all)")
    parser.add_argument("--standalone-only", action="store_true",
                        help="Only run chunking + analysis, skip RAGExperiment")

    # --- Chunk sizes (baseline) ---
    parser.add_argument("--chunk-size", type=int, default=1024,
                        help="Baseline chunk size in tokens")
    parser.add_argument("--chunk-overlap", type=int, default=128,
                        help="Baseline overlap in tokens")

    # --- Semantic ---
    parser.add_argument("--semantic-threshold", type=float, default=0.5,
                        help="Cosine similarity threshold for semantic chunking")
    parser.add_argument("--semantic-min-sentences", type=int, default=2)
    parser.add_argument("--semantic-max-sentences", type=int, default=40)

    # --- Adaptive ---
    parser.add_argument("--adaptive-min", type=int, default=256,
                        help="Min chunk size for adaptive")
    parser.add_argument("--adaptive-max", type=int, default=2048,
                        help="Max chunk size for adaptive")

    # --- Parent-Child ---
    parser.add_argument("--parent-size", type=int, default=2048)
    parser.add_argument("--parent-overlap", type=int, default=256)
    parser.add_argument("--child-size", type=int, default=512)
    parser.add_argument("--child-overlap", type=int, default=64)

    # --- Late ---
    parser.add_argument("--late-chunk-size", type=int, default=512)
    parser.add_argument("--late-chunk-overlap", type=int, default=64)
    parser.add_argument("--late-model", type=str,
                        default="jinaai/jina-embeddings-v2-base-en",
                        help="Long-context encoder for late chunking")
    parser.add_argument("--late-max-tokens", type=int, default=8192)
    parser.add_argument("--late-window-stride", type=int, default=512)
    parser.add_argument("--late-pooling", type=str, default="mean")

    # --- Contextual ---
    parser.add_argument("--context-budget", type=int, default=128,
                        help="Token budget for context prefix in contextual chunking")

    # --- RAGExperiment parameters ---
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--llm-model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--vector-store-dir", default="./vector_stores")
    parser.add_argument("--no-8bit", action="store_true")
    parser.add_argument("--use-api", action="store_true")
    parser.add_argument("--api-base-url", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--use-all-pdfs", action="store_true")
    parser.add_argument("--eval-type", default="none", choices=["none", "llm", "exact", "both"])
    parser.add_argument("--eval-mode", default="accuracy", choices=["static"])
    parser.add_argument("--chunk-tokenizer-name", default="BAAI/bge-m3")
    parser.add_argument("--use-faiss-chunking", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-run even if outputs exist")

    # --- Unified pipeline ---
    parser.add_argument("--unified-hyde", action="store_true")
    parser.add_argument("--unified-hyde-k", type=int, default=3)
    parser.add_argument("--unified-retrieval", default="dense",
                        choices=["dense", "sparse", "hybrid"])
    parser.add_argument("--unified-rerank", action="store_true")
    parser.add_argument("--unified-reranker-style", default="cross-encoder")

    args = parser.parse_args()

    # --- Build configs ---
    configs = build_strategy_configs(args)
    logger.info(f"Running {len(configs)} chunking strategies: "
                f"{[c['strategy'] for c in configs]}")

    summary_rows: List[Dict[str, Any]] = []
    os.makedirs(args.output_root, exist_ok=True)

    for config in configs:
        strategy = config["strategy"]
        logger.info(f"\n{'='*60}\n  Strategy: {strategy}  ({config['name']})\n{'='*60}")

        # ------ Phase 1: Standalone chunking (always run for analysis) ------
        chunk_output_dir = os.path.join(args.output_root, "chunk_data")
        all_chunks = run_standalone_chunking(
            config, args.pdf_dir, chunk_output_dir
        )

        # Compute basic chunk stats
        all_token_counts = []
        all_page_spans = []
        n_chunks_per_doc = []
        for doc_id, chunks in all_chunks.items():
            indexable = [c for c in chunks if c.strategy != "parent_child_parent"]
            n_chunks_per_doc.append(len(indexable))
            for c in indexable:
                all_token_counts.append(c.raw_token_count)
                all_page_spans.append(len(c.page_nums))

        chunk_stats = {}
        if all_token_counts:
            chunk_stats = {
                "n_docs": len(all_chunks),
                "total_chunks": sum(n_chunks_per_doc),
                "mean_chunks_per_doc": round(np.mean(n_chunks_per_doc), 1),
                "mean_token_count": round(np.mean(all_token_counts), 1),
                "std_token_count": round(np.std(all_token_counts), 1),
                "median_token_count": round(float(np.median(all_token_counts)), 1),
                "min_token_count": int(np.min(all_token_counts)),
                "max_token_count": int(np.max(all_token_counts)),
                "mean_page_span": round(np.mean(all_page_spans), 2),
            }
            logger.info(f"  Chunk stats: {json.dumps(chunk_stats, indent=2)}")

        # ------ Phase 2: RAG experiment (if available and not standalone-only) ------
        output_file, scored_file = None, None
        if not args.standalone_only and HAS_RAG_EXPERIMENT:
            try:
                output_file, scored_file = run_rag_experiment(config, args)
            except Exception as e:
                logger.error(f"  RAGExperiment failed for {strategy}: {e}")
                import traceback
                traceback.print_exc()

        # ------ Collect summary ------
        row = collect_summary(config, output_file, scored_file, chunk_stats)
        summary_rows.append(row)
        logger.info(f"  Summary: {json.dumps(row, indent=2)}")

    # ------ Write summary ------
    summary_path = os.path.join(args.output_root, "chunking_sweep_summary.csv")
    write_summary_csv(summary_rows, summary_path)

    # Also write as JSON for easier downstream consumption
    summary_json_path = os.path.join(args.output_root, "chunking_sweep_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary_rows, f, indent=2)
    logger.info(f"\nAll done.  Summary: {summary_path}")


if __name__ == "__main__":
    main()