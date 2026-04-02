#!/usr/bin/env python3
"""
Chunking sweep runner — evaluates multiple chunking strategies on FinanceBench
through either the single-vector or unified pipeline (or both).

Fixes applied vs original:
  1. output_dir passed to RAGExperiment no longer pre-appends experiment_type.
     RAGExperiment.__init__ appends <experiment_type>/<YYYYMMDD> itself; doing
     it twice produced .../unified/unified/20250303 paths.
  2. Post-run glob uses exp.output_dir (resolved inside RAGExperiment) instead
     of the local output_dir variable, so _latest_file reliably finds outputs.
  3. Default late_window_stride: 128 → 512  (≥ chunk_size; old value created
     16× overlapping windows that dilute embeddings).
  4. Default late_max_tokens: 2048 → 8192  (covers meaningful document context
     on long FinanceBench PDFs).
  5. Default late_model: None → jinaai/jina-embeddings-v2-base-en  (a real
     long-context encoder; without this late chunking degenerates to standard
     chunking and produces no benefit).
"""

import argparse
import csv
import glob
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.core.rag_experiments import RAGExperiment


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _overlap_value(size: int, overlap: float) -> int:
    """Convert fractional or absolute overlap spec to an integer token/char count."""
    if overlap <= 1.0:
        return int(round(size * overlap))
    return int(overlap)


def _latest_file(path_glob: str) -> str:
    candidates = glob.glob(path_glob, recursive=True)
    if not candidates:
        return ""
    return max(candidates, key=os.path.getmtime)


def _find_existing_outputs(
    output_root: str, cfg_name: str, experiment_type: str
) -> Tuple[str, str]:
    """
    Search for outputs under the path structure RAGExperiment creates:
        <output_root>/<cfg_name>/<experiment_type>/**/<experiment_type>_*.json
    """
    output_glob = os.path.join(
        output_root,
        cfg_name,
        experiment_type,
        "**",
        f"{experiment_type}_*.json",
    )
    scored_glob = os.path.join(
        output_root,
        cfg_name,
        "results",
        experiment_type,
        "**",
        f"{experiment_type}_*_scored.json",
    )
    return _latest_file(output_glob), _latest_file(scored_glob)


def _flatten_metrics(
    prefix: str, metrics: Dict[str, Any], row: Dict[str, Any]
) -> None:
    for key, value in (metrics or {}).items():
        row[f"{prefix}_{key}"] = value


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def _build_chunking_configs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []

    # Fixed-token configs
    for size in _parse_int_list(args.fixed_token_sizes):
        for ov in _parse_float_list(args.fixed_token_overlaps):
            ov_val = _overlap_value(size, ov)
            configs.append({
                "name": f"fixed_tokens_{size}_ov{ov_val}",
                "chunking_strategy": "fixed",
                "chunking_unit": "tokens",
                "chunk_size": size,
                "chunk_overlap": ov_val,
            })

    # Fixed-char configs
    for size in _parse_int_list(args.fixed_char_sizes):
        for ov in _parse_float_list(args.fixed_char_overlaps):
            ov_val = _overlap_value(size, ov)
            configs.append({
                "name": f"fixed_chars_{size}_ov{ov_val}",
                "chunking_strategy": "fixed",
                "chunking_unit": "chars",
                "chunk_size": size,
                "chunk_overlap": ov_val,
            })

    # Sentence config
    configs.append({
        "name": f"sentence_{args.sentence_chunk_size}_ov{args.sentence_overlap}",
        "chunking_strategy": "sentence",
        "chunking_unit": "chars",
        "chunk_size": args.sentence_chunk_size,
        "chunk_overlap": args.sentence_overlap,
        "sentence_chunk_size": args.sentence_chunk_size,
        "sentence_overlap": args.sentence_overlap,
        "sentence_max_chars": args.sentence_max_chars,
    })

    # Semantic config
    configs.append({
        "name": (
            f"semantic_t{args.semantic_similarity_threshold}"
            f"_max{args.semantic_max_sentences}"
        ),
        "chunking_strategy": "semantic",
        "chunking_unit": "chars",
        "chunk_size": args.semantic_max_sentences,
        "chunk_overlap": 0,
        "semantic_similarity_threshold": args.semantic_similarity_threshold,
        "semantic_min_sentences": args.semantic_min_sentences,
        "semantic_max_sentences": args.semantic_max_sentences,
        "semantic_max_chunk_chars": args.semantic_max_chunk_chars,
    })

    # Late-chunking config
    configs.append({
        "name": f"late_tokens_{args.late_chunk_size}_ov{args.late_chunk_overlap}",
        "chunking_strategy": "late",
        "chunking_unit": "tokens",
        "chunk_size": args.late_chunk_size,
        "chunk_overlap": args.late_chunk_overlap,
        "late_model": args.late_model,
        "late_max_tokens": args.late_max_tokens,
        "late_window_stride": args.late_window_stride,
        "late_pooling": args.late_pooling,
    })

    return configs


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def _run_experiment(
    cfg: Dict[str, Any],
    experiment_type: str,
    args: argparse.Namespace,
) -> Tuple[str, str]:
    # FIX 1: pass cfg-level root only; RAGExperiment appends experiment_type/date.
    cfg_root = os.path.join(args.output_root, cfg["name"])

    exp = RAGExperiment(
        experiment_type=experiment_type,
        llm_model=args.llm_model,
        chunk_size=cfg.get("chunk_size", args.chunk_size),
        chunk_overlap=cfg.get("chunk_overlap", args.chunk_overlap),
        top_k=args.top_k,
        embedding_model=args.embedding_model,
        output_dir=cfg_root,                  # ← corrected
        vector_store_dir=args.vector_store_dir,
        pdf_local_dir=args.pdf_dir,
        load_in_8bit=not args.no_8bit,
        use_api=args.use_api,
        api_base_url=args.api_base_url,
        api_key_env=args.api_key_env,
        use_all_pdfs=args.use_all_pdfs,
        eval_type=None if args.eval_type == "none" else args.eval_type,
        eval_mode=args.eval_mode,
        chunking_strategy=cfg.get("chunking_strategy", args.chunking_strategy),
        chunking_unit=cfg.get("chunking_unit", args.chunking_unit),
        parent_chunk_size=args.parent_chunk_size,
        parent_chunk_overlap=args.parent_chunk_overlap,
        child_chunk_size=args.child_chunk_size,
        child_chunk_overlap=args.child_chunk_overlap,
        sentence_chunk_size=cfg.get("sentence_chunk_size", args.sentence_chunk_size),
        sentence_overlap=cfg.get("sentence_overlap", args.sentence_overlap),
        sentence_max_chars=cfg.get("sentence_max_chars", args.sentence_max_chars),
        semantic_similarity_threshold=cfg.get(
            "semantic_similarity_threshold", args.semantic_similarity_threshold
        ),
        semantic_min_sentences=cfg.get(
            "semantic_min_sentences", args.semantic_min_sentences
        ),
        semantic_max_sentences=cfg.get(
            "semantic_max_sentences", args.semantic_max_sentences
        ),
        semantic_max_chunk_chars=cfg.get(
            "semantic_max_chunk_chars", args.semantic_max_chunk_chars
        ),
        chunk_tokenizer_name=args.chunk_tokenizer_name,
        partition_model=args.partition_model,
        render_dpi=args.render_dpi,
        vision_encoder=args.vision_encoder,
        patch_size=args.patch_size,
        pipeline_version=args.pipeline_version,
        late_model=cfg.get("late_model", args.late_model),
        late_max_tokens=cfg.get("late_max_tokens", args.late_max_tokens),
        late_window_stride=cfg.get("late_window_stride", args.late_window_stride),
        late_pooling=cfg.get("late_pooling", args.late_pooling),
        use_faiss_chunking=args.use_faiss_chunking,
    )

    # Unified pipeline knobs
    exp.unified_use_hyde           = args.unified_hyde
    exp.unified_hyde_k             = args.unified_hyde_k
    exp.unified_retrieval          = args.unified_retrieval
    exp.unified_use_rerank         = args.unified_rerank
    exp.unified_reranker_style     = args.unified_reranker_style
    exp.unified_ot_model           = args.unified_ot_model
    exp.unified_ot_query_sentences = args.unified_ot_query_sentences
    exp.unified_ot_doc_sentences   = args.unified_ot_doc_sentences
    exp.unified_ot_reg             = args.unified_ot_reg
    exp.unified_ot_iters           = args.unified_ot_iters
    exp.unified_ot_prune_k         = args.unified_ot_prune_k

    exp.run_experiment(num_samples=args.num_samples)

    # FIX 2: use exp.output_dir (resolved by RAGExperiment) for the glob.
    output_file = _latest_file(
        os.path.join(exp.output_dir, f"{experiment_type}_*.json")
    )
    scored_file = _latest_file(
        os.path.join(exp.results_dir, f"{experiment_type}_*_scored.json")
    )
    return output_file, scored_file


# ---------------------------------------------------------------------------
# Summary collector
# ---------------------------------------------------------------------------

def _collect_summary(
    cfg: Dict[str, Any],
    run_type: str,
    output_file: str,
    scored_file: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    target = scored_file or output_file
    if not target or not os.path.exists(target):
        return rows

    import json

    with open(target) as f:
        data = json.load(f)

    meta         = data.get("metadata", {})         if isinstance(data, dict) else {}
    eval_summary = data.get("evaluation_summary", {}) if isinstance(data, dict) else {}

    row: Dict[str, Any] = {
        "config":            cfg.get("name"),
        "run_type":          run_type,
        "chunking_strategy": cfg.get("chunking_strategy"),
        "chunking_unit":     cfg.get("chunking_unit"),
        "chunk_size":        cfg.get("chunk_size"),
        "chunk_overlap":     cfg.get("chunk_overlap"),
        "embedding_model":   meta.get("embedding_model"),
        "llm_model":         meta.get("llm_model"),
    }
    _flatten_metrics("retrieval",  eval_summary.get("retrieval",  {}), row)
    _flatten_metrics("generative", eval_summary.get("generative", {}), row)
    rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Chunking sweep runner.")

    # I/O
    parser.add_argument("--pdf-dir",          default="pdfs")
    parser.add_argument("--output-root",      default="outputs/chunking_sweep")
    parser.add_argument("--vector-store-dir", default="vector_stores")

    # Models
    parser.add_argument("--llm-model",        default=RAGExperiment.QWEN_2_5_7B)
    parser.add_argument("--embedding-model",  default="BAAI/bge-m3")

    # Evaluation
    parser.add_argument(
        "--eval-type", default="both",
        choices=["retrieval", "generative", "both", "none"],
    )
    parser.add_argument(
        "--eval-mode", default="static", choices=["static", "semantic"]
    )
    parser.add_argument("--num-samples",          type=int, default=None)
    parser.add_argument("--top-k",                type=int, default=5)
    parser.add_argument("--chunk-size",           type=int, default=1024)
    parser.add_argument("--chunk-overlap",        type=int, default=128)
    parser.add_argument("--chunking-strategy",    default="recursive")
    parser.add_argument(
        "--chunking-unit", default="chars", choices=["chars", "tokens"]
    )
    parser.add_argument("--chunk-tokenizer-name", default=None)

    # Fixed-size sweep ranges
    parser.add_argument("--fixed-token-sizes",    default="256,512,1024")
    parser.add_argument("--fixed-token-overlaps", default="0,0.1,0.25")
    parser.add_argument("--fixed-char-sizes",     default="256,512,1024")
    parser.add_argument("--fixed-char-overlaps",  default="0,0.1,0.25")

    # Sentence chunking
    parser.add_argument("--sentence-chunk-size",  type=int, default=8)
    parser.add_argument("--sentence-overlap",     type=int, default=1)
    parser.add_argument("--sentence-max-chars",   type=int, default=3000)

    # Semantic chunking
    parser.add_argument("--semantic-similarity-threshold", type=float, default=0.6)
    parser.add_argument("--semantic-min-sentences",        type=int,   default=1)
    parser.add_argument("--semantic-max-sentences",        type=int,   default=12)
    parser.add_argument("--semantic-max-chunk-chars",      type=int,   default=4000)

    # Late-chunking params (all defaults corrected)
    parser.add_argument(
        "--late-model",
        default="jinaai/jina-embeddings-v2-base-en",   # FIX 5
        help=(
            "Long-context encoder for late chunking. "
            "Requires a model with a context window ≥ late_max_tokens. "
            "Recommended: jinaai/jina-embeddings-v2-base-en or "
            "nomic-ai/nomic-embed-text-v1 (both support 8 192 tokens)."
        ),
    )
    parser.add_argument(
        "--late-max-tokens", type=int, default=8192,   # FIX 4
        help="Max tokens per context window for the late encoder.",
    )
    parser.add_argument(
        "--late-window-stride", type=int, default=512,  # FIX 3
        help=(
            "Sliding-window stride in tokens. "
            "Must be ≥ late_chunk_size to avoid redundant overlap. "
            "(Old default of 128 caused 16× overlap on 512-token chunks.)"
        ),
    )
    parser.add_argument("--late-pooling",      default="mean")
    parser.add_argument("--late-chunk-size",   type=int, default=512)
    parser.add_argument("--late-chunk-overlap",type=int, default=64)
    parser.add_argument(
        "--use-faiss-chunking", action="store_true", default=True
    )

    # Which experiment pipelines to run
    parser.add_argument("--include-bm25",    action="store_true")
    parser.add_argument("--include-single",  action="store_true")
    parser.add_argument("--include-unified", action="store_true")

    # API / quantisation
    parser.add_argument("--use-api",       action="store_true")
    parser.add_argument("--api-base-url",  default="https://api.openai.com/v1")
    parser.add_argument("--api-key-env",   default="OPENAI_API_KEY")
    parser.add_argument("--no-8bit",       action="store_true")
    parser.add_argument("--use-all-pdfs",  action="store_true")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run experiments even if outputs already exist.",
    )

    # Hierarchical / vision / chipper params
    parser.add_argument("--parent-chunk-size",    type=int, default=None)
    parser.add_argument("--parent-chunk-overlap", type=int, default=None)
    parser.add_argument("--child-chunk-size",     type=int, default=None)
    parser.add_argument("--child-chunk-overlap",  type=int, default=None)
    parser.add_argument("--partition-model",      default="chipper")
    parser.add_argument("--render-dpi",           type=int, default=None)
    parser.add_argument("--vision-encoder",       default=None)
    parser.add_argument("--patch-size",           type=int, default=None)
    parser.add_argument("--pipeline-version",     default=None)

    # Unified pipeline knobs
    parser.add_argument("--unified-hyde",            action="store_true")
    parser.add_argument("--unified-hyde-k",          type=int, default=1)
    parser.add_argument(
        "--unified-retrieval", default="dense",
        choices=["dense", "sparse", "hybrid"],
    )
    parser.add_argument("--unified-rerank",          action="store_true")
    parser.add_argument(
        "--unified-reranker-style", default="cross_encoder",
        choices=["cross_encoder", "ot", "ot_then_cross_encoder"],
    )
    parser.add_argument("--unified-ot-model",          default="BAAI/bge-m3")
    parser.add_argument("--unified-ot-query-sentences",type=int, default=8)
    parser.add_argument("--unified-ot-doc-sentences",  type=int, default=24)
    parser.add_argument("--unified-ot-reg",            type=float, default=0.05)
    parser.add_argument("--unified-ot-iters",          type=int, default=40)
    parser.add_argument("--unified-ot-prune-k",        type=int, default=20)

    args = parser.parse_args()

    # Default: run both single and unified if nothing specified
    if not args.include_single and not args.include_unified and not args.include_bm25:
        args.include_single  = True
        args.include_unified = True

    configs = _build_chunking_configs(args)
    summary_rows: List[Dict[str, Any]] = []
    skip_existing = not args.force

    for cfg in configs:
        is_late = cfg.get("chunking_strategy") == "late"

        if args.include_single:
            if skip_existing:
                out, scored = _find_existing_outputs(
                    args.output_root, cfg["name"], RAGExperiment.SINGLE_VECTOR
                )
                if out or scored:
                    print(f"[skip] single  {cfg['name']} (already exists)")
                else:
                    out, scored = _run_experiment(
                        cfg, RAGExperiment.SINGLE_VECTOR, args
                    )
            else:
                out, scored = _run_experiment(cfg, RAGExperiment.SINGLE_VECTOR, args)
            summary_rows.extend(_collect_summary(cfg, "single", out, scored))

        if args.include_unified:
            if skip_existing:
                out, scored = _find_existing_outputs(
                    args.output_root, cfg["name"], RAGExperiment.UNIFIED
                )
                if out or scored:
                    print(f"[skip] unified {cfg['name']} (already exists)")
                else:
                    out, scored = _run_experiment(cfg, RAGExperiment.UNIFIED, args)
            else:
                out, scored = _run_experiment(cfg, RAGExperiment.UNIFIED, args)
            summary_rows.extend(_collect_summary(cfg, "unified", out, scored))

        if args.include_bm25 and not is_late:
            if skip_existing:
                out, scored = _find_existing_outputs(
                    args.output_root, cfg["name"], RAGExperiment.BM25
                )
                if out or scored:
                    print(f"[skip] bm25    {cfg['name']} (already exists)")
                else:
                    out, scored = _run_experiment(cfg, RAGExperiment.BM25, args)
            else:
                out, scored = _run_experiment(cfg, RAGExperiment.BM25, args)
            summary_rows.extend(_collect_summary(cfg, "bm25", out, scored))
        elif args.include_bm25 and is_late:
            print(f"[skip] bm25 not supported for late chunking — skipping {cfg['name']}")

    if summary_rows:
        summary_path = os.path.join(args.output_root, "chunking_sweep_summary.csv")
        os.makedirs(args.output_root, exist_ok=True)
        fieldnames = sorted(summary_rows[0].keys())
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()