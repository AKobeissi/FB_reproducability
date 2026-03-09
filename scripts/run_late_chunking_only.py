#!/usr/bin/env python3
"""
Run late chunking experiment only (unified pipeline).

Fixes applied vs original:
  1. output_dir passed to RAGExperiment is now cfg_root only (no experiment_type
     appended) — RAGExperiment.__init__ appends experiment_type/YYYYMMDD itself,
     so the old code produced a doubled path: .../unified/unified/20250303.
  2. Post-run glob now uses exp.output_dir (the resolved path) instead of the
     pre-constructed output_dir variable, so _latest_file always finds the file.
  3. Default late_window_stride changed from 128 → 512 (stride should be ≥
     chunk_size to avoid a 16x over-lapping window explosion that dilutes
     embeddings and balloons index build time).
  4. Default late_max_tokens changed from 2048 → 8192 to give the encoder
     meaningful document context on long FinanceBench PDFs.
  5. Default late_model set to jinaai/jina-embeddings-v2-base-en, a true
     long-context encoder (8 192-token context window) required for late
     chunking to differ from standard chunking. Override with --late-model.
"""

import argparse
import glob
import os
from typing import Any, Dict, Tuple

from src.core.rag_experiments import RAGExperiment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _latest_file(path_glob: str) -> str:
    candidates = glob.glob(path_glob, recursive=True)
    if not candidates:
        return ""
    return max(candidates, key=os.path.getmtime)


def _find_existing_outputs(
    output_root: str, cfg_name: str, experiment_type: str
) -> Tuple[str, str]:
    """
    Look for existing output / scored files under the directory tree that
    RAGExperiment creates:
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
        "results",          # RAGExperiment puts results one level up from outputs
        experiment_type,
        "**",
        f"{experiment_type}_*_scored.json",
    )
    return _latest_file(output_glob), _latest_file(scored_glob)


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def _run_experiment(
    cfg: Dict[str, Any], args: argparse.Namespace
) -> Tuple[str, str]:
    # FIX 1: do NOT append RAGExperiment.UNIFIED here.
    # RAGExperiment.__init__ appends  <experiment_type>/<YYYYMMDD>  itself.
    # The old code produced:  .../late_tokens_512_ov64/unified/unified/20250303
    cfg_root = os.path.join(args.output_root, cfg["name"])

    exp = RAGExperiment(
        experiment_type=RAGExperiment.UNIFIED,
        llm_model=args.llm_model,
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
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
        chunking_strategy="late",
        chunking_unit="tokens",
        chunk_tokenizer_name=args.chunk_tokenizer_name,
        late_model=args.late_model,
        late_max_tokens=args.late_max_tokens,
        late_window_stride=args.late_window_stride,
        late_pooling=args.late_pooling,
        use_faiss_chunking=args.use_faiss_chunking,
    )

    # Unified pipeline knobs
    exp.unified_use_hyde         = args.unified_hyde
    exp.unified_hyde_k           = args.unified_hyde_k
    exp.unified_retrieval        = args.unified_retrieval
    exp.unified_use_rerank       = args.unified_rerank
    exp.unified_reranker_style   = args.unified_reranker_style
    exp.unified_ot_model         = args.unified_ot_model
    exp.unified_ot_query_sentences = args.unified_ot_query_sentences
    exp.unified_ot_doc_sentences = args.unified_ot_doc_sentences
    exp.unified_ot_reg           = args.unified_ot_reg
    exp.unified_ot_iters         = args.unified_ot_iters
    exp.unified_ot_prune_k       = args.unified_ot_prune_k

    exp.run_experiment(num_samples=args.num_samples)

    # FIX 2: use exp.output_dir (resolved by RAGExperiment) for the glob,
    # not the cfg_root we passed in.
    output_file = _latest_file(
        os.path.join(exp.output_dir, f"{RAGExperiment.UNIFIED}_*.json")
    )
    scored_file = _latest_file(
        os.path.join(exp.results_dir, f"{RAGExperiment.UNIFIED}_*_scored.json")
    )
    return output_file, scored_file


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run late chunking only (unified pipeline)."
    )

    # I/O
    parser.add_argument("--pdf-dir",          default="pdfs")
    parser.add_argument("--output-root",      default="outputs/chunking_sweep")
    parser.add_argument("--vector-store-dir", default="vector_stores")

    # Models
    parser.add_argument("--llm-model",        default=RAGExperiment.QWEN_2_5_7B)
    parser.add_argument("--embedding-model",  default="bge-m3")

    # Evaluation
    parser.add_argument(
        "--eval-type", default="both",
        choices=["retrieval", "generative", "both", "none"],
    )
    parser.add_argument(
        "--eval-mode", default="static", choices=["static", "semantic"]
    )
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--top-k",       type=int, default=5)
    parser.add_argument("--chunk-tokenizer-name", default=None)

    # Late-chunking params
    # FIX 3: default late_model → a real long-context encoder.
    parser.add_argument(
        "--late-model",
        default="jinaai/jina-embeddings-v2-base-en",
        help=(
            "Long-context encoder used to build contextual chunk embeddings. "
            "Must support at least late_max_tokens tokens. "
            "Recommended: jinaai/jina-embeddings-v2-base-en (8 192 tok) or "
            "nomic-ai/nomic-embed-text-v1 (8 192 tok)."
        ),
    )
    # FIX 4: late_max_tokens 2048 → 8192 to cover meaningful doc context.
    parser.add_argument(
        "--late-max-tokens", type=int, default=8192,
        help="Max tokens per context window for the late encoder.",
    )
    # FIX 3 (stride): 128 → 512  (≥ chunk_size avoids redundant window overlap).
    parser.add_argument(
        "--late-window-stride", type=int, default=512,
        help=(
            "Stride of the sliding context window in tokens. "
            "Should be ≥ late_chunk_size. "
            "Old default of 128 caused 16× overlap, diluting embeddings."
        ),
    )
    parser.add_argument("--late-pooling",      default="mean")
    parser.add_argument("--late-chunk-size",   type=int, default=512)
    parser.add_argument("--late-chunk-overlap",type=int, default=64)
    parser.add_argument(
        "--use-faiss-chunking", action="store_true", default=True,
        help="Use FAISS-based approximate NN for the late chunking index.",
    )

    # API / quantisation
    parser.add_argument("--use-api",       action="store_true")
    parser.add_argument("--api-base-url",  default="https://api.openai.com/v1")
    parser.add_argument("--api-key-env",   default="OPENAI_API_KEY")
    parser.add_argument("--no-8bit",       action="store_true")
    parser.add_argument("--use-all-pdfs",  action="store_true")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run even if outputs already exist.",
    )

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

    cfg = {
        "name":         f"late_tokens_{args.late_chunk_size}_ov{args.late_chunk_overlap}",
        "chunk_size":   args.late_chunk_size,
        "chunk_overlap":args.late_chunk_overlap,
    }

    if not args.force:
        output_file, scored_file = _find_existing_outputs(
            args.output_root, cfg["name"], RAGExperiment.UNIFIED
        )
        if output_file or scored_file:
            print(
                "Existing late-chunking outputs found — skipping. "
                "Pass --force to re-run."
            )
            return

    output_file, scored_file = _run_experiment(cfg, args)

    if scored_file:
        print(f"Scored output: {scored_file}")
    elif output_file:
        print(f"Output: {output_file}")
    else:
        print("WARNING: no output file found after experiment run.")


if __name__ == "__main__":
    main()