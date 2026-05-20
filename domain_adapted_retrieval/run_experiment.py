"""
Master entry point for the Domain-Adapted Financial RAG experiment.

Usage examples:

  # Full pipeline (train → page index → chunk index → eval → plots)
  python run_experiment.py --mode all --qwen-model Qwen/Qwen2.5-7B-Instruct

  # Train only
  python run_experiment.py --mode train

  # Build indexes only (page-level + chunk-level for hierarchical retrieval)
  python run_experiment.py --mode index --force-reindex

  # Evaluate only (after training + indexing)
  python run_experiment.py --mode eval --no-generation

  # Just regenerate plots from saved metrics
  python run_experiment.py --mode plots

  # Ablation without HyDE (if no GPU for generation)
  python run_experiment.py --mode eval --no-hyde --no-generation

Key changes from prior run:
  • Base model: bge-m3 (was bge-base-en-v1.5) — matches the baseline model
  • Layer freezing: last 3 layers trainable → prevents catastrophic forgetting
  • Doc-filter: ChromaDB where={"doc_name": ...} per question — key improvement
  • Hierarchical index: chunk-level ChromaDB + chunk→page aggregation
  • Per-question-type metrics: metrics-generated / domain-relevant / novel-generated
  • New collection names (v2) — force-reindex to rebuild with correct embed format

Saved outputs (in domain_adapted_retrieval/results/):
  metrics/
    all_variants_metrics.json     — full metrics for all variants
    ablation_table.csv            — overall ablation (LaTeX-ready)
    ablation_by_type.csv          — per-question-type breakdown
    {variant}_metrics.json        — per-variant JSON
  predictions/
    {variant}_predictions.json    — full retrieval outputs
  plots/
    recall_at_k_curves.pdf/png
    ablation_bar_chart_k5.pdf/png
    by_question_type_k5.pdf/png   — NEW: per-type bar chart
    combined_panel_k5.pdf/png
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_experiment")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Domain-Adapted Financial RAG — full experiment pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["train", "index", "eval", "all", "plots"],
        default="all",
    )

    # Model overrides
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--ft-model-path", type=str, default=None,
                        help="Path to already fine-tuned model (skip training)")
    parser.add_argument("--qwen-model", type=str, default=None)

    # Training overrides
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--trainable-layers", type=int, default=None,
                        help="Number of transformer layers to keep trainable (0=all)")

    # Retrieval overrides
    parser.add_argument("--candidate-pages", type=int, default=None)
    parser.add_argument("--num-hypotheticals", type=int, default=None)

    # Flags
    parser.add_argument("--force-reindex", action="store_true",
                        help="Delete and rebuild ChromaDB indexes")
    parser.add_argument("--no-hyde", action="store_true")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--no-generation", action="store_true")
    parser.add_argument("--no-baseline-index", action="store_true",
                        help="Skip building the BGE-M3 baseline index")
    parser.add_argument("--no-chunk-index", action="store_true",
                        help="Skip building the chunk-level index (skips hier variant)")
    parser.add_argument("--no-doc-filter", action="store_true",
                        help="Disable doc-filtered retrieval globally")

    # Output
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def build_config(args) -> "ExperimentConfig":
    from domain_adapted_retrieval.config import ExperimentConfig

    cfg = ExperimentConfig()
    cfg.seed = args.seed

    if args.base_model:
        cfg.training.base_model = args.base_model
    if args.ft_model_path:
        cfg.training.output_model_path = args.ft_model_path
    if args.qwen_model:
        cfg.hyde.qwen_model_name = args.qwen_model
    if args.epochs:
        cfg.training.num_epochs = args.epochs
    if args.batch_size:
        cfg.training.batch_size = args.batch_size
    if args.lr:
        cfg.training.learning_rate = args.lr
    if args.trainable_layers is not None:
        cfg.training.trainable_layers = args.trainable_layers
    if args.candidate_pages:
        cfg.retrieval.candidate_pages = args.candidate_pages
    if args.num_hypotheticals:
        cfg.hyde.num_hypotheticals = args.num_hypotheticals
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.no_hyde:
        cfg.hyde.enabled = False
    if args.no_rerank:
        cfg.retrieval.enable_reranking = False
    if args.no_doc_filter:
        cfg.retrieval.doc_filter_retrieval = False

    for d in [
        cfg.output_dir,
        os.path.join(cfg.output_dir, "metrics"),
        os.path.join(cfg.output_dir, "predictions"),
        os.path.join(cfg.output_dir, "plots"),
        os.path.join(cfg.output_dir, "checkpoints"),
    ]:
        os.makedirs(d, exist_ok=True)

    return cfg


# ---------------------------------------------------------------------------
# Step 1: Train
# ---------------------------------------------------------------------------

def step_train(cfg) -> "SentenceTransformer":
    from domain_adapted_retrieval.data_prep import build_training_pairs
    from domain_adapted_retrieval.train_biencoder import train_biencoder

    logger.info("=" * 60)
    logger.info("STEP 1: Fine-tuning bi-encoder")
    logger.info(f"  Base model:       {cfg.training.base_model}")
    logger.info(f"  Trainable layers: {cfg.training.trainable_layers}")
    logger.info(f"  LR / Epochs:      {cfg.training.learning_rate} / {cfg.training.num_epochs}")
    logger.info("=" * 60)

    t0 = time.time()
    all_pairs, train_pairs, val_pairs = build_training_pairs(cfg)
    ft_model = train_biencoder(cfg, train_pairs, val_pairs)
    logger.info(f"Training completed in {(time.time()-t0)/60:.1f} min")
    return ft_model


def load_ft_model(cfg) -> "SentenceTransformer":
    from sentence_transformers import SentenceTransformer
    model_path = cfg.training.output_model_path
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Fine-tuned model not found at {model_path}. "
            "Run with --mode train first."
        )
    logger.info(f"Loading fine-tuned model from {model_path}")
    return SentenceTransformer(model_path)


def load_baseline_model() -> "SentenceTransformer":
    from sentence_transformers import SentenceTransformer
    logger.info("Loading baseline model: BAAI/bge-m3")
    return SentenceTransformer("BAAI/bge-m3")


# ---------------------------------------------------------------------------
# Step 2: Build indexes
# ---------------------------------------------------------------------------

def step_build_indexes(
    cfg,
    ft_model,
    baseline_model=None,
    force_rebuild: bool = False,
    build_baseline: bool = True,
    build_chunks: bool = True,
) -> Tuple:
    """
    Build page-level and (optionally) chunk-level ChromaDB indexes.

    Returns:
        (ft_page_col, baseline_page_col, ft_chunk_col, baseline_chunk_col)
        Any of the last three may be None if not requested.
    """
    from domain_adapted_retrieval.build_index import build_chroma_index, build_chunk_index

    logger.info("=" * 60)
    logger.info("STEP 2: Building ChromaDB indexes")
    logger.info("=" * 60)

    # Fine-tuned page index
    logger.info("Building fine-tuned page index …")
    ft_page_col = build_chroma_index(
        cfg, ft_model,
        collection_name=cfg.data.collection_name,
        force_rebuild=force_rebuild,
    )

    # Fine-tuned chunk index (for hierarchical retrieval)
    ft_chunk_col = None
    if build_chunks:
        logger.info("Building fine-tuned chunk index …")
        ft_chunk_col = build_chunk_index(
            cfg, ft_model,
            collection_name=cfg.data.chunk_collection_name,
            force_rebuild=force_rebuild,
        )

    # Baseline page index
    baseline_page_col = None
    if build_baseline and baseline_model is not None:
        logger.info("Building baseline page index …")
        baseline_page_col = build_chroma_index(
            cfg, baseline_model,
            collection_name=cfg.data.baseline_collection_name,
            force_rebuild=force_rebuild,
        )

    return ft_page_col, baseline_page_col, ft_chunk_col


# ---------------------------------------------------------------------------
# Step 3: Load auxiliary models
# ---------------------------------------------------------------------------

def load_auxiliary_models(cfg, args) -> Tuple:
    from domain_adapted_retrieval.pipeline import HyDEGenerator, CrossEncoderReranker

    hyde_gen = None
    if cfg.hyde.enabled and not args.no_hyde:
        logger.info(f"Loading HyDE generator: {cfg.hyde.qwen_model_name}")
        try:
            hyde_gen = HyDEGenerator(cfg)
        except Exception as e:
            logger.warning(f"HyDE generator failed: {e}. HyDE variants skipped.")

    reranker = None
    if cfg.retrieval.enable_reranking and not args.no_rerank:
        logger.info(f"Loading reranker: {cfg.retrieval.reranker_model}")
        try:
            reranker = CrossEncoderReranker(cfg.retrieval.reranker_model)
        except Exception as e:
            logger.warning(f"Reranker failed: {e}. Reranking variants skipped.")

    return hyde_gen, reranker


def load_generator(cfg, args) -> Optional["QwenGenerator"]:
    if args.no_generation:
        return None
    from domain_adapted_retrieval.pipeline import QwenGenerator
    logger.info(f"Loading generator: {cfg.hyde.qwen_model_name}")
    try:
        return QwenGenerator(cfg.hyde.qwen_model_name, load_in_4bit=cfg.hyde.load_in_4bit)
    except Exception as e:
        logger.warning(f"Generator failed: {e}. Generation skipped.")
        return None


# ---------------------------------------------------------------------------
# Step 4: Evaluate
# ---------------------------------------------------------------------------

def step_evaluate(
    cfg,
    ft_model,
    baseline_model,
    ft_page_col,
    baseline_page_col,
    ft_chunk_col,
    hyde_gen,
    reranker,
    generator,
) -> Dict:
    from domain_adapted_retrieval.evaluate import run_ablation

    logger.info("=" * 60)
    logger.info("STEP 3: Running ablation evaluation")
    logger.info(f"  Doc-filter: {cfg.retrieval.doc_filter_retrieval}")
    logger.info(f"  Hierarchical: {cfg.retrieval.hierarchical_retrieval}")
    logger.info("=" * 60)

    return run_ablation(
        config=cfg,
        ft_embed_model=ft_model,
        baseline_embed_model=baseline_model,
        ft_page_collection=ft_page_col,
        baseline_page_collection=baseline_page_col,
        ft_chunk_collection=ft_chunk_col,
        hyde_generator=hyde_gen,
        reranker=reranker,
        generator=generator,
        run_generation=(generator is not None),
    )


# ---------------------------------------------------------------------------
# Step 5: Visualise
# ---------------------------------------------------------------------------

def step_visualise(cfg, all_results: Dict) -> None:
    from domain_adapted_retrieval.visualize import generate_all_plots

    logger.info("=" * 60)
    logger.info("STEP 4: Generating plots")
    logger.info("=" * 60)

    generate_all_plots(cfg, all_results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = build_config(args)

    # Save run config
    cfg_snapshot = {
        "mode": args.mode,
        "base_model": cfg.training.base_model,
        "trainable_layers": cfg.training.trainable_layers,
        "ft_model_path": cfg.training.output_model_path,
        "qwen_model": cfg.hyde.qwen_model_name,
        "num_epochs": cfg.training.num_epochs,
        "learning_rate": cfg.training.learning_rate,
        "batch_size": cfg.training.batch_size,
        "candidate_pages": cfg.retrieval.candidate_pages,
        "final_k": cfg.retrieval.final_k,
        "num_hypotheticals": cfg.hyde.num_hypotheticals,
        "doc_filter_retrieval": cfg.retrieval.doc_filter_retrieval,
        "hierarchical_retrieval": cfg.retrieval.hierarchical_retrieval,
        "hyde_enabled": cfg.hyde.enabled,
        "reranking_enabled": cfg.retrieval.enable_reranking,
        "k_eval_values": cfg.k_eval_values,
        "seed": cfg.seed,
    }
    with open(os.path.join(cfg.output_dir, "run_config.json"), "w") as f:
        json.dump(cfg_snapshot, f, indent=2)

    t_total = time.time()

    # ---- plots-only mode ----
    if args.mode == "plots":
        metrics_path = os.path.join(cfg.output_dir, "metrics", "all_variants_metrics.json")
        if not os.path.exists(metrics_path):
            logger.error(f"No saved metrics at {metrics_path}")
            sys.exit(1)
        with open(metrics_path) as f:
            all_results = json.load(f)
        step_visualise(cfg, all_results)
        return

    do_train = args.mode in ("train", "all")
    do_index = args.mode in ("index", "all")
    do_eval  = args.mode in ("eval", "all")

    build_baseline = not args.no_baseline_index
    build_chunks   = not args.no_chunk_index

    # Fine-tuned model
    ft_model = step_train(cfg) if do_train else load_ft_model(cfg)

    # Baseline model
    baseline_model = None
    if (do_index or do_eval) and build_baseline:
        baseline_model = load_baseline_model()

    # Indexes
    ft_page_col = baseline_page_col = ft_chunk_col = None

    if do_index:
        ft_page_col, baseline_page_col, ft_chunk_col = step_build_indexes(
            cfg, ft_model, baseline_model,
            force_rebuild=args.force_reindex,
            build_baseline=build_baseline,
            build_chunks=build_chunks,
        )
        # After indexing the embeddings are persisted in ChromaDB; move embedding
        # models off the GPU so the next stage (HyDE pre-gen) gets a clean slate.
        if do_eval:
            import torch as _torch
            logger.info("Indexing done — freeing embedding models from GPU before eval stage.")
            if ft_model is not None:
                ft_model = ft_model.cpu()
                if hasattr(ft_model, '_target_device'):
                    ft_model._target_device = _torch.device('cpu')
            if baseline_model is not None:
                baseline_model = baseline_model.cpu()
                if hasattr(baseline_model, '_target_device'):
                    baseline_model._target_device = _torch.device('cpu')
            _torch.cuda.empty_cache()
            logger.info("GPU memory cleared after indexing.")
    elif do_eval:
        from domain_adapted_retrieval.build_index import load_chroma_index
        ft_page_col = load_chroma_index(cfg, cfg.data.collection_name)
        if build_chunks:
            try:
                ft_chunk_col = load_chroma_index(cfg, cfg.data.chunk_collection_name)
            except Exception:
                logger.warning("Chunk index not found — hierarchical variant will be skipped.")
        if build_baseline:
            try:
                baseline_page_col = load_chroma_index(cfg, cfg.data.baseline_collection_name)
            except Exception:
                logger.warning("Baseline index not found — using ft index for baseline variants.")
                baseline_page_col = ft_page_col

    # Evaluation
    if do_eval:
        # --- Memory-efficient model loading strategy ---
        # Peak VRAM without this: ft_model + baseline_model + Qwen 7B + reranker ≈ 14 GB
        # With this: each stage uses at most ~5 GB
        #
        # Step A: Pre-generate all HyDE hypotheticals with Qwen alone on GPU.
        #         Embedding models are freed from GPU first so Qwen has headroom.
        # Step B: Qwen frees itself after pre-generation.
        # Step C: Reload embedding models for retrieval (reranker also loads here).

        from domain_adapted_retrieval.pipeline import HyDEGenerator, CrossEncoderReranker
        hyde_gen = None
        if cfg.hyde.enabled and not args.no_hyde:
            import json as _json
            # Load the questions we'll evaluate so we can pre-generate for all of them
            _questions = []
            try:
                with open(cfg.data.financebench_data_path) as _fh:
                    for _line in _fh:
                        _line = _line.strip()
                        if _line:
                            _q = _json.loads(_line).get("question", "")
                            if _q:
                                _questions.append(_q)
            except Exception as _e:
                logger.warning(f"Could not pre-load questions for HyDE pre-gen: {_e}")

            if _questions:
                # Free embedding models from GPU so Qwen can load without competing
                import torch
                logger.info("Freeing embedding models from GPU before HyDE pre-generation…")
                if ft_model is not None:
                    ft_model = ft_model.cpu()
                    if hasattr(ft_model, '_target_device'):
                        ft_model._target_device = torch.device('cpu')
                if baseline_model is not None:
                    baseline_model = baseline_model.cpu()
                    if hasattr(baseline_model, '_target_device'):
                        baseline_model._target_device = torch.device('cpu')
                torch.cuda.empty_cache()
                logger.info("GPU freed. Loading HyDE generator (Qwen only in VRAM)…")

                try:
                    hyde_gen = HyDEGenerator(cfg)
                    # Pre-generate for all questions → frees Qwen after completion
                    hyde_gen.pre_generate_all(_questions)
                except Exception as _e:
                    logger.warning(f"HyDE pre-generation failed: {_e}. HyDE variants skipped.")
                    hyde_gen = None

                # Move embedding models back to GPU for retrieval
                logger.info("Reloading embedding models to GPU for retrieval…")
                if ft_model is not None:
                    ft_model = ft_model.cuda()
                    if hasattr(ft_model, '_target_device'):
                        ft_model._target_device = torch.device('cuda')
                if baseline_model is not None:
                    baseline_model = baseline_model.cuda()
                    if hasattr(baseline_model, '_target_device'):
                        baseline_model._target_device = torch.device('cuda')
                torch.cuda.empty_cache()

        reranker = None
        if cfg.retrieval.enable_reranking and not args.no_rerank:
            logger.info(f"Loading reranker: {cfg.retrieval.reranker_model}")
            try:
                reranker = CrossEncoderReranker(cfg.retrieval.reranker_model)
            except Exception as _e:
                logger.warning(f"Reranker failed: {_e}. Reranking variants skipped.")

        generator = load_generator(cfg, args) if not args.no_generation else None

        _baseline_page_col = baseline_page_col if baseline_page_col is not None else ft_page_col
        _baseline_model    = baseline_model    if baseline_model    is not None else ft_model

        all_results = step_evaluate(
            cfg,
            ft_model=ft_model,
            baseline_model=_baseline_model,
            ft_page_col=ft_page_col,
            baseline_page_col=_baseline_page_col,
            ft_chunk_col=ft_chunk_col,
            hyde_gen=hyde_gen,
            reranker=reranker,
            generator=generator,
        )
        step_visualise(cfg, all_results)

        # Summary
        best_key = "ft_global_hyde_rerank"
        if best_key in all_results:
            best = all_results[best_key]
            pr5 = best.get("page_recall@5", 0)
            dr5 = best.get("doc_recall@5", 0)
            logger.info(
                f"\n{'='*50}\n"
                f"BEST METHOD (FT + HyDE + ReRanker, global):\n"
                f"  DocRec@5  = {dr5:.4f}\n"
                f"  PageRec@5 = {pr5:.4f}  (target ≥ 0.50)\n"
                f"  {'Target reached!' if pr5 >= 0.50 else 'Below target'}\n"
                f"{'='*50}"
            )

        # Show FT gain vs global baseline
        if "baseline_global" in all_results and "ft_global" in all_results:
            pr_base = all_results["baseline_global"].get("page_recall@5", 0)
            pr_ft   = all_results["ft_global"].get("page_recall@5", 0)
            logger.info(
                f"FT gain (global): "
                f"PageRec@5 {pr_base:.3f} → {pr_ft:.3f} "
                f"(+{pr_ft - pr_base:.3f})"
            )

    elapsed_total = (time.time() - t_total) / 60
    logger.info(f"\nTotal wall-clock time: {elapsed_total:.1f} min")
    logger.info(f"Results saved in: {cfg.output_dir}")


if __name__ == "__main__":
    main()
