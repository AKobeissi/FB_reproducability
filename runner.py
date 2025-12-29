"""
Simple runner to execute FinanceBench-style experiments from the command line.

Usage examples:
  python runner.py both closed
  python runner.py llama --experiment single --num-samples 50
  python runner.py qwen open --output-dir ./results
"""
from __future__ import annotations

import argparse
import logging
import sys
import gc
from pathlib import Path

# Handle import depending on whether run as module or script. Be explicit about errors
# and try both absolute and relative imports so the script works when executed from
# the project directory or as a package.
RAGExperiment = None
_import_errors = []
try:
    # Try package-relative import first (works when running via `python -m FB_reproducability.runner`)
    if __package__:
        try:
            from .rag_experiments import RAGExperiment  # type: ignore
        except Exception as e:
            _import_errors.append(("package-relative", e))
            raise
    else:
        raise ImportError("no package context")
except Exception:
    # Fall back to absolute/top-level import (works when running `python runner.py` from package dir)
    try:
        from rag_experiments import RAGExperiment
    except Exception as e:
        _import_errors.append(("absolute", e))

        # As a last attempt, try importing using the package name (useful if package name is different)
        try:
            import importlib
            pkg = __package__ or Path(__file__).resolve().parent.name
            mod = importlib.import_module(f"{pkg}.rag_experiments")
            RAGExperiment = getattr(mod, "RAGExperiment")
        except Exception as e2:
            _import_errors.append(("pkg-qualified", e2))

if RAGExperiment is None:
    print("ERROR: Cannot import RAGExperiment")
    for k, err in _import_errors:
        print(f"Import attempt [{k}] failed: {err}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run FinanceBench RAG experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    script_dir = Path(__file__).resolve().parent
    default_vector_store_dir = str(script_dir / "vector_stores")
    default_output_dir = str(script_dir / "outputs")
    
    parser.add_argument(
        "model",
        nargs="?",
        default="both",
        choices=["llama", "qwen", "both"],
        help="Model to run"
    )
    
    parser.add_argument(
        "experiment",
        nargs="?",
        default="closed",
        choices=["closed", "single", "random_single", "shared", "open"],
        help="Experiment type"
    )
    
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to process (default: all)"
    )
    
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default=None,
        help="Local PDF directory (default: ./pdfs)"
    )
    
    parser.add_argument(
        "--vector-store-dir",
        type=str,
        default=default_vector_store_dir,
        help="Vector store directory"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=default_output_dir,
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--no-8bit",
        action="store_true",
        help="Disable 8-bit quantization"
    )
    
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--use-api",
        action="store_true",
        help="Route generation through the OpenAI-compatible API client instead of local weights"
    )
    parser.add_argument(
        "--api-base-url",
        type=str,
        default="https://api.openai.com/v1",
        help="Base URL for the OpenAI-compatible API (use HF router or other endpoints if needed)"
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable that holds the API key for --use-api runs"
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="Override the built-in model mapping with an explicit model id (e.g., gpt-4o-mini)"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="Embedding model identifier (HF or OpenAI, e.g., text-embedding-ada-002)"
    )

    # Evaluation configuration
    parser.add_argument(
        "--eval-type",
        type=str,
        default="both",
        choices=["retrieval", "generative", "both"],
        help="Type of evaluation to run"
    )
    
    parser.add_argument(
        "--eval-mode",
        type=str,
        default="static",
        choices=["static", "semantic"],
        help="Evaluation mode: 'static' (BLEU/ROUGE/BERTScore) or 'semantic' (RAGAS + LLM Judge)"
    )

    parser.add_argument(
        "--judge-model",
        type=str,
        default="openai/gpt-4o",
        help="Model to use for LLM judge (if eval-mode is semantic)"
    )
    
    return parser.parse_args()


def get_models(model_arg: str, override_model: str | None = None) -> list[tuple[str, str]]:
    """Get model configurations based on CLI argument."""
    if override_model:
        return [("custom", override_model)]
    model_map = {
        "llama": [("llama", RAGExperiment.LLAMA_3_2_3B)],
        "qwen": [("qwen", RAGExperiment.QWEN_2_5_7B)],
        "both": [
            ("llama", RAGExperiment.LLAMA_3_2_3B),
            ("qwen", RAGExperiment.QWEN_2_5_7B)
        ]
    }
    return model_map[model_arg.lower()]


def get_experiment_type(exp_arg: str) -> str:
    """Map CLI argument to experiment type constant."""
    exp_map = {
        "closed": RAGExperiment.CLOSED_BOOK,
        "single": RAGExperiment.SINGLE_VECTOR,
        "random_single": RAGExperiment.RANDOM_SINGLE,
        "shared": RAGExperiment.SHARED_VECTOR,
        "open": RAGExperiment.OPEN_BOOK,
    }
    return exp_map[exp_arg]


def main():
    """Main entry point."""
    args = parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # Set default PDF directory if not specified
    pdf_dir = args.pdf_dir
    if pdf_dir is None:
        pdf_dir = str(Path(__file__).resolve().parent / "pdfs")
    
    # Get models and experiment type
    models = get_models(args.model, override_model=args.llm_model)
    experiment_type = get_experiment_type(args.experiment)
    
    # Run experiments
    for label, model_name in models:
        print("\n" + "=" * 80)
        print(f"Experiment: {args.experiment} | Model: {label}")
        if args.num_samples:
            print(f"Samples: {args.num_samples}")
        print("=" * 80)

        exp = None
        try:
            exp = RAGExperiment(
                experiment_type=experiment_type,
                llm_model=model_name,
                embedding_model=args.embedding_model,
                pdf_local_dir=pdf_dir,
                vector_store_dir=args.vector_store_dir,
                output_dir=args.output_dir,
                load_in_8bit=not args.no_8bit,
                use_api=args.use_api,
                api_base_url=args.api_base_url,
                api_key_env=args.api_key_env,
                eval_type=args.eval_type,
                eval_mode=args.eval_mode,
                judge_model=args.judge_model,
            )
            
            exp.run_experiment(num_samples=args.num_samples)
            
        except Exception as e:
            logger.error(f"Experiment failed for {label}: {e}", exc_info=True)
        finally:
            if exp:
                del exp
            gc.collect()


if __name__ == "__main__":
    main()
