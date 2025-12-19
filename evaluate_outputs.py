#!/usr/bin/env python3
"""
Batch evaluator for FinanceBench experiment outputs.

Example:
    python evaluate_outputs.py outputs/*.json \
        --judge-model meta-llama/Meta-Llama-3-8B-Instruct \
        --output-dir outputs/scored
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from pathlib import Path

# Set memory management env var to avoid fragmentation OOM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

try:
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover - optional dependency
    ChatOpenAI = None

try:
    from langchain_community.llms import HuggingFacePipeline as LangchainHFPipeline  # type: ignore
except Exception:  # pragma: no cover
    try:
        from langchain_huggingface import HuggingFacePipeline as LangchainHFPipeline  # type: ignore
    except Exception:
        LangchainHFPipeline = None

from .evaluator import Evaluator


class OpenAIChatPipeline:
    """Adapter that mimics the transformers pipeline interface for OpenAI chat models."""

    def __init__(self, client: OpenAI, model_id: str):
        self._client = client
        self._model_id = model_id
        self.tokenizer = SimpleNamespace(eos_token_id=None)

    def __call__(self, prompt: str, max_new_tokens: int = 200, **_: Any) -> List[Dict[str, str]]:
        response = self._client.chat.completions.create(
            model=self._model_id,
            messages=[
                {"role": "system", "content": "You are an impartial grader for FinanceBench answers."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=max_new_tokens,
        )
        text = response.choices[0].message.content or ""
        generated_text = f"{prompt}\n{text.strip()}"
        return [{"generated_text": generated_text}]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attach BERTScore + HF-judge metrics to saved experiment outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Paths (or glob-expanded paths) to JSON files inside ./outputs.",
    )
    parser.add_argument(
        "--output-dir",
        help="Where to write scored JSON files. Defaults to each input's directory.",
    )
    parser.add_argument(
        "--suffix",
        default="_scored",
        help="Suffix injected before .json when --overwrite is not set.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the original JSON file with the scored version.",
    )
    parser.add_argument(
        "--judge-model",
        default="openai/gpt-oss-20b",
        help="HuggingFace model id to use as the LLM judge.",
    )
    parser.add_argument(
        "--judge-provider",
        choices=["huggingface", "openai"],
        default="huggingface",
        help="Backend for the LLM judge. 'openai' expects a valid API key.",
    )
    parser.add_argument(
        "--judge-dtype",
        default="float16",
        help="Torch dtype (e.g., float16, bfloat16, float32) for the judge model.",
    )
    parser.add_argument(
        "--judge-max-new-tokens",
        type=int,
        default=200,
        help="Max new tokens to sample when the judge produces its rationale.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Device map hint passed to transformers (auto, cuda, cpu, cuda:0, etc.).",
    )
    parser.add_argument(
        "--max-gpu-memory",
        default=None,
        help="Max memory to use per GPU (e.g. '10GiB'). Helps with OOM by forcing offload.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow loading models that require custom code (HF trust_remote_code).",
    )
    parser.add_argument(
        "--openai-api-base",
        default="https://api.openai.com/v1",
        help="Base URL for OpenAI-compatible endpoints (judge + embeddings).",
    )
    parser.add_argument(
        "--openai-api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable that stores the OpenAI-compatible API key.",
    )
    parser.add_argument(
        "--no-bertscore",
        action="store_true",
        help="Skip BERTScore if resources are tight (not recommended).",
    )
    parser.add_argument(
        "--skip-llm-judge",
        action="store_true",
        help="Disable the judge entirely (overrides --judge-model).",
    )
    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=None,
        help="If set, compute retrieval diagnostics up to this k per sample.",
    )
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help="Disable RAGAS holistic metrics.",
    )
    parser.add_argument(
        "--ragas-embedding-model",
        default=None,
        help="Optional override for the embedding model used by RAGAS.",
    )
    parser.add_argument(
        "--ragas-device",
        default=None,
        help="Optional device hint (cpu/cuda) for the RAGAS embedding encoder.",
    )
    parser.add_argument(
        "--ragas-llm-provider",
        choices=["auto", "huggingface", "openai"],
        default="auto",
        help="Provider for the LangChain LLM fed to RAGAS (defaults to judge provider).",
    )
    parser.add_argument(
        "--ragas-llm-model",
        default=None,
        help="LLM identifier for RAGAS (defaults to --judge-model).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce log verbosity.",
    )
    return parser.parse_args()


def resolve_dtype(name: Optional[str]) -> Optional[torch.dtype]:
    if not name:
        return None
    if not hasattr(torch, name):
        raise ValueError(f"Unknown torch dtype: {name}")
    return getattr(torch, name)


def build_judge_pipeline(
    model_id: str,
    device_map: Optional[str],
    dtype_name: Optional[str],
    trust_remote_code: bool,
    provider: str = "huggingface",
    openai_api_key_env: str = "OPENAI_API_KEY",
    openai_api_base: str = "https://api.openai.com/v1",
    max_gpu_memory: Optional[str] = None,
) -> Any:
    if provider == "openai":
        return build_openai_judge_pipeline(
            model_id=model_id,
            api_key_env=openai_api_key_env,
            api_base=openai_api_base,
        )
    torch_dtype = resolve_dtype(dtype_name)
    model_kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    if device_map:
        model_kwargs["device_map"] = device_map
    if max_gpu_memory:
        model_kwargs["max_memory"] = {0: max_gpu_memory, "cpu": "200GiB"}

    logging.info("Loading judge model %s (dtype=%s, device_map=%s)", model_id, dtype_name, device_map)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        **model_kwargs,
    )
    return pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,
    )


def build_openai_judge_pipeline(model_id: str, api_key_env: str, api_base: str) -> Any:
    if OpenAI is None:
        raise RuntimeError("The 'openai' package is required for --judge-provider openai.")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Missing API key for OpenAI judge. Set the '{api_key_env}' environment variable."
        )
    logging.info("Using OpenAI judge model %s (base=%s)", model_id, api_base)
    client = OpenAI(api_key=api_key, base_url=api_base)
    return OpenAIChatPipeline(client, model_id)


def build_ragas_langchain_llm(
    provider: str,
    model_id: str,
    *,
    openai_api_key_env: str,
    openai_api_base: str,
    hf_pipeline: Optional[Any],
    hf_builder_params: Optional[Dict[str, Any]],
) -> Optional[Any]:
    if provider == "openai":
        if ChatOpenAI is None:
            logging.warning("langchain-openai is not installed; cannot create RAGAS LLM.")
            return None
        api_key = os.environ.get(openai_api_key_env)
        if not api_key:
            logging.warning(
                "OpenAI API key not found in %s; skipping RAGAS LLM.", openai_api_key_env
            )
            return None
        try:
            return ChatOpenAI(
                model=model_id,
                api_key=api_key,
                base_url=openai_api_base,
                temperature=0.0,
                max_tokens=None,
            )
        except Exception as exc:
            logging.warning("Failed to initialize ChatOpenAI for RAGAS: %s", exc)
            return None

    if provider == "huggingface":
        if LangchainHFPipeline is None:
            logging.warning(
                "LangChain HuggingFacePipeline not available; cannot wrap HF model for RAGAS."
            )
            return None
        pipeline_obj = hf_pipeline
        if pipeline_obj is None:
            if hf_builder_params is None:
                logging.warning("No HF pipeline or builder parameters supplied for RAGAS.")
                return None
            pipeline_obj = build_judge_pipeline(
                provider="huggingface",
                openai_api_key_env=openai_api_key_env,
                openai_api_base=openai_api_base,
                **hf_builder_params,
            )
        try:
            return LangchainHFPipeline(pipeline=pipeline_obj)
        except Exception as exc:
            logging.warning("Failed to wrap HuggingFace pipeline for RAGAS: %s", exc)
            return None

    if provider not in {"auto", None}:
        logging.warning("Unsupported RAGAS LLM provider '%s'; skipping RAGAS LLM.", provider)
    return None


def load_results(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, list):
        container = {"results": data}
        return data, {}, container

    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return data["results"], data.get("metadata", {}), data

    raise ValueError(f"Unsupported JSON schema in {path}")


def extract_contexts(chunks: Optional[Sequence[Dict[str, Any]]]) -> List[str]:
    contexts: List[str] = []
    if not chunks:
        return contexts
    for chunk in chunks:
        text = chunk.get("text")
        if text:
            contexts.append(text)
    return contexts


def extract_gold_contexts(sample: Dict[str, Any]) -> List[str]:
    contexts: List[str] = []
    segments = sample.get("gold_evidence_segments")
    if isinstance(segments, list):
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            text = seg.get("text") or seg.get("evidence_text") or seg.get("evidence_text_full_page")
            if text:
                contexts.append(text)
    elif isinstance(segments, dict):
        text = segments.get("text") or segments.get("evidence_text")
        if text:
            contexts.append(text)
    else:
        legacy = sample.get("gold_evidence")
        if isinstance(legacy, str) and legacy.strip():
            contexts.append(legacy)
    return contexts


def compute_retrieval_metrics_if_available(
    evaluator: Evaluator,
    sample: Dict[str, Any],
    top_k: Optional[int],
) -> Optional[Dict[str, Any]]:
    if top_k is None:
        return None
    retrieved = sample.get("retrieved_chunks") or []
    gold = sample.get("gold_evidence_segments") or []
    if not retrieved or not gold:
        return None
    return evaluator.compute_retrieval_metrics(retrieved, gold, top_k=top_k)


def evaluate_samples(
    evaluator: Evaluator,
    samples: List[Dict[str, Any]],
    retrieval_top_k: Optional[int],
    ragas_llm: Any = None,
) -> List[Dict[str, Any]]:
    for idx, sample in enumerate(samples):
        prediction = (sample.get("generated_answer") or "").strip()
        reference = (sample.get("reference_answer") or "").strip()
        question = sample.get("question") or ""
        contexts = extract_contexts(sample.get("retrieved_chunks"))
        gold_contexts = extract_gold_contexts(sample)

        metrics = evaluator.evaluate_generation(
            prediction=prediction,
            reference=reference,
            question=question,
            contexts=contexts,
            gold_contexts=gold_contexts,
            langchain_llm=ragas_llm,
        )
        sample["generation_evaluation"] = metrics

        retrieval_metrics = compute_retrieval_metrics_if_available(
            evaluator,
            sample,
            retrieval_top_k,
        )
        if retrieval_metrics:
            sample["retrieval_evaluation"] = retrieval_metrics

        logging.debug("Scored sample %s", idx)

    return samples


def summarize_results(
    evaluator: Evaluator,
    samples: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not samples:
        return {}
    return evaluator.summarize_experiment(samples)


def determine_output_path(
    source: Path,
    output_dir: Optional[str],
    suffix: str,
    overwrite: bool,
) -> Path:
    if overwrite:
        return source
    target_dir = Path(output_dir) if output_dir else source.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / f"{source.stem}{suffix}{source.suffix}"


def format_float(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def render_summary_line(
    path: Path,
    num_samples: int,
    summary: Dict[str, Any],
) -> str:
    overall = summary.get("overall", {})
    parts = [
        f"{path.name}:",
        f"samples={num_samples}",
        f"BLEU4={format_float(overall.get('bleu_4_mean'))}",
        f"ROUGEL={format_float(overall.get('rouge_l_f1_mean'))}",
        f"BERTScore={format_float(overall.get('bertscore_f1_mean'))}",
        f"JudgeAcc={format_float(overall.get('llm_judge_accuracy'))}",
    ]
    return " | ".join(parts)


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s - %(message)s",
    )

    try:
        import triton
        logging.info(f"Triton version {triton.__version__} is available.")
    except ImportError:
        logging.warning("Triton is NOT installed or cannot be imported. MXFP4 models will fail or fallback to slow/heavy dequantization.")
    except Exception as e:
        logging.warning(f"Error checking Triton: {e}")

    evaluator = Evaluator(
        use_bertscore=not args.no_bertscore,
        use_llm_judge=not args.skip_llm_judge,
        use_ragas=not args.skip_ragas,
        ragas_embedding_model=args.ragas_embedding_model,
        ragas_device=args.ragas_device,
    )

    judge_pipeline = None
    if evaluator.use_llm_judge:
        judge_pipeline = build_judge_pipeline(
            model_id=args.judge_model,
            device_map=args.device_map,
            dtype_name=args.judge_dtype,
            trust_remote_code=args.trust_remote_code,
            provider=args.judge_provider,
            openai_api_key_env=args.openai_api_key_env,
            openai_api_base=args.openai_api_base,
            max_gpu_memory=args.max_gpu_memory,
        )
        evaluator.set_judge_pipeline(judge_pipeline, max_new_tokens=args.judge_max_new_tokens)

    ragas_llm = None
    if evaluator.use_ragas:
        ragas_provider = args.ragas_llm_provider
        if ragas_provider == "auto":
            ragas_provider = args.judge_provider if judge_pipeline is not None else "openai"
        ragas_model = args.ragas_llm_model or args.judge_model
        hf_pipeline_for_ragas: Optional[Any] = None
        if ragas_provider == "huggingface" and args.judge_provider == "huggingface":
            hf_pipeline_for_ragas = judge_pipeline
        hf_builder_params = (
            {
                "model_id": ragas_model,
                "device_map": args.device_map,
                "dtype_name": args.judge_dtype,
                "trust_remote_code": args.trust_remote_code,
            }
            if ragas_provider == "huggingface"
            else None
        )
        ragas_llm = build_ragas_langchain_llm(
            provider=ragas_provider,
            model_id=ragas_model,
            openai_api_key_env=args.openai_api_key_env,
            openai_api_base=args.openai_api_base,
            hf_pipeline=hf_pipeline_for_ragas,
            hf_builder_params=hf_builder_params,
        )

    scored_any = False
    for input_pattern in args.inputs:
        if any(ch in input_pattern for ch in "*?[]"):
            paths = sorted(Path(p) for p in glob.glob(input_pattern))
        else:
            paths = [Path(input_pattern)]
        for path in paths:
            if not path.is_file():
                logging.warning("Skipping %s (not a file)", path)
                continue

            samples, _metadata, container = load_results(path)
            logging.info("Scoring %s (%d samples)", path, len(samples))
            evaluated_samples = evaluate_samples(evaluator, samples, args.retrieval_top_k, ragas_llm=ragas_llm)
            summary = summarize_results(evaluator, evaluated_samples)
            container["evaluation_summary"] = summary

            target_path = determine_output_path(path, args.output_dir, args.suffix, args.overwrite)
            action = "Overwriting" if target_path == path else "Writing"
            logging.info("%s %s", action, target_path)
            target_path.write_text(json.dumps(container, indent=2), encoding="utf-8")
            scored_any = True

            print(render_summary_line(target_path, len(samples), summary))

    if not scored_any:
        raise SystemExit("No files were scored. Check your --inputs pattern.")


if __name__ == "__main__":
    main()