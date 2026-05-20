#!/usr/bin/env python3
"""
compute_benchmark_extension_answer_metrics.py
=============================================
Augment benchmark-extension retrieval metrics with answer-level metrics:

  - answer_rougeL      : mean ROUGE-L F1 over the evaluated subset
  - numeric_match      : mean numeric match over FinanceBench metrics-generated questions
  - numeric_match_all  : mean numeric match over all questions in the subset

The benchmark-extension prediction files already contain `generated_answer`
fields, but current retrieval-only runs leave them empty. This script can:

  1. recompute and inject answer-level metrics into the existing
     `all_variants_{global,financebench,finqa}.json` files, and
  2. optionally generate missing answers before scoring.

Typical usage:

  # Add zero/placeholder answer metrics to existing JSONs (no generation)
  venv/bin/python baselines/compute_benchmark_extension_answer_metrics.py

  # Generate answers for the FinanceBench subset of the best variant, then score
  venv/bin/python baselines/compute_benchmark_extension_answer_metrics.py \
      --variants multi_hyde_ft_reranker \
      --dataset-filter financebench \
      --generate-missing
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PREDICTIONS_DIR = PROJECT_ROOT / "baselines/results/benchmark_extension/predictions"
METRICS_DIR = PROJECT_ROOT / "baselines/results/benchmark_extension/metrics"
SUMMARY_PATH = METRICS_DIR / "benchmark_extension_answer_metrics_summary.json"

LEVEL_DATASETS = {
    "global": None,
    "financebench": "financebench",
    "finqa": "finqa",
}

MAIN_K = 5
MAX_CONTEXT_TOKENS = 3600
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

GEN_PROMPT = (
    "You are a financial analyst answering questions based on SEC filings. "
    "Use ONLY the provided context. If the context does not contain the answer, "
    "say 'I cannot determine this from the provided information.' "
    "Be concise and precise, especially for numerical answers.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("absl").setLevel(logging.ERROR)
logger = logging.getLogger("benchmark_ext_answer_metrics")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add answer-level metrics to benchmark-extension outputs."
    )
    parser.add_argument(
        "--predictions-dir",
        default=str(PREDICTIONS_DIR),
        help="Directory containing benchmark-extension prediction JSON files.",
    )
    parser.add_argument(
        "--metrics-dir",
        default=str(METRICS_DIR),
        help="Directory containing benchmark-extension metrics JSON files.",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=None,
        help="Variants to process. Defaults to all *_retrieval.json files in predictions-dir.",
    )
    parser.add_argument(
        "--dataset-filter",
        choices=["all", "financebench", "finqa"],
        default="all",
        help="Subset to generate answers for when --generate-missing is used.",
    )
    parser.add_argument(
        "--generate-missing",
        action="store_true",
        help="Generate answers for samples whose generated_answer is empty.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Causal LM to use when generating missing answers.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum generation length per answer.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on the number of samples to generate per variant.",
    )
    parser.add_argument(
        "--report-variant",
        default="multi_hyde_ft_reranker",
        help="Variant to print a concise final report for.",
    )
    parser.add_argument(
        "--report-level",
        choices=["global", "financebench", "finqa"],
        default="financebench",
        help="Subset level to print in the final report.",
    )
    return parser.parse_args()


def discover_variants(predictions_dir: Path) -> List[str]:
    variants = []
    for path in sorted(predictions_dir.glob("*_retrieval.json")):
        variants.append(path.name[: -len("_retrieval.json")])
    return variants


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def extract_numbers(text: str) -> List[float]:
    if not text:
        return []
    clean = re.sub(r"[,$]", "", str(text))
    matches = re.findall(r"-?\d*\.?\d+", clean)
    values = []
    for match in matches:
        try:
            if match in {".", "-"}:
                continue
            values.append(float(match))
        except ValueError:
            continue
    return values


def numeric_match(reference: str, prediction: str) -> float:
    ref_nums = extract_numbers(reference)
    pred_nums = extract_numbers(prediction)
    if not ref_nums or not pred_nums:
        return 0.0
    for ref_num in ref_nums:
        for pred_num in pred_nums:
            if np.isclose(ref_num, pred_num, atol=0.03, rtol=0.03):
                return 1.0
    return 0.0


def compute_answer_metrics(samples: List[Dict]) -> Dict[str, float]:
    try:
        from rouge_score import rouge_scorer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "rouge-score is required. Run this script from the project venv."
        ) from exc

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    rouge_scores: List[float] = []
    metrics_numeric: List[float] = []
    all_numeric: List[float] = []
    answered = 0

    for sample in samples:
        reference = sample.get("reference_answer", "") or ""
        prediction = (
            sample.get("generated_answer")
            or sample.get("predicted_answer")
            or ""
        ).strip()
        if prediction:
            answered += 1

        if reference and prediction:
            rouge = scorer.score(reference, prediction)["rougeL"].fmeasure
        else:
            rouge = 0.0
        rouge_scores.append(rouge)

        if reference:
            match = numeric_match(reference, prediction)
            all_numeric.append(match)
            if str(sample.get("question_type", "")).strip().lower() == "metrics-generated":
                metrics_numeric.append(match)

    return {
        "answer_rougeL": float(np.mean(rouge_scores)) if rouge_scores else 0.0,
        "numeric_match": float(np.mean(metrics_numeric)) if metrics_numeric else 0.0,
        "numeric_match_all": float(np.mean(all_numeric)) if all_numeric else 0.0,
        "n_answered": answered,
        "n_samples": len(samples),
        "n_metrics_qs": len(metrics_numeric),
    }


def filter_samples(samples: List[Dict], dataset: Optional[str]) -> List[Dict]:
    if dataset is None:
        return list(samples)
    return [sample for sample in samples if sample.get("dataset") == dataset]


def group_samples(samples: List[Dict], key_fn) -> Dict[str, List[Dict]]:
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for sample in samples:
        groups[key_fn(sample)].append(sample)
    return groups


def update_entry_with_answer_metrics(entry: Dict, samples: List[Dict]) -> None:
    entry.setdefault("overall", {}).update(compute_answer_metrics(samples))

    by_question_type = entry.setdefault("by_question_type", {})
    for key, group in group_samples(samples, lambda s: s.get("question_type", "unknown")).items():
        by_question_type.setdefault(key, {}).update(compute_answer_metrics(group))

    by_doc_type = entry.setdefault("by_doc_type", {})
    for key, group in group_samples(samples, lambda s: s.get("doc_type", "unknown")).items():
        by_doc_type.setdefault(key, {}).update(compute_answer_metrics(group))

    by_qt_dt = entry.setdefault("by_question_type_x_doc_type", {})
    for key, group in group_samples(
        samples,
        lambda s: f"{s.get('question_type', 'unknown')}|{s.get('doc_type', 'unknown')}",
    ).items():
        by_qt_dt.setdefault(key, {}).update(compute_answer_metrics(group))

    entry["n"] = len(samples)


def build_context(sample: Dict, tokenizer) -> str:
    chunks = sample.get("retrieved_chunks", [])[:MAIN_K]
    context_parts: List[str] = []
    budget = MAX_CONTEXT_TOKENS

    for chunk in chunks:
        text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
        if not text or budget <= 0:
            continue
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= budget:
            context_parts.append(text)
            budget -= len(token_ids)
        elif budget >= 50:
            context_parts.append(
                tokenizer.decode(token_ids[:budget], skip_special_tokens=True)
            )
            budget = 0

    return "\n\n".join(context_parts)


def generate_missing_answers(
    samples: List[Dict],
    dataset_filter: str,
    model_name: str,
    max_new_tokens: int,
    max_samples: Optional[int],
) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    target_dataset = None if dataset_filter == "all" else dataset_filter
    candidates = [
        sample
        for sample in samples
        if (target_dataset is None or sample.get("dataset") == target_dataset)
        and not (sample.get("generated_answer") or "").strip()
    ]
    if max_samples is not None:
        candidates = candidates[:max_samples]

    if not candidates:
        return 0

    logger.info(
        "Loading %s to generate %d missing answers (%s subset)",
        model_name,
        len(candidates),
        dataset_filter,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    generated = 0
    for sample in candidates:
        context = build_context(sample, tokenizer)
        prompt = GEN_PROMPT.format(context=context, question=sample.get("question", ""))
        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, "apply_chat_template"):
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            formatted = prompt

        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        )
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

        try:
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            prompt_len = inputs["input_ids"].shape[-1]
            sample["generated_answer"] = tokenizer.decode(
                output[0][prompt_len:],
                skip_special_tokens=True,
            ).strip()
            generated += 1
        except Exception as exc:  # pragma: no cover
            logger.warning("Generation failed for %s: %s", sample.get("id", "?"), exc)
            sample["generated_answer"] = ""

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return generated


def main() -> None:
    args = parse_args()
    predictions_dir = Path(args.predictions_dir)
    metrics_dir = Path(args.metrics_dir)

    variants = args.variants or discover_variants(predictions_dir)
    if not variants:
        raise FileNotFoundError(f"No *_retrieval.json files found in {predictions_dir}")

    metrics_by_level = {
        level: load_json(metrics_dir / f"all_variants_{level}.json")
        for level in LEVEL_DATASETS
    }
    summary: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    total_generated = 0

    for variant in variants:
        prediction_path = predictions_dir / f"{variant}_retrieval.json"
        if not prediction_path.exists():
            logger.warning("Skipping missing predictions file: %s", prediction_path)
            continue

        samples = load_json(prediction_path)
        if not isinstance(samples, list):
            raise ValueError(f"Expected a list in {prediction_path}")

        generated_now = 0
        if args.generate_missing:
            generated_now = generate_missing_answers(
                samples=samples,
                dataset_filter=args.dataset_filter,
                model_name=args.model_name,
                max_new_tokens=args.max_new_tokens,
                max_samples=args.max_samples,
            )
            if generated_now:
                save_json(prediction_path, samples)
                logger.info(
                    "[%s] saved %d new generated answers → %s",
                    variant,
                    generated_now,
                    prediction_path,
                )
        total_generated += generated_now

        for level, dataset in LEVEL_DATASETS.items():
            subset = filter_samples(samples, dataset)
            if not subset:
                continue

            metrics_by_level[level].setdefault(variant, {})
            update_entry_with_answer_metrics(metrics_by_level[level][variant], subset)
            summary[variant][level] = dict(metrics_by_level[level][variant]["overall"])

    for level, data in metrics_by_level.items():
        save_json(metrics_dir / f"all_variants_{level}.json", data)

    save_json(SUMMARY_PATH, summary)

    report = summary.get(args.report_variant, {}).get(args.report_level)
    if report:
        logger.info(
            "[report] %s / %s → ROUGE-L=%.4f  NumMatch(metrics)=%.4f  NumMatch(all)=%.4f  answered=%d/%d",
            args.report_variant,
            args.report_level,
            report.get("answer_rougeL", 0.0),
            report.get("numeric_match", 0.0),
            report.get("numeric_match_all", 0.0),
            report.get("n_answered", 0),
            report.get("n_samples", 0),
        )

    logger.info("Done. Total newly generated answers: %d", total_generated)


if __name__ == "__main__":
    main()
