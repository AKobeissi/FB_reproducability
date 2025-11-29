#!/usr/bin/env python3
"""
Post-hoc evaluation utility.

Usage:
  python posthoc_evaluator.py --input outputs/single_vector_20251118_174111.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate FinanceBench RAG evaluation metrics from saved JSON results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to a JSON results file produced by rag_* experiments.",
    )
    parser.add_argument(
        "--save-report",
        "-o",
        default=None,
        help="Optional path to write the aggregated summary as JSON.",
    )
    parser.add_argument(
        "--top-n-failure-modes",
        type=int,
        default=5,
        help="Number of retrieval failure modes to display.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def load_results(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data, {}

    if isinstance(data, dict):
        if "results" in data and isinstance(data["results"], list):
            return data["results"], data.get("metadata", {})
        # treat dict with integer keys as list-like
        if all(isinstance(k, str) and k.isdigit() for k in data.keys()):
            ordered = [data[k] for k in sorted(data.keys(), key=int)]
            return ordered, {}

    raise ValueError(
        "Unsupported JSON structure. Expected either a list or a dict with a 'results' field."
    )


def summarize(values: Iterable[float]) -> Dict[str, float]:
    vals = [v for v in values if isinstance(v, (int, float))]
    if not vals:
        return {}
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def flatten_generation_metrics(samples: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, List[float]] = defaultdict(list)
    ragas_buckets: Dict[str, List[float]] = defaultdict(list)
    judge_scores: List[float] = []
    judge_correct_flags: List[bool] = []

    for sample in samples:
        gen = sample.get("generation_evaluation") or {}
        for family in ("bleu", "rouge", "bertscore"):
            metrics = gen.get(family) or {}
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    buckets[f"{family}.{key}"].append(value)

        ragas_scores = gen.get("ragas") or {}
        for metric_name, value in ragas_scores.items():
            if isinstance(value, (int, float)):
                ragas_buckets[metric_name].append(value)

        judge = gen.get("llm_judge") or {}
        score = judge.get("score")
        if isinstance(score, (int, float)):
            judge_scores.append(score)
        correct = judge.get("correct")
        if isinstance(correct, bool):
            judge_correct_flags.append(correct)

    summary = {name: summarize(values) for name, values in buckets.items()}
    ragas_summary = {f"ragas.{name}": summarize(values) for name, values in ragas_buckets.items()}
    judge_summary: Dict[str, Any] = {}
    if judge_scores:
        judge_summary["llm_judge.score"] = summarize(judge_scores)
    if judge_correct_flags:
        judge_summary["llm_judge.accuracy"] = float(np.mean(judge_correct_flags))

    summary.update(ragas_summary)
    summary.update(judge_summary)
    return summary


def flatten_retrieval_metrics(
    samples: List[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, int]]:
    buckets: Dict[str, List[float]] = defaultdict(list)
    binary_keys = {
        "doc_hit_at_k",
        "page_hit_at_k",
        "chunk_hit_at_k",
    }
    failure_counter: Counter[str] = Counter()
    for sample in samples:
        retrieval = sample.get("retrieval_evaluation") or {}
        for key, value in retrieval.items():
            if key in ("failure_reason",):
                if isinstance(value, str):
                    failure_counter[value] += 1
                continue
            if key in binary_keys and isinstance(value, bool):
                buckets[key].append(float(value))
                continue
            if isinstance(value, (int, float)):
                buckets[key].append(float(value))
    summary = {name: summarize(values) for name, values in buckets.items()}
    return summary, dict(failure_counter)


def summarize_lengths(samples: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    gen_lengths = [len(sample.get("generated_answer") or "") for sample in samples]
    context_lengths = [sample.get("context_length") for sample in samples if sample.get("context_length") is not None]
    summary = {"generated_answer.length": summarize(gen_lengths)}
    if context_lengths:
        summary["context.length"] = summarize(context_lengths)
    return summary


def print_section(title: str):
    print("\n" + title)
    print("-" * len(title))


def render_summary(summary: Dict[str, Dict[str, float]]):
    for metric, stats in sorted(summary.items()):
        if not stats:
            continue
        mean = stats.get("mean")
        std = stats.get("std")
        min_ = stats.get("min")
        max_ = stats.get("max")
        parts = []
        if mean is not None:
            parts.append(f"mean={mean:.4f}")
        if std is not None:
            parts.append(f"std={std:.4f}")
        if min_ is not None and max_ is not None:
            parts.append(f"range=[{min_:.4f}, {max_:.4f}]")
        print(f"  {metric:30s} {' | '.join(parts)}")


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s - %(message)s",
    )

    samples, metadata = load_results(Path(args.input))
    if not samples:
        raise ValueError("No samples found in the provided JSON file.")

    print("=" * 80)
    print(f"Results file : {args.input}")
    if metadata:
        print("Metadata     :")
        for key, value in metadata.items():
            print(f"  - {key}: {value}")
    print(f"Samples      : {len(samples)}")
    print("=" * 80)

    length_summary = summarize_lengths(samples)
    gen_summary = flatten_generation_metrics(samples)
    retrieval_summary, failure_modes = flatten_retrieval_metrics(samples)

    print_section("Generation Metrics")
    render_summary(gen_summary)

    print_section("Answer / Context Lengths")
    render_summary(length_summary)

    if retrieval_summary:
        print_section("Retrieval Metrics")
        render_summary(retrieval_summary)

    if failure_modes:
        print_section("Retrieval Failure Modes")
        total_failures = sum(failure_modes.values())
        for reason, count in Counter(failure_modes).most_common(args.top_n_failure_modes):
            pct = (count / total_failures) * 100 if total_failures else 0.0
            print(f"  {reason:30s} {count:5d} ({pct:5.1f}%)")

    report = {
        "samples": len(samples),
        "metadata": metadata,
        "generation_metrics": gen_summary,
        "length_metrics": length_summary,
        "retrieval_metrics": retrieval_summary,
        "retrieval_failure_modes": failure_modes,
    }

    if args.save_report:
        report_path = Path(args.save_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
