#!/usr/bin/env python3
"""
Aggregate evaluation metrics from FB_reproducability result files.

Scans:
- final_results.json (kfold experiments)
- outputs/results/unified/**/*.json

Outputs 4 CSVs:
1) overall
2) by doc_type
3) by question_type
4) by question_reasoning
"""

import argparse
import csv
import json
import logging
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = SCRIPT_ROOT / "src" / "evaluation"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

try:
    from evaluator import Evaluator
except Exception as exc:  # pragma: no cover
    Evaluator = None
    logger.warning("Failed to import Evaluator: %s", exc)


MAX_CONTEXT_CHUNKS = 5


def normalize_doc_name(doc_name: str) -> str:
    if not doc_name:
        return ""
    name = doc_name.lower()
    name = re.sub(r"\.pdf$", "", name)
    name = re.sub(r"[^a-z0-9]", "", name)
    return name


def extract_doc_type_from_name(doc_name: str) -> str:
    if not doc_name:
        return "unknown"
    doc_lower = doc_name.lower()
    if "10-k" in doc_lower or "10k" in doc_lower:
        return "10-K"
    if "10-q" in doc_lower or "10q" in doc_lower:
        return "10-Q"
    if "8-k" in doc_lower or "8k" in doc_lower:
        return "8-K"
    if "earnings" in doc_lower or "earning" in doc_lower:
        return "earnings"
    return "other"


def compute_numeric_match(reference: str, hypothesis: str) -> bool:
    if not reference or not hypothesis:
        return False
    pattern = r"\$?\d+(?:,\d{3})*(?:\.\d+)?%?"
    ref_numbers = {n.replace("$", "").replace(",", "") for n in re.findall(pattern, reference)}
    hyp_numbers = {n.replace("$", "").replace(",", "") for n in re.findall(pattern, hypothesis)}
    return bool(ref_numbers & hyp_numbers)


def extract_gold_segments(sample: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    gold_segments: List[Dict[str, Any]] = []
    gold_text_fallback: Optional[str] = None

    raw_gold = (
        sample.get("gold_evidence_segments")
        or sample.get("gold_evidence")
        or sample.get("evidence")
        or sample.get("gold_segments")
        or []
    )

    if isinstance(raw_gold, list):
        for seg in raw_gold:
            if not isinstance(seg, dict):
                continue
            raw = seg.get("raw") if isinstance(seg.get("raw"), dict) else {}
            doc_name = (
                seg.get("doc_name")
                or raw.get("doc_name")
                or seg.get("document")
                or seg.get("doc")
            )
            page_num = seg.get("page") if seg.get("page") is not None else seg.get("page_number")
            if page_num is None:
                page_num = seg.get("evidence_page_num")
            if page_num is None:
                page_num = raw.get("evidence_page_num")

            gold_segments.append(
                {
                    "text": seg.get("text", "") or raw.get("evidence_text") or "",
                    "doc_name": normalize_doc_name(doc_name),
                    "page": page_num,
                }
            )
    elif isinstance(raw_gold, str):
        gold_text_fallback = raw_gold

    return gold_segments, gold_text_fallback


def extract_retrieved_chunks(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    retrieved_chunks: List[Dict[str, Any]] = []

    if isinstance(sample.get("retrieved_chunks"), list):
        for chunk in sample.get("retrieved_chunks", []):
            if not isinstance(chunk, dict):
                continue
            metadata = chunk.get("metadata", {}) if isinstance(chunk.get("metadata"), dict) else {}
            doc_name = chunk.get("doc_name") or metadata.get("doc_name")
            page = chunk.get("page") if chunk.get("page") is not None else metadata.get("page")
            if page is None:
                page = chunk.get("page_number") if chunk.get("page_number") is not None else metadata.get("page_number")
            retrieved_chunks.append(
                {
                    "text": chunk.get("text", "") or "",
                    "metadata": {
                        "doc_name": normalize_doc_name(doc_name),
                        "page": page,
                    },
                }
            )
    elif isinstance(sample.get("retrieved_pages"), list):
        for page in sample.get("retrieved_pages", []):
            if not isinstance(page, dict):
                continue
            doc_name = page.get("doc_name") or page.get("metadata", {}).get("doc_name")
            page_num = page.get("page") if page.get("page") is not None else page.get("page_number")
            retrieved_chunks.append(
                {
                    "text": "",
                    "metadata": {
                        "doc_name": normalize_doc_name(doc_name),
                        "page": page_num,
                    },
                }
            )

    return retrieved_chunks


def compute_doc_page_recall(
    retrieved_chunks: List[Dict[str, Any]],
    gold_segments: List[Dict[str, Any]],
    k: int,
) -> Tuple[float, float]:
    gold_docs = {seg.get("doc_name") for seg in gold_segments if seg.get("doc_name")}
    gold_pages = {
        (seg.get("doc_name"), seg.get("page"))
        for seg in gold_segments
        if seg.get("doc_name") and seg.get("page") is not None
    }

    ordered_docs: List[str] = []
    ordered_pages: List[Tuple[str, Any]] = []
    for chunk in retrieved_chunks:
        meta = chunk.get("metadata", {}) if isinstance(chunk.get("metadata"), dict) else {}
        doc_name = meta.get("doc_name")
        page_num = meta.get("page")
        if doc_name and doc_name not in ordered_docs:
            ordered_docs.append(doc_name)
        if doc_name and page_num is not None:
            page_key = (doc_name, page_num)
            if page_key not in ordered_pages:
                ordered_pages.append(page_key)

    if gold_docs:
        doc_val = len(set(ordered_docs[:k]) & gold_docs) / len(gold_docs)
    else:
        doc_val = 0.0

    if gold_pages:
        page_val = len(set(ordered_pages[:k]) & gold_pages) / len(gold_pages)
    else:
        page_val = 0.0

    return float(doc_val), float(page_val)


def compute_metrics_from_results(results: List[Dict[str, Any]]) -> Dict[str, float]:
    metrics = {
        "total_samples": 0,
        # retrieval
        "doc_recall_1": 0.0,
        "doc_recall_3": 0.0,
        "doc_recall_5": 0.0,
        "page_recall_1": 0.0,
        "page_recall_3": 0.0,
        "page_recall_5": 0.0,
        "retrieval_samples": 0,
        # context metrics
        "ctx_bleu": 0.0,
        "ctx_rouge1": 0.0,
        "ctx_rouge2": 0.0,
        "ctx_rougeL": 0.0,
        "context_samples": 0,
        # generation
        "gen_bleu": 0.0,
        "gen_rouge1": 0.0,
        "gen_rouge2": 0.0,
        "gen_rougeL": 0.0,
        "numeric_correct": 0,
        "numeric_total": 0,
        "generation_samples": 0,
    }

    if not results:
        return metrics

    if Evaluator is None:
        logger.warning("Evaluator unavailable; returning zeroed metrics")
        return metrics

    evaluator = Evaluator(use_bertscore=False, use_llm_judge=False, use_ragas=False)
    metrics["total_samples"] = len(results)

    doc_r1: List[float] = []
    doc_r3: List[float] = []
    doc_r5: List[float] = []
    page_r1: List[float] = []
    page_r3: List[float] = []
    page_r5: List[float] = []

    ctx_bleu: List[float] = []
    ctx_r1: List[float] = []
    ctx_r2: List[float] = []
    ctx_rl: List[float] = []

    gen_bleu: List[float] = []
    gen_r1: List[float] = []
    gen_r2: List[float] = []
    gen_rl: List[float] = []

    numeric_correct = 0
    numeric_total = 0

    for sample in results:
        retrieved_chunks = extract_retrieved_chunks(sample)
        gold_segments, gold_text_fallback = extract_gold_segments(sample)

        if retrieved_chunks and gold_segments:
            d1, p1 = compute_doc_page_recall(retrieved_chunks, gold_segments, 1)
            d3, p3 = compute_doc_page_recall(retrieved_chunks, gold_segments, 3)
            d5, p5 = compute_doc_page_recall(retrieved_chunks, gold_segments, 5)
            doc_r1.append(d1)
            doc_r3.append(d3)
            doc_r5.append(d5)
            page_r1.append(p1)
            page_r3.append(p3)
            page_r5.append(p5)

        # context metrics (max over chunks)
        gold_text = "\n\n".join([seg.get("text", "") for seg in gold_segments if seg.get("text")])
        if not gold_text and gold_text_fallback:
            gold_text = gold_text_fallback
        if gold_text and retrieved_chunks:
            max_bleu = 0.0
            max_r1 = 0.0
            max_r2 = 0.0
            max_rl = 0.0
            for chunk in retrieved_chunks[:MAX_CONTEXT_CHUNKS]:
                chunk_text = chunk.get("text", "") or ""
                if not chunk_text:
                    continue
                bleu = evaluator.compute_bleu(prediction=chunk_text, reference=gold_text)
                rouge = evaluator.compute_rouge(prediction=chunk_text, reference=gold_text)
                if isinstance(bleu, dict) and bleu.get("bleu_4") is not None:
                    max_bleu = max(max_bleu, float(bleu["bleu_4"]))
                if isinstance(rouge, dict):
                    if rouge.get("rouge_1_f1") is not None:
                        max_r1 = max(max_r1, float(rouge["rouge_1_f1"]))
                    if rouge.get("rouge_2_f1") is not None:
                        max_r2 = max(max_r2, float(rouge["rouge_2_f1"]))
                    if rouge.get("rouge_l_f1") is not None:
                        max_rl = max(max_rl, float(rouge["rouge_l_f1"]))
            ctx_bleu.append(max_bleu)
            ctx_r1.append(max_r1)
            ctx_r2.append(max_r2)
            ctx_rl.append(max_rl)

        # generation metrics
        pred = None
        pred_source = None
        for key in ["generated_answer", "prediction", "model_answer", "response", "output"]:
            if sample.get(key):
                pred = sample.get(key)
                pred_source = key
                break
        if pred is None and sample.get("answer"):
            pred = sample.get("answer")
            pred_source = "answer"

        ref = sample.get("reference_answer") or sample.get("reference") or sample.get("ground_truth") or sample.get("gold_answer")
        if not ref and pred_source != "answer":
            ref = sample.get("answer")

        if pred and ref:
            bleu = evaluator.compute_bleu(prediction=pred, reference=ref)
            rouge = evaluator.compute_rouge(prediction=pred, reference=ref)

            if isinstance(bleu, dict) and bleu.get("bleu_4") is not None:
                gen_bleu.append(float(bleu["bleu_4"]))

            if isinstance(rouge, dict):
                if rouge.get("rouge_1_f1") is not None:
                    gen_r1.append(float(rouge["rouge_1_f1"]))
                if rouge.get("rouge_2_f1") is not None:
                    gen_r2.append(float(rouge["rouge_2_f1"]))
                if rouge.get("rouge_l_f1") is not None:
                    gen_rl.append(float(rouge["rouge_l_f1"]))

            q_type = (sample.get("question_type") or "").lower()
            if q_type == "metrics-generated":
                numeric_total += 1
                if compute_numeric_match(ref, pred):
                    numeric_correct += 1

    if doc_r1:
        metrics["doc_recall_1"] = float(np.mean(doc_r1))
        metrics["retrieval_samples"] = len(doc_r1)
    if doc_r3:
        metrics["doc_recall_3"] = float(np.mean(doc_r3))
    if doc_r5:
        metrics["doc_recall_5"] = float(np.mean(doc_r5))
    if page_r1:
        metrics["page_recall_1"] = float(np.mean(page_r1))
    if page_r3:
        metrics["page_recall_3"] = float(np.mean(page_r3))
    if page_r5:
        metrics["page_recall_5"] = float(np.mean(page_r5))

    if ctx_bleu:
        metrics["ctx_bleu"] = float(np.mean(ctx_bleu))
        metrics["context_samples"] = len(ctx_bleu)
    if ctx_r1:
        metrics["ctx_rouge1"] = float(np.mean(ctx_r1))
    if ctx_r2:
        metrics["ctx_rouge2"] = float(np.mean(ctx_r2))
    if ctx_rl:
        metrics["ctx_rougeL"] = float(np.mean(ctx_rl))

    if gen_bleu:
        metrics["gen_bleu"] = float(np.mean(gen_bleu))
        metrics["generation_samples"] = len(gen_bleu)
    if gen_r1:
        metrics["gen_rouge1"] = float(np.mean(gen_r1))
    if gen_r2:
        metrics["gen_rouge2"] = float(np.mean(gen_r2))
    if gen_rl:
        metrics["gen_rougeL"] = float(np.mean(gen_rl))

    metrics["numeric_correct"] = numeric_correct
    metrics["numeric_total"] = numeric_total

    return metrics


def extract_experiment_metadata(file_path: Path, data: Dict[str, Any]) -> Dict[str, Any]:
    metadata = {
        "file_path": str(file_path),
        "file_name": file_path.name,
        "experiment_name": "",
        "subfolder": "",
        "timestamp": "",
        "date": "",
        "retrieval_method": "",
        "model_name": "",
        "embedding_model": "",
        "reranker_model": "",
    }

    # Extract timestamp from parent directory name (for final_results.json in timestamped folders)
    parent_dir = file_path.parent.name
    timestamp_match = re.search(r"(\d{8})_(\d{6})", parent_dir)
    if timestamp_match:
        date_str = timestamp_match.group(1)
        time_str = timestamp_match.group(2)
        metadata["timestamp"] = f"{date_str}_{time_str}"
        try:
            dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
            metadata["date"] = dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            metadata["date"] = date_str
    else:
        # Try to extract from filename (for unified results)
        date_match = re.search(r"(\d{8})_(\d{6})", file_path.name)
        if date_match:
            date_str = date_match.group(1)
            time_str = date_match.group(2)
            metadata["timestamp"] = f"{date_str}_{time_str}"
            try:
                dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
                metadata["date"] = dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                metadata["date"] = date_str
        elif date_match := re.search(r"(\d{8})", file_path.name):
            metadata["date"] = date_match.group(1)
            metadata["timestamp"] = date_match.group(1)

    parts = file_path.parts
    if "results" in parts:
        idx = parts.index("results")
        if idx + 1 < len(parts):
            # For results/subfolder/timestamp/final_results.json
            subfolder = parts[idx + 1]
            metadata["subfolder"] = subfolder
            metadata["experiment_name"] = subfolder
            # If there's a timestamp directory, append it
            if idx + 2 < len(parts) and re.match(r"\d{8}_\d{6}", parts[idx + 2]):
                metadata["experiment_name"] = f"{subfolder}/{parts[idx + 2]}"
    elif "outputs" in parts:
        name = file_path.stem
        name = re.sub(r"_\d{8}_\d{6}", "", name)
        name = re.sub(r"_scored$", "", name)
        metadata["experiment_name"] = name
        metadata["subfolder"] = "unified"

    if isinstance(data, dict) and "metadata" in data:
        meta = data["metadata"]
        if "experiment_type" in meta:
            metadata["retrieval_method"] = str(meta["experiment_type"]).upper()
        if "llm_model" in meta:
            metadata["model_name"] = meta["llm_model"]
        if "embedding_model" in meta:
            metadata["embedding_model"] = meta["embedding_model"]
        if "reranker_model" in meta:
            metadata["reranker_model"] = meta["reranker_model"]

    if not metadata["retrieval_method"]:
        filename_lower = file_path.name.lower()
        if "bm25" in filename_lower:
            metadata["retrieval_method"] = "BM25"
        elif "hybrid" in filename_lower:
            metadata["retrieval_method"] = "Hybrid"
        elif "dense" in filename_lower:
            metadata["retrieval_method"] = "Dense"
        elif "splade" in filename_lower:
            metadata["retrieval_method"] = "SPLADE"
        elif "page_then_chunk" in filename_lower or "page_chunk" in filename_lower:
            metadata["retrieval_method"] = "Page-Then-Chunk"
        elif "learned_page" in filename_lower:
            metadata["retrieval_method"] = "Learned-Page"

    return metadata


def load_result_file(file_path: Path) -> Optional[Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except Exception as exc:
        logger.debug("Failed to load %s: %s", file_path, exc)
        return None

    if isinstance(data, dict):
        if "results" in data:
            results = data["results"]
        elif "samples" in data:
            results = data["samples"]
        else:
            if all(isinstance(v, dict) for v in data.values()):
                results = list(data.values())
            else:
                results = []
    elif isinstance(data, list):
        results = data
    else:
        return None

    if not isinstance(results, list):
        results = list(results) if hasattr(results, "__iter__") else []

    if len(results) < 5:
        return None

    sample_to_check = results[:min(5, len(results))]
    if not any(isinstance(r, dict) and ("question" in r or "query" in r) for r in sample_to_check):
        return None

    metadata = extract_experiment_metadata(file_path, data)
    return metadata, results


def collect_result_files(root_dir: Path) -> List[Path]:
    files: Set[Path] = set()
    for file_path in root_dir.rglob("all_predictions.json"):
        files.add(file_path)

    unified_dir = root_dir / "outputs" / "results" / "unified"
    if unified_dir.exists():
        for file_path in unified_dir.rglob("*.json"):
            files.add(file_path)

    return sorted(files)


def aggregate_overall_results(files: List[Path]) -> List[Dict[str, Any]]:
    all_experiments = []
    for file_path in files:
        print(f"Processing: {file_path}")
        loaded = load_result_file(file_path)
        if loaded is None:
            print(f"  Skipped (not a results file): {file_path}")
            continue
        metadata, results = loaded
        print(f"  Samples: {len(results)}")
        metrics = compute_metrics_from_results(results)
        all_experiments.append({**metadata, **metrics})
        logger.info("  ✓ %s: %s samples", file_path.name, metrics["total_samples"])
    return all_experiments


def aggregate_by_group(files: List[Path], group_by: str) -> List[Dict[str, Any]]:
    experiment_groups: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))

    for file_path in files:
        print(f"Grouping by {group_by}: {file_path}")
        loaded = load_result_file(file_path)
        if loaded is None:
            print(f"  Skipped (not a results file): {file_path}")
            continue
        metadata, results = loaded
        experiment_name = f"{metadata['experiment_name']}_{metadata['date']}"

        for r in results:
            if group_by == "doc_type":
                group_value = r.get("doc_type", "")
                if not group_value:
                    group_value = extract_doc_type_from_name(r.get("doc_name", ""))
            elif group_by in ["question_type", "question_reasoning"]:
                group_value = r.get(group_by, "unknown")
            else:
                group_value = "unknown"

            experiment_groups[experiment_name][group_value].append(r)

    aggregated = []
    for experiment_name, groups in experiment_groups.items():
        for group_value, group_results in groups.items():
            if not group_results:
                continue
            metrics = compute_metrics_from_results(group_results)
            aggregated.append(
                {
                    "experiment_name": experiment_name,
                    "group_by": group_by,
                    "group_value": group_value,
                    **metrics,
                }
            )

    return aggregated


def write_overall_csv(experiments: List[Dict[str, Any]], output_path: Path) -> None:
    if not experiments:
        logger.warning("No experiments to write")
        return

    fieldnames = [
        "experiment_name",
        "subfolder",
        "timestamp",
        "date",
        "file_name",
        "retrieval_method",
        "model_name",
        "embedding_model",
        "reranker_model",
        "total_samples",
        "retrieval_samples",
        "doc_recall_1",
        "doc_recall_3",
        "doc_recall_5",
        "page_recall_1",
        "page_recall_3",
        "page_recall_5",
        "context_samples",
        "ctx_bleu",
        "ctx_rouge1",
        "ctx_rouge2",
        "ctx_rougeL",
        "generation_samples",
        "gen_bleu",
        "gen_rouge1",
        "gen_rouge2",
        "gen_rougeL",
        "numeric_correct",
        "numeric_total",
        "file_path",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(experiments)

    logger.info("✓ Wrote %s experiments to %s", len(experiments), output_path)


def write_grouped_csv(results: List[Dict[str, Any]], output_path: Path) -> None:
    if not results:
        logger.warning("No results to write")
        return

    fieldnames = [
        "experiment_name",
        "group_by",
        "group_value",
        "total_samples",
        "retrieval_samples",
        "doc_recall_1",
        "doc_recall_3",
        "doc_recall_5",
        "page_recall_1",
        "page_recall_3",
        "page_recall_5",
        "context_samples",
        "ctx_bleu",
        "ctx_rouge1",
        "ctx_rouge2",
        "ctx_rougeL",
        "generation_samples",
        "gen_bleu",
        "gen_rouge1",
        "gen_rouge2",
        "gen_rougeL",
        "numeric_correct",
        "numeric_total",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    logger.info("✓ Wrote %s grouped rows to %s", len(results), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate evaluation metrics for FB_reproducability")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_ROOT / "aggregated_results",
        help="Directory to write output CSV files",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix for output filenames",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = collect_result_files(SCRIPT_ROOT)
    print(f"Found {len(files)} files to process")
    logger.info("Found %s JSON files", len(files))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{args.prefix}_" if args.prefix else ""

    overall = aggregate_overall_results(files)
    print("Writing overall CSV...")
    overall_path = args.output_dir / f"{prefix}overall_results_{timestamp}.csv"
    write_overall_csv(overall, overall_path)

    doc_type = aggregate_by_group(files, "doc_type")
    print("Writing by_doc_type CSV...")
    doc_type_path = args.output_dir / f"{prefix}by_doc_type_{timestamp}.csv"
    write_grouped_csv(doc_type, doc_type_path)

    question_type = aggregate_by_group(files, "question_type")
    print("Writing by_question_type CSV...")
    question_type_path = args.output_dir / f"{prefix}by_question_type_{timestamp}.csv"
    write_grouped_csv(question_type, question_type_path)

    reasoning = aggregate_by_group(files, "question_reasoning")
    print("Writing by_question_reasoning CSV...")
    reasoning_path = args.output_dir / f"{prefix}by_question_reasoning_{timestamp}.csv"
    write_grouped_csv(reasoning, reasoning_path)

    logger.info(
        "Summary: %s overall, %s doc_type, %s question_type, %s question_reasoning",
        len(overall),
        len(doc_type),
        len(question_type),
        len(reasoning),
    )


if __name__ == "__main__":
    main()
