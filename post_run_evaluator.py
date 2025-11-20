"""
Post-run evaluation script for RAG experiments.

This module ingests previously saved experiment output JSON files,
enriches each sample with FinanceBench metadata (doc name, evidence pages),
and computes comprehensive retrieval + generation diagnostics:
    - Retrieval: Hit@k, Recall@k, MRR at doc/page/chunk granularity
    - Generation: EM, precision/recall/F1, BLEU, ROUGE, BERTScore (optional), LLM judge (optional)
    - Failure attribution: wrong doc vs wrong page vs wrong chunk vs low-rank chunk

Usage:
    python post_run_evaluator.py --outputs outputs/single_vector_*.json --use-bertscore
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import logging
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import statistics

try:
    # Package-style import
    from .data_loader import FinanceBenchLoader
    from .evaluator import Evaluator
except Exception:  # pragma: no cover - fallback for script execution
    from data_loader import FinanceBenchLoader  # type: ignore
    from evaluator import Evaluator  # type: ignore

LOGGER = logging.getLogger("post_run_evaluator")
TOKEN_PATTERN = re.compile(r"\w+")
DEFAULT_TOPK = (1, 3, 5)


def normalize_question(text: Optional[str]) -> str:
    """Lowercase and collapse whitespace to normalize question keys."""
    if not text:
        return ""
    return " ".join(text.strip().lower().split())


def normalize_doc_name(name: Optional[str]) -> str:
    """Normalize document names for comparisons."""
    if not name:
        return ""
    name = name.strip().lower()
    name = name.replace(".pdf", "")
    return re.sub(r"[^a-z0-9]+", "", name)


def squad_normalize(text: str) -> str:
    """Standard normalization used in QA evaluations (lowercase, remove punctuation/articles)."""
    import string

    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    def remove_punc(s: str) -> str:
        return "".join(ch for ch in s if ch not in string.punctuation)

    text = text.lower()
    text = remove_articles(text)
    text = remove_punc(text)
    text = white_space_fix(text)
    return text


def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text.lower())


def safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return None


def mean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if isinstance(v, (int, float))]
    if not vals:
        return None
    return float(statistics.fmean(vals))


@dataclass
class GoldEvidenceEntry:
    entry_id: str
    doc_name: Optional[str]
    page: Optional[int]
    text: str
    page_text: Optional[str] = None


@dataclass
class GoldEvidence:
    doc_names: Set[str]
    doc_names_norm: Set[str]
    pages: Set[int]
    entries: List[GoldEvidenceEntry]


class PostRunEvaluator:
    """Orchestrates post-run evaluation over saved experiment outputs."""

    def __init__(
        self,
        dataset_split: str = "train",
        topk: Sequence[int] = DEFAULT_TOPK,
        chunk_match_threshold: float = 0.4,
        low_rank_threshold: int = 3,
        failure_threshold: float = 0.5,
        use_bertscore: bool = False,
        use_llm_judge: bool = False,
    ):
        self.topk = sorted(set(int(k) for k in topk if k > 0))
        if not self.topk:
            raise ValueError("topk must contain at least one positive integer")
        self.chunk_match_threshold = chunk_match_threshold
        self.low_rank_threshold = low_rank_threshold
        self.failure_threshold = failure_threshold

        loader = FinanceBenchLoader()
        try:
            self.reference_df = loader.load_data(split=dataset_split)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load FinanceBench split '{dataset_split}'. "
                "Ensure datasets library and HF credentials are configured."
            ) from exc

        self.reference_index = self._build_reference_index(self.reference_df)
        self.evaluator = Evaluator(use_bertscore=use_bertscore, use_llm_judge=use_llm_judge)

    # ------------------------------------------------------------------ #
    # Reference data helpers
    # ------------------------------------------------------------------ #
    def _build_reference_index(self, df):
        index: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for _, row in df.iterrows():
            record = row.to_dict()
            question = normalize_question(record.get("question"))
            if question:
                index[question].append(record)
        return index

    def _find_reference_record(
        self, question: Optional[str], doc_name: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        q_key = normalize_question(question)
        candidates = list(self.reference_index.get(q_key, []))
        if not candidates:
            return None
        if doc_name:
            norm_doc = normalize_doc_name(doc_name)
            filtered = [
                c for c in candidates if normalize_doc_name(c.get("doc_name")) == norm_doc
            ]
            if filtered:
                candidates = filtered
        if len(candidates) > 1:
            LOGGER.debug(
                "Multiple dataset rows found for question '%s'; using first match.", question
            )
        return candidates[0]

    # ------------------------------------------------------------------ #
    # Evidence normalization
    # ------------------------------------------------------------------ #
    def _normalize_evidence_entries(
        self,
        raw: Any,
        fallback_doc: Optional[str],
        source: str,
    ) -> List[GoldEvidenceEntry]:
        if raw is None:
            return []

        parsed = raw
        if isinstance(raw, str):
            stripped = raw.strip()
            if not stripped:
                return []
            if stripped.startswith("{") or stripped.startswith("["):
                try:
                    parsed = ast.literal_eval(stripped)
                except Exception:
                    parsed = raw
            else:
                parsed = stripped

        entries: List[GoldEvidenceEntry] = []
        if isinstance(parsed, dict):
            items = [parsed]
        elif isinstance(parsed, list):
            items = parsed
        else:
            items = [parsed]

        for idx, item in enumerate(items):
            entry_id = f"{source}_{idx}"
            doc = fallback_doc
            page = None
            text = ""
            page_text = None

            if isinstance(item, dict):
                doc = item.get("doc_name") or fallback_doc
                page = safe_int(
                    item.get("evidence_page_num")
                    or item.get("page")
                    or item.get("page_num")
                    or item.get("page_number")
                )
                text = str(
                    item.get("evidence_text")
                    or item.get("text")
                    or item.get("snippet")
                    or ""
                )
                page_text = item.get("evidence_text_full_page") or item.get("page_text")
            else:
                text = str(item)

            entries.append(
                GoldEvidenceEntry(
                    entry_id=entry_id,
                    doc_name=doc,
                    page=page,
                    text=text,
                    page_text=page_text,
                )
            )

        return entries

    def _build_gold_evidence(
        self, sample: Dict[str, Any], reference: Optional[Dict[str, Any]]
    ) -> GoldEvidence:
        doc_candidates = []
        if sample.get("doc_name"):
            doc_candidates.append(sample["doc_name"])
        if reference and reference.get("doc_name"):
            doc_candidates.append(reference["doc_name"])

        entries: List[GoldEvidenceEntry] = []
        sources = [
            ("sample_gold", sample.get("gold_evidence")),
            ("sample_evidence", sample.get("evidence")),
            ("reference_evidence", reference.get("evidence") if reference else None),
            ("reference_gold", reference.get("gold_evidence") if reference else None),
        ]
        fallback_doc = doc_candidates[0] if doc_candidates else None

        for label, raw in sources:
            entries.extend(
                self._normalize_evidence_entries(raw, fallback_doc=fallback_doc, source=label)
            )

        doc_names = {entry.doc_name for entry in entries if entry.doc_name} or set(doc_candidates)
        doc_names_norm = {normalize_doc_name(name) for name in doc_names if name}
        pages = {entry.page for entry in entries if entry.page is not None}

        return GoldEvidence(
            doc_names=doc_names,
            doc_names_norm=doc_names_norm,
            pages=pages,
            entries=entries,
        )

    # ------------------------------------------------------------------ #
    # Retrieval analysis
    # ------------------------------------------------------------------ #
    def _token_recall(self, text: str, gold_text: str) -> float:
        gold_tokens = tokenize(gold_text)
        if not gold_tokens:
            return 0.0
        text_tokens = tokenize(text)
        if not text_tokens:
            return 0.0
        overlap = set(text_tokens) & set(gold_tokens)
        return len(overlap) / len(set(gold_tokens))

    def _match_chunk(
        self,
        chunk_text: str,
        chunk_doc_norm: str,
        chunk_page: Optional[int],
        gold: GoldEvidence,
    ) -> Optional[Dict[str, Any]]:
        best_entry = None
        best_overlap = 0.0
        for entry in gold.entries:
            entry_doc_norm = normalize_doc_name(entry.doc_name)
            if gold.doc_names_norm and entry_doc_norm and chunk_doc_norm:
                if entry_doc_norm != chunk_doc_norm:
                    continue
            if entry.page is not None and chunk_page is not None and entry.page != chunk_page:
                continue
            overlap = self._token_recall(chunk_text, entry.text)
            if overlap >= self.chunk_match_threshold and overlap > best_overlap:
                best_overlap = overlap
                best_entry = entry
        if best_entry is None:
            return None
        return {
            "unit": best_entry.entry_id,
            "doc_name": best_entry.doc_name,
            "page": best_entry.page,
            "score": best_overlap,
        }

    def _calc_level_metrics(
        self, hits: List[Dict[str, Any]], gold_units: Set[str]
    ) -> Dict[str, Any]:
        hits_sorted = sorted(hits, key=lambda x: x["rank"])
        metrics: Dict[str, Any] = {
            "matches": hits_sorted,
            "best_rank": hits_sorted[0]["rank"] if hits_sorted else None,
            "mrr": (1.0 / hits_sorted[0]["rank"]) if hits_sorted else 0.0,
        }
        for k in self.topk:
            metrics[f"hit@{k}"] = 1.0 if any(h["rank"] <= k for h in hits_sorted) else 0.0
            if gold_units:
                found_units = {h["unit"] for h in hits_sorted if h["rank"] <= k}
                metrics[f"recall@{k}"] = len(found_units & gold_units) / len(gold_units)
            else:
                metrics[f"recall@{k}"] = None
        return metrics

    def _analyze_retrieval(
        self,
        sample: Dict[str, Any],
        gold: GoldEvidence,
    ) -> Dict[str, Any]:
        retrieved = sample.get("retrieved_chunks") or []
        if not isinstance(retrieved, list):
            retrieved = []

        doc_hits: List[Dict[str, Any]] = []
        page_hits: List[Dict[str, Any]] = []
        chunk_hits: List[Dict[str, Any]] = []

        doc_units_gold = gold.doc_names_norm
        page_units_gold = {
            f"{normalize_doc_name(entry.doc_name)}|p{entry.page}"
            for entry in gold.entries
            if entry.doc_name and entry.page is not None
        }
        chunk_units_gold = {entry.entry_id for entry in gold.entries}

        for idx, chunk in enumerate(retrieved):
            rank = chunk.get("rank") or (idx + 1)
            metadata = chunk.get("metadata") or {}
            chunk_doc = metadata.get("doc_name") or sample.get("doc_name")
            chunk_doc_norm = normalize_doc_name(chunk_doc)
            chunk_page = safe_int(
                metadata.get("page")
                or metadata.get("page_number")
                or metadata.get("evidence_page_num")
            )
            chunk_text = chunk.get("text") or ""

            if chunk_doc_norm and doc_units_gold:
                if chunk_doc_norm in doc_units_gold:
                    doc_hits.append(
                        {
                            "rank": rank,
                            "unit": chunk_doc_norm,
                            "doc_name": chunk_doc,
                            "page": chunk_page,
                        }
                    )
            elif chunk_doc_norm:
                # If no gold doc recorded, still track matches for analysis
                doc_hits.append(
                    {
                        "rank": rank,
                        "unit": chunk_doc_norm,
                        "doc_name": chunk_doc,
                        "page": chunk_page,
                    }
                )

            if (
                chunk_doc_norm
                and chunk_page is not None
                and page_units_gold
                and f"{chunk_doc_norm}|p{chunk_page}" in page_units_gold
            ):
                page_hits.append(
                    {
                        "rank": rank,
                        "unit": f"{chunk_doc_norm}|p{chunk_page}",
                        "doc_name": chunk_doc,
                        "page": chunk_page,
                    }
                )

            chunk_match = self._match_chunk(chunk_text, chunk_doc_norm, chunk_page, gold)
            if chunk_match:
                chunk_hits.append({"rank": rank, **chunk_match})

        doc_metrics = self._calc_level_metrics(doc_hits, doc_units_gold)
        page_metrics = self._calc_level_metrics(page_hits, page_units_gold)
        chunk_metrics = self._calc_level_metrics(chunk_hits, chunk_units_gold)

        classification = self._classify_retrieval(
            num_retrieved=len(retrieved),
            doc_hits=doc_hits,
            page_hits=page_hits,
            chunk_hits=chunk_hits,
            chunk_metrics=chunk_metrics,
        )

        return {
            "num_retrieved": len(retrieved),
            "doc": doc_metrics,
            "page": page_metrics,
            "chunk": chunk_metrics,
            "classification": classification,
        }

    def _classify_retrieval(
        self,
        num_retrieved: int,
        doc_hits: List[Dict[str, Any]],
        page_hits: List[Dict[str, Any]],
        chunk_hits: List[Dict[str, Any]],
        chunk_metrics: Dict[str, Any],
    ) -> str:
        if num_retrieved == 0:
            return "no_retrieval"
        if not doc_hits:
            return "wrong_document"
        if not page_hits:
            return "right_doc_wrong_page"
        if not chunk_hits:
            return "right_page_wrong_chunk"
        best_rank = chunk_metrics.get("best_rank")
        if best_rank and best_rank > self.low_rank_threshold:
            return "right_chunk_low_rank"
        return "retrieval_on_point"

    # ------------------------------------------------------------------ #
    # Generation analysis
    # ------------------------------------------------------------------ #
    def _precision_recall_f1(
        self, pred_tokens: List[str], ref_tokens: List[str]
    ) -> Tuple[float, float, float]:
        if not pred_tokens and not ref_tokens:
            return 1.0, 1.0, 1.0
        if not pred_tokens or not ref_tokens:
            return 0.0, 0.0, 0.0
        common = Counter(pred_tokens) & Counter(ref_tokens)  # type: ignore
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0, 0.0, 0.0
        precision = num_same / len(pred_tokens)
        recall = num_same / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return precision, recall, f1

    def _analyze_generation(
        self,
        sample: Dict[str, Any],
        reference: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        prediction = sample.get("generated_answer") or ""
        reference_answer = (
            sample.get("reference_answer")
            or (reference.get("answer") if reference else "")
            or ""
        )

        normalized_pred = squad_normalize(prediction)
        normalized_ref = squad_normalize(reference_answer)
        pred_tokens = tokenize(normalized_pred)
        ref_tokens = tokenize(normalized_ref)
        precision, recall, f1 = self._precision_recall_f1(pred_tokens, ref_tokens)
        exact_match = 1.0 if normalized_pred == normalized_ref and normalized_ref else 0.0

        bleu_scores = self.evaluator.compute_bleu(prediction, reference_answer)
        rouge_scores = self.evaluator.compute_rouge(prediction, reference_answer)

        bert_scores = None
        if self.evaluator.use_bertscore and reference_answer.strip():
            bert_raw = self.evaluator.compute_bertscore([prediction], [reference_answer])
            if bert_raw and bert_raw.get("precision"):
                bert_scores = {
                    "precision": bert_raw["precision"][0],
                    "recall": bert_raw["recall"][0],
                    "f1": bert_raw["f1"][0],
                }

        llm_judge = None
        if self.evaluator.use_llm_judge and reference_answer.strip():
            llm_judge = self.evaluator.llm_judge_correctness(
                question=sample.get("question", ""),
                prediction=prediction,
                reference=reference_answer,
            )

        metrics = {
            "em": exact_match,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "bleu": bleu_scores,
            "rouge": rouge_scores,
            "bertscore": bert_scores,
        }

        return {
            "generated_answer": prediction,
            "reference_answer": reference_answer,
            "metrics": metrics,
            "llm_judge": llm_judge,
        }

    # ------------------------------------------------------------------ #
    # Failure attribution
    # ------------------------------------------------------------------ #
    def _analyze_failure(
        self,
        generation_info: Dict[str, Any],
        retrieval_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        metrics = generation_info["metrics"]
        f1 = metrics.get("f1")
        llm_judge = generation_info.get("llm_judge")
        llm_flag = (
            llm_judge is not None and llm_judge.get("correct") is False
        )
        is_failure = (
            f1 is None or f1 < self.failure_threshold or llm_flag
        )
        return {
            "is_failure": bool(is_failure),
            "f1": f1,
            "failure_threshold": self.failure_threshold,
            "retrieval_reason": retrieval_info.get("classification"),
            "llm_judge": llm_judge,
        }

    # ------------------------------------------------------------------ #
    # High-level evaluation API
    # ------------------------------------------------------------------ #
    def evaluate_files(
        self, output_paths: Sequence[str]
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        samples: List[Dict[str, Any]] = []
        file_counts: Counter = Counter()

        for path in output_paths:
            with open(path, "r") as fh:
                payload = json.load(fh)
            metadata = payload.get("metadata", {})
            results = payload.get("results", [])
            LOGGER.info("Loaded %d samples from %s", len(results), path)
            file_counts[path] = len(results)

            for sample in results:
                reference = self._find_reference_record(
                    question=sample.get("question"),
                    doc_name=sample.get("doc_name"),
                )
                gold = self._build_gold_evidence(sample, reference)
                retrieval = self._analyze_retrieval(sample, gold)
                generation = self._analyze_generation(sample, reference)
                failure = self._analyze_failure(generation, retrieval)

                sample_record = {
                    "question": sample.get("question"),
                    "doc_name": sample.get("doc_name")
                    or (reference.get("doc_name") if reference else None),
                    "reference_answer": generation["reference_answer"],
                    "generated_answer": generation["generated_answer"],
                    "experiment_type": sample.get("experiment_type")
                    or metadata.get("experiment_type"),
                    "source_file": path,
                    "gold": {
                        "doc_names": sorted(gold.doc_names),
                        "pages": sorted(gold.pages),
                        "num_entries": len(gold.entries),
                    },
                    "retrieval": retrieval,
                    "generation": generation["metrics"],
                    "llm_judge": generation.get("llm_judge"),
                    "failure": failure,
                }
                samples.append(sample_record)

        summary = self._summarize(samples, file_counts)
        return summary, samples

    def _aggregate_level(self, samples: List[Dict[str, Any]], level_key: str) -> Dict[str, Any]:
        aggregated: Dict[str, Any] = {}
        for k in self.topk:
            hits = [
                sample["retrieval"][level_key].get(f"hit@{k}")
                for sample in samples
                if sample.get("retrieval") and sample["retrieval"].get(level_key)
            ]
            aggregated[f"hit@{k}"] = mean_or_none(hits)

            recalls = [
                sample["retrieval"][level_key].get(f"recall@{k}")
                for sample in samples
                if sample.get("retrieval") and sample["retrieval"].get(level_key)
            ]
            aggregated[f"recall@{k}"] = mean_or_none(recalls)

        mrr_values = [
            sample["retrieval"][level_key].get("mrr")
            for sample in samples
            if sample.get("retrieval") and sample["retrieval"].get(level_key)
        ]
        aggregated["mrr"] = mean_or_none(mrr_values)
        return aggregated

    def _aggregate_nested_metric(
        self, samples: List[Dict[str, Any]], key: str
    ) -> Dict[str, float]:
        aggregated: Dict[str, List[float]] = defaultdict(list)
        for sample in samples:
            nested = sample["generation"].get(key) if sample.get("generation") else None
            if not isinstance(nested, dict):
                continue
            for metric_name, value in nested.items():
                if isinstance(value, (int, float)):
                    aggregated[metric_name].append(float(value))
        return {k: mean_or_none(v) for k, v in aggregated.items() if v}

    def _summarize(
        self,
        samples: List[Dict[str, Any]],
        file_counts: Counter,
    ) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "num_samples": len(samples),
            "files": file_counts,
            "topk": list(self.topk),
        }

        retrieval_summary = {
            "doc": self._aggregate_level(samples, "doc"),
            "page": self._aggregate_level(samples, "page"),
            "chunk": self._aggregate_level(samples, "chunk"),
        }
        classification_counts = Counter(
            sample["retrieval"].get("classification")
            for sample in samples
            if sample.get("retrieval")
        )
        total = sum(classification_counts.values()) or 1
        retrieval_summary["classification_breakdown"] = {
            label: count / total for label, count in classification_counts.items()
        }

        summary["retrieval"] = retrieval_summary

        generation_summary: Dict[str, Any] = {}
        for metric in ("em", "precision", "recall", "f1"):
            values = [
                sample["generation"].get(metric)
                for sample in samples
                if sample.get("generation") and sample["generation"].get(metric) is not None
            ]
            generation_summary[metric] = mean_or_none(values)

        generation_summary["bleu"] = self._aggregate_nested_metric(samples, "bleu")
        generation_summary["rouge"] = self._aggregate_nested_metric(samples, "rouge")
        if any(sample["generation"].get("bertscore") for sample in samples):
            generation_summary["bertscore"] = self._aggregate_nested_metric(
                samples, "bertscore"
            )

        llm_flags = [
            sample.get("llm_judge")
            for sample in samples
            if sample.get("llm_judge") is not None
        ]
        if llm_flags:
            judged = [
                1.0 if flag.get("correct") else 0.0
                for flag in llm_flags
                if flag.get("correct") is not None
            ]
            correct_rate = statistics.fmean(judged) if judged else None
            generation_summary["llm_judge_correct_rate"] = float(correct_rate)

        summary["generation"] = generation_summary

        failures = [sample for sample in samples if sample["failure"]["is_failure"]]
        failure_counts = Counter(
            sample["failure"].get("retrieval_reason") for sample in failures
        )
        failure_summary = {
            "failure_rate": len(failures) / len(samples) if samples else 0.0,
            "num_failures": len(failures),
            "threshold": self.failure_threshold,
            "by_retrieval_reason": {
                reason or "unknown": count / len(failures) if failures else 0.0
                for reason, count in failure_counts.items()
            },
        }
        summary["failure_analysis"] = failure_summary

        return summary


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #
def expand_output_paths(patterns: Sequence[str]) -> List[str]:
    paths: List[str] = []
    for pattern in patterns:
        if os.path.isdir(pattern):
            paths.extend(
                sorted(
                    str(p)
                    for p in Path(pattern).glob("*.json")
                    if p.is_file()
                )
            )
        else:
            expanded = glob.glob(pattern)
            if expanded:
                paths.extend(sorted(expanded))
            elif os.path.isfile(pattern):
                paths.append(pattern)
    unique = list(dict.fromkeys(paths))
    if not unique:
        raise FileNotFoundError(
            f"No output JSON files found for patterns: {patterns}"
        )
    return unique


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Offline evaluator for RAG experiment outputs."
    )
    parser.add_argument(
        "--outputs",
        nargs="+",
        required=True,
        help="Paths or glob patterns to experiment output JSON files.",
    )
    parser.add_argument("--split", default="train", help="FinanceBench split to join with.")
    parser.add_argument(
        "--topk",
        nargs="+",
        type=int,
        default=list(DEFAULT_TOPK),
        help="Top-k values for retrieval metrics (e.g., --topk 1 3 5).",
    )
    parser.add_argument(
        "--chunk-match-threshold",
        type=float,
        default=0.4,
        help="Minimum token recall for a chunk to count as containing the gold evidence.",
    )
    parser.add_argument(
        "--low-rank-threshold",
        type=int,
        default=3,
        help="Rank threshold for labeling a correct chunk as 'low_rank'.",
    )
    parser.add_argument(
        "--failure-threshold",
        type=float,
        default=0.5,
        help="F1 threshold below which a sample is treated as a failed answer.",
    )
    parser.add_argument(
        "--use-bertscore",
        action="store_true",
        help="Enable BERTScore (requires bert-score package).",
    )
    parser.add_argument(
        "--use-llm-judge",
        action="store_true",
        help="Enable LLM-as-judge (requires judge pipeline configuration).",
    )
    parser.add_argument(
        "--save-summary",
        type=str,
        default=None,
        help="Path to write the aggregated summary JSON (default: outputs/post_eval/<ts>_summary.json).",
    )
    parser.add_argument(
        "--save-details",
        type=str,
        default=None,
        help="Path to write per-sample diagnostics JSON (default: outputs/post_eval/<ts>_details.json).",
    )
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    configure_logging(args.verbose)

    output_paths = expand_output_paths(args.outputs)
    LOGGER.info("Evaluating %d files", len(output_paths))

    evaluator = PostRunEvaluator(
        dataset_split=args.split,
        topk=args.topk,
        chunk_match_threshold=args.chunk_match_threshold,
        low_rank_threshold=args.low_rank_threshold,
        failure_threshold=args.failure_threshold,
        use_bertscore=args.use_bertscore,
        use_llm_judge=args.use_llm_judge,
    )

    summary, samples = evaluator.evaluate_files(output_paths)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs") / "post_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = (
        Path(args.save_summary)
        if args.save_summary
        else output_dir / f"post_eval_summary_{timestamp}.json"
    )
    details_path = (
        Path(args.save_details)
        if args.save_details
        else output_dir / f"post_eval_details_{timestamp}.json"
    )

    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    LOGGER.info("Summary written to %s", summary_path)

    with open(details_path, "w") as fh:
        json.dump(samples, fh, indent=2, default=str)
    LOGGER.info("Per-sample details written to %s", details_path)

    # Pretty-print key highlights to console
    print("\n=== POST-RUN EVALUATION SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
