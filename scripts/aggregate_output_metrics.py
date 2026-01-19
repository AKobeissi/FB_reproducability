#!/usr/bin/env python3
"""
Aggregate FinanceBench metrics across all hidden output folders.
What it does
------------
- Recursively searches for JSON/JSONL experiment output files under any folder
  named "outputs" (this is gitignored in this repo). It intentionally does NOT
  scan "results/" to avoid mixing derived artifacts with raw experiment outputs.
- For each file, computes:
  Retrieval (at k=5 by default):
    - Doc: Hit@5, Recall@5
    - Page: Hit@5, Recall@5
    - Retrieved chunks vs gold evidence: max(BLEU-4), max(ROUGE-L F1)
    - Retrieved chunks vs reference answer: max(BLEU-4), max(ROUGE-L F1)
    - Ranking vs reference answer: MRR@5, NDCG@5 (based on token-overlap relevance)
    - Ranking vs gold evidence text: MRR@5, NDCG@5 (based on token-overlap relevance)
  Generation:
    - BLEU-4, ROUGE-L F1, BERTScore F1
    - For question_type == "metrics-generated": numeric string-match accuracy
Outputs
-------
Writes 3 tables (CSV):
  1) Aggregated by experiment metadata
  2) Aggregated by experiment metadata + question_type
  3) Aggregated by experiment metadata + question_reasoning
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# Silence extremely verbose per-sample logs from the shared Evaluator.
# (Users can still enable verbosity via --verbose.)
logging.getLogger("src.evaluation.evaluator").setLevel(logging.WARNING)

# Ensure repo root is importable when running as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.evaluator import Evaluator  # noqa: E402


DEFAULT_METADATA_COLUMNS: List[str] = [
    "experiment_type",
    "llm_model",
    "use_api",
    "api_base_url",
    "chunk_size",
    "chunk_overlap",
    "top_k",
    "embedding_model",
    "device",
    "load_in_8bit",
    "max_new_tokens",
    "pdf_local_dir",
    "use_all_pdfs",
    "timestamp",
    "eval_type",
    "eval_mode",
    "judge_model",
    "chunking_strategy",
    "chunking_unit",
    "parent_chunk_size",
    "parent_chunk_overlap",
    "child_chunk_size",
    "child_chunk_overlap",
]


def _iso_mtime(path: Path) -> str:
    """
    File modified time as ISO-8601 string (UTC-naive).
    Only used as a fallback when the experiment metadata has no timestamp.
    """
    try:
        return pd.Timestamp(path.stat().st_mtime, unit="s").isoformat()
    except Exception:
        return ""


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def _normalize_doc_name(name: Any) -> str:
    if name is None:
        return ""
    s = str(name).strip().lower()
    if s.endswith(".pdf"):
        s = s[:-4]
    return s


def _normalize_page(page: Any) -> str:
    if page is None:
        return ""
    return str(page).strip()


_TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _token_overlap_ratio(text_a: str, text_b: str) -> float:
    """
    Symmetric token overlap ratio in [0,1].
    Mirrors the logic used in Evaluator._token_overlap_ratio.
    """
    if not text_a or not text_b:
        return 0.0
    a = set(_tokenize(text_a))
    b = set(_tokenize(text_b))
    if not a or not b:
        return 0.0
    return len(a & b) / max(len(a), len(b))


def _safe_mean(xs: Sequence[Optional[float]]) -> Optional[float]:
    vals = [x for x in xs if isinstance(x, (int, float)) and not math.isnan(float(x))]
    if not vals:
        return None
    return float(np.mean(vals))


def _extract_numbers(text: str) -> List[float]:
    """
    Extract numbers from text, tolerating commas and currency symbols.
    """
    clean = re.sub(r"[,$]", "", str(text or ""))
    matches = re.findall(r"-?\d*\.?\d+", clean)
    out: List[float] = []
    for m in matches:
        if m in {"", "-", ".", "-."}:
            continue
        try:
            out.append(float(m))
        except ValueError:
            continue
    return out


def _numeric_match(reference: str, prediction: str, *, atol: float = 0.01, rtol: float = 0.01) -> float:
    """
    Returns 1.0 if any number in the reference is found in the prediction (within tolerance).
    """
    ref_nums = _extract_numbers(reference)
    pred_nums = _extract_numbers(prediction)
    if not ref_nums or not pred_nums:
        return 0.0
    for r in ref_nums:
        for p in pred_nums:
            if np.isclose(r, p, atol=atol) or np.isclose(r, p, rtol=rtol):
                return 1.0
    return 0.0


def _dcg(rels: Sequence[float]) -> float:
    total = 0.0
    for idx, rel in enumerate(rels, start=1):
        denom = math.log2(idx + 1.0)
        total += float(rel) / denom
    return total


def _ndcg(rels: Sequence[float]) -> float:
    if not rels:
        return 0.0
    dcg_val = _dcg(rels)
    ideal = sorted((float(r) for r in rels), reverse=True)
    idcg_val = _dcg(ideal)
    if idcg_val <= 0.0:
        return 0.0
    return float(dcg_val / idcg_val)


def _extract_gold_evidence_text(sample: Dict[str, Any]) -> str:
    """
    Returns a single gold evidence text string to compare retrieved chunks against.
    """
    segments = sample.get("gold_evidence_segments")
    texts: List[str] = []
    if isinstance(segments, list):
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            t = seg.get("text") or seg.get("evidence_text") or seg.get("evidence_text_full_page")
            if t:
                texts.append(str(t))
    elif isinstance(segments, dict):
        t = segments.get("text") or segments.get("evidence_text") or segments.get("evidence_text_full_page")
        if t:
            texts.append(str(t))

    if not texts:
        legacy = sample.get("gold_evidence")
        if isinstance(legacy, str) and legacy.strip():
            texts.append(legacy.strip())

    # Keep a separator that is stable for BLEU/ROUGE comparisons
    return "\n".join(t.strip() for t in texts if t and str(t).strip())


def _extract_gold_doc_pages(sample: Dict[str, Any]) -> Tuple[set, set]:
    """
    Extract gold doc and (doc,page) sets from gold evidence segments.
    """
    segments = sample.get("gold_evidence_segments")
    segs: List[Dict[str, Any]] = []
    if isinstance(segments, list):
        segs = [s for s in segments if isinstance(s, dict)]
    elif isinstance(segments, dict):
        segs = [segments]
    else:
        segs = []

    gold_docs = set()
    gold_pages = set()
    for seg in segs:
        doc = seg.get("doc_name") or seg.get("document") or seg.get("source")
        page = seg.get("page") or seg.get("page_number")
        doc_n = _normalize_doc_name(doc)
        if doc_n:
            gold_docs.add(doc_n)
            if page is not None:
                gold_pages.add((doc_n, _normalize_page(page)))
    return gold_docs, gold_pages


def _extract_retrieved_doc_pages(retrieved_chunks: Sequence[Dict[str, Any]], k: int) -> Tuple[List[str], List[Tuple[str, str]]]:
    docs: List[str] = []
    pages: List[Tuple[str, str]] = []
    for chunk in list(retrieved_chunks)[:k]:
        if not isinstance(chunk, dict):
            # Some outputs store retrieved chunks as plain strings (no metadata).
            continue
        meta = chunk.get("metadata") or {}
        doc = meta.get("doc_name") or meta.get("source") or meta.get("document")
        page = meta.get("page") or meta.get("page_number")
        doc_n = _normalize_doc_name(doc)
        if doc_n:
            docs.append(doc_n)
            if page is not None:
                pages.append((doc_n, _normalize_page(page)))
    return docs, pages


@dataclass(frozen=True)
class PerSampleMetrics:
    # Retrieval
    retrieval_doc_hit_at_5: Optional[float]
    retrieval_doc_recall_at_5: Optional[float]
    retrieval_page_hit_at_5: Optional[float]
    retrieval_page_recall_at_5: Optional[float]
    retrieval_chunk_bleu4_max_vs_evidence_at_5: Optional[float]
    retrieval_chunk_rougeL_f1_max_vs_evidence_at_5: Optional[float]
    retrieval_chunk_bleu4_max_vs_reference_at_5: Optional[float]
    retrieval_chunk_rougeL_f1_max_vs_reference_at_5: Optional[float]
    retrieval_mrr_at_5_vs_reference: Optional[float]
    retrieval_ndcg_at_5_vs_reference: Optional[float]
    retrieval_mrr_at_5_vs_evidence: Optional[float]
    retrieval_ndcg_at_5_vs_evidence: Optional[float]
    # Generation
    generation_bleu4: Optional[float]
    generation_rougeL_f1: Optional[float]
    generation_bertscore_f1: Optional[float]
    generation_numeric_match: Optional[float]


def compute_per_sample_metrics(
    evaluator: Evaluator,
    sample: Dict[str, Any],
    *,
    k: int,
    overlap_threshold: float,
) -> PerSampleMetrics:
    retrieved = _as_list(sample.get("retrieved_chunks"))
    gold_docs, gold_pages = _extract_gold_doc_pages(sample)
    ref_answer = str(sample.get("reference_answer") or "").strip()
    pred_answer = str(sample.get("generated_answer") or "").strip()

    # Retrieval: doc/page hit + recall @k
    ret_docs_k, ret_pages_k = _extract_retrieved_doc_pages(retrieved, k)
    ret_docs_set = set(ret_docs_k)
    ret_pages_set = set(ret_pages_k)

    doc_hit = None
    doc_recall = None
    if gold_docs:
        doc_hit = 1.0 if not gold_docs.isdisjoint(ret_docs_set) else 0.0
        doc_recall = float(len(gold_docs & ret_docs_set) / len(gold_docs))

    page_hit = None
    page_recall = None
    if gold_pages:
        page_hit = 1.0 if not gold_pages.isdisjoint(ret_pages_set) else 0.0
        page_recall = float(len(gold_pages & ret_pages_set) / len(gold_pages))

    # Retrieval: chunk text similarity vs evidence/reference (take max across retrieved @k)
    gold_evidence_text = _extract_gold_evidence_text(sample)
    chunk_texts_k: List[str] = []
    for c in list(retrieved)[:k]:
        # Support both schemas:
        # - dict: {"text": "...", "metadata": {...}}
        # - str:  "..."
        if isinstance(c, str):
            t = c.strip()
            if t:
                chunk_texts_k.append(t)
            continue
        if isinstance(c, dict):
            t = str(c.get("text") or c.get("page_content") or "").strip()
            if t:
                chunk_texts_k.append(t)

    bleu4_evi: List[float] = []
    rougeL_evi: List[float] = []
    if gold_evidence_text and chunk_texts_k:
        for t in chunk_texts_k:
            bleu4_evi.append(float(evaluator.compute_bleu(t, gold_evidence_text).get("bleu_4", 0.0)))
            rougeL_evi.append(float(evaluator.compute_rouge(t, gold_evidence_text).get("rouge_l_f1", 0.0)))

    bleu4_ref: List[float] = []
    rougeL_ref: List[float] = []
    if ref_answer and chunk_texts_k:
        for t in chunk_texts_k:
            bleu4_ref.append(float(evaluator.compute_bleu(t, ref_answer).get("bleu_4", 0.0)))
            rougeL_ref.append(float(evaluator.compute_rouge(t, ref_answer).get("rouge_l_f1", 0.0)))

    chunk_bleu4_max_vs_evidence = max(bleu4_evi) if bleu4_evi else None
    chunk_rougeL_max_vs_evidence = max(rougeL_evi) if rougeL_evi else None
    chunk_bleu4_max_vs_reference = max(bleu4_ref) if bleu4_ref else None
    chunk_rougeL_max_vs_reference = max(rougeL_ref) if rougeL_ref else None

    # --- NDCG FIX: Assume 1 perfect match exists in the corpus for IDCG ---
    ideal_rels = [1.0]

    # Retrieval ranking metrics vs reference answer: MRR@k, NDCG@k
    mrr = None
    ndcg = None
    if ref_answer and chunk_texts_k:
        rels = [_token_overlap_ratio(t, ref_answer) for t in chunk_texts_k]
        
        # Calculate NDCG using the fixed ideal_rels
        dcg_val = _dcg(rels)
        idcg_val = _dcg(ideal_rels)
        ndcg = float(dcg_val / idcg_val) if idcg_val > 0 else 0.0

        first_rank: Optional[int] = None
        for i, rel in enumerate(rels, start=1):
            if rel >= overlap_threshold:
                first_rank = i
                break
        mrr = 1.0 / first_rank if first_rank else 0.0

    # Retrieval ranking metrics vs gold evidence: MRR@k, NDCG@k
    mrr_evidence = None
    ndcg_evidence = None
    if gold_evidence_text and chunk_texts_k:
        rels_evi = [_token_overlap_ratio(t, gold_evidence_text) for t in chunk_texts_k]
        
        # Calculate NDCG using the fixed ideal_rels
        dcg_val = _dcg(rels_evi)
        idcg_val = _dcg(ideal_rels)
        ndcg_evidence = float(dcg_val / idcg_val) if idcg_val > 0 else 0.0

        first_rank_evi: Optional[int] = None
        for i, rel in enumerate(rels_evi, start=1):
            if rel >= overlap_threshold:
                first_rank_evi = i
                break
        mrr_evidence = 1.0 / first_rank_evi if first_rank_evi else 0.0

    # Generation metrics vs reference answer
    gen_bleu4 = None
    gen_rougeL = None
    gen_bertscore_f1 = None
    if ref_answer and pred_answer:
        gen_bleu4 = float(evaluator.compute_bleu(pred_answer, ref_answer).get("bleu_4", 0.0))
        gen_rougeL = float(evaluator.compute_rouge(pred_answer, ref_answer).get("rouge_l_f1", 0.0))
        if evaluator.use_bertscore:
            try:
                bs = evaluator.compute_bertscore([pred_answer], [ref_answer])
                f1s = bs.get("f1") or []
                gen_bertscore_f1 = float(f1s[0]) if f1s else None
            except Exception:
                gen_bertscore_f1 = None

    # Numeric match for metrics-generated
    q_type = str(sample.get("question_type") or "").strip().lower()
    gen_numeric = 0.0
    if q_type == "metrics-generated":
        gen_numeric = _numeric_match(ref_answer, pred_answer)

    return PerSampleMetrics(
        retrieval_doc_hit_at_5=doc_hit,
        retrieval_doc_recall_at_5=doc_recall,
        retrieval_page_hit_at_5=page_hit,
        retrieval_page_recall_at_5=page_recall,
        retrieval_chunk_bleu4_max_vs_evidence_at_5=chunk_bleu4_max_vs_evidence,
        retrieval_chunk_rougeL_f1_max_vs_evidence_at_5=chunk_rougeL_max_vs_evidence,
        retrieval_chunk_bleu4_max_vs_reference_at_5=chunk_bleu4_max_vs_reference,
        retrieval_chunk_rougeL_f1_max_vs_reference_at_5=chunk_rougeL_max_vs_reference,
        retrieval_mrr_at_5_vs_reference=mrr,
        retrieval_ndcg_at_5_vs_reference=ndcg,
        retrieval_mrr_at_5_vs_evidence=mrr_evidence,
        retrieval_ndcg_at_5_vs_evidence=ndcg_evidence,
        generation_bleu4=gen_bleu4,
        generation_rougeL_f1=gen_rougeL,
        generation_bertscore_f1=gen_bertscore_f1,
        generation_numeric_match=gen_numeric,
    )

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def load_experiment_file(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Supports:
      - dict with { "metadata": {...}, "results": [...] }
      - list of samples
      - JSONL: one sample per line
    """
    if path.suffix.lower() == ".jsonl":
        return _load_jsonl(path), {}

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data, {}
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return data["results"], (data.get("metadata") or {})
    raise ValueError(f"Unsupported schema: {path}")


def discover_experiment_files(root: Path, *, dir_names: Sequence[str]) -> List[Path]:
    """
    Search beneath any directory whose name is in `dir_names`.
    """
    roots: List[Path] = []
    for d in dir_names:
        direct = root / d
        if direct.exists() and direct.is_dir():
            roots.append(direct)
    for p in root.rglob("*"):
        if p.is_dir() and p.name in set(dir_names):
            roots.append(p)

    seen = set()
    out: List[Path] = []
    for r in roots:
        for fp in r.rglob("*"):
            if not fp.is_file():
                continue
            # Avoid scanning nested "results/" directories under outputs/,
            # which are typically derived artifacts.
            try:
                rel = fp.relative_to(r)
                if any(part == "results" for part in rel.parts[:-1]):
                    continue
            except Exception:
                pass
            if fp.suffix.lower() not in {".json", ".jsonl"}:
                continue
            if fp in seen:
                continue
            seen.add(fp)
            out.append(fp)
    return sorted(out)


def _load_dataset_metadata(dataset_path: Path) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns (by_financebench_id, by_index).
    """
    by_id: Dict[str, Dict[str, Any]] = {}
    by_idx: List[Dict[str, Any]] = []
    if not dataset_path.exists():
        return by_id, by_idx
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            by_idx.append(row)
            fid = row.get("financebench_id")
            if fid:
                by_id[str(fid)] = row
    return by_id, by_idx


def _inject_question_metadata(
    samples: List[Dict[str, Any]],
    *,
    dataset_by_id: Dict[str, Dict[str, Any]],
    dataset_by_index: List[Dict[str, Any]],
) -> None:
    """
    Best-effort injection of question_type/question_reasoning if missing.
    """
    for i, s in enumerate(samples):
        if s.get("question_type") and s.get("question_reasoning"):
            continue
        fid = s.get("financebench_id") or s.get("sample_id") or s.get("id")
        ds: Dict[str, Any] = {}
        if fid is not None:
            ds = dataset_by_id.get(str(fid), {})
        if not ds and i < len(dataset_by_index):
            ds = dataset_by_index[i]
        if not s.get("question_type"):
            s["question_type"] = ds.get("question_type", "unknown")
        if not s.get("question_reasoning"):
            s["question_reasoning"] = ds.get("question_reasoning", "unknown")


def _aggregate_table(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    metric_cols = [c for c in df.columns if c.startswith(("retrieval_", "generation_"))]
    agg_dict = {c: "mean" for c in metric_cols}
    agg_dict["n_samples"] = "sum"
    out = (
        df.groupby(group_cols, dropna=False, as_index=False)
        .agg(agg_dict)
        .sort_values(group_cols)
        .reset_index(drop=True)
    )
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scan outputs/ folders and compute aggregated FinanceBench metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--root", type=str, default=".", help="Repo/workspace root to scan from.")
    p.add_argument(
        "--search-dirs",
        type=str,
        default="outputs",
        help='Comma-separated directory names to scan (e.g. "outputs").',
    )
    p.add_argument("--dataset", type=str, default="data/financebench_open_source.jsonl")
    p.add_argument("--k", type=int, default=5, help="Top-k used for retrieval metrics.")
    p.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.70,
        help="Token-overlap threshold used for MRR (binary relevance) vs reference answer.",
    )
    p.add_argument("--skip-bertscore", action="store_true", help="Skip BERTScore (faster/lighter).")
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (including per-sample metric logs).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="results/metrics_aggregated",
        help="Directory to write the 3 CSV tables to.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.verbose:
        logging.getLogger("src.evaluation.evaluator").setLevel(logging.INFO)
    root = Path(args.root).resolve()
    dataset_path = (root / args.dataset).resolve() if not Path(args.dataset).is_absolute() else Path(args.dataset)
    out_dir = (root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    search_dirs = [s.strip() for s in args.search_dirs.split(",") if s.strip()]
    files = discover_experiment_files(root, dir_names=search_dirs)
    if not files:
        print(f"No experiment files found under directories: {search_dirs} (root={root})")
        return 0

    dataset_by_id, dataset_by_index = _load_dataset_metadata(dataset_path)

    evaluator = Evaluator(use_bertscore=not args.skip_bertscore, use_llm_judge=False, use_ragas=False)

    rows: List[Dict[str, Any]] = []
    for fp in files:
        try:
            samples, metadata = load_experiment_file(fp)
        except Exception:
            continue

        if not samples or not isinstance(samples, list):
            continue

        # Require at least one sample with the expected answer fields.
        if not any(isinstance(s, dict) and ("generated_answer" in s or "reference_answer" in s) for s in samples[:5]):
            continue

        _inject_question_metadata(samples, dataset_by_id=dataset_by_id, dataset_by_index=dataset_by_index)

        # Normalize metadata columns (only those requested; extra metadata is ignored for grouping)
        meta_row = {k: metadata.get(k) for k in DEFAULT_METADATA_COLUMNS}
        # Ensure a timestamp is always present so multiple runs don't get merged.
        if not meta_row.get("timestamp"):
            meta_row["timestamp"] = _iso_mtime(fp)

        for s in samples:
            if not isinstance(s, dict):
                continue
            per = compute_per_sample_metrics(
                evaluator,
                s,
                k=args.k,
                overlap_threshold=args.overlap_threshold,
            )
            rows.append(
                {
                    **meta_row,
                    "source_file": str(fp.relative_to(root)) if fp.is_relative_to(root) else str(fp),
                    "question_type": s.get("question_type", "unknown"),
                    "question_reasoning": s.get("question_reasoning", "unknown"),
                    "n_samples": 1,
                    **per.__dict__,
                }
            )

    if not rows:
        print("No valid experiment samples found under outputs/results.")
        return 0

    df = pd.DataFrame(rows)

    meta_cols = DEFAULT_METADATA_COLUMNS
    overall = _aggregate_table(df, group_cols=meta_cols)
    by_type = _aggregate_table(df, group_cols=meta_cols + ["question_type"])
    by_reasoning = _aggregate_table(df, group_cols=meta_cols + ["question_reasoning"])

    overall_path = out_dir / "table__overall_by_metadata.csv"
    type_path = out_dir / "table__by_question_type.csv"
    reasoning_path = out_dir / "table__by_question_reasoning.csv"

    overall.to_csv(overall_path, index=False)
    by_type.to_csv(type_path, index=False)
    by_reasoning.to_csv(reasoning_path, index=False)

    print(f"Wrote:\n- {overall_path}\n- {type_path}\n- {reasoning_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())