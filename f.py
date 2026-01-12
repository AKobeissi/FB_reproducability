#!/usr/bin/env python3
"""
Summarize FinanceBench retrieval experiments into a @5-only table.

For each input JSON, this script will:
- If the JSON already looks like an aggregated metrics dict (has doc_recall@5 etc), it uses it directly.
- Else it assumes it's raw samples (list OR dict with "results") and runs RetrievalEvaluator.compute_metrics(k_values=[5]).

Outputs a Markdown table with:
chunk_size, doc_recall@5, page_recall@5, context_bleu@5, context_rougeL@5, mrr
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


NEEDED_KEYS = ["doc_recall@5", "page_recall@5", "context_bleu@5", "context_rougeL@5", "mrr"]


@dataclass
class Row:
    label: str
    label_num: Optional[int]
    doc_recall_5: float
    page_recall_5: float
    bleu_5: float
    rougeL_5: float
    mrr: float


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _looks_like_aggregated_metrics(obj: Any) -> bool:
    return isinstance(obj, dict) and any(k in obj for k in NEEDED_KEYS)


def _extract_samples(obj: Any) -> List[Dict[str, Any]]:
    """
    Supports:
    - list[dict]  (already samples)
    - {"results": [...]} (common wrapper)
    """
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and "results" in obj and isinstance(obj["results"], list):
        return obj["results"]
    raise ValueError("Unsupported JSON structure: expected list or dict with key 'results'.")


def _infer_label_and_num(path: Path) -> Tuple[str, Optional[int]]:
    """
    Tries to infer chunk size from filename, e.g.:
      ...128... , ...256... , ...512... , ...1024...
    Returns (label_str, label_num).
    """
    s = path.stem
    m = re.search(r"(128|256|512|1024)", s)
    if m:
        n = int(m.group(1))
        return f"{n}", n
    # fallback: any number in name
    m2 = re.search(r"(\d+)", s)
    if m2:
        n = int(m2.group(1))
        return f"{n}", n
    return s, None


def _dynamic_import_retrieval_evaluator(evaluator_path: Path):
    """
    Load RetrievalEvaluator from a path (so it doesn't need to be installed as a package).
    """
    evaluator_path = evaluator_path.resolve()
    if not evaluator_path.exists():
        raise FileNotFoundError(f"Evaluator file not found: {evaluator_path}")

    spec = importlib.util.spec_from_file_location("retrieval_evaluator", str(evaluator_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import evaluator from: {evaluator_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]

    if not hasattr(module, "RetrievalEvaluator"):
        raise ImportError(f"{evaluator_path} does not define RetrievalEvaluator")

    return module.RetrievalEvaluator


def _format_pct(x: float, decimals: int = 2) -> str:
    return f"{x * 100:.{decimals}f}%"


def _format_mrr(x: float, decimals: int = 3) -> str:
    return f"{x:.{decimals}f}"


def _format_md_table(rows: List[Row]) -> str:
    header = (
        "| Chunk size | doc_recall@5 | page_recall@5 | context_bleu@5 | context_rougeL@5 | MRR |\n"
        "|---:|---:|---:|---:|---:|---:|\n"
    )
    lines = []
    for r in rows:
        lines.append(
            f"| {r.label} | {_format_pct(r.doc_recall_5)} | {_format_pct(r.page_recall_5)} | "
            f"{_format_pct(r.bleu_5)} | {_format_pct(r.rougeL_5)} | {_format_mrr(r.mrr)} |"
        )
    return header + "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Paths to 4 experiment JSON files (raw samples or already-aggregated metrics).",
    )
    p.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional labels for each input (e.g., 128 256 512 1024). If omitted, inferred from filename.",
    )
    p.add_argument(
        "--evaluator",
        default="retrieval_evaluator.py",
        help="Path to retrieval_evaluator.py (default: ./retrieval_evaluator.py).",
    )
    p.add_argument(
        "--csv-out",
        default=None,
        help="Optional path to save CSV (e.g., results.csv).",
    )
    args = p.parse_args()

    input_paths = [Path(x) for x in args.inputs]
    for ip in input_paths:
        if not ip.exists():
            print(f"ERROR: input file not found: {ip}", file=sys.stderr)
            return 2

    labels: Optional[List[str]] = args.labels if args.labels else None
    if labels is not None and len(labels) != len(input_paths):
        print("ERROR: --labels must have the same number of entries as --inputs", file=sys.stderr)
        return 2

    RetrievalEvaluator = _dynamic_import_retrieval_evaluator(Path(args.evaluator))

    rows: List[Row] = []
    evaluator = RetrievalEvaluator()

    for i, path in enumerate(input_paths):
        obj = _load_json(path)

        if labels is not None:
            label = labels[i]
            label_num = int(label) if label.isdigit() else None
        else:
            label, label_num = _infer_label_and_num(path)

        # Case A: already aggregated metrics dict
        if _looks_like_aggregated_metrics(obj):
            metrics = obj
        else:
            # Case B: raw samples -> compute metrics @5
            samples = _extract_samples(obj)
            metrics = evaluator.compute_metrics(samples, k_values=[5])

        missing = [k for k in NEEDED_KEYS if k not in metrics]
        if missing:
            print(
                f"ERROR: {path} missing keys {missing}. "
                f"Got keys: {sorted(metrics.keys())[:30]}...",
                file=sys.stderr,
            )
            return 3

        rows.append(
            Row(
                label=label,
                label_num=label_num,
                doc_recall_5=float(metrics["doc_recall@5"]),
                page_recall_5=float(metrics["page_recall@5"]),
                bleu_5=float(metrics["context_bleu@5"]),
                rougeL_5=float(metrics["context_rougeL@5"]),
                mrr=float(metrics["mrr"]),
            )
        )

    # Sort by numeric label if possible
    rows.sort(key=lambda r: (r.label_num is None, r.label_num if r.label_num is not None else r.label))

    md = _format_md_table(rows)
    print(md)

    if args.csv_out:
        out = Path(args.csv_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            f.write("chunk_size,doc_recall@5,page_recall@5,context_bleu@5,context_rougeL@5,mrr\n")
            for r in rows:
                f.write(f"{r.label},{r.doc_recall_5},{r.page_recall_5},{r.bleu_5},{r.rougeL_5},{r.mrr}\n")
        print(f"\nSaved CSV to: {out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
