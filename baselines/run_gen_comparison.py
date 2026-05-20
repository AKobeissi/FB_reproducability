#!/usr/bin/env python3
"""
run_gen_comparison.py
=====================
Head-to-head generative comparison: Qwen2.5-7B-Instruct vs Llama-3.1-8B-Instruct.

Retrieval is held fixed (multi_hyde_ft_reranker by default — best pipeline).
Each model generates answers for all 150 FinanceBench questions.

Metrics
-------
  ROUGE-L           vs reference answer
  BERTScore F1      (roberta-large, rescaled)
  NumericMatch      within ±3 % tolerance
                    — metrics-generated:  rate over the 50 numeric questions only
                    — all question types: rate over all 150 questions

Outputs  →  outputs/gen_comparison/
  summary.json / summary.csv              overall metrics per model
  by_question_type.json / .csv            breakdown by question type
  by_doc_type.json / .csv                 breakdown by doc type (10k / 10q / 8k)
  predictions/{model}.json                annotated per-sample predictions
  plots/                                  comparison figures
"""

import argparse
import copy
import json
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gen_comparison")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_RETRIEVAL_FILE = (
    PROJECT_ROOT / "baselines/results/predictions/multi_hyde_ft_reranker_retrieval.json"
)
DEFAULT_OUTPUT_DIR    = PROJECT_ROOT / "outputs/gen_comparison"
DOC_INFO_FILE         = PROJECT_ROOT / "data/financebench_document_information.jsonl"
FINANCEBENCH_FILE     = PROJECT_ROOT / "data/financebench_open_source.jsonl"

MAIN_K             = 5
MAX_CONTEXT_TOKENS = 3600

MODELS: Dict[str, str] = {
    "Qwen2.5-7B":  "Qwen/Qwen2.5-7B-Instruct",
    "Llama3.1-8B": "meta-llama/Llama-3.1-8B-Instruct",
}

QUESTION_TYPES = ["metrics-generated", "domain-relevant", "novel-generated"]

GEN_PROMPT = (
    "You are a financial analyst answering questions based on SEC filings. "
    "Use ONLY the provided context. If the context does not contain the answer, "
    "say 'I cannot determine this from the provided information.' "
    "Be concise and precise, especially for numerical answers.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_doc_info(path: Path) -> Dict[str, Dict]:
    """Return {doc_name → {doc_type, gics_sector, doc_period, company}}."""
    info: Dict[str, Dict] = {}
    if not path.exists():
        logger.warning(f"doc_info not found: {path}")
        return info
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                info[r["doc_name"]] = r
    return info


def annotate_doc_type(samples: List[Dict], doc_info: Dict[str, Dict]) -> List[Dict]:
    """Attach doc_type and gics_sector to each sample in-place."""
    for s in samples:
        meta = doc_info.get(s.get("doc_name", ""), {})
        s["doc_type"]    = meta.get("doc_type", "unknown")
        s["gics_sector"] = meta.get("gics_sector", "unknown")
    return samples


def load_gold_evidence(path: Path) -> Dict[str, str]:
    """Return {financebench_id → concatenated gold evidence text}."""
    gold: Dict[str, str] = {}
    if not path.exists():
        return gold
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            fid = str(r.get("financebench_id", ""))
            pieces = [ev.get("evidence_text", "")
                      for ev in r.get("evidence", [])
                      if ev.get("evidence_text")]
            if pieces:
                gold[fid] = "\n\n".join(pieces)
    return gold


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_with_model(
    samples: List[Dict],
    model_name: str,
    label: str,
) -> List[Dict]:
    """
    Load *model_name* in 4-bit, generate answers for all samples, free GPU.
    Returns a new list with 'generated_answer' populated.
    """
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
    )

    logger.info(f"Loading {label} ({model_name}) in 4-bit…")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    results = copy.deepcopy(samples)
    for sample in tqdm(results, desc=f"Generating ({label})"):
        chunks = sample.get("retrieved_chunks", [])[:MAIN_K]

        context_parts: List[str] = []
        budget = MAX_CONTEXT_TOKENS
        for chunk in chunks:
            if budget <= 0:
                break
            ids = tokenizer.encode(chunk["text"], add_special_tokens=False)
            if len(ids) <= budget:
                context_parts.append(chunk["text"])
                budget -= len(ids)
            elif budget >= 50:
                context_parts.append(
                    tokenizer.decode(ids[:budget], skip_special_tokens=True)
                )
                budget = 0

        context = "\n\n".join(context_parts)
        prompt  = GEN_PROMPT.format(context=context, question=sample["question"])

        msgs = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, "apply_chat_template"):
            formatted = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted = prompt

        inputs = tokenizer(
            formatted, return_tensors="pt", truncation=True, max_length=4096
        ).to(model.device)

        try:
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=350,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            plen = inputs["input_ids"].shape[-1]
            sample["generated_answer"] = tokenizer.decode(
                out[0][plen:], skip_special_tokens=True
            ).strip()
        except Exception as e:
            logger.warning(
                f"Generation failed for {sample.get('financebench_id')}: {e}"
            )
            sample["generated_answer"] = ""

    del model
    import torch as _torch
    if _torch.cuda.is_available():
        _torch.cuda.empty_cache()
    logger.info(f"Freed {label} from GPU.")
    return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

_SCALE_PATTERNS: List[Tuple] = [
    (re.compile(r"\btrillion[s]?\b",               re.I), 1e12),
    (re.compile(r"\bbillion[s]?\b|\bbn\b|\bbln\b", re.I), 1e9),
    (re.compile(r"\bmillion[s]?\b|\bmm\b|\bmln\b", re.I), 1e6),
    (re.compile(r"\bthousand[s]?\b|\bk\b",         re.I), 1e3),
]


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_main_number(text: str) -> Optional[float]:
    text = re.sub(r"[$,€£%()]", "", text)
    m = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if not m:
        return None
    try:
        return float(m[0].replace(",", ""))
    except ValueError:
        return None


def _extract_all_numbers(text: str) -> List[float]:
    text = re.sub(r"[$,€£%()]", "", text)
    out = []
    for m in re.findall(r"-?\d[\d,]*\.?\d*", text):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def _scaled_value(raw: float, text: str) -> float:
    for pat, factor in _SCALE_PATTERNS:
        if pat.search(text):
            return raw * factor
    return raw


def numeric_match(pred: str, ref: str, rtol: float = 0.03) -> float:
    ref_raw = _extract_main_number(ref)
    if ref_raw is None:
        return 0.0
    ref_scaled = _scaled_value(ref_raw, ref)
    pred_nums  = _extract_all_numbers(pred)
    if not pred_nums:
        return 0.0
    ref_has_scale  = any(p.search(ref)  for p, _ in _SCALE_PATTERNS)
    pred_has_scale = any(p.search(pred) for p, _ in _SCALE_PATTERNS)
    for p_raw in pred_nums:
        p_scaled = _scaled_value(p_raw, pred)
        if ref_has_scale or pred_has_scale:
            denom = abs(ref_scaled) if ref_scaled != 0 else 1.0
            if abs(p_scaled - ref_scaled) / denom <= rtol:
                return 1.0
        if ref_raw == 0:
            if p_raw == 0:
                return 1.0
        else:
            if abs(p_raw - ref_raw) / abs(ref_raw) <= rtol:
                return 1.0
    return 0.0


def compute_metrics(samples: List[Dict], bert_scorer=None) -> Dict:
    """
    Compute ROUGE-L, BERTScore, and two NumericMatch variants:
      numeric_match     — mean over metrics-generated questions only
      numeric_match_all — mean over ALL questions
    """
    from rouge_score import rouge_scorer as rs_mod

    scorer = rs_mod.RougeScorer(["rougeL"], use_stemmer=True)

    rougeL_scores: List[float] = []
    nm_metrics:    List[float] = []   # metrics-generated only
    nm_all:        List[float] = []   # all questions

    for s in samples:
        gen = s.get("generated_answer", "").strip()
        ref = s.get("reference_answer",  "").strip()

        if gen and ref:
            r = scorer.score(ref, gen)
            rougeL_scores.append(r["rougeL"].fmeasure)
        else:
            rougeL_scores.append(0.0)

        nm_score = numeric_match(gen, ref)
        nm_all.append(nm_score)
        if s.get("question_type") == "metrics-generated":
            nm_metrics.append(nm_score)

    def _mean(lst: List[float]) -> float:
        return float(np.mean(lst)) if lst else 0.0

    out: Dict = {
        "rougeL":            _mean(rougeL_scores),
        "numeric_match":     _mean(nm_metrics),
        "numeric_match_all": _mean(nm_all),
        "n_samples":         len(samples),
        "n_metrics_qs":      len(nm_metrics),
    }

    if bert_scorer is not None:
        preds     = [s.get("generated_answer", "").strip() for s in samples]
        refs      = [s.get("reference_answer",  "").strip() for s in samples]
        valid_idx = [i for i, (p, r) in enumerate(zip(preds, refs)) if p and r]
        bs = [0.0] * len(samples)
        if valid_idx:
            try:
                _, _, F1 = bert_scorer.score(
                    [preds[i] for i in valid_idx],
                    [refs[i]  for i in valid_idx],
                    verbose=False, batch_size=32,
                )
                for i, f1 in zip(valid_idx, F1.tolist()):
                    bs[i] = float(f1)
            except Exception as e:
                logger.warning(f"BERTScore failed: {e}")
        out["bertscore_f1"] = _mean(bs)

    return out


def compute_breakdown(
    samples: List[Dict],
    key_fn,
    bert_scorer=None,
) -> Dict[str, Dict]:
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for s in samples:
        groups[key_fn(s)].append(s)
    return {k: compute_metrics(v, bert_scorer) for k, v in sorted(groups.items())}


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

METRIC_COLS = [
    ("rougeL",            "ROUGE-L"),
    ("bertscore_f1",      "BERTScore-F1"),
    ("numeric_match",     "NumMatch(metrics)"),
    ("numeric_match_all", "NumMatch(all)"),
]


def print_table(title: str, metrics: Dict[str, Dict]) -> None:
    col_w = 18
    header = f"{'Model':<20}" + "".join(f"  {label:>{col_w}}" for _, label in METRIC_COLS)
    print(f"\n{'='*len(header)}")
    print(f"  {title}")
    print("="*len(header))
    print(header)
    print("-"*len(header))
    for model_label, m in metrics.items():
        row = f"{model_label:<20}"
        for key, _ in METRIC_COLS:
            val = m.get(key, float("nan"))
            row += f"  {val:>{col_w}.4f}"
        print(row)
    print("="*len(header))


def print_breakdown_table(title: str, breakdown: Dict[str, Dict[str, Dict]]) -> None:
    """breakdown: {model_label → {group → metrics}}"""
    all_groups = sorted({g for m in breakdown.values() for g in m})
    col_w = 16
    group_w = 24

    print(f"\n{'='*80}")
    print(f"  {title}")
    print("="*80)

    for _, metric_label in METRIC_COLS:
        print(f"\n  {metric_label}")
        header = f"  {'Group':<{group_w}}" + "".join(
            f"  {lbl:>{col_w}}" for lbl in breakdown
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for g in all_groups:
            row = f"  {g:<{group_w}}"
            for model_label in breakdown:
                val = breakdown[model_label].get(g, {}).get(
                    [k for k, l in METRIC_COLS if l == metric_label][0], float("nan")
                )
                row += f"  {val:>{col_w}.4f}"
            print(row)


def save_csv(path: Path, metrics: Dict[str, Dict]) -> None:
    col_keys = [k for k, _ in METRIC_COLS] + ["n_samples", "n_metrics_qs"]
    lines = ["model," + ",".join(col_keys)]
    for label, m in metrics.items():
        lines.append(label + "," + ",".join(str(m.get(k, "")) for k in col_keys))
    path.write_text("\n".join(lines) + "\n")


def save_breakdown_csv(path: Path, breakdown: Dict[str, Dict[str, Dict]]) -> None:
    """breakdown: {model → {group → metrics}}"""
    col_keys = [k for k, _ in METRIC_COLS] + ["n_samples", "n_metrics_qs"]
    lines = ["model,group," + ",".join(col_keys)]
    for model_label, groups in breakdown.items():
        for g, m in sorted(groups.items()):
            lines.append(
                f"{model_label},{g}," + ",".join(str(m.get(k, "")) for k in col_keys)
            )
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def save_plots(
    overall:  Dict[str, Dict],
    by_qtype: Dict[str, Dict[str, Dict]],
    by_dtype: Dict[str, Dict[str, Dict]],
    plots_dir: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping plots.")
        return

    plots_dir.mkdir(parents=True, exist_ok=True)
    model_labels = list(overall.keys())
    colors       = ["#3F51B5", "#E91E63", "#FF9800", "#4CAF50"]
    n_models     = len(model_labels)
    x            = np.arange(n_models)

    # ── Plot 1: Overall bar chart ──────────────────────────────────────────
    metrics_to_plot = [
        ("rougeL",            "ROUGE-L",          "#3F51B5"),
        ("bertscore_f1",      "BERTScore-F1",     "#4CAF50"),
        ("numeric_match",     "NumMatch(metrics)", "#E91E63"),
        ("numeric_match_all", "NumMatch(all)",     "#FF9800"),
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.18
    for i, (key, lbl, col) in enumerate(metrics_to_plot):
        vals   = [overall[m].get(key, 0) for m in model_labels]
        offset = (i - 1.5) * width
        bars   = ax.bar(x + offset, vals, width, label=lbl, color=col, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=12)
    ax.set_ylabel("Score")
    ax.set_title("Qwen2.5-7B vs Llama3.1-8B — Generative Quality (Best RAG Pipeline)")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_ylim(0, min(1.0, ax.get_ylim()[1] * 1.2))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(plots_dir / f"overall_comparison.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {plots_dir}/overall_comparison.*")

    # ── Plot 2: By question type (ROUGE-L + NumMatch side by side) ─────────
    qt_list   = [qt for qt in QUESTION_TYPES
                 if any(qt in by_qtype.get(m, {}) for m in model_labels)]
    n_qt      = len(qt_list)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (metric_key, metric_lbl) in zip(
        axes, [("rougeL", "ROUGE-L"), ("numeric_match_all", "NumMatch (all qs)")]
    ):
        width = 0.35 if n_models == 2 else 0.2
        for mi, (mlab, col) in enumerate(zip(model_labels, colors)):
            vals   = [by_qtype.get(mlab, {}).get(qt, {}).get(metric_key, 0)
                      for qt in qt_list]
            offset = (mi - (n_models - 1) / 2) * width
            bars   = ax.bar(np.arange(n_qt) + offset, vals, width,
                            label=mlab, color=col, alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(np.arange(n_qt))
        ax.set_xticklabels([qt.replace("-", "\n") for qt in qt_list], fontsize=10)
        ax.set_ylabel(metric_lbl)
        ax.set_title(f"{metric_lbl} by Question Type")
        ax.legend(fontsize=9)
        ax.set_ylim(0, min(1.0, ax.get_ylim()[1] * 1.2))
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(plots_dir / f"by_question_type.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {plots_dir}/by_question_type.*")

    # ── Plot 3: By doc type (ROUGE-L) ─────────────────────────────────────
    dt_list = sorted({dt for m in model_labels for dt in by_dtype.get(m, {})})
    if dt_list:
        fig, ax = plt.subplots(figsize=(10, 5))
        width = 0.35 if n_models == 2 else 0.2
        for mi, (mlab, col) in enumerate(zip(model_labels, colors)):
            vals   = [by_dtype.get(mlab, {}).get(dt, {}).get("rougeL", 0)
                      for dt in dt_list]
            offset = (mi - (n_models - 1) / 2) * width
            bars   = ax.bar(np.arange(len(dt_list)) + offset, vals, width,
                            label=mlab, color=col, alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(np.arange(len(dt_list)))
        ax.set_xticklabels([dt.upper() for dt in dt_list], fontsize=11)
        ax.set_ylabel("ROUGE-L")
        ax.set_title("ROUGE-L by Document Type")
        ax.legend(fontsize=9)
        ax.set_ylim(0, min(1.0, ax.get_ylim()[1] * 1.2))
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(plots_dir / f"by_doc_type.{ext}", bbox_inches="tight", dpi=150)
        plt.close(fig)
        logger.info(f"Saved: {plots_dir}/by_doc_type.*")

    # ── Plot 4: NumericMatch by question type (grouped bar) ────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.35 if n_models == 2 else 0.2
    for mi, (mlab, col) in enumerate(zip(model_labels, colors)):
        vals   = [by_qtype.get(mlab, {}).get(qt, {}).get("numeric_match_all", 0)
                  for qt in qt_list]
        offset = (mi - (n_models - 1) / 2) * width
        bars   = ax.bar(np.arange(n_qt) + offset, vals, width,
                        label=mlab, color=col, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(np.arange(n_qt))
    ax.set_xticklabels([qt.replace("-", "\n") for qt in qt_list], fontsize=10)
    ax.set_ylabel("Numeric Match Rate")
    ax.set_title("Numeric Match by Question Type")
    ax.legend(fontsize=9)
    ax.set_ylim(0, min(1.0, ax.get_ylim()[1] * 1.2))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(plots_dir / f"numeric_by_qtype.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {plots_dir}/numeric_by_qtype.*")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retrieval-file", type=Path, default=DEFAULT_RETRIEVAL_FILE,
        help="Pre-computed retrieval JSON (default: multi_hyde_ft_reranker)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--models", nargs="*", default=None,
        help="Subset of model labels to run (default: all). "
             f"Available: {list(MODELS.keys())}",
    )
    parser.add_argument(
        "--skip-generation", action="store_true",
        help="Skip generation; re-evaluate existing prediction files only.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    preds_dir  = output_dir / "predictions"
    plots_dir  = output_dir / "plots"
    for d in (output_dir, preds_dir, plots_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Select which models to run
    models_to_run: Dict[str, str] = (
        {k: v for k, v in MODELS.items() if k in args.models}
        if args.models else MODELS
    )

    # Load data
    logger.info(f"Loading retrieval results from {args.retrieval_file}")
    if not args.retrieval_file.exists():
        raise FileNotFoundError(f"Retrieval file not found: {args.retrieval_file}")
    base_samples = json.load(open(args.retrieval_file))
    logger.info(f"Loaded {len(base_samples)} samples.")

    doc_info = load_doc_info(DOC_INFO_FILE)
    base_samples = annotate_doc_type(base_samples, doc_info)

    # BERTScorer — loaded once, reused for all models
    bert_scorer = None
    try:
        import torch
        from bert_score import BERTScorer as _BS
        device = "cuda" if torch.cuda.is_available() else "cpu"
        bert_scorer = _BS(lang="en", rescale_with_baseline=True, device=device)
        logger.info(f"BERTScorer ready (roberta-large, rescaled, {device}).")
    except Exception as e:
        logger.warning(f"BERTScore disabled: {e}")

    # Per-model results
    overall:  Dict[str, Dict] = {}
    by_qtype: Dict[str, Dict[str, Dict]] = {}
    by_dtype: Dict[str, Dict[str, Dict]] = {}

    for label, model_name in models_to_run.items():
        pred_file = preds_dir / f"{label}.json"

        # Generation
        if pred_file.exists() or args.skip_generation:
            if pred_file.exists():
                logger.info(f"[{label}] Loading existing predictions…")
                samples = json.load(open(pred_file))
                # Re-annotate doc_type in case it's missing
                samples = annotate_doc_type(samples, doc_info)
            else:
                logger.error(f"[{label}] No prediction file found and --skip-generation set.")
                continue
        else:
            samples = generate_with_model(base_samples, model_name, label)
            samples = annotate_doc_type(samples, doc_info)

        # Save predictions (with metrics)
        with open(pred_file, "w") as f:
            json.dump(samples, f, indent=2)

        # Evaluate
        logger.info(f"[{label}] Computing metrics…")
        overall[label]  = compute_metrics(samples, bert_scorer)
        by_qtype[label] = compute_breakdown(
            samples, lambda s: s.get("question_type", "unknown"), bert_scorer
        )
        by_dtype[label] = compute_breakdown(
            samples, lambda s: s.get("doc_type", "unknown"), bert_scorer
        )

        m = overall[label]
        logger.info(
            f"[{label}]  ROUGE-L={m['rougeL']:.4f}  "
            f"BERTScore={m.get('bertscore_f1', float('nan')):.4f}  "
            f"NumMatch(metrics)={m['numeric_match']:.4f}  "
            f"NumMatch(all)={m['numeric_match_all']:.4f}"
        )

    if not overall:
        logger.error("No results to report.")
        return

    # Print tables
    print_table("OVERALL", overall)
    print_breakdown_table("BY QUESTION TYPE", by_qtype)
    print_breakdown_table("BY DOCUMENT TYPE", by_dtype)

    # Save JSON
    with open(output_dir / "summary.json", "w") as f:
        json.dump(overall, f, indent=2)
    with open(output_dir / "by_question_type.json", "w") as f:
        json.dump(by_qtype, f, indent=2)
    with open(output_dir / "by_doc_type.json", "w") as f:
        json.dump(by_dtype, f, indent=2)

    # Save CSVs
    save_csv(output_dir / "summary.csv", overall)
    save_breakdown_csv(output_dir / "by_question_type.csv", by_qtype)
    save_breakdown_csv(output_dir / "by_doc_type.csv",      by_dtype)

    logger.info(f"All results saved to {output_dir}")

    # Plots
    save_plots(overall, by_qtype, by_dtype, plots_dir)


if __name__ == "__main__":
    main()
