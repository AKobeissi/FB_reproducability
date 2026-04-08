#!/usr/bin/env python3
"""
llm_scaling_study.py
====================
Studies the impact of LLM scale on RAG generation quality.

Retrieval is held fixed (dense BGE-M3, top-5 chunks from the pre-computed
baseline). Only the generator changes:
  - Qwen/Qwen2.5-3B-Instruct
  - Qwen/Qwen2.5-7B-Instruct
  - Qwen/Qwen2.5-14B-Instruct

Metrics (no LLM judge):
  - ROUGE-1 / ROUGE-2 / ROUGE-L   (vs reference answer)
  - BLEU-4                         (vs reference answer)
  - Exact match                    (normalised string match)
  - Numeric match                  (within 3% tolerance, metrics-generated only)

All metrics are reported overall and broken down by question_type.
"""

import copy
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("llm_scaling")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RETRIEVAL_FILE    = PROJECT_ROOT / "baselines/results/predictions/dense_bge_m3_retrieval.json"
FINANCEBENCH_FILE = PROJECT_ROOT / "data/financebench_open_source.jsonl"
OUTPUT_DIR        = PROJECT_ROOT / "outputs/llm_scaling"
MAIN_K            = 5    # top-k chunks to pass to generator
# Token budget reserved for context.  The remainder of the 4096-token window
# covers: system prompt (~50 tok), question (~60 tok), chat-template overhead
# (~30 tok), and max_new_tokens (200).  3700 is conservative but fits the
# median full context (≈3 650 tok) and is well within the L40S memory budget.
MAX_CONTEXT_TOKENS = 3700

MODELS = [
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
]

MODEL_LABELS = {
    "Qwen/Qwen2.5-3B-Instruct":  "Qwen2.5-3B",
    "Qwen/Qwen2.5-7B-Instruct":  "Qwen2.5-7B",
    "Qwen/Qwen2.5-14B-Instruct": "Qwen2.5-14B",
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
# Generation
# ---------------------------------------------------------------------------

def generate_with_model(samples: List[Dict], model_name: str) -> List[Dict]:
    """
    Load model_name in 4-bit, generate answers for all samples, free GPU memory.
    Returns a new list of samples with 'generated_answer' populated.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    logger.info(f"Loading {model_name} (4-bit)…")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    results = copy.deepcopy(samples)
    for sample in tqdm(results, desc=f"Generating ({MODEL_LABELS[model_name]})"):
        chunks = sample.get("retrieved_chunks", [])[:MAIN_K]

        # Build context token-by-token so the budget is exact and the question
        # is never at risk of being silently truncated.  We fill whole chunks
        # first; if a chunk would overflow the budget we fit as much of it as
        # the remaining space allows (minimum 50 tokens, otherwise skip).
        context_parts: list = []
        budget = MAX_CONTEXT_TOKENS
        for chunk in chunks:
            if budget <= 0:
                break
            ids = tokenizer.encode(chunk["text"], add_special_tokens=False)
            if len(ids) <= budget:
                context_parts.append(chunk["text"])
                budget -= len(ids)
            elif budget >= 50:
                # Fit a partial chunk — at least 50 tokens worth
                context_parts.append(
                    tokenizer.decode(ids[:budget], skip_special_tokens=True)
                )
                budget = 0
        context = "\n\n".join(context_parts)
        prompt = GEN_PROMPT.format(context=context, question=sample["question"])

        if hasattr(tokenizer, "apply_chat_template"):
            msgs = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(
                formatted, return_tensors="pt", truncation=True, max_length=4096
            ).to(model.device)
        else:
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=4096
            ).to(model.device)

        try:
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            plen = inputs["input_ids"].shape[-1]
            sample["generated_answer"] = tokenizer.decode(
                out[0][plen:], skip_special_tokens=True
            ).strip()
        except Exception as e:
            logger.warning(f"Generation failed for sample {sample.get('financebench_id')}: {e}")
            sample["generated_answer"] = ""

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info(f"Freed {model_name} from GPU.")
    return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_main_number(text: str) -> Optional[float]:
    text = re.sub(r"[$,€£%()]", "", text)
    matches = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if not matches:
        return None
    try:
        return float(matches[0].replace(",", ""))
    except ValueError:
        return None


def numeric_match(pred: str, ref: str, rtol: float = 0.03) -> float:
    pn, rn = _extract_main_number(pred), _extract_main_number(ref)
    if pn is None or rn is None:
        return 0.0
    if rn == 0:
        return 1.0 if pn == 0 else 0.0
    return 1.0 if abs(pn - rn) / abs(rn) <= rtol else 0.0


def exact_match(pred: str, ref: str) -> float:
    return 1.0 if _normalize(pred) == _normalize(ref) else 0.0


def compute_metrics(samples: List[Dict], bert_scorer=None) -> Dict:
    """Compute all non-LLM generative metrics over a list of samples."""
    from rouge_score import rouge_scorer as rs_module
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        _has_nltk = True
    except ImportError:
        _has_nltk = False

    scorer = rs_module.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    smooth = SmoothingFunction().method1 if _has_nltk else None

    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
    bleu4_scores = []
    em_scores = []
    numeric_scores = []
    numeric_samples = []

    for s in samples:
        gen = s.get("generated_answer", "").strip()
        ref = s.get("reference_answer", "").strip()

        if gen and ref:
            r = scorer.score(ref, gen)
            rouge1_scores.append(r["rouge1"].fmeasure)
            rouge2_scores.append(r["rouge2"].fmeasure)
            rougeL_scores.append(r["rougeL"].fmeasure)
            em_scores.append(exact_match(gen, ref))

            if _has_nltk:
                ref_tokens  = _normalize(ref).split()
                pred_tokens = _normalize(gen).split()
                if ref_tokens and pred_tokens:
                    bleu4_scores.append(
                        sentence_bleu([ref_tokens], pred_tokens,
                                      smoothing_function=smooth)
                    )
                else:
                    bleu4_scores.append(0.0)
        else:
            rouge1_scores.append(0.0)
            rouge2_scores.append(0.0)
            rougeL_scores.append(0.0)
            em_scores.append(0.0)
            if _has_nltk:
                bleu4_scores.append(0.0)

        if s.get("question_type") == "metrics-generated":
            numeric_scores.append(numeric_match(gen, ref))
            numeric_samples.append(s)

    def _mean(lst):
        return float(np.mean(lst)) if lst else 0.0

    out = {
        "rouge1":        _mean(rouge1_scores),
        "rouge2":        _mean(rouge2_scores),
        "rougeL":        _mean(rougeL_scores),
        "exact_match":   _mean(em_scores),
        "numeric_match": _mean(numeric_scores),
        "n_samples":     len(samples),
        "n_metrics_qs":  len(numeric_scores),
    }
    if _has_nltk and bleu4_scores:
        out["bleu4"] = _mean(bleu4_scores)

    # BERTScore — computed in a single batch for efficiency.
    # Uses roberta-large (lang="en") with baseline rescaling so scores sit in a
    # more interpretable range (rescaled F1 ≈ 0 means avg performance).
    # Empty / refusal predictions contribute 0.0 to the mean so the penalty for
    # failing to answer is preserved.
    if bert_scorer is not None:
        preds = [s.get("generated_answer", "").strip() for s in samples]
        refs  = [s.get("reference_answer",  "").strip() for s in samples]

        # Identify pairs where both sides are non-empty
        valid_idx = [i for i, (p, r) in enumerate(zip(preds, refs)) if p and r]

        bs_scores = [0.0] * len(samples)
        if valid_idx:
            valid_preds = [preds[i] for i in valid_idx]
            valid_refs  = [refs[i]  for i in valid_idx]
            try:
                _, _, F1 = bert_scorer.score(valid_preds, valid_refs,
                                             verbose=False, batch_size=32)
                for i, f1 in zip(valid_idx, F1.tolist()):
                    bs_scores[i] = float(f1)
            except Exception as e:
                logger.warning(f"BERTScore batch failed: {e}")

        out["bertscore_f1"] = _mean(bs_scores)

    return out


def compute_breakdown(samples: List[Dict], bert_scorer=None) -> Dict[str, Dict]:
    """Return per-question-type metrics."""
    from collections import defaultdict
    groups: Dict[str, List] = defaultdict(list)
    for s in samples:
        groups[s.get("question_type", "unknown")].append(s)
    return {qt: compute_metrics(gs, bert_scorer) for qt, gs in groups.items()}


# ---------------------------------------------------------------------------
# Oracle-context helpers
# ---------------------------------------------------------------------------

def load_gold_evidence(fb_path: Path) -> Dict[str, str]:
    """
    Return {financebench_id → gold_evidence_text} from the FinanceBench JSONL.
    We concatenate all evidence segments for the same question.
    """
    gold: Dict[str, str] = {}
    if not fb_path.exists():
        logger.warning(f"FinanceBench data file not found: {fb_path}. Oracle mode disabled.")
        return gold
    with open(fb_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            fid = str(raw.get("financebench_id", ""))
            pieces = [ev.get("evidence_text", "") for ev in raw.get("evidence", []) if ev.get("evidence_text")]
            gold[fid] = "\n\n".join(pieces)
    logger.info(f"Loaded gold evidence for {len(gold)} questions.")
    return gold


def build_oracle_samples(base_samples: List[Dict], gold_evidence: Dict[str, str]) -> List[Dict]:
    """
    Replace retrieved_chunks context with the gold evidence page text.
    Samples without gold evidence are kept but will have empty context.
    """
    oracle = copy.deepcopy(base_samples)
    n_found = 0
    for s in oracle:
        fid = str(s.get("financebench_id", ""))
        evidence = gold_evidence.get(fid, "")
        if evidence:
            n_found += 1
        # Overwrite retrieved_chunks with a single pseudo-chunk containing gold text
        s["retrieved_chunks"] = [{"text": evidence}] if evidence else []
    logger.info(f"Oracle samples: {n_found}/{len(oracle)} have gold evidence.")
    return oracle


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_table(all_metrics: Dict[str, Dict]) -> None:
    cols = ["rougeL", "rouge1", "rouge2", "bertscore_f1", "exact_match", "numeric_match"]
    header = f"{'Model':<22}" + "".join(f"  {c:>14}" for c in cols)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for label, m in all_metrics.items():
        row = f"{label:<22}"
        for c in cols:
            val = m.get(c, float('nan'))
            row += f"  {val:>14.4f}"
        print(row)
    print("=" * len(header) + "\n")


def save_plots(all_metrics: Dict[str, Dict], breakdown: Dict[str, Dict[str, Dict]],
               plots_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping plots.")
        return

    plots_dir.mkdir(parents=True, exist_ok=True)
    labels = list(all_metrics.keys())
    x = np.arange(len(labels))

    # ── Plot 1: Overall metrics bar chart ─────────────────────────────────
    metrics_to_plot = [
        ("rougeL",        "ROUGE-L",         "#3F51B5"),
        ("rouge1",        "ROUGE-1",         "#4CAF50"),
        ("rouge2",        "ROUGE-2",         "#FF9800"),
        ("numeric_match", "Numeric Match",   "#E91E63"),
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.18
    for i, (key, label, color) in enumerate(metrics_to_plot):
        vals = [all_metrics[l].get(key, 0) for l in labels]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, vals, width, label=label, color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Score")
    ax.set_title("LLM Scaling Impact on Generation Quality (Fixed BGE-M3 Retrieval)")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(0, min(1.0, ax.get_ylim()[1] * 1.15))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(plots_dir / f"scaling_overall.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {plots_dir}/scaling_overall.*")

    # ── Plot 2: ROUGE-L by question type ──────────────────────────────────
    qt_colors = {
        "metrics-generated": "#E91E63",
        "domain-relevant":   "#3F51B5",
        "novel-generated":   "#4CAF50",
    }
    fig, ax = plt.subplots(figsize=(9, 5))
    qt_list = [qt for qt in QUESTION_TYPES if any(
        qt in breakdown.get(l, {}) for l in labels
    )]
    width = 0.25 if len(labels) == 3 else 0.2
    for i, label in enumerate(labels):
        vals = [breakdown.get(label, {}).get(qt, {}).get("rougeL", 0) for qt in qt_list]
        offset = (i - (len(labels) - 1) / 2) * width
        bars = ax.bar(np.arange(len(qt_list)) + offset, vals, width,
                      label=label, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(np.arange(len(qt_list)))
    ax.set_xticklabels([qt.replace("-", "\n") for qt in qt_list], fontsize=10)
    ax.set_ylabel("ROUGE-L")
    ax.set_title("ROUGE-L by Question Type — LLM Scaling")
    ax.legend(fontsize=9)
    ax.set_ylim(0, min(1.0, ax.get_ylim()[1] * 1.15))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(plots_dir / f"scaling_by_qtype.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {plots_dir}/scaling_by_qtype.*")

    # ── Plot 3: Numeric match for metrics-generated questions ─────────────
    nm_vals = [all_metrics[l].get("numeric_match", 0) for l in labels]
    n_qs    = [all_metrics[l].get("n_metrics_qs", 0) for l in labels]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, nm_vals, color="#E91E63", alpha=0.85, width=0.4)
    for bar, v, n in zip(bars, nm_vals, n_qs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.3f}\n(n={n})", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Numeric Match Rate")
    ax.set_title("Numeric Match — Metrics-Generated Questions")
    ax.set_ylim(0, min(1.0, max(nm_vals) * 1.3 + 0.1))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(plots_dir / f"scaling_numeric_match.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {plots_dir}/scaling_numeric_match.*")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plots_dir = OUTPUT_DIR / "plots"
    preds_dir = OUTPUT_DIR / "predictions"
    oracle_dir = OUTPUT_DIR / "oracle_predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)
    oracle_dir.mkdir(parents=True, exist_ok=True)

    # Initialise BERTScorer once so the model is loaded a single time and
    # reused across all compute_metrics / compute_breakdown calls.
    # rescale_with_baseline=True maps raw cosine similarities onto a more
    # interpretable range using pre-computed language-specific baselines,
    # which is standard practice for reporting BERTScore in papers.
    bert_scorer = None
    try:
        import torch
        from bert_score import BERTScorer as _BERTScorer
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        bert_scorer = _BERTScorer(
            lang="en",
            rescale_with_baseline=True,
            device=_device,
        )
        logger.info(f"BERTScorer ready (roberta-large, rescaled, device={_device}).")
    except Exception as e:
        logger.warning(f"BERTScore disabled: {e}")

    # Load pre-computed retrieval results (fixed across all models)
    logger.info(f"Loading retrieval results from {RETRIEVAL_FILE}")
    if not RETRIEVAL_FILE.exists():
        raise FileNotFoundError(
            f"Retrieval file not found: {RETRIEVAL_FILE}\n"
            "Run the dense BGE-M3 baseline first (scripts/run_baselines.sh)."
        )
    base_samples = json.load(open(RETRIEVAL_FILE))
    logger.info(f"Loaded {len(base_samples)} samples.")

    # Build oracle-context samples (gold evidence instead of retrieved chunks)
    gold_evidence  = load_gold_evidence(FINANCEBENCH_FILE)
    oracle_samples = build_oracle_samples(base_samples, gold_evidence)
    has_oracle     = bool(gold_evidence)

    all_metrics:        Dict[str, Dict] = {}
    all_breakdown:      Dict[str, Dict] = {}
    oracle_metrics:     Dict[str, Dict] = {}
    oracle_breakdown:   Dict[str, Dict] = {}

    for model_name in MODELS:
        label = MODEL_LABELS[model_name]

        # ── Standard (retrieved context) ──────────────────────────────────
        pred_file = preds_dir / f"{label.replace('/', '_')}.json"
        if pred_file.exists():
            logger.info(f"Found existing predictions for {label}, loading…")
            samples = json.load(open(pred_file))
        else:
            samples = generate_with_model(base_samples, model_name)
            with open(pred_file, "w") as f:
                json.dump(samples, f, indent=2)
            logger.info(f"Saved predictions: {pred_file}")

        all_metrics[label]   = compute_metrics(samples, bert_scorer)
        all_breakdown[label] = compute_breakdown(samples, bert_scorer)

        logger.info(
            f"{label}  ROUGE-L={all_metrics[label]['rougeL']:.4f}  "
            f"NumMatch={all_metrics[label]['numeric_match']:.4f}  "
            f"(n_metrics={all_metrics[label]['n_metrics_qs']})"
        )

        # ── Oracle context ─────────────────────────────────────────────────
        if has_oracle:
            oracle_pred_file = oracle_dir / f"{label.replace('/', '_')}_oracle.json"
            if oracle_pred_file.exists():
                logger.info(f"Found existing oracle predictions for {label}, loading…")
                oracle_preds = json.load(open(oracle_pred_file))
            else:
                oracle_preds = generate_with_model(oracle_samples, model_name)
                with open(oracle_pred_file, "w") as f:
                    json.dump(oracle_preds, f, indent=2)
                logger.info(f"Saved oracle predictions: {oracle_pred_file}")

            oracle_metrics[label]   = compute_metrics(oracle_preds, bert_scorer)
            oracle_breakdown[label] = compute_breakdown(oracle_preds, bert_scorer)

            logger.info(
                f"{label} [ORACLE]  ROUGE-L={oracle_metrics[label]['rougeL']:.4f}  "
                f"NumMatch={oracle_metrics[label]['numeric_match']:.4f}"
            )

    # Summary tables
    print("\n=== STANDARD (Retrieved Context) ===")
    print_table(all_metrics)

    if oracle_metrics:
        print("\n=== ORACLE (Gold Evidence Context) ===")
        print_table(oracle_metrics)

        # Print delta table to clearly show generation-only scaling effect
        cols = ["rougeL", "numeric_match"]
        print(f"\n{'':22}  {'ROUGE-L (std)':>14}  {'ROUGE-L (oracle)':>16}  {'NumMatch (std)':>14}  {'NumMatch (oracle)':>17}")
        print("-" * 90)
        for label in all_metrics:
            sm = all_metrics.get(label, {})
            om = oracle_metrics.get(label, {})
            print(
                f"{label:<22}  {sm.get('rougeL', 0):>14.4f}  {om.get('rougeL', 0):>16.4f}"
                f"  {sm.get('numeric_match', 0):>14.4f}  {om.get('numeric_match', 0):>17.4f}"
            )

    # Save JSON outputs
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    with open(OUTPUT_DIR / "breakdown_by_qtype.json", "w") as f:
        json.dump(all_breakdown, f, indent=2)

    if oracle_metrics:
        with open(OUTPUT_DIR / "oracle_summary.json", "w") as f:
            json.dump(oracle_metrics, f, indent=2)
        with open(OUTPUT_DIR / "oracle_breakdown_by_qtype.json", "w") as f:
            json.dump(oracle_breakdown, f, indent=2)

    # CSV for easy import
    cols = ["rouge1", "rouge2", "rougeL", "bleu4", "bertscore_f1",
            "exact_match", "numeric_match", "n_samples", "n_metrics_qs"]
    csv_lines = ["model," + ",".join(cols)]
    for label, m in all_metrics.items():
        row = [label] + [str(m.get(c, "")) for c in cols]
        csv_lines.append(",".join(row))
    with open(OUTPUT_DIR / "summary.csv", "w") as f:
        f.write("\n".join(csv_lines) + "\n")

    if oracle_metrics:
        oracle_csv = ["model," + ",".join(cols)]
        for label, m in oracle_metrics.items():
            row = [label] + [str(m.get(c, "")) for c in cols]
            oracle_csv.append(",".join(row))
        with open(OUTPUT_DIR / "oracle_summary.csv", "w") as f:
            f.write("\n".join(oracle_csv) + "\n")

    # Plots
    save_plots(all_metrics, all_breakdown, plots_dir)

    logger.info(f"All results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
