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
# (~30 tok), and max_new_tokens (350).  3600 is conservative but fits the
# median full context (≈3 550 tok) and is well within the L40S memory budget.
MAX_CONTEXT_TOKENS = 3600

MODELS = [
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
]

MODEL_LABELS = {
    "Qwen/Qwen2.5-3B-Instruct":  "Qwen2.5-3B",
    "Qwen/Qwen2.5-7B-Instruct":  "Qwen2.5-7B",
    "Qwen/Qwen2.5-14B-Instruct": "Qwen2.5-14B",
    "Qwen/Qwen2.5-72B-Instruct": "Qwen2.5-72B",
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


def _extract_all_numbers(text: str) -> list:
    """Return all numbers found in text after stripping currency symbols."""
    text = re.sub(r"[$,€£%()]", "", text)
    matches = re.findall(r"-?\d[\d,]*\.?\d*", text)
    out = []
    for m in matches:
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


# Scale words → multiplier.  Ordered longest-first to avoid partial matches.
_SCALE_PATTERNS: List[tuple] = [
    (re.compile(r"\btrillion[s]?\b",  re.I), 1e12),
    (re.compile(r"\bbillion[s]?\b|\bbn\b|\bbln\b",  re.I), 1e9),
    (re.compile(r"\bmillion[s]?\b|\bmm\b|\bmln\b",  re.I), 1e6),
    (re.compile(r"\bthousand[s]?\b|\bk\b",           re.I), 1e3),
]


def _scaled_value(raw_number: float, text: str) -> float:
    """
    Apply the first scale word found in text to raw_number.
    e.g. _scaled_value(8.70, "8.70 billion") → 8.70e9
         _scaled_value(8738, "8,738 million") → 8.738e9
         _scaled_value(1577, "$1,577")        → 1577   (no scale word)
    """
    for pattern, factor in _SCALE_PATTERNS:
        if pattern.search(text):
            return raw_number * factor
    return raw_number


def numeric_match(pred: str, ref: str, rtol: float = 0.03) -> float:
    """
    Checks whether ANY number in pred matches the reference number within rtol.

    Two comparison strategies are tried for each candidate number:
      1. Scale-normalised: both values are multiplied by their respective scale
         words before comparing.  Handles "$8.70 billion" vs "8,738 million"
         (both normalize to ~8.70e9, within 0.5%).
      2. Raw (fallback): compare the bare extracted numbers directly, for cases
         where neither text contains an explicit scale word.

    A match on either strategy counts as correct.
    """
    ref_raw = _extract_main_number(ref)
    if ref_raw is None:
        return 0.0
    ref_scaled = _scaled_value(ref_raw, ref)

    pred_nums = _extract_all_numbers(pred)
    if not pred_nums:
        return 0.0

    for p_raw in pred_nums:
        p_scaled = _scaled_value(p_raw, pred)

        # Strategy 1 — scale-normalised comparison
        # Only use when at least one side has a scale word (otherwise scaling
        # noise can create false positives between e.g. 100 and 100,000)
        ref_has_scale = any(pat.search(ref)  for pat, _ in _SCALE_PATTERNS)
        pred_has_scale = any(pat.search(pred) for pat, _ in _SCALE_PATTERNS)
        if ref_has_scale or pred_has_scale:
            denom = abs(ref_scaled) if ref_scaled != 0 else 1.0
            if abs(p_scaled - ref_scaled) / denom <= rtol:
                return 1.0

        # Strategy 2 — raw comparison (handles "$8.70" vs "8.738" rounding)
        if ref_raw == 0:
            if p_raw == 0:
                return 1.0
        else:
            if abs(p_raw - ref_raw) / abs(ref_raw) <= rtol:
                return 1.0

    return 0.0


def exact_match(pred: str, ref: str) -> float:
    return 1.0 if _normalize(pred) == _normalize(ref) else 0.0


def squad_f1(pred: str, ref: str) -> float:
    """
    Token-level F1 from SQuAD evaluation — harmonic mean of precision and recall
    over shared tokens after normalisation.  More reliable than ROUGE-L for short
    financial answers because it is symmetric: a short correct answer inside a
    long explanation still scores well, and it does not penalise word reordering.
    """
    pred_tokens = _normalize(pred).split()
    ref_tokens  = _normalize(ref).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    from collections import Counter
    common = Counter(pred_tokens) & Counter(ref_tokens)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_tokens)
    recall    = n_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


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
    squad_f1_scores = []
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
            squad_f1_scores.append(squad_f1(gen, ref))
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
            squad_f1_scores.append(0.0)
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
        "squad_f1":      _mean(squad_f1_scores),
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


def annotate_samples_with_metrics(
    samples: List[Dict], bert_scorer=None
) -> List[Dict]:
    """
    Attach per-sample evaluation metrics to each sample dict under the key
    'eval_metrics'.  This makes the prediction JSON human-readable: you can
    open any prediction file and immediately see how each answer scored and
    why it may have failed.

    Fields added per sample
    -----------------------
    rouge1, rouge2, rougeL  — lexical overlap (unreliable for short numeric answers)
    squad_f1                — symmetric token F1 (SQuAD-style); more robust than ROUGE
    bleu4                   — n-gram precision
    bertscore_f1            — semantic similarity (negative = worse than average)
    exact_match             — normalised string equality
    numeric_match           — any number in prediction within 3% of reference number
                              (fixed: scans ALL numbers, not just the first one)
    ref_number              — the number extracted from the reference answer
    pred_numbers            — all numbers found in the generated answer
    is_refusal              — True if model said "cannot determine"
    answer_words            — word count of the generated answer (0 = empty)
    """
    from rouge_score import rouge_scorer as rs_module
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        _has_nltk = True
        smooth = SmoothingFunction().method1
    except ImportError:
        _has_nltk = False
        smooth = None

    rouge_scorer_obj = rs_module.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    # Compute BERTScore in one batch for all valid pairs
    bs_per_sample = [None] * len(samples)
    if bert_scorer is not None:
        preds     = [s.get("generated_answer", "").strip() for s in samples]
        refs      = [s.get("reference_answer",  "").strip() for s in samples]
        valid_idx = [i for i, (p, r) in enumerate(zip(preds, refs)) if p and r]
        if valid_idx:
            try:
                _, _, F1 = bert_scorer.score(
                    [preds[i] for i in valid_idx],
                    [refs[i]  for i in valid_idx],
                    verbose=False, batch_size=32,
                )
                for i, f1 in zip(valid_idx, F1.tolist()):
                    bs_per_sample[i] = float(f1)
            except Exception as e:
                logger.warning(f"Per-sample BERTScore failed: {e}")

    for idx, s in enumerate(samples):
        gen = s.get("generated_answer", "").strip()
        ref = s.get("reference_answer",  "").strip()

        is_refusal   = "cannot determine" in gen.lower()
        answer_words = len(gen.split()) if gen else 0

        if gen and ref:
            r       = rouge_scorer_obj.score(ref, gen)
            r1      = r["rouge1"].fmeasure
            r2      = r["rouge2"].fmeasure
            rl      = r["rougeL"].fmeasure
            sq_f1   = squad_f1(gen, ref)
            em      = exact_match(gen, ref)

            if _has_nltk:
                rt = _normalize(ref).split()
                pt = _normalize(gen).split()
                b4 = sentence_bleu([rt], pt, smoothing_function=smooth) if rt and pt else 0.0
            else:
                b4 = None
        else:
            r1 = r2 = rl = sq_f1 = em = b4 = 0.0

        # Numeric fields — expose raw and scaled values for manual inspection
        ref_num_raw    = _extract_main_number(ref)
        ref_num_scaled = _scaled_value(ref_num_raw, ref) if ref_num_raw is not None else None
        pred_nums_raw  = _extract_all_numbers(gen)
        pred_nums_scaled = [_scaled_value(p, gen) for p in pred_nums_raw]

        if s.get("question_type") == "metrics-generated":
            nm = numeric_match(gen, ref)
        else:
            nm = None   # not applicable

        eval_metrics: Dict = {
            "rouge1":             round(r1,     4),
            "rouge2":             round(r2,     4),
            "rougeL":             round(rl,     4),
            "squad_f1":           round(sq_f1,  4),
            "exact_match":        int(em),
            "numeric_match":      nm,
            # Numeric debugging — lets you see why a match passed or failed
            "ref_number":         ref_num_raw,
            "ref_number_scaled":  ref_num_scaled,
            "pred_numbers":       pred_nums_raw[:5],
            "pred_numbers_scaled": pred_nums_scaled[:5],
            "is_refusal":         is_refusal,
            "answer_words":       answer_words,
        }
        if b4 is not None:
            eval_metrics["bleu4"] = round(b4, 4)
        if bs_per_sample[idx] is not None:
            eval_metrics["bertscore_f1"] = round(bs_per_sample[idx], 4)

        s["eval_metrics"] = eval_metrics

    return samples


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
    cols = ["rougeL", "squad_f1", "rouge1", "bertscore_f1", "exact_match", "numeric_match"]
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-file", type=Path, default=RETRIEVAL_FILE,
                        help="Pre-computed retrieval JSON (default: dense BGE-M3)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Where to write outputs (default: outputs/llm_scaling)")
    parser.add_argument("--model-filter", type=str, default=None,
                        help="Comma-separated model labels to run (e.g. 'Qwen2.5-72B'). "
                             "Skipped models still appear in summary if predictions exist.")
    args = parser.parse_args()

    retrieval_file = args.retrieval_file
    output_dir     = args.output_dir

    # Build the set of labels we are allowed to *generate* for.
    # If --model-filter is not set, all models are eligible.
    allowed_labels: set = (
        {l.strip() for l in args.model_filter.split(",")}
        if args.model_filter else set(MODEL_LABELS.values())
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    preds_dir = output_dir / "predictions"
    oracle_dir = output_dir / "oracle_predictions"
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
    logger.info(f"Loading retrieval results from {retrieval_file}")
    if not retrieval_file.exists():
        raise FileNotFoundError(
            f"Retrieval file not found: {retrieval_file}\n"
            "Run the retrieval baseline first."
        )
    base_samples = json.load(open(retrieval_file))
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
        elif label not in allowed_labels:
            logger.info(f"Skipping generation for {label} (not in --model-filter).")
            continue
        else:
            samples = generate_with_model(base_samples, model_name)

        # Annotate with per-sample metrics (always refresh so eval_metrics
        # reflects any changes to metric functions)
        samples = annotate_samples_with_metrics(samples, bert_scorer)
        with open(pred_file, "w") as f:
            json.dump(samples, f, indent=2)
        logger.info(f"Saved annotated predictions: {pred_file}")

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
            elif label not in allowed_labels:
                logger.info(f"Skipping oracle generation for {label} (not in --model-filter).")
                continue
            else:
                oracle_preds = generate_with_model(oracle_samples, model_name)

            oracle_preds = annotate_samples_with_metrics(oracle_preds, bert_scorer)
            with open(oracle_pred_file, "w") as f:
                json.dump(oracle_preds, f, indent=2)
            logger.info(f"Saved annotated oracle predictions: {oracle_pred_file}")

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
    with open(output_dir / "summary.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    with open(output_dir / "breakdown_by_qtype.json", "w") as f:
        json.dump(all_breakdown, f, indent=2)

    if oracle_metrics:
        with open(output_dir / "oracle_summary.json", "w") as f:
            json.dump(oracle_metrics, f, indent=2)
        with open(output_dir / "oracle_breakdown_by_qtype.json", "w") as f:
            json.dump(oracle_breakdown, f, indent=2)

    # CSV for easy import
    cols = ["rouge1", "rouge2", "rougeL", "squad_f1", "bleu4", "bertscore_f1",
            "exact_match", "numeric_match", "n_samples", "n_metrics_qs"]
    csv_lines = ["model," + ",".join(cols)]
    for label, m in all_metrics.items():
        row = [label] + [str(m.get(c, "")) for c in cols]
        csv_lines.append(",".join(row))
    with open(output_dir / "summary.csv", "w") as f:
        f.write("\n".join(csv_lines) + "\n")

    if oracle_metrics:
        oracle_csv = ["model," + ",".join(cols)]
        for label, m in oracle_metrics.items():
            row = [label] + [str(m.get(c, "")) for c in cols]
            oracle_csv.append(",".join(row))
        with open(output_dir / "oracle_summary.csv", "w") as f:
            f.write("\n".join(oracle_csv) + "\n")

    # Plots
    save_plots(all_metrics, all_breakdown, plots_dir)

    logger.info(f"All results saved to {output_dir}")


if __name__ == "__main__":
    main()
