#!/usr/bin/env python3
"""
run_benchmark_ext_gen_eval.py
==============================
Generate answers for all benchmark-extension retrieval variants (680 questions:
150 FinanceBench + 530 FinQA) and compute global answer-quality metrics:

  - ROUGE-L F1
  - BERTScore F1   (roberta-large, rescaled)
  - NumericMatch   (exact/±3% tolerance over ALL questions)

Metrics are reported globally only — no question-type breakdown — because the
FinQA subset does not carry the same fine-grained question-type labels as FB.

Resume-safe: questions that already have a non-empty generated_answer are
skipped; prediction files are updated in-place after each variant finishes.

Usage:
    python baselines/run_benchmark_ext_gen_eval.py [--variants V [V ...]] [--no-bertscore]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

PROJECT_ROOT   = Path(__file__).resolve().parent.parent
PRED_DIR       = PROJECT_ROOT / "baselines/results/benchmark_extension/predictions"
OUT_DIR        = PROJECT_ROOT / "baselines/results/benchmark_extension/gen_eval"
SYS_PATH_INSERT = str(PROJECT_ROOT)

if SYS_PATH_INSERT not in sys.path:
    sys.path.insert(0, SYS_PATH_INSERT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bench_ext_gen")

# ── Config ────────────────────────────────────────────────────────────────────
MAIN_K             = 5
MAX_CONTEXT_TOKENS = 3600
DEFAULT_MODEL      = "Qwen/Qwen2.5-7B-Instruct"
MAX_NEW_TOKENS     = 128

# All 16 variants in the benchmark extension (oracle included as upper bounds)
ALL_VARIANTS = [
    "bm25",
    "splade",
    "dense_bge_m3",
    "dense_bge_base",
    "dense_investopedia",
    "hybrid_50_50",
    "hybrid_75_25",
    "hybrid_25_75",
    "parent_child",
    "query_expansion",
    "hyde",
    "multi_hyde",
    "bge_reranker",
    "multi_hyde_reranker",
    "dense_bge_m3_ft_reranker",
    "multi_hyde_ft_reranker",
    "oracle_doc",
    "oracle_page",
]

GEN_PROMPT = (
    "You are a financial analyst answering questions based on SEC filings. "
    "Use ONLY the provided context. If the context does not contain the answer, "
    "say 'I cannot determine this from the provided information.' "
    "Be concise and precise, especially for numerical answers.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def extract_numbers(text: str) -> List[float]:
    clean = re.sub(r"[,$%]", "", str(text))
    matches = re.findall(r"-?\d*\.?\d+", clean)
    out = []
    for m in matches:
        try:
            if m not in {".", "-"}:
                out.append(float(m))
        except ValueError:
            pass
    return out


def numeric_match(pred: str, ref: str, rtol: float = 0.03) -> float:
    ref_nums  = extract_numbers(ref)
    pred_nums = extract_numbers(pred)
    if not ref_nums or not pred_nums:
        return 0.0
    for r in ref_nums:
        for p in pred_nums:
            if r == 0.0:
                if abs(p) < 0.03:
                    return 1.0
            elif abs(p - r) / abs(r) <= rtol:
                return 1.0
    return 0.0


def build_context(sample: Dict, tokenizer) -> str:
    chunks  = sample.get("retrieved_chunks", [])[:MAIN_K]
    parts   = []
    budget  = MAX_CONTEXT_TOKENS
    for chunk in chunks:
        text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
        if not text or budget <= 0:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= budget:
            parts.append(text)
            budget -= len(ids)
        elif budget >= 50:
            parts.append(tokenizer.decode(ids[:budget], skip_special_tokens=True))
            budget = 0
    return "\n\n".join(parts)


# ── Generation ────────────────────────────────────────────────────────────────

def generate_answers(samples: List[Dict], model_name: str) -> int:
    """Fill in generated_answer for samples that don't have one. Returns count."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    pending = [s for s in samples if not (s.get("generated_answer") or "").strip()]
    if not pending:
        return 0

    logger.info("Loading %s to generate %d answers …", model_name, len(pending))
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
    for sample in tqdm(pending, desc="generating", leave=False):
        context = build_context(sample, tokenizer)
        prompt  = GEN_PROMPT.format(context=context, question=sample.get("question", ""))
        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, "apply_chat_template"):
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted = prompt

        inputs = tokenizer(
            formatted, return_tensors="pt", truncation=True, max_length=4096
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        try:
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                )
            prompt_len = inputs["input_ids"].shape[-1]
            sample["generated_answer"] = tokenizer.decode(
                out[0][prompt_len:], skip_special_tokens=True
            ).strip()
            generated += 1
        except Exception as exc:
            logger.warning("Generation failed for %s: %s", sample.get("id", "?"), exc)
            sample["generated_answer"] = ""

    del model
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return generated


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_samples(samples: List[Dict], bert_scorer=None) -> Dict:
    from rouge_score import rouge_scorer as rs_mod

    rouge = rs_mod.RougeScorer(["rougeL"], use_stemmer=True)

    rougeL_scores: List[float] = []
    nm_all:        List[float] = []
    answered = 0

    for s in samples:
        gen = (s.get("generated_answer") or "").strip()
        ref = (s.get("reference_answer")  or "").strip()
        if gen:
            answered += 1
        if gen and ref:
            rougeL_scores.append(rouge.score(ref, gen)["rougeL"].fmeasure)
        else:
            rougeL_scores.append(0.0)
        nm_all.append(numeric_match(gen, ref))

    def _mean(lst):
        return float(np.mean(lst)) if lst else 0.0

    out = {
        "rougeL":          _mean(rougeL_scores),
        "numeric_match":   _mean(nm_all),
        "n_answered":      answered,
        "n_samples":       len(samples),
        "bertscore_f1":    None,
    }

    if bert_scorer is not None:
        preds     = [(s.get("generated_answer") or "").strip() for s in samples]
        refs      = [(s.get("reference_answer")  or "").strip() for s in samples]
        valid_idx = [i for i, (p, r) in enumerate(zip(preds, refs)) if p and r]
        bs = [0.0] * len(samples)
        if valid_idx:
            try:
                _, _, F1 = bert_scorer.score(
                    [preds[i] for i in valid_idx],
                    [refs[i]  for i in valid_idx],
                    verbose=False,
                    batch_size=16,
                )
                for i, f1 in zip(valid_idx, F1.tolist()):
                    bs[i] = float(f1)
            except Exception as exc:
                logger.warning("BERTScore batch failed: %s", exc)
        out["bertscore_f1"] = _mean(bs)

    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--variants",  nargs="*", default=ALL_VARIANTS,
                   help="Variants to evaluate. Default: all 18.")
    p.add_argument("--model",     default=DEFAULT_MODEL,
                   help="Causal LM for generation.")
    p.add_argument("--no-bertscore", action="store_true",
                   help="Skip BERTScore (faster, CPU-only runs).")
    p.add_argument("--skip-generation", action="store_true",
                   help="Score existing generated_answer fields without generating new ones.")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # BERTScorer — loaded once, reused across all variants
    bert_scorer = None
    if not args.no_bertscore:
        try:
            import torch
            from bert_score import BERTScorer as _BS
            device = "cuda" if torch.cuda.is_available() else "cpu"
            bert_scorer = _BS(lang="en", rescale_with_baseline=True, device=device)
            logger.info("BERTScorer ready (roberta-large, rescaled, %s)", device)
        except Exception as exc:
            logger.warning("BERTScore disabled: %s", exc)

    results: Dict[str, Dict] = {}

    for variant in args.variants:
        pred_path = PRED_DIR / f"{variant}_retrieval.json"
        if not pred_path.exists():
            logger.warning("Skipping %s — prediction file not found", variant)
            continue

        samples = load_json(pred_path)
        already = sum(1 for s in samples if (s.get("generated_answer") or "").strip())
        logger.info("[%s] %d/%d already answered", variant, already, len(samples))

        if not args.skip_generation:
            n_gen = generate_answers(samples, args.model)
            if n_gen:
                save_json(pred_path, samples)
                logger.info("[%s] generated %d new answers → saved", variant, n_gen)

        metrics = score_samples(samples, bert_scorer)
        results[variant] = metrics
        logger.info(
            "[%s] ROUGE-L=%.4f  BERTScore-F1=%s  NumMatch=%.4f  answered=%d/%d",
            variant,
            metrics["rougeL"],
            f"{metrics['bertscore_f1']:.4f}" if metrics["bertscore_f1"] is not None else "N/A",
            metrics["numeric_match"],
            metrics["n_answered"],
            metrics["n_samples"],
        )

    # ── Save JSON summary ─────────────────────────────────────────────────────
    save_json(OUT_DIR / "gen_eval_results.json", results)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    import csv
    csv_path = OUT_DIR / "gen_eval_results.csv"
    with open(csv_path, "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=[
            "variant", "rougeL", "bertscore_f1", "numeric_match",
            "n_answered", "n_samples",
        ])
        writer.writeheader()
        for variant, m in results.items():
            writer.writerow({
                "variant":       variant,
                "rougeL":        f"{m['rougeL']:.4f}",
                "bertscore_f1":  f"{m['bertscore_f1']:.4f}" if m["bertscore_f1"] is not None else "",
                "numeric_match": f"{m['numeric_match']:.4f}",
                "n_answered":    m["n_answered"],
                "n_samples":     m["n_samples"],
            })

    # ── Print text report ─────────────────────────────────────────────────────
    report_path = OUT_DIR / "gen_eval_report.txt"
    with open(report_path, "w") as rpt:

        def w(line=""):
            print(line)
            rpt.write(line + "\n")

        has_bert = any(m["bertscore_f1"] is not None for m in results.values())
        w("=" * 80)
        w("BENCHMARK EXTENSION — GENERATIVE EVALUATION (global, n=680)")
        w("FinanceBench (150q) + FinQA (530q) — unified 206-doc index")
        w("Model: " + args.model)
        w("Metrics: ROUGE-L F1 | BERTScore F1 (roberta-large, rescaled) | NumericMatch (±3%)")
        w("=" * 80)
        w()

        if has_bert:
            w(f"  {'Method':<45} {'ROUGE-L':>8} {'BERT-F1':>9} {'NumMatch':>10} {'Answered':>10}")
            w("  " + "-" * 85)
        else:
            w(f"  {'Method':<45} {'ROUGE-L':>8} {'NumMatch':>10} {'Answered':>10}")
            w("  " + "-" * 76)

        # Sort by ROUGE-L descending
        for variant, m in sorted(results.items(), key=lambda x: -x[1]["rougeL"]):
            answered_str = f"{m['n_answered']}/{m['n_samples']}"
            if has_bert:
                bs_str = f"{m['bertscore_f1']:.4f}" if m["bertscore_f1"] is not None else "  N/A  "
                w(f"  {variant:<45} {m['rougeL']:>8.4f} {bs_str:>9} {m['numeric_match']:>10.4f} {answered_str:>10}")
            else:
                w(f"  {variant:<45} {m['rougeL']:>8.4f} {m['numeric_match']:>10.4f} {answered_str:>10}")

        w()
        w("NumericMatch: fraction of questions where prediction contains the reference")
        w("             number within ±3% tolerance (computed over all 680 questions).")
        w("BERTScore:    0.0 for unanswered questions; mean over all 680 samples.")
        w()
        w(f"Report: {report_path}")
        w(f"CSV:    {csv_path}")
        w(f"JSON:   {OUT_DIR / 'gen_eval_results.json'}")

    logger.info("Done. Outputs in %s", OUT_DIR)


if __name__ == "__main__":
    main()
