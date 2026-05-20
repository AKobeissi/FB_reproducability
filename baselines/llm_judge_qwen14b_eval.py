#!/usr/bin/env python3
"""
LLM-as-a-Judge Evaluation — Qwen2.5-14B Generated Answers on FinanceBench
===========================================================================

Extracts generated answers from report_Qwen2.5-14B.html, fetches the
corresponding question, expected answer, and gold evidence_text from
financebench_open_source.jsonl, then calls GPT-4o to produce a binary
CORRECT / INCORRECT verdict for each item.

Evaluation prompt follows the binary judge protocol from:
  Islam et al. (2023) "FinanceBench: A New Benchmark for Financial Question
  Answering" — which uses GPT-4 as judge with a structured binary rubric.
  Additional rubric elements from G-Eval (Liu et al., 2023) and RAGAS
  (Es et al., 2023) for numerical tolerance and refusal handling.

Usage
-----
  export OPENAI_API_KEY="sk-..."
  python baselines/llm_judge_qwen14b_eval.py

  # resume after partial run (skips already-scored rows in output CSV)
  python baselines/llm_judge_qwen14b_eval.py --resume

  # omit gold evidence from the prompt (slightly cheaper, slightly lower accuracy)
  python baselines/llm_judge_qwen14b_eval.py --no-evidence

Output
------
  baselines/results/llm_judge/qwen14b_judge_results.csv
  baselines/results/llm_judge/qwen14b_judge_summary.txt
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("llm_judge")

# ── paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
HTML_REPORT   = PROJECT_ROOT / "report_Qwen2.5-14B.html"
FB_DATA       = PROJECT_ROOT / "data" / "financebench_open_source.jsonl"
OUT_DIR       = PROJECT_ROOT / "baselines" / "results" / "llm_judge"
OUT_CSV       = OUT_DIR / "qwen14b_judge_results.csv"
OUT_SUMMARY   = OUT_DIR / "qwen14b_judge_summary.txt"

# ── OpenAI config ──────────────────────────────────────────────────────────────
JUDGE_MODEL      = "gpt-4o"
MAX_RETRIES      = 5
RETRY_BACKOFF    = 2.0   # seconds, doubled each retry
REQUESTS_PER_MIN = 50    # stay under RPM limit; adjust if on Tier 4+
SLEEP_BETWEEN    = 60.0 / REQUESTS_PER_MIN   # ~1.2 s between calls


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  HTML EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_from_html(html_path: Path) -> List[Dict]:
    """
    Parse the Qwen2.5-14B HTML report and return a list of dicts:
      {financebench_id, generated_answer, html_label}

    The HTML structure uses:
      <div class="sample-card" id="financebench_id_XXXXX" ...>
        ...
        <div class="answer-box generated answer-correct|answer-wrong">
          <div class="answer-label">Generated Answer</div>
          <div class="answer-content">GENERATED TEXT</div>
        </div>
    """
    logger.info(f"Reading HTML: {html_path}")
    html = html_path.read_text(encoding="utf-8")

    # Split into per-card blocks so we can associate ID + answer safely
    card_blocks = re.split(r'(?=<div class="sample-card" id="financebench_id_\d+")', html)
    card_blocks = [b for b in card_blocks if b.strip().startswith('<div class="sample-card"')]

    records = []
    for block in card_blocks:
        fid_m = re.search(r'id="(financebench_id_\d+)"', block)
        if not fid_m:
            continue
        fid = fid_m.group(1)

        label_m = re.search(r'<div class="answer-box generated (answer-correct|answer-wrong)">', block)
        html_label = label_m.group(1).replace("answer-", "") if label_m else "unknown"

        # Generated answer content — between the two <div class="answer-content"> tags
        # (first is expected, second is generated — but we're inside the generated box block)
        gen_content_m = re.search(
            r'<div class="answer-box generated answer-(?:correct|wrong)">'
            r'\s*<div class="answer-label">Generated Answer</div>'
            r'\s*<div class="answer-content">(.*?)</div>',
            block, re.DOTALL
        )
        if not gen_content_m:
            logger.warning(f"No generated answer found for {fid}")
            continue
        generated = gen_content_m.group(1).strip()

        records.append({
            "financebench_id": fid,
            "generated_answer": generated,
            "html_label": html_label,
        })

    logger.info(f"Extracted {len(records)} generated answers from HTML")
    return records


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  FINANCEBENCH DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_financebench(fb_path: Path) -> Dict[str, Dict]:
    """
    Returns a dict keyed by financebench_id containing:
      {question, expected_answer, evidence_texts, question_type, doc_name}

    evidence_texts: list of evidence_text strings (NOT full-page) from the
    evidence array. The shorter evidence_text is sufficient for the judge to
    calibrate numerical answers without overwhelming the context window.
    """
    data = {}
    with open(fb_path) as f:
        for line in f:
            raw = json.loads(line.strip())
            if not raw:
                continue
            fid = raw.get("financebench_id", "")
            ev_texts = []
            for ev in raw.get("evidence", []):
                t = ev.get("evidence_text", "").strip()
                if t:
                    ev_texts.append(t)
            data[fid] = {
                "question":        raw.get("question", ""),
                "expected_answer": raw.get("answer", ""),
                "evidence_texts":  ev_texts,
                "question_type":   raw.get("question_type", ""),
                "doc_name":        raw.get("doc_name", ""),
                "justification":   raw.get("justification", ""),
            }
    logger.info(f"Loaded {len(data)} FinanceBench items")
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  JUDGE PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are an expert financial analyst and impartial evaluator. Your task is to \
assess whether an AI system's generated answer is correct for a given financial \
question, using the expected (gold) answer as the reference.\
"""

# Prompt template following the binary judge protocol from FinanceBench
# (Islam et al., 2023) extended with numerical-tolerance guidance from G-Eval
# (Liu et al., 2023) and refusal-handling from RAGAS (Es et al., 2023).
JUDGE_PROMPT_WITH_EVIDENCE = """\
## Financial Question Answering Evaluation

You are evaluating the correctness of an AI system's answer to a financial question.

### Question
{question}

### Gold Evidence (source context)
{evidence}

### Expected Answer (ground truth)
{expected_answer}

### Generated Answer (to evaluate)
{generated_answer}

---

### Evaluation Rubric

**CORRECT** — assign this verdict when:
- The generated answer conveys the same key financial fact(s) as the expected answer.
- Numerical values match within ±2% relative tolerance (e.g., $5.1B vs $5.05B is acceptable).
- Minor differences in phrasing, unit notation (e.g., "million" vs "M"), or additional \
correct explanation are acceptable as long as the core answer is right.
- A yes/no answer matches directionally AND is supported by correct reasoning.

**INCORRECT** — assign this verdict when:
- The generated answer states a materially different value, percentage, or financial metric.
- The generated answer contradicts the expected answer on a factual point.
- The model refuses to answer or says it cannot determine the answer ("I don't know", \
"cannot be determined", "insufficient information").
- The generated answer addresses a different question or metric than what was asked.
- The answer is directionally wrong (e.g., says "yes" when the correct answer is "no").

### Output Format
Respond with exactly:
Verdict: CORRECT
Reason: <one concise sentence explaining your verdict>

or:
Verdict: INCORRECT
Reason: <one concise sentence explaining your verdict>

Do not output anything else.\
"""

JUDGE_PROMPT_NO_EVIDENCE = """\
## Financial Question Answering Evaluation

You are evaluating the correctness of an AI system's answer to a financial question.

### Question
{question}

### Expected Answer (ground truth)
{expected_answer}

### Generated Answer (to evaluate)
{generated_answer}

---

### Evaluation Rubric

**CORRECT** — assign this verdict when:
- The generated answer conveys the same key financial fact(s) as the expected answer.
- Numerical values match within ±2% relative tolerance (e.g., $5.1B vs $5.05B is acceptable).
- Minor differences in phrasing, unit notation (e.g., "million" vs "M"), or additional \
correct explanation are acceptable as long as the core answer is right.
- A yes/no answer matches directionally AND is supported by correct reasoning.

**INCORRECT** — assign this verdict when:
- The generated answer states a materially different value, percentage, or financial metric.
- The generated answer contradicts the expected answer on a factual point.
- The model refuses to answer or says it cannot determine the answer ("I don't know", \
"cannot be determined", "insufficient information").
- The generated answer addresses a different question or metric than what was asked.
- The answer is directionally wrong (e.g., says "yes" when the correct answer is "no").

### Output Format
Respond with exactly:
Verdict: CORRECT
Reason: <one concise sentence explaining your verdict>

or:
Verdict: INCORRECT
Reason: <one concise sentence explaining your verdict>

Do not output anything else.\
"""


def build_prompt(question: str, expected: str, generated: str,
                 evidence_texts: List[str], include_evidence: bool) -> str:
    if include_evidence and evidence_texts:
        evidence_str = "\n\n---\n\n".join(evidence_texts)
        return JUDGE_PROMPT_WITH_EVIDENCE.format(
            question=question,
            evidence=evidence_str,
            expected_answer=expected,
            generated_answer=generated,
        )
    else:
        return JUDGE_PROMPT_NO_EVIDENCE.format(
            question=question,
            expected_answer=expected,
            generated_answer=generated,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  GPT-4o API CALL
# ═══════════════════════════════════════════════════════════════════════════════

def call_judge(client, prompt: str) -> tuple[str, str]:
    """
    Returns (verdict, reason) where verdict is 'CORRECT' or 'INCORRECT'.
    Raises on unrecoverable failure after MAX_RETRIES.
    """
    delay = RETRY_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.0,
                max_tokens=150,
            )
            text = response.choices[0].message.content.strip()
            return _parse_verdict(text)
        except Exception as e:
            err_str = str(e)
            if "rate_limit" in err_str.lower() or "429" in err_str:
                wait = delay * (2 ** (attempt - 1))
                logger.warning(f"Rate limit hit, waiting {wait:.0f}s (attempt {attempt}/{MAX_RETRIES})")
                time.sleep(wait)
            elif attempt == MAX_RETRIES:
                raise
            else:
                logger.warning(f"API error attempt {attempt}: {e}")
                time.sleep(delay)
    raise RuntimeError("Max retries exceeded")


def _parse_verdict(text: str) -> tuple[str, str]:
    """Parse 'Verdict: CORRECT\nReason: ...' from model output."""
    verdict_m = re.search(r"Verdict:\s*(CORRECT|INCORRECT)", text, re.IGNORECASE)
    reason_m  = re.search(r"Reason:\s*(.+)", text, re.DOTALL)

    verdict = verdict_m.group(1).upper() if verdict_m else "PARSE_ERROR"
    reason  = reason_m.group(1).strip().split("\n")[0] if reason_m else text[:200]

    if verdict == "PARSE_ERROR":
        # fallback: look for keyword anywhere
        if "incorrect" in text.lower():
            verdict = "INCORRECT"
        elif "correct" in text.lower():
            verdict = "CORRECT"

    return verdict, reason


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

CSV_FIELDS = [
    "financebench_id", "question_type", "doc_name",
    "question", "expected_answer", "generated_answer",
    "html_label",          # correct/wrong from ROUGE/EM in the report
    "judge_verdict",       # CORRECT / INCORRECT from GPT-4o
    "judge_binary",        # 1 / 0
    "judge_reason",
]


def load_existing_results(csv_path: Path) -> set:
    """Return set of financebench_ids already scored."""
    if not csv_path.exists():
        return set()
    done = set()
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("judge_verdict") not in ("", None, "PARSE_ERROR"):
                done.add(row["financebench_id"])
    logger.info(f"Resume: {len(done)} items already scored")
    return done


def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge eval for Qwen2.5-14B on FinanceBench")
    parser.add_argument("--resume",      action="store_true", help="Skip already-scored rows")
    parser.add_argument("--no-evidence", action="store_true", help="Omit gold evidence from prompt")
    parser.add_argument("--limit",       type=int, default=None, help="Only evaluate first N items (for testing)")
    args = parser.parse_args()

    include_evidence = not args.no_evidence

    # ── API client ──────────────────────────────────────────────────────────────
    # load .env from project root if OPENAI_API_KEY not already in environment
    if not os.environ.get("OPENAI_API_KEY"):
        env_file = PROJECT_ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set. "
                     "Run: export OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    try:
        from openai import OpenAI
    except ImportError:
        logger.error("openai package not found. Run: pip install openai")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    logger.info(f"Judge model: {JUDGE_MODEL}  |  Evidence in prompt: {include_evidence}")

    # ── load data ───────────────────────────────────────────────────────────────
    extracted  = extract_from_html(HTML_REPORT)
    fb_data    = load_financebench(FB_DATA)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── resume ──────────────────────────────────────────────────────────────────
    already_done = load_existing_results(OUT_CSV) if args.resume else set()

    # ── open CSV for writing ────────────────────────────────────────────────────
    write_mode = "a" if (args.resume and OUT_CSV.exists()) else "w"
    csv_file   = open(OUT_CSV, write_mode, newline="", encoding="utf-8")
    writer     = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if write_mode == "w":
        writer.writeheader()

    # ── evaluation loop ─────────────────────────────────────────────────────────
    n_total = len(extracted)
    if args.limit:
        extracted = extracted[:args.limit]
        n_total   = len(extracted)

    n_correct = 0
    n_incorrect = 0
    n_skip = 0
    parse_errors = []

    logger.info(f"Evaluating {n_total} items ...")
    for i, item in enumerate(extracted, 1):
        fid = item["financebench_id"]

        if fid in already_done:
            n_skip += 1
            continue

        fb = fb_data.get(fid)
        if fb is None:
            logger.warning(f"[{i}/{n_total}] {fid} not found in FinanceBench data — skipping")
            continue

        prompt = build_prompt(
            question=fb["question"],
            expected=fb["expected_answer"],
            generated=item["generated_answer"],
            evidence_texts=fb["evidence_texts"],
            include_evidence=include_evidence,
        )

        verdict, reason = "PARSE_ERROR", ""
        try:
            verdict, reason = call_judge(client, prompt)
        except Exception as e:
            logger.error(f"[{i}/{n_total}] {fid} — API failure: {e}")
            reason = f"API_ERROR: {e}"

        binary = 1 if verdict == "CORRECT" else 0
        if verdict == "CORRECT":
            n_correct += 1
        elif verdict == "INCORRECT":
            n_incorrect += 1
        else:
            parse_errors.append(fid)

        row = {
            "financebench_id":  fid,
            "question_type":    fb["question_type"],
            "doc_name":         fb["doc_name"],
            "question":         fb["question"],
            "expected_answer":  fb["expected_answer"],
            "generated_answer": item["generated_answer"],
            "html_label":       item["html_label"],
            "judge_verdict":    verdict,
            "judge_binary":     binary,
            "judge_reason":     reason,
        }
        writer.writerow(row)
        csv_file.flush()

        evaluated = n_correct + n_incorrect
        pct = n_correct / evaluated * 100 if evaluated > 0 else 0.0
        logger.info(f"[{i}/{n_total}] {fid}  →  {verdict}  "
                    f"(running: {n_correct}/{evaluated} = {pct:.1f}% correct)")

        time.sleep(SLEEP_BETWEEN)

    csv_file.close()

    # ── summary ──────────────────────────────────────────────────────────────────
    evaluated = n_correct + n_incorrect
    accuracy  = n_correct / evaluated * 100 if evaluated > 0 else 0.0

    summary_lines = [
        "LLM-AS-A-JUDGE EVALUATION SUMMARY",
        "=" * 50,
        f"Model evaluated : Qwen2.5-14B",
        f"Judge model     : {JUDGE_MODEL}",
        f"Evidence in prompt: {include_evidence}",
        f"",
        f"Total items     : {n_total}",
        f"Skipped (resume): {n_skip}",
        f"Evaluated       : {evaluated}",
        f"  CORRECT       : {n_correct}",
        f"  INCORRECT     : {n_incorrect}",
        f"  Parse errors  : {len(parse_errors)}",
        f"",
        f"Judge Accuracy  : {accuracy:.2f}%",
        f"",
    ]

    if parse_errors:
        summary_lines.append(f"Parse error IDs: {parse_errors}")

    # per question_type breakdown (re-read CSV)
    try:
        import pandas as pd
        df = pd.read_csv(OUT_CSV)
        df = df[df["judge_verdict"].isin(["CORRECT", "INCORRECT"])]
        summary_lines.append("── BY QUESTION TYPE ──")
        for qt, g in df.groupby("question_type"):
            acc = g["judge_binary"].mean() * 100
            summary_lines.append(f"  {qt:25s}: {acc:.1f}%  (n={len(g)})")
        summary_lines.append("")
        summary_lines.append("── JUDGE vs HTML LABEL AGREEMENT ──")
        df["html_binary"] = (df["html_label"] == "correct").astype(int)
        agree = (df["judge_binary"] == df["html_binary"]).mean() * 100
        summary_lines.append(f"  Agreement rate: {agree:.1f}%")
        # breakdown
        for combo, g in df.groupby(["html_label", "judge_verdict"]):
            summary_lines.append(f"  html={combo[0]}  judge={combo[1]}  n={len(g)}")
    except Exception as e:
        summary_lines.append(f"[breakdown error: {e}]")

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)
    OUT_SUMMARY.write_text(summary_text)
    logger.info(f"Saved results → {OUT_CSV}")
    logger.info(f"Saved summary → {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
