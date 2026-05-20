"""
LLM Scaling Error Analysis
==========================
Decomposes failures into retrieval vs. generation causes, classifies
failure modes, and writes a readable HTML report + summary CSV.

Usage:
    python analyze_llm_scaling_errors.py \
        --pred-dir outputs/llm_scaling/predictions \
        --oracle-dir outputs/llm_scaling/oracle_predictions \
        --out-dir outputs/llm_scaling/error_analysis

The script answers three questions:
  1. Is the model failing because the right evidence was not retrieved?
  2. Given the right evidence, can the model reason correctly?
  3. When the model fails, what specific failure mode is it?
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ── helpers ───────────────────────────────────────────────────────────────────

def safe_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def retrieval_hit(p: dict, min_snippet_len: int = 50) -> str:
    """
    Returns 'full', 'partial', or 'miss' depending on how many gold evidence
    segments appear in the retrieved chunks.
    """
    gold_segs = p.get("gold_evidence_segments", [])
    retrieved  = p.get("retrieved_chunks", [])
    if not gold_segs:
        return "no_gold"

    retrieved_text = " ".join(
        r["text"].lower() if isinstance(r, dict) else str(r).lower()
        for r in retrieved
    )

    hits = 0
    for seg in gold_segs:
        seg_text = seg.get("text", "") if isinstance(seg, dict) else str(seg)
        snippet  = seg_text[:min_snippet_len].strip().lower()
        if snippet and snippet in retrieved_text:
            hits += 1

    if hits == 0:
        return "miss"
    elif hits < len(gold_segs):
        return "partial"
    else:
        return "full"


def is_correct(p: dict, squad_thresh: float = 0.4, rouge_thresh: float = 0.35) -> bool:
    """
    Numeric questions  → numeric_match == 1.0
    Text questions     → squad_f1 > thresh  OR  rougeL > thresh
    """
    m  = p.get("eval_metrics", {}) or {}
    nm = m.get("numeric_match")
    if nm is not None:
        return nm == 1.0
    return m.get("squad_f1", 0) > squad_thresh or m.get("rougeL", 0) > rouge_thresh


def classify_failure(p: dict) -> list[str]:
    """
    Returns a list of failure-mode tags (can be multiple).

    Tags
    ----
    REFUSAL            – model says it cannot answer
    UNIT_SCALE_ERROR   – answer is right number, wrong scale (×1000 or ÷1000)
    MULTI_STEP_CALC    – model shows working but gets wrong result (chain-of-thought failure)
    WRONG_NUMBER       – some number was extracted but it is wrong
    FORMAT_MISMATCH    – answer contains correct number but in wrong format (%, $B vs $M, etc.)
    HALLUCINATION      – model confabulates numbers not in context
    OTHER              – does not fit above
    """
    gen      = p.get("generated_answer", "") or ""
    m        = p.get("eval_metrics", {}) or {}
    ref_num  = safe_float(m.get("ref_number"))
    pred_nums = [safe_float(x) for x in (m.get("pred_numbers") or []) if safe_float(x) is not None]
    ref_ans  = str(p.get("reference_answer", ""))

    modes = []

    # Refusal
    refusal_phrases = [
        "cannot determine", "cannot calculate", "can't determine",
        "cannot be determined", "insufficient information",
        "not enough information", "i don't have", "unable to determine",
        "not provided", "not available in", "no information",
    ]
    if m.get("is_refusal") or any(ph in gen.lower() for ph in refusal_phrases):
        modes.append("REFUSAL")
        return modes   # once you refuse, other tags don't apply

    # Unit / scale error: number is right but off by 1000 (millions vs billions)
    if ref_num and pred_nums and ref_num != 0:
        for pn in pred_nums:
            ratio = pn / ref_num if ref_num != 0 else None
            if ratio and (abs(ratio - 1000) < 1 or abs(ratio - 0.001) < 0.0001):
                modes.append("UNIT_SCALE_ERROR")
                break

    # Multi-step reasoning attempt (shows working, gets it wrong)
    chain_markers = [
        "step 1", "step 2", "to calculate", "using the formula",
        "formula:", "we need to", "first, we", "let me calculate",
        "let's calculate",
    ]
    if any(m in gen.lower() for m in chain_markers) and "UNIT_SCALE_ERROR" not in modes:
        modes.append("MULTI_STEP_CALC")

    # Wrong number — a number was predicted but it's wrong
    if pred_nums and "UNIT_SCALE_ERROR" not in modes and "MULTI_STEP_CALC" not in modes:
        modes.append("WRONG_NUMBER")

    # Format mismatch — answer might be there but wrong suffix/prefix
    # e.g., ref = "$0.40B", gen = "400 million"
    if ref_ans:
        ref_clean = re.sub(r"[^\d.]", "", ref_ans)
        ref_float = safe_float(ref_clean)
        if ref_float and any(
            abs(pn - ref_float) < 1e-6 for pn in pred_nums
        ):
            modes.append("FORMAT_MISMATCH")

    # Hallucination: numbers in answer but none close to anything real
    if pred_nums and not modes:
        modes.append("HALLUCINATION")

    if not modes:
        modes.append("OTHER")

    return modes


# ── per-sample analysis ────────────────────────────────────────────────────────

def analyse_file(pred_path: Path, oracle_path: Path | None = None) -> list[dict]:
    with open(pred_path) as f:
        preds = json.load(f)

    oracle_map = {}
    if oracle_path and oracle_path.exists():
        with open(oracle_path) as f:
            for op in json.load(f):
                oracle_map[op["financebench_id"]] = op

    rows = []
    for p in preds:
        fid  = p["financebench_id"]
        qt   = p.get("question_type", "unknown")
        m    = p.get("eval_metrics", {}) or {}
        nm   = m.get("numeric_match")
        gen  = p.get("generated_answer", "") or ""

        ret_status = retrieval_hit(p)
        correct    = is_correct(p)

        failure_modes = []
        if not correct:
            failure_modes = classify_failure(p)

        # oracle correctness (retrieval-independent)
        oracle_correct = None
        oracle_failure_modes = []
        if fid in oracle_map:
            op = oracle_map[fid]
            oracle_correct = is_correct(op)
            if not oracle_correct:
                oracle_failure_modes = classify_failure(op)

        rows.append({
            "id":                   fid,
            "question_type":        qt,
            "doc_name":             p.get("doc_name", ""),
            "question":             p.get("question", ""),
            "reference_answer":     p.get("reference_answer", ""),
            "generated_answer":     gen,
            "oracle_answer":        oracle_map.get(fid, {}).get("generated_answer", "") if fid in oracle_map else "",
            "retrieval_status":     ret_status,
            "rag_correct":          correct,
            "oracle_correct":       oracle_correct,
            "numeric_match":        nm,
            "squad_f1":             m.get("squad_f1"),
            "rougeL":               m.get("rougeL"),
            "is_refusal":           bool(m.get("is_refusal") or any(ph in gen.lower() for ph in ["cannot determine","cannot calculate","unable to"])),
            "pred_numbers":         str(m.get("pred_numbers", [])),
            "ref_number":           m.get("ref_number"),
            "failure_modes":        "|".join(failure_modes) if failure_modes else "",
            "oracle_failure_modes": "|".join(oracle_failure_modes) if oracle_failure_modes else "",
        })

    return rows


# ── summary stats ──────────────────────────────────────────────────────────────

def compute_summary(rows: list[dict], model_name: str) -> dict:
    total = len(rows)
    ret_hit    = sum(1 for r in rows if r["retrieval_status"] == "full")
    ret_part   = sum(1 for r in rows if r["retrieval_status"] == "partial")
    ret_miss   = sum(1 for r in rows if r["retrieval_status"] == "miss")

    rag_correct    = sum(1 for r in rows if r["rag_correct"])
    oracle_correct = sum(1 for r in rows if r["oracle_correct"])

    # 2×2: retrieval × correctness
    rh_correct = sum(1 for r in rows if r["retrieval_status"] in ("full","partial") and r["rag_correct"])
    rh_wrong   = sum(1 for r in rows if r["retrieval_status"] in ("full","partial") and not r["rag_correct"])
    rm_correct = sum(1 for r in rows if r["retrieval_status"] == "miss" and r["rag_correct"])
    rm_wrong   = sum(1 for r in rows if r["retrieval_status"] == "miss" and not r["rag_correct"])

    retrieval_ok = ret_hit + ret_part

    # failure mode counts (all RAG failures)
    all_modes = []
    for r in rows:
        if not r["rag_correct"] and r["failure_modes"]:
            all_modes.extend(r["failure_modes"].split("|"))
    fm_counts = Counter(all_modes)

    # oracle failure modes
    oracle_modes = []
    for r in rows:
        if r["oracle_correct"] is False and r["oracle_failure_modes"]:
            oracle_modes.extend(r["oracle_failure_modes"].split("|"))
    ofm_counts = Counter(oracle_modes)

    # by question type
    by_qt = {}
    for qt in ["metrics-generated", "domain-relevant", "novel-generated"]:
        sub = [r for r in rows if r["question_type"] == qt]
        by_qt[qt] = {
            "n": len(sub),
            "rag_correct": sum(1 for r in sub if r["rag_correct"]),
            "oracle_correct": sum(1 for r in sub if r["oracle_correct"]),
            "ret_miss": sum(1 for r in sub if r["retrieval_status"] == "miss"),
        }

    return {
        "model":           model_name,
        "total":           total,
        "ret_full_hit_%":  round(100*ret_hit/total),
        "ret_partial_%":   round(100*ret_part/total),
        "ret_miss_%":      round(100*ret_miss/total),
        "rag_correct_%":   round(100*rag_correct/total),
        "oracle_correct_%":round(100*oracle_correct/total) if oracle_correct else None,
        "rag_gap_from_oracle_%": round(100*(oracle_correct-rag_correct)/total) if oracle_correct else None,
        "pct_gap_from_retrieval": round(100*ret_miss/total),
        "success_rate_given_ret_hit_%": round(100*rh_correct/retrieval_ok) if retrieval_ok else None,
        "success_rate_given_ret_miss_%": round(100*rm_correct/ret_miss) if ret_miss else None,
        "failure_modes_rag":    dict(fm_counts.most_common()),
        "failure_modes_oracle": dict(ofm_counts.most_common()),
        "by_question_type":     by_qt,
    }


# ── HTML report ───────────────────────────────────────────────────────────────

STYLE = """
<style>
body { font-family: -apple-system, sans-serif; margin: 2rem; color: #222; }
h1 { color: #1a1a2e; }
h2 { color: #16213e; border-bottom: 2px solid #0f3460; padding-bottom: 4px; }
h3 { color: #533483; }
table { border-collapse: collapse; width: 100%; margin-bottom: 1rem; font-size: 0.85rem; }
th { background: #0f3460; color: white; padding: 6px 10px; text-align: left; }
td { padding: 5px 10px; border-bottom: 1px solid #ddd; vertical-align: top; }
tr:hover { background: #f0f4ff; }
.tag { display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 0.75rem; font-weight: bold; margin: 1px; }
.tag-REFUSAL { background:#ff6b6b;color:white; }
.tag-MULTI_STEP_CALC { background:#ffd93d;color:#222; }
.tag-UNIT_SCALE_ERROR { background:#6bcb77;color:white; }
.tag-WRONG_NUMBER { background:#4d96ff;color:white; }
.tag-HALLUCINATION { background:#a855f7;color:white; }
.tag-FORMAT_MISMATCH { background:#f97316;color:white; }
.tag-OTHER { background:#aaa;color:white; }
.good { color: green; font-weight:bold; }
.bad  { color: red;   font-weight:bold; }
.hit  { color: green; }
.miss { color: red; }
.partial { color: orange; }
.box { background:#f8f9fa; border-left:4px solid #0f3460; padding:1rem; margin:1rem 0; border-radius:4px; }
</style>
"""

def tag_html(modes_str: str) -> str:
    if not modes_str:
        return ""
    tags = []
    for m in modes_str.split("|"):
        tags.append(f'<span class="tag tag-{m}">{m}</span>')
    return " ".join(tags)


def ret_badge(status: str) -> str:
    cls = {"full": "hit", "partial": "partial", "miss": "miss", "no_gold": "partial"}.get(status, "")
    return f'<span class="{cls}">{status}</span>'


def build_html(all_summaries: list[dict], all_rows: dict[str, list[dict]]) -> str:
    parts = [f"<html><head><meta charset='utf-8'><title>LLM Scaling Error Analysis</title>{STYLE}</head><body>"]
    parts.append("<h1>LLM Scaling Error Analysis</h1>")

    # ── Overall summary table ──────────────────────────────────────────────────
    parts.append("<h2>1. Overall Performance &amp; Retrieval–Generation Decomposition</h2>")
    parts.append("<div class='box'><b>Reading guide</b>: "
                 "The gap between RAG and Oracle is caused by retrieval failures. "
                 "Oracle failure is a pure generation/reasoning failure (model had the right context). "
                 "Together they tell you whether to invest in better retrieval or a better prompt/model.</div>")

    cols = ["model", "ret_miss_%", "rag_correct_%", "oracle_correct_%",
            "rag_gap_from_oracle_%", "success_rate_given_ret_hit_%", "success_rate_given_ret_miss_%"]
    headers = ["Model", "Ret Miss %", "RAG Correct %", "Oracle Correct %",
               "Gap (Oracle-RAG) %", "P(correct | ret hit)", "P(correct | ret miss)"]

    parts.append("<table><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>")
    for s in all_summaries:
        parts.append("<tr>" + "".join(
            f"<td>{s.get(c, '–')}</td>" for c in cols
        ) + "</tr>")
    parts.append("</table>")

    # ── By question type ──────────────────────────────────────────────────────
    parts.append("<h2>2. Performance by Question Type</h2>")
    parts.append("<table><tr><th>Model</th><th>Question Type</th><th>N</th>"
                 "<th>RAG Correct</th><th>Oracle Correct</th><th>Ret Miss</th></tr>")
    for s in all_summaries:
        for qt, v in s["by_question_type"].items():
            n = v["n"]
            rc = v["rag_correct"]
            oc = v["oracle_correct"]
            rm = v["ret_miss"]
            parts.append(
                f"<tr><td>{s['model']}</td><td>{qt}</td><td>{n}</td>"
                f"<td>{rc}/{n} ({100*rc//n}%)</td>"
                f"<td>{oc}/{n} ({100*oc//n}%)</td>"
                f"<td>{rm}/{n} ({100*rm//n}%)</td></tr>"
            )
    parts.append("</table>")

    # ── Failure mode breakdown ─────────────────────────────────────────────────
    parts.append("<h2>3. Failure Mode Breakdown</h2>")
    parts.append("<div class='box'>"
                 "<b>RAG failures</b> include both retrieval-caused and generation-caused errors. "
                 "<b>Oracle failures</b> are purely generation / reasoning failures (retrieval was perfect). "
                 "Compare the two columns to isolate generation issues.</div>")

    parts.append("<table><tr><th>Model</th><th>Failure Mode</th><th>RAG count</th><th>Oracle count</th></tr>")
    all_modes = set()
    for s in all_summaries:
        all_modes |= set(s["failure_modes_rag"].keys()) | set(s["failure_modes_oracle"].keys())
    for mode in sorted(all_modes):
        for s in all_summaries:
            rc = s["failure_modes_rag"].get(mode, 0)
            oc = s["failure_modes_oracle"].get(mode, 0)
            parts.append(f"<tr><td>{s['model']}</td><td>{tag_html(mode)}</td><td>{rc}</td><td>{oc}</td></tr>")
    parts.append("</table>")

    # ── Per-sample deep dive ───────────────────────────────────────────────────
    parts.append("<h2>4. Per-Sample Analysis</h2>")

    for model_name, rows in all_rows.items():
        parts.append(f"<h3>{model_name}</h3>")

        # Sort: failures first, then retrieval miss, then correct
        def sort_key(r):
            return (int(r["rag_correct"]), r["retrieval_status"] == "full", r["question_type"])

        sorted_rows = sorted(rows, key=sort_key)

        parts.append(
            "<table><tr>"
            "<th>#</th><th>QType</th><th>Retrieval</th>"
            "<th>RAG</th><th>Oracle</th>"
            "<th>Question</th><th>Reference</th>"
            "<th>RAG Answer</th><th>Oracle Answer</th>"
            "<th>Failure modes (RAG)</th><th>Failure modes (Oracle)</th>"
            "</tr>"
        )

        for i, r in enumerate(sorted_rows, 1):
            rag_cls   = "good" if r["rag_correct"]    else "bad"
            ora_cls   = "good" if r["oracle_correct"] else "bad"
            gen_short = (r["generated_answer"] or "")[:200].replace("<","&lt;").replace(">","&gt;")
            ora_short = (r["oracle_answer"]    or "")[:200].replace("<","&lt;").replace(">","&gt;")
            q_short   = (r["question"]         or "")[:120].replace("<","&lt;").replace(">","&gt;")
            ref_short = (r["reference_answer"] or "")[:80].replace("<","&lt;").replace(">","&gt;")

            parts.append(
                f"<tr>"
                f"<td>{i}</td>"
                f"<td>{r['question_type']}</td>"
                f"<td>{ret_badge(r['retrieval_status'])}</td>"
                f"<td class='{rag_cls}'>{'✓' if r['rag_correct'] else '✗'}</td>"
                f"<td class='{ora_cls}'>{'✓' if r['oracle_correct'] else ('✗' if r['oracle_correct'] is False else '–')}</td>"
                f"<td>{q_short}</td>"
                f"<td><b>{ref_short}</b></td>"
                f"<td>{gen_short}</td>"
                f"<td>{ora_short}</td>"
                f"<td>{tag_html(r['failure_modes'])}</td>"
                f"<td>{tag_html(r['oracle_failure_modes'])}</td>"
                f"</tr>"
            )
        parts.append("</table>")

    parts.append("</body></html>")
    return "\n".join(parts)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM Scaling Error Analysis")
    parser.add_argument("--pred-dir",   default="outputs/llm_scaling/predictions",       help="RAG prediction JSON dir")
    parser.add_argument("--oracle-dir", default="outputs/llm_scaling/oracle_predictions", help="Oracle prediction JSON dir")
    parser.add_argument("--out-dir",    default="outputs/llm_scaling/error_analysis",     help="Output directory")
    parser.add_argument("--models",     nargs="+", default=["Qwen2.5-3B","Qwen2.5-7B","Qwen2.5-14B"])
    args = parser.parse_args()

    pred_dir   = Path(args.pred_dir)
    oracle_dir = Path(args.oracle_dir)
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    all_rows      = {}

    for model in args.models:
        pred_path   = pred_dir   / f"{model}.json"
        oracle_path = oracle_dir / f"{model}_oracle.json"

        if not pred_path.exists():
            print(f"[SKIP] {pred_path} not found", file=sys.stderr)
            continue

        print(f"Analysing {model}...")
        rows = analyse_file(pred_path, oracle_path if oracle_path.exists() else None)
        summ = compute_summary(rows, model)

        all_rows[model]  = rows
        all_summaries.append(summ)

        # per-model CSV
        csv_path = out_dir / f"{model}_samples.csv"
        if rows:
            with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        print(f"  → {len(rows)} samples written to {csv_path}")

    # combined summary CSV
    summary_rows = []
    for s in all_summaries:
        flat = {k: v for k, v in s.items() if not isinstance(v, dict)}
        flat["failure_modes_rag"]    = str(s["failure_modes_rag"])
        flat["failure_modes_oracle"] = str(s["failure_modes_oracle"])
        summary_rows.append(flat)
    if summary_rows:
        with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)

    # HTML report
    html = build_html(all_summaries, all_rows)
    html_path = out_dir / "error_analysis.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"\nHTML report: {html_path}")

    # ── Print console summary ─────────────────────────────────────────────────
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for s in all_summaries:
        print(f"\n{s['model']}")
        print(f"  Retrieval miss:          {s['ret_miss_%']}%  ({s['total']-round(s['total']*s['ret_miss_%']/100)}/{s['total']} retrieved)")
        print(f"  RAG correct:             {s['rag_correct_%']}%")
        print(f"  Oracle correct:          {s['oracle_correct_%']}%")
        print(f"  Gap (retrieval cause):   ~{s['rag_gap_from_oracle_%']}%")
        print(f"  P(correct | ret hit):    {s['success_rate_given_ret_hit_%']}%")
        print(f"  P(correct | ret miss):   {s['success_rate_given_ret_miss_%']}%  (lucky guesses)")
        print(f"  Oracle failure modes:    {s['failure_modes_oracle']}")
        print(f"  RAG failure modes:       {s['failure_modes_rag']}")


if __name__ == "__main__":
    main()
