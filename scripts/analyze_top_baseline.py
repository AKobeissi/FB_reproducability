#!/usr/bin/env python3
"""
analyze_top_baseline.py
=======================
Full error-decomposition report for the DENSE+MultiHyDE+ReRANKER top baseline.

Usage:
    python analyze_top_baseline.py \
        --pred   baselines/results/predictions/multi_hyde_reranker_generated.json \
        --out    outputs/top_baseline_analysis/report.html

The HTML report contains:
  • Executive summary dashboard (overall stats, 2×2 retrieval×correctness matrix,
    failure-mode breakdown, per-question-type breakdown)
  • One card per sample (all 150 FB questions), with:
      – Question (full)
      – Expected vs Generated answer (full, side-by-side)
      – All computed metrics (ROUGE-L, squad_F1, numeric match, exact match)
      – Failure mode tags with explanations
      – Retrieval status (full hit / partial / miss) against gold evidence
      – Gold evidence segment(s) — full text, highlighted
      – Top-5 retrieved chunks — full text, with rank / dense-score / rerank-score /
        page / doc; chunks whose (doc, page) match gold evidence are highlighted green
  • JS filtering (correct / incorrect / by question type / by failure mode /
    by retrieval status) and keyword search — no page reload needed
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ─── Metrics ──────────────────────────────────────────────────────────────────

_SCALE_PATTERNS = [
    (re.compile(r"\btrillion[s]?\b", re.I), 1e12),
    (re.compile(r"\bbillion[s]?\b|\bbn\b|\bbln\b", re.I), 1e9),
    (re.compile(r"\bmillion[s]?\b|\bmm\b|\bmln\b", re.I), 1e6),
    (re.compile(r"\bthousand[s]?\b|\bk\b", re.I), 1e3),
]

REFUSAL_PHRASES = [
    "cannot determine", "cannot calculate", "can't determine",
    "cannot be determined", "insufficient information",
    "not enough information", "i don't have", "unable to determine",
    "not provided", "not available in", "no information",
    "i cannot determine", "i am unable",
]


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_all_numbers(text: str) -> list:
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
    ref_nums = _extract_all_numbers(ref)
    if not ref_nums:
        return 0.0
    ref_raw = ref_nums[0]
    ref_scaled = _scaled_value(ref_raw, ref)

    pred_nums = _extract_all_numbers(pred)
    if not pred_nums:
        return 0.0

    for p_raw in pred_nums:
        p_scaled = _scaled_value(p_raw, pred)
        ref_has_scale = any(p.search(ref) for p, _ in _SCALE_PATTERNS)
        pred_has_scale = any(p.search(pred) for p, _ in _SCALE_PATTERNS)
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


def squad_f1(pred: str, ref: str) -> float:
    pred_tokens = _normalize(pred).split()
    ref_tokens = _normalize(ref).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    p = n_common / len(pred_tokens)
    r = n_common / len(ref_tokens)
    return 2 * p * r / (p + r)


def rouge_l(pred: str, ref: str) -> float:
    try:
        from rouge_score import rouge_scorer as rs
        scorer = rs.RougeScorer(["rougeL"], use_stemmer=True)
        return scorer.score(ref, pred)["rougeL"].fmeasure
    except Exception:
        # fallback: token overlap LCS approximation
        p_tok = _normalize(pred).split()
        r_tok = _normalize(ref).split()
        if not p_tok or not r_tok:
            return 0.0
        common = set(p_tok) & set(r_tok)
        if not common:
            return 0.0
        prec = len([t for t in p_tok if t in common]) / len(p_tok)
        rec = len([t for t in r_tok if t in common]) / len(r_tok)
        return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def exact_match(pred: str, ref: str) -> float:
    return 1.0 if _normalize(pred) == _normalize(ref) else 0.0


def is_refusal(text: str) -> bool:
    t = text.lower()
    return any(ph in t for ph in REFUSAL_PHRASES)


# ─── Correctness ──────────────────────────────────────────────────────────────

def is_correct(sample: dict) -> bool:
    gen = sample.get("generated_answer", "") or ""
    ref = sample.get("reference_answer", "") or ""
    qt  = sample.get("question_type", "")

    if is_refusal(gen):
        return False

    if qt == "metrics-generated":
        nm = numeric_match(gen, ref)
        if nm == 1.0:
            return True
        # also accept if squad_f1 is high (covers edge cases like "N/A")
        return squad_f1(gen, ref) > 0.7

    # text questions
    return squad_f1(gen, ref) > 0.4 or rouge_l(gen, ref) > 0.35


# ─── Retrieval status ─────────────────────────────────────────────────────────

def _reranked_order(chunks: list) -> list:
    """
    Return chunks sorted by post-reranker position (the order fed to the LLM).
    Prefer _score descending (reranker output); fall back to rank ascending
    only when _score is absent for all chunks.
    """
    has_score = any(c.get("_score") is not None for c in chunks)
    if has_score:
        return sorted(chunks, key=lambda c: -(c.get("_score") or 0))
    return sorted(chunks, key=lambda c: c.get("rank", 999))


def retrieval_status(sample: dict, top_k: int = 5) -> tuple[str, list[int]]:
    """
    Returns (status, list_of_matching_gold_indices).
    status ∈ {'full', 'partial', 'miss', 'no_gold'}
    matching_gold_indices = indices into gold_evidence_segments that were found.
    Uses the post-reranker top-k (sorted by _score) — the chunks actually fed
    to the generator.
    """
    gold_segs = sample.get("gold_evidence_segments") or []
    chunks = (sample.get("retrieved_chunks") or [])
    top5 = _reranked_order(chunks)[:top_k]

    if not gold_segs:
        return "no_gold", []

    # Strategy 1: (doc_name, page) exact match
    top5_pages = {
        (c.get("metadata", {}).get("doc_name", ""), c.get("metadata", {}).get("page"))
        for c in top5
    }

    found_by_page = []
    for i, seg in enumerate(gold_segs):
        seg_doc  = seg.get("doc_name", "")
        seg_page = seg.get("page")
        if (seg_doc, seg_page) in top5_pages:
            found_by_page.append(i)

    if found_by_page:
        if len(found_by_page) == len(gold_segs):
            return "full", found_by_page
        return "partial", found_by_page

    # Strategy 2: text snippet match (first 60 chars of gold segment)
    top5_text = " ".join(c.get("text", "").lower() for c in top5)
    found_by_text = []
    for i, seg in enumerate(gold_segs):
        snippet = (seg.get("text", "")[:60]).strip().lower()
        if snippet and snippet in top5_text:
            found_by_text.append(i)

    if found_by_text:
        if len(found_by_text) == len(gold_segs):
            return "full", found_by_text
        return "partial", found_by_text

    return "miss", []


# ─── Failure mode classifier ──────────────────────────────────────────────────

FAILURE_MODE_EXPLANATIONS = {
    "REFUSAL":          "Model said it cannot determine the answer from the provided information.",
    "UNIT_SCALE_ERROR": "Model produced the right number but in the wrong unit scale (e.g., millions vs billions, off by ×1000).",
    "MULTI_STEP_CALC":  "Model attempted a multi-step calculation (showed working) but arrived at the wrong result.",
    "WRONG_NUMBER":     "Model produced a specific number but it does not match the reference answer.",
    "FORMAT_MISMATCH":  "The correct numerical value is present but expressed in the wrong format (e.g., % vs absolute, $M vs $B).",
    "HALLUCINATION":    "Model generated numbers or statements that are not grounded in the retrieved context.",
    "RETRIEVAL_MISS":   "The correct evidence page was not in the top-5 retrieved chunks.",
    "OTHER":            "Failure does not fit neatly into the above categories.",
}


def classify_failure(sample: dict, ret_status: str) -> list[str]:
    gen = (sample.get("generated_answer", "") or "").strip()
    ref = (sample.get("reference_answer", "") or "").strip()

    modes = []

    if is_refusal(gen):
        modes.append("REFUSAL")
        if ret_status == "miss":
            modes.append("RETRIEVAL_MISS")
        return modes

    if ret_status == "miss":
        modes.append("RETRIEVAL_MISS")

    ref_nums = _extract_all_numbers(ref)
    pred_nums = _extract_all_numbers(gen)
    ref_raw = ref_nums[0] if ref_nums else None
    ref_scaled = _scaled_value(ref_raw, ref) if ref_raw is not None else None

    # Unit/scale error: any pred number is within 0.2% of ref×1000 or ref÷1000
    if ref_raw and pred_nums:
        for pn in pred_nums:
            ratio = pn / ref_raw if ref_raw != 0 else None
            if ratio and (abs(ratio - 1000) < 2 or abs(ratio - 0.001) < 0.0002):
                modes.append("UNIT_SCALE_ERROR")
                break

    chain_markers = [
        "step 1", "step 2", "to calculate", "using the formula",
        "formula:", "we need to", "first, we", "let me calculate",
        "let's calculate", "therefore", "thus,",
    ]
    if "UNIT_SCALE_ERROR" not in modes and any(m in gen.lower() for m in chain_markers):
        modes.append("MULTI_STEP_CALC")

    # Format mismatch: the correct number is literally there but wrong suffix
    if ref_raw and pred_nums and "UNIT_SCALE_ERROR" not in modes:
        ref_clean = re.sub(r"[^\d.]", "", ref)
        ref_float = None
        try:
            ref_float = float(ref_clean)
        except Exception:
            pass
        if ref_float and any(abs(pn - ref_float) < 1e-4 for pn in pred_nums):
            modes.append("FORMAT_MISMATCH")

    # Wrong number: numbers predicted but none match
    if pred_nums and not any(m in modes for m in ["UNIT_SCALE_ERROR", "FORMAT_MISMATCH", "MULTI_STEP_CALC"]):
        if ref_raw is None or numeric_match(gen, ref) < 1.0:
            modes.append("WRONG_NUMBER")

    # Hallucination: predicted numbers exist but we couldn't find gold
    if pred_nums and ret_status == "miss" and not modes:
        modes.append("HALLUCINATION")

    if not modes:
        modes.append("OTHER")

    return modes


# ─── Per-sample analysis ──────────────────────────────────────────────────────

def analyse_samples(samples: list[dict]) -> list[dict]:
    results = []
    for s in samples:
        gen = (s.get("generated_answer", "") or "").strip()
        ref = (s.get("reference_answer", "") or "").strip()
        qt  = s.get("question_type", "unknown")

        correct  = is_correct(s)
        ret_stat, gold_found_indices = retrieval_status(s)

        nm   = numeric_match(gen, ref) if qt == "metrics-generated" else None
        sf   = squad_f1(gen, ref)
        rl   = rouge_l(gen, ref)
        em   = exact_match(gen, ref)
        ref_ = is_refusal(gen)

        failure_modes = [] if correct else classify_failure(s, ret_stat)

        # Top-5 chunks in post-reranker order (sorted by _score, as fed to the LLM)
        all_chunks = _reranked_order(s.get("retrieved_chunks", []))
        top5 = all_chunks[:5]

        # Identify which top-5 chunks overlap with gold evidence (by doc+page)
        gold_segs = s.get("gold_evidence_segments") or []
        gold_pages = {(g.get("doc_name", ""), g.get("page")) for g in gold_segs}

        chunks_annotated = []
        for reranked_pos, c in enumerate(top5, start=1):
            meta = c.get("metadata", {})
            is_gold = (meta.get("doc_name", ""), meta.get("page")) in gold_pages
            chunks_annotated.append({
                "rank":          reranked_pos,          # post-reranker position (1-5)
                "orig_rank":     c.get("rank"),         # original dense retrieval rank
                "dense_score":   c.get("_score"),
                "rerank_score":  c.get("_reranker_score") or c.get("_rerank_score"),
                "doc_name":      meta.get("doc_name", ""),
                "page":          meta.get("page"),
                "text":          c.get("text", ""),
                "is_gold_page":  is_gold,
            })

        results.append({
            "financebench_id":      s.get("financebench_id", ""),
            "question_type":        qt,
            "doc_name":             s.get("doc_name", ""),
            "question":             s.get("question", ""),
            "reference_answer":     ref,
            "generated_answer":     gen,
            "correct":              correct,
            "is_refusal":           ref_,
            "retrieval_status":     ret_stat,
            "gold_found_indices":   gold_found_indices,
            "failure_modes":        failure_modes,
            "metrics": {
                "numeric_match":    nm,
                "squad_f1":         sf,
                "rougeL":           rl,
                "exact_match":      em,
            },
            "gold_evidence_segments": gold_segs,
            "top5_chunks":          chunks_annotated,
        })
    return results


# ─── Summary stats ────────────────────────────────────────────────────────────

def compute_summary(results: list[dict]) -> dict:
    n = len(results)
    n_correct = sum(1 for r in results if r["correct"])
    n_miss    = sum(1 for r in results if r["retrieval_status"] == "miss")
    n_partial = sum(1 for r in results if r["retrieval_status"] == "partial")
    n_full    = sum(1 for r in results if r["retrieval_status"] == "full")

    # 2×2
    n_hit = n_full + n_partial
    hit_correct  = sum(1 for r in results if r["retrieval_status"] in ("full","partial") and r["correct"])
    hit_wrong    = n_hit - hit_correct
    miss_correct = sum(1 for r in results if r["retrieval_status"] == "miss" and r["correct"])
    miss_wrong   = n_miss - miss_correct

    all_modes = []
    for r in results:
        if not r["correct"]:
            all_modes.extend(r["failure_modes"])
    mode_counts = Counter(all_modes)

    by_qt = {}
    for qt in ["metrics-generated", "domain-relevant", "novel-generated"]:
        sub = [r for r in results if r["question_type"] == qt]
        by_qt[qt] = {
            "n":        len(sub),
            "correct":  sum(1 for r in sub if r["correct"]),
            "ret_full": sum(1 for r in sub if r["retrieval_status"] == "full"),
            "ret_partial": sum(1 for r in sub if r["retrieval_status"] == "partial"),
            "ret_miss": sum(1 for r in sub if r["retrieval_status"] == "miss"),
        }

    avg_rl   = sum(r["metrics"]["rougeL"] for r in results) / n
    avg_sf   = sum(r["metrics"]["squad_f1"] for r in results) / n
    nm_vals  = [r["metrics"]["numeric_match"] for r in results if r["metrics"]["numeric_match"] is not None]
    avg_nm   = sum(nm_vals) / len(nm_vals) if nm_vals else 0.0

    return {
        "n": n,
        "n_correct": n_correct,
        "n_wrong": n - n_correct,
        "acc_pct": round(100 * n_correct / n, 1),
        "ret_full": n_full,
        "ret_partial": n_partial,
        "ret_miss": n_miss,
        "ret_full_pct":    round(100 * n_full / n, 1),
        "ret_partial_pct": round(100 * n_partial / n, 1),
        "ret_miss_pct":    round(100 * n_miss / n, 1),
        "hit_correct": hit_correct,
        "hit_wrong":   hit_wrong,
        "miss_correct": miss_correct,
        "miss_wrong":   miss_wrong,
        "p_correct_given_hit":  round(100 * hit_correct / n_hit, 1) if n_hit else 0,
        "p_correct_given_miss": round(100 * miss_correct / n_miss, 1) if n_miss else 0,
        "mode_counts": dict(mode_counts.most_common()),
        "by_qt":  by_qt,
        "avg_rougeL":       round(avg_rl, 4),
        "avg_squad_f1":     round(avg_sf, 4),
        "avg_numeric_match": round(avg_nm, 4),
    }


# ─── HTML helpers ─────────────────────────────────────────────────────────────

TAG_COLORS = {
    "REFUSAL":          ("#dc2626", "#fff"),
    "RETRIEVAL_MISS":   ("#7c3aed", "#fff"),
    "UNIT_SCALE_ERROR": ("#059669", "#fff"),
    "MULTI_STEP_CALC":  ("#d97706", "#fff"),
    "WRONG_NUMBER":     ("#2563eb", "#fff"),
    "FORMAT_MISMATCH":  ("#ea580c", "#fff"),
    "HALLUCINATION":    ("#db2777", "#fff"),
    "OTHER":            ("#6b7280", "#fff"),
}

QT_COLORS = {
    "metrics-generated": "#1d4ed8",
    "domain-relevant":   "#0f766e",
    "novel-generated":   "#7c3aed",
}

RET_COLORS = {
    "full":    ("#15803d", "Full Hit"),
    "partial": ("#b45309", "Partial Hit"),
    "miss":    ("#dc2626", "Miss"),
    "no_gold": ("#6b7280", "No Gold"),
}


def esc(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def tag_badge(mode: str) -> str:
    bg, fg = TAG_COLORS.get(mode, ("#6b7280", "#fff"))
    tip = esc(FAILURE_MODE_EXPLANATIONS.get(mode, ""))
    return f'<span class="badge" style="background:{bg};color:{fg}" title="{tip}">{esc(mode)}</span>'


def ret_badge(status: str) -> str:
    color, label = RET_COLORS.get(status, ("#6b7280", status))
    return f'<span class="ret-badge" style="background:{color}">{label}</span>'


def qt_badge(qt: str) -> str:
    color = QT_COLORS.get(qt, "#374151")
    short = {"metrics-generated": "Metrics", "domain-relevant": "Domain", "novel-generated": "Novel"}.get(qt, qt)
    return f'<span class="qt-badge" style="border-color:{color};color:{color}">{short}</span>'


def correct_badge(correct: bool) -> str:
    if correct:
        return '<span class="correct-badge correct">CORRECT</span>'
    return '<span class="correct-badge wrong">WRONG</span>'


def metric_pill(label: str, value, threshold_good: float = None, threshold_ok: float = None) -> str:
    if value is None:
        return f'<span class="metric-pill neutral"><b>{label}</b> —</span>'
    display = f"{value:.3f}" if isinstance(value, float) else str(value)
    cls = "neutral"
    if threshold_good is not None and isinstance(value, float):
        if value >= threshold_good:
            cls = "good"
        elif threshold_ok is not None and value >= threshold_ok:
            cls = "ok"
        else:
            cls = "bad"
    return f'<span class="metric-pill {cls}"><b>{label}</b> {display}</span>'


def chunk_card(c: dict, idx: int) -> str:
    is_gold = c["is_gold_page"]
    border  = "#15803d" if is_gold else "#e5e7eb"
    bg      = "#f0fdf4" if is_gold else "#f9fafb"
    gold_marker = '<span class="gold-marker">★ Gold Page</span>' if is_gold else ""
    score   = f"{c['dense_score']:.4f}" if c['dense_score'] is not None else "—"
    rerank  = f"{c['rerank_score']:.4f}" if c['rerank_score'] is not None else "—"
    orig    = f"orig dense rank: {c['orig_rank']}" if c.get('orig_rank') is not None else ""
    orig_label = f'<span class="chunk-orig-rank">{orig}</span>' if orig else ""
    return f"""
<div class="chunk-card" style="border-color:{border};background:{bg}">
  <div class="chunk-header">
    <span class="chunk-rank">Rank {c['rank']}</span>
    <span class="chunk-meta">{esc(c['doc_name'])} · p.{c['page']}</span>
    <span class="chunk-scores">Score: {score} &nbsp;|&nbsp; Rerank: {rerank}</span>
    {orig_label}
    {gold_marker}
  </div>
  <pre class="chunk-text">{esc(c['text'])}</pre>
</div>"""


def gold_seg_card(seg: dict, found: bool) -> str:
    border = "#15803d" if found else "#9ca3af"
    bg     = "#f0fdf4" if found else "#f3f4f6"
    label  = "Retrieved ✓" if found else "NOT retrieved ✗"
    label_color = "#15803d" if found else "#dc2626"
    return f"""
<div class="gold-card" style="border-color:{border};background:{bg}">
  <div class="gold-header">
    <span style="font-weight:700">Gold Evidence</span>
    <span class="chunk-meta">{esc(seg.get('doc_name',''))} · p.{seg.get('page','?')}</span>
    <span style="color:{label_color};font-weight:700">{label}</span>
  </div>
  <pre class="chunk-text">{esc(seg.get('text',''))}</pre>
</div>"""


def sample_card(r: dict, idx: int) -> str:
    fid     = r["financebench_id"]
    qt      = r["question_type"]
    correct = r["correct"]
    ret     = r["retrieval_status"]
    modes   = r["failure_modes"]
    m       = r["metrics"]

    # failure mode tags
    tags_html = " ".join(tag_badge(m_) for m_ in modes) if modes else ""

    # metrics row
    nm_pill = metric_pill("Numeric Match", m["numeric_match"], 1.0, 0.5) if qt == "metrics-generated" else ""
    metrics_html = (
        metric_pill("ROUGE-L",    m["rougeL"],    0.35, 0.2)
        + metric_pill("Squad-F1",   m["squad_f1"],  0.4,  0.2)
        + metric_pill("Exact Match", m["exact_match"], 1.0, 0.5)
        + nm_pill
    )

    # gold segments
    gold_segs = r["gold_evidence_segments"]
    gold_found_set = set(r["gold_found_indices"])
    gold_html = "".join(
        gold_seg_card(seg, i in gold_found_set)
        for i, seg in enumerate(gold_segs)
    )
    if not gold_html:
        gold_html = '<p class="no-data">No gold evidence segments available.</p>'

    # chunks
    chunks_html = "".join(chunk_card(c, i) for i, c in enumerate(r["top5_chunks"]))
    if not chunks_html:
        chunks_html = '<p class="no-data">No retrieved chunks.</p>'

    # data attrs for JS filtering
    mode_str   = "|".join(modes)
    correct_s  = "correct" if correct else "wrong"
    refusal_s  = "refusal" if r["is_refusal"] else ""

    card_border = "#15803d" if correct else "#dc2626"

    return f"""
<div class="sample-card" id="{fid}"
     data-correct="{correct_s}"
     data-qt="{qt}"
     data-ret="{ret}"
     data-modes="{mode_str}"
     data-refusal="{refusal_s}"
     data-text="{esc(r['question'].lower())} {esc(r['reference_answer'].lower())} {esc(r['generated_answer'].lower()[:200])}"
     style="border-left-color:{card_border}">

  <!-- Card header -->
  <div class="card-header">
    <span class="card-num">#{idx}</span>
    {correct_badge(correct)}
    {qt_badge(qt)}
    {ret_badge(ret)}
    <span class="fid">{fid}</span>
    <span class="doc-name">{esc(r['doc_name'])}</span>
  </div>

  <!-- Failure modes -->
  {'<div class="failure-row">' + tags_html + '</div>' if tags_html else ''}

  <!-- Question -->
  <div class="section-label">Question</div>
  <div class="question-text">{esc(r['question'])}</div>

  <!-- Answers side by side -->
  <div class="answer-grid">
    <div class="answer-box expected">
      <div class="answer-label">Expected Answer</div>
      <div class="answer-content">{esc(r['reference_answer'])}</div>
    </div>
    <div class="answer-box generated {'answer-correct' if correct else 'answer-wrong'}">
      <div class="answer-label">Generated Answer</div>
      <div class="answer-content">{esc(r['generated_answer']) or '<em>empty</em>'}</div>
    </div>
  </div>

  <!-- Metrics -->
  <div class="metrics-row">{metrics_html}</div>

  <!-- Collapsible evidence + chunks -->
  <details class="evidence-details">
    <summary class="evidence-summary">
      Gold Evidence &amp; Retrieved Chunks
      <span class="ret-inline" style="color:{RET_COLORS[ret][0]}">{RET_COLORS[ret][1]}</span>
    </summary>
    <div class="evidence-body">
      <div class="section-label" style="margin-top:0">Gold Evidence Segment(s)</div>
      {gold_html}
      <div class="section-label">Top-5 Retrieved Chunks <small>(post-reranker order — as fed to generator)</small></div>
      {chunks_html}
    </div>
  </details>

</div>"""


# ─── HTML page ────────────────────────────────────────────────────────────────

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       background: #f1f5f9; color: #1e293b; }
.page-wrap { max-width: 1300px; margin: 0 auto; padding: 2rem 1.5rem; }

/* Dashboard */
.dashboard { background: #fff; border-radius: 12px; padding: 2rem;
             box-shadow: 0 1px 4px rgba(0,0,0,0.1); margin-bottom: 2rem; }
.dashboard h1 { font-size: 1.6rem; margin-bottom: 0.3rem; }
.dashboard .subtitle { color: #64748b; margin-bottom: 1.5rem; }
.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
             gap: 1rem; margin-bottom: 1.5rem; }
.stat-box { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;
            padding: 1rem; text-align: center; }
.stat-box .val { font-size: 2rem; font-weight: 700; }
.stat-box .lbl { font-size: 0.8rem; color: #64748b; margin-top: 0.2rem; }
.green { color: #15803d; } .red { color: #dc2626; } .blue { color: #2563eb; }
.orange { color: #b45309; } .purple { color: #7c3aed; }

/* 2x2 matrix */
.matrix-wrap { margin-bottom: 1.5rem; }
.matrix-title { font-weight: 600; margin-bottom: 0.6rem; }
.matrix { display: inline-grid; grid-template-columns: auto 1fr 1fr;
          border: 1px solid #cbd5e1; border-radius: 8px; overflow: hidden; }
.matrix-cell { padding: 0.6rem 1.2rem; border: 1px solid #e2e8f0; text-align:center; }
.matrix-cell.header { background:#f1f5f9; font-weight:600; font-size:0.85rem; }
.matrix-cell.row-header { background:#f1f5f9; font-weight:600; font-size:0.85rem;
                           writing-mode: horizontal-tb; }
.mc-good { background:#f0fdf4; color:#15803d; font-size:1.1rem; font-weight:700; }
.mc-bad  { background:#fef2f2; color:#dc2626; font-size:1.1rem; font-weight:700; }

/* Charts / breakdown tables */
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom:1.5rem; }
@media(max-width:700px){ .two-col { grid-template-columns:1fr; } }
.breakdown-table { width: 100%; border-collapse: collapse; font-size:0.88rem; }
.breakdown-table th { background:#f1f5f9; padding:0.5rem 0.8rem; text-align:left;
                       border-bottom:2px solid #cbd5e1; }
.breakdown-table td { padding:0.45rem 0.8rem; border-bottom:1px solid #e2e8f0; }
.breakdown-table tr:last-child td { border-bottom:none; }
.bar { height:10px; border-radius:5px; display:inline-block; min-width:4px; }
.section-h2 { font-size:1.1rem; font-weight:700; margin-bottom:0.8rem;
              border-bottom:2px solid #e2e8f0; padding-bottom:0.4rem; }

/* Filters */
.filter-bar { background:#fff; border-radius:12px; padding:1.2rem 1.5rem;
              box-shadow:0 1px 4px rgba(0,0,0,0.08); margin-bottom:1.5rem; }
.filter-bar h3 { font-size:0.95rem; font-weight:700; margin-bottom:0.8rem; color:#475569; }
.filter-row { display:flex; flex-wrap:wrap; gap:0.5rem; align-items:center; margin-bottom:0.5rem; }
.filter-btn { padding:0.3rem 0.8rem; border-radius:20px; border:1.5px solid #cbd5e1;
              background:#f8fafc; cursor:pointer; font-size:0.82rem; font-weight:500;
              transition:all 0.15s; }
.filter-btn:hover { border-color:#94a3b8; }
.filter-btn.active { background:#1d4ed8; color:#fff; border-color:#1d4ed8; }
.search-input { padding:0.4rem 0.8rem; border:1.5px solid #cbd5e1; border-radius:8px;
                font-size:0.88rem; width:280px; }
.count-display { font-size:0.85rem; color:#64748b; margin-left:auto; }

/* Sample cards */
.sample-card { background:#fff; border-radius:10px; border-left:5px solid #e2e8f0;
               box-shadow:0 1px 3px rgba(0,0,0,0.08); margin-bottom:1.2rem;
               padding:1.2rem 1.5rem; }
.sample-card.hidden { display:none; }
.card-header { display:flex; align-items:center; flex-wrap:wrap; gap:0.5rem;
               margin-bottom:0.8rem; }
.card-num { font-size:0.85rem; font-weight:700; color:#94a3b8; min-width:2rem; }
.fid { font-size:0.75rem; color:#94a3b8; margin-left:auto; }
.doc-name { font-size:0.78rem; color:#64748b; background:#f1f5f9; padding:0.15rem 0.5rem;
             border-radius:4px; }
.failure-row { display:flex; flex-wrap:wrap; gap:0.4rem; margin-bottom:0.8rem; }

/* Badges */
.badge { display:inline-block; padding:0.2rem 0.6rem; border-radius:12px;
         font-size:0.75rem; font-weight:700; cursor:help; }
.correct-badge { display:inline-block; padding:0.2rem 0.7rem; border-radius:12px;
                 font-size:0.8rem; font-weight:700; }
.correct-badge.correct { background:#dcfce7; color:#15803d; }
.correct-badge.wrong   { background:#fee2e2; color:#dc2626; }
.qt-badge { display:inline-block; padding:0.15rem 0.6rem; border-radius:12px;
            font-size:0.75rem; font-weight:600; border:1.5px solid; background:#fff; }
.ret-badge { display:inline-block; padding:0.15rem 0.6rem; border-radius:12px;
             font-size:0.75rem; font-weight:700; color:#fff; }

/* Content */
.section-label { font-size:0.72rem; font-weight:700; letter-spacing:0.08em;
                  text-transform:uppercase; color:#94a3b8; margin:0.8rem 0 0.3rem; }
.question-text { font-size:0.95rem; line-height:1.6; color:#1e293b;
                 background:#f8fafc; padding:0.7rem 1rem; border-radius:6px;
                 border-left:3px solid #94a3b8; }
.answer-grid { display:grid; grid-template-columns:1fr 2fr; gap:0.8rem; margin-top:0.8rem; }
@media(max-width:700px){ .answer-grid { grid-template-columns:1fr; } }
.answer-box { border-radius:8px; padding:0.8rem 1rem; border:1.5px solid #e2e8f0; }
.answer-box.expected { background:#eff6ff; border-color:#bfdbfe; }
.answer-box.answer-correct { background:#f0fdf4; border-color:#86efac; }
.answer-box.answer-wrong   { background:#fef2f2; border-color:#fecaca; }
.answer-label { font-size:0.72rem; font-weight:700; text-transform:uppercase;
                letter-spacing:0.07em; color:#64748b; margin-bottom:0.3rem; }
.answer-content { font-size:0.92rem; line-height:1.6; white-space:pre-wrap; word-wrap:break-word; }

/* Metrics */
.metrics-row { display:flex; flex-wrap:wrap; gap:0.4rem; margin-top:0.8rem; }
.metric-pill { display:inline-block; padding:0.2rem 0.6rem; border-radius:6px;
               font-size:0.8rem; }
.metric-pill.good    { background:#dcfce7; color:#15803d; }
.metric-pill.ok      { background:#fef9c3; color:#854d0e; }
.metric-pill.bad     { background:#fee2e2; color:#dc2626; }
.metric-pill.neutral { background:#f1f5f9; color:#475569; }

/* Evidence */
.evidence-details { margin-top:1rem; }
.evidence-summary { cursor:pointer; font-size:0.88rem; font-weight:600; color:#475569;
                    padding:0.5rem 0; list-style:none; display:flex; align-items:center; gap:0.5rem; }
.evidence-summary::-webkit-details-marker { display:none; }
.evidence-summary::before { content:"▶"; font-size:0.7rem; transition:transform 0.15s; }
details[open] .evidence-summary::before { transform:rotate(90deg); }
.ret-inline { font-size:0.8rem; font-weight:700; }
.evidence-body { padding:0.8rem 0 0; }

.chunk-card { border:1.5px solid #e5e7eb; border-radius:8px; margin-bottom:0.7rem; overflow:hidden; }
.gold-card  { border:1.5px solid #9ca3af; border-radius:8px; margin-bottom:0.7rem; overflow:hidden; }
.chunk-header, .gold-header { display:flex; align-items:center; flex-wrap:wrap; gap:0.5rem;
                               padding:0.5rem 0.8rem; background:rgba(0,0,0,0.03);
                               border-bottom:1px solid rgba(0,0,0,0.07); font-size:0.82rem; }
.chunk-rank      { font-weight:700; color:#1e293b; }
.chunk-meta      { color:#64748b; }
.chunk-scores    { color:#64748b; font-size:0.78rem; margin-left:auto; }
.chunk-orig-rank { color:#94a3b8; font-size:0.72rem; font-style:italic; }
.gold-marker  { background:#15803d; color:#fff; font-size:0.72rem; font-weight:700;
                padding:0.1rem 0.5rem; border-radius:10px; }
.chunk-text   { font-size:0.82rem; line-height:1.6; padding:0.8rem 1rem;
                white-space:pre-wrap; word-wrap:break-word; font-family:inherit;
                color:#374151; overflow-x:auto; }
.no-data { font-size:0.85rem; color:#94a3b8; font-style:italic; padding:0.4rem 0; }
"""

JS = """
const cards = Array.from(document.querySelectorAll('.sample-card'));
const countEl = document.getElementById('count-display');
let activeFilters = { correct: 'all', qt: 'all', ret: 'all', mode: 'all' };
let searchText = '';

function applyFilters() {
  let shown = 0;
  cards.forEach(card => {
    const ok =
      (activeFilters.correct === 'all' || card.dataset.correct === activeFilters.correct ||
        (activeFilters.correct === 'refusal' && card.dataset.refusal === 'refusal')) &&
      (activeFilters.qt === 'all' || card.dataset.qt === activeFilters.qt) &&
      (activeFilters.ret === 'all' || card.dataset.ret === activeFilters.ret) &&
      (activeFilters.mode === 'all' || card.dataset.modes.includes(activeFilters.mode)) &&
      (searchText === '' || card.dataset.text.includes(searchText));
    card.classList.toggle('hidden', !ok);
    if (ok) shown++;
  });
  countEl.textContent = shown + ' / ' + cards.length + ' samples shown';
}

function setFilter(group, value, btn) {
  activeFilters[group] = value;
  document.querySelectorAll('[data-filter-group="' + group + '"]').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  applyFilters();
}

document.getElementById('search').addEventListener('input', e => {
  searchText = e.target.value.toLowerCase();
  applyFilters();
});

applyFilters();
"""


def build_html(results: list[dict], summary: dict, model_label: str) -> str:
    n = summary["n"]

    # ── Dashboard ──────────────────────────────────────────────────────────
    stat_grid = f"""
<div class="stat-grid">
  <div class="stat-box"><div class="val green">{summary['n_correct']}</div><div class="lbl">Correct ({summary['acc_pct']}%)</div></div>
  <div class="stat-box"><div class="val red">{summary['n_wrong']}</div><div class="lbl">Wrong ({round(100-summary['acc_pct'],1)}%)</div></div>
  <div class="stat-box"><div class="val blue">{summary['ret_full']}</div><div class="lbl">Full Ret. Hit ({summary['ret_full_pct']}%)</div></div>
  <div class="stat-box"><div class="val orange">{summary['ret_partial']}</div><div class="lbl">Partial Hit ({summary['ret_partial_pct']}%)</div></div>
  <div class="stat-box"><div class="val red">{summary['ret_miss']}</div><div class="lbl">Ret. Miss ({summary['ret_miss_pct']}%)</div></div>
  <div class="stat-box"><div class="val blue">{summary['avg_rougeL']:.3f}</div><div class="lbl">Avg ROUGE-L</div></div>
  <div class="stat-box"><div class="val purple">{summary['avg_squad_f1']:.3f}</div><div class="lbl">Avg Squad-F1</div></div>
  <div class="stat-box"><div class="val orange">{summary['avg_numeric_match']:.3f}</div><div class="lbl">Avg Numeric Match</div></div>
</div>"""

    # 2x2 matrix
    hc = summary["hit_correct"]
    hw = summary["hit_wrong"]
    mc = summary["miss_correct"]
    mw = summary["miss_wrong"]
    n_hit  = hc + hw
    n_miss = mc + mw
    matrix_html = f"""
<div class="matrix-wrap">
  <div class="matrix-title">Retrieval × Correctness Matrix</div>
  <div class="matrix">
    <div class="matrix-cell header"></div>
    <div class="matrix-cell header">✓ Correct</div>
    <div class="matrix-cell header">✗ Wrong</div>
    <div class="matrix-cell row-header">Ret. Hit (top-5)</div>
    <div class="matrix-cell mc-good">{hc}<br><small>({round(100*hc/n_hit)}% of hit)</small></div>
    <div class="matrix-cell mc-bad">{hw}<br><small>({round(100*hw/n_hit)}% of hit)</small></div>
    <div class="matrix-cell row-header">Ret. Miss</div>
    <div class="matrix-cell mc-good">{mc}<br><small>({round(100*mc/n_miss) if n_miss else 0}% of miss)</small></div>
    <div class="matrix-cell mc-bad">{mw}<br><small>({round(100*mw/n_miss) if n_miss else 0}% of miss)</small></div>
  </div>
  <p style="font-size:0.8rem;color:#64748b;margin-top:0.5rem">
    P(correct | retrieval hit) = <b>{summary['p_correct_given_hit']}%</b> &nbsp;|&nbsp;
    P(correct | retrieval miss) = <b>{summary['p_correct_given_miss']}%</b> (lucky guesses)
  </p>
</div>"""

    # Failure modes table
    max_mode = max(summary["mode_counts"].values()) if summary["mode_counts"] else 1
    mode_rows = ""
    for mode, cnt in sorted(summary["mode_counts"].items(), key=lambda x: -x[1]):
        bar_w = round(120 * cnt / max_mode)
        mode_rows += f"""<tr>
          <td>{tag_badge(mode)}</td>
          <td style="font-size:0.82rem;color:#475569;max-width:260px">{esc(FAILURE_MODE_EXPLANATIONS.get(mode,''))}</td>
          <td><span class="bar" style="width:{bar_w}px;background:#dc2626"></span> {cnt}</td>
        </tr>"""
    mode_table = f"""
<div class="section-h2">Failure Mode Breakdown (wrong predictions only)</div>
<table class="breakdown-table">
  <thead><tr><th>Mode</th><th>Description</th><th>Count</th></tr></thead>
  <tbody>{mode_rows}</tbody>
</table>"""

    # Per-question-type breakdown
    qt_rows = ""
    for qt, v in summary["by_qt"].items():
        n_qt = v["n"]
        acc = round(100 * v["correct"] / n_qt) if n_qt else 0
        rh  = v["ret_full"] + v["ret_partial"]
        qt_rows += f"""<tr>
          <td><span style="font-weight:600;color:{QT_COLORS.get(qt,'#374151')}">{qt}</span></td>
          <td>{n_qt}</td>
          <td>{v['correct']}/{n_qt} ({acc}%)</td>
          <td>{v['ret_full']}/{n_qt} ({round(100*v['ret_full']/n_qt)}%)</td>
          <td>{v['ret_partial']}/{n_qt} ({round(100*v['ret_partial']/n_qt)}%)</td>
          <td>{v['ret_miss']}/{n_qt} ({round(100*v['ret_miss']/n_qt)}%)</td>
        </tr>"""
    qt_table = f"""
<div class="section-h2">Breakdown by Question Type</div>
<table class="breakdown-table">
  <thead><tr><th>Type</th><th>N</th><th>Correct</th><th>Full Ret.</th><th>Partial</th><th>Miss</th></tr></thead>
  <tbody>{qt_rows}</tbody>
</table>"""

    dashboard = f"""
<div class="dashboard">
  <h1>DENSE + MultiHyDE + ReRanker — Error Analysis</h1>
  <p class="subtitle">FinanceBench · {n} questions · model: {esc(model_label)}</p>
  {stat_grid}
  <div class="two-col">
    <div>{matrix_html}</div>
    <div>{mode_table}</div>
  </div>
  {qt_table}
</div>"""

    # ── Filter bar ────────────────────────────────────────────────────────
    def btn(label, group, value, extra_cls=""):
        return f'<button class="filter-btn{extra_cls}" data-filter-group="{group}" onclick="setFilter(\'{group}\',\'{value}\',this)">{label}</button>'

    filter_bar = f"""
<div class="filter-bar">
  <h3>Filter &amp; Search</h3>
  <div class="filter-row">
    <b style="font-size:0.8rem;color:#64748b;min-width:80px">Correctness</b>
    {btn("All","correct","all"," active")}
    {btn("✓ Correct","correct","correct")}
    {btn("✗ Wrong","correct","wrong")}
    {btn("Refusals","correct","refusal")}
  </div>
  <div class="filter-row">
    <b style="font-size:0.8rem;color:#64748b;min-width:80px">Question Type</b>
    {btn("All","qt","all"," active")}
    {btn("Metrics","qt","metrics-generated")}
    {btn("Domain","qt","domain-relevant")}
    {btn("Novel","qt","novel-generated")}
  </div>
  <div class="filter-row">
    <b style="font-size:0.8rem;color:#64748b;min-width:80px">Retrieval</b>
    {btn("All","ret","all"," active")}
    {btn("Full Hit","ret","full")}
    {btn("Partial","ret","partial")}
    {btn("Miss","ret","miss")}
  </div>
  <div class="filter-row">
    <b style="font-size:0.8rem;color:#64748b;min-width:80px">Failure Mode</b>
    {btn("All","mode","all"," active")}
    {"".join(btn(m, "mode", m) for m in sorted(summary["mode_counts"].keys()))}
  </div>
  <div class="filter-row" style="margin-top:0.3rem">
    <input id="search" class="search-input" type="text" placeholder="Search question / answer text…">
    <span id="count-display" class="count-display"></span>
  </div>
</div>"""

    # ── Cards ─────────────────────────────────────────────────────────────
    # Sort: wrong first (most informative), then by question type
    sorted_results = sorted(results, key=lambda r: (int(r["correct"]), r["question_type"]))
    cards_html = "\n".join(sample_card(r, i + 1) for i, r in enumerate(sorted_results))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Top Baseline Error Analysis — FinanceBench</title>
<style>{CSS}</style>
</head>
<body>
<div class="page-wrap">
{dashboard}
{filter_bar}
<div id="cards">
{cards_html}
</div>
</div>
<script>{JS}</script>
</body>
</html>"""


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Top-baseline error analysis HTML report")
    parser.add_argument("--pred", default="baselines/results/predictions/multi_hyde_reranker_generated.json",
                        help="Generated predictions JSON")
    parser.add_argument("--out", default="outputs/top_baseline_analysis/report.html",
                        help="Output HTML path")
    parser.add_argument("--model-label", default="Qwen2.5-7B-Instruct (4-bit)",
                        help="Model name shown in the report header")
    args = parser.parse_args()

    pred_path = Path(args.pred)
    out_path  = Path(args.out)

    if not pred_path.exists():
        print(f"ERROR: {pred_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Loading predictions from {pred_path} …")
    samples = json.load(open(pred_path))
    print(f"  {len(samples)} samples loaded.")

    print("Computing metrics and classifying failures …")
    results = analyse_samples(samples)
    summary = compute_summary(results)

    n  = summary["n"]
    print(f"\n{'='*55}")
    print(f"  DENSE+MultiHyDE+ReRanker  ({args.model_label})")
    print(f"{'='*55}")
    print(f"  Accuracy        : {summary['n_correct']}/{n}  ({summary['acc_pct']}%)")
    print(f"  Ret full hit    : {summary['ret_full']}/{n}  ({summary['ret_full_pct']}%)")
    print(f"  Ret partial     : {summary['ret_partial']}/{n}  ({summary['ret_partial_pct']}%)")
    print(f"  Ret miss        : {summary['ret_miss']}/{n}  ({summary['ret_miss_pct']}%)")
    print(f"  Avg ROUGE-L     : {summary['avg_rougeL']:.4f}")
    print(f"  Avg Squad-F1    : {summary['avg_squad_f1']:.4f}")
    print(f"  Avg Num. Match  : {summary['avg_numeric_match']:.4f}")
    print(f"  P(✓|ret hit)    : {summary['p_correct_given_hit']}%")
    print(f"  P(✓|ret miss)   : {summary['p_correct_given_miss']}% (lucky guesses)")
    print(f"\n  Failure modes:")
    for mode, cnt in sorted(summary["mode_counts"].items(), key=lambda x: -x[1]):
        print(f"    {mode:<22} {cnt}")
    print(f"{'='*55}")

    print(f"\nBuilding HTML report …")
    html = build_html(results, summary, args.model_label)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    size_kb = out_path.stat().st_size // 1024
    print(f"Report written → {out_path}  ({size_kb} KB)")


if __name__ == "__main__":
    main()
