#!/usr/bin/env python3
"""
Lexical / Vocabulary Mismatch Analysis — FinanceBench + FinQA (Benchmark Extension)
=====================================================================================

Quantitative metrics per query (OOV Rate, Jaccard, Term Coverage, JSD) computed
separately for:
  - FinanceBench subset  (150 questions, original FB data)
  - FinQA subset         (530 questions, original FinQA questions + LoFin gold-page annotations)
  - Combined             (680 questions, global benchmark extension index)

Key comparisons:
  1. Overall distributions per dataset
  2. Dataset vs dataset side-by-side (FB vs FinQA)
  3. Per question_type (metrics-generated / domain-relevant / novel-generated / finqa)
  4. Per doc_type (10k / 10q / 8k / earnings / unknown)
  5. Corpus-level term-frequency gap — query vs document, per dataset
  6. Written report with all stats
"""

import json
import math
import re
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.spatial.distance import jensenshannon
from scipy.stats import kruskal, mannwhitneyu

# ── paths ──────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE / "data"
OUT_DIR  = Path(__file__).resolve().parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

FB_QUERIES_FILE  = DATA_DIR / "financebench_open_source.jsonl"
FB_DOCINFO_FILE  = DATA_DIR / "financebench_document_information.jsonl"
FINQA_FILE       = DATA_DIR / "finqa_test_gold_pages.jsonl"

# ── style ──────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.15)

QTYPE_PALETTE = {
    "metrics-generated": "#4C72B0",
    "domain-relevant":   "#DD8452",
    "novel-generated":   "#55A868",
    "finqa":             "#C44E52",
}
DATASET_PALETTE = {
    "financebench": "#4C72B0",
    "finqa":        "#C44E52",
    "combined":     "#55A868",
}
DOC_PALETTE = sns.color_palette("Set2", 6)

ALL_METRICS      = ["oov_rate", "jaccard", "coverage", "jsd"]
MISMATCH_METRICS = ["oov_rate", "jsd"]
OVERLAP_METRICS  = ["jaccard", "coverage"]
METRIC_LABELS    = {
    "oov_rate": "OOV Rate (↑ = more mismatch)",
    "jaccard":  "Jaccard Similarity (↑ = more overlap)",
    "coverage": "Term Coverage (↑ = more overlap)",
    "jsd":      "Jensen-Shannon Divergence (↑ = more mismatch)",
}

# ── stop-words ─────────────────────────────────────────────────────────────────
STOPWORDS = {
    "a","an","the","and","or","of","in","to","for","with","on","at","by","from",
    "is","are","was","were","be","been","being","have","has","had","do","does",
    "did","will","would","could","should","may","might","shall","can","not",
    "this","that","these","those","it","its","we","our","you","your","they",
    "their","i","my","he","his","she","her","as","if","but","so","up","out",
    "about","what","which","who","when","where","how","all","any","both","each",
    "few","more","most","other","some","such","no","nor","than","too","very",
    "s","t","just","don","into","then","there","also","only","over","after",
    "well","same","per","us","use","used","based","please","give","answer",
    "following","information","shown","using","assume","relying","details",
    "primarily","state","explain","response","question",
}

_NUM_RE = re.compile(r"^\d[\d,.\-/:%$]*$")


def tokenise(text: str, remove_numbers: bool = True) -> list:
    tokens = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9',.\-/$%]*", text.lower())
    tokens = [t.strip("',.-/$%") for t in tokens]
    tokens = [t for t in tokens if len(t) > 1 and t not in STOPWORDS]
    if remove_numbers:
        tokens = [t for t in tokens if not _NUM_RE.match(t)]
    return tokens


def jsd(p_counter: Counter, q_counter: Counter) -> float:
    vocab = list(set(p_counter) | set(q_counter))
    p = np.array([p_counter.get(w, 0) for w in vocab], dtype=float)
    q = np.array([q_counter.get(w, 0) for w in vocab], dtype=float)
    p /= p.sum() if p.sum() > 0 else 1
    q /= q.sum() if q.sum() > 0 else 1
    return float(jensenshannon(p, q, base=2))


def _infer_doc_type(doc_name: str) -> str:
    d = doc_name.lower()
    if "10q" in d or "10-q" in d:
        return "10q"
    if "8k" in d or "8-k" in d:
        return "8k"
    if "10k" in d or "10-k" in d:
        return "10k"
    if "earnings" in d or "earn" in d:
        return "earnings"
    return "unknown"


def savefig(name):
    path = OUT_DIR / name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {path.relative_to(BASE)}")


# ── data loading ───────────────────────────────────────────────────────────────

def load_financebench():
    doc_info = {}
    if FB_DOCINFO_FILE.exists():
        with open(FB_DOCINFO_FILE) as f:
            for line in f:
                d = json.loads(line.strip())
                if d:
                    doc_info[d["doc_name"]] = d

    rows = []
    with open(FB_QUERIES_FILE) as f:
        for line in f:
            raw = json.loads(line.strip())
            if not raw:
                continue
            doc_name = raw.get("doc_name", "")
            dm = doc_info.get(doc_name, {})

            # collect evidence texts
            ev_texts = []
            for ev in raw.get("evidence", []):
                t = ev.get("evidence_text_full_page") or ev.get("evidence_text", "")
                if t:
                    ev_texts.append(t)

            rows.append({
                "id":            raw.get("financebench_id", ""),
                "question":      raw.get("question", ""),
                "question_type": raw.get("question_type", "unknown"),
                "doc_name":      doc_name,
                "doc_type":      dm.get("doc_type", _infer_doc_type(doc_name)),
                "company":       raw.get("company", ""),
                "dataset":       "financebench",
                "ev_texts":      ev_texts,
            })
    print(f"  FinanceBench: {len(rows)} questions")
    return rows, doc_info


def load_finqa():
    rows = []
    with open(FINQA_FILE) as f:
        for line in f:
            raw = json.loads(line.strip())
            if not raw:
                continue

            evs = raw.get("evidences_updated", raw.get("evidences", []))
            ev_texts = []
            for ev in evs:
                t = ev.get("evidence_text", ev.get("pre_text", ""))
                if t:
                    ev_texts.append(t)

            if not ev_texts:
                continue

            doc_name = evs[0].get("doc_name", "") if evs else ""
            rows.append({
                "id":            raw.get("qid", ""),
                "question":      raw.get("question", ""),
                "question_type": "finqa",
                "doc_name":      doc_name,
                "doc_type":      _infer_doc_type(doc_name),
                "company":       doc_name.split("/")[0] if doc_name else "",
                "dataset":       "finqa",
                "ev_texts":      ev_texts,
            })
    print(f"  FinQA: {len(rows)} questions")
    return rows


# ── per-query metrics ─────────────────────────────────────────────────────────

def compute_metrics(raw_rows: list) -> pd.DataFrame:
    records = []
    for r in raw_rows:
        qtoks = tokenise(r["question"])
        if not qtoks:
            continue
        q_set = set(qtoks)
        q_cnt = Counter(qtoks)

        doc_full = " ".join(r["ev_texts"])
        dtoks = tokenise(doc_full)
        if not dtoks:
            continue
        d_set = set(dtoks)
        d_cnt = Counter(dtoks)

        intersection = q_set & d_set
        union        = q_set | d_set

        records.append({
            "id":            r["id"],
            "question":      r["question"],
            "question_type": r["question_type"],
            "doc_name":      r["doc_name"],
            "doc_type":      r["doc_type"],
            "company":       r["company"],
            "dataset":       r["dataset"],
            "jaccard":       len(intersection) / len(union) if union else 0.0,
            "coverage":      len(intersection) / len(q_set) if q_set else 0.0,
            "oov_rate":      1.0 - (len(intersection) / len(q_set) if q_set else 0.0),
            "jsd":           jsd(q_cnt, d_cnt),
            "n_query_tokens":  len(qtoks),
            "n_query_unique":  len(q_set),
            "n_doc_tokens":    len(dtoks),
            "n_doc_unique":    len(d_set),
            "n_overlap":       len(intersection),
            "exclusive_q":     sorted(q_set - d_set),
        })
    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Overall distributions (combined)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_overall(df):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Overall Lexical Mismatch Distribution (FinanceBench + FinQA, n=680)", fontsize=14, y=1.01)

    for ax, metric in zip(axes.flat, ALL_METRICS):
        sns.histplot(df[metric], bins=30, kde=True, ax=ax, color="#55A868", edgecolor="white")
        mu, med = df[metric].mean(), df[metric].median()
        ax.axvline(mu,  color="red",    lw=1.5, ls="--", label=f"mean={mu:.3f}")
        ax.axvline(med, color="orange", lw=1.5, ls=":",  label=f"median={med:.3f}")
        ax.set_xlabel(METRIC_LABELS[metric])
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)

    plt.tight_layout()
    savefig("01_overall_distributions.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — FB vs FinQA side-by-side (the key comparison)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_dataset_comparison(df):
    datasets = ["financebench", "finqa"]
    labels   = {"financebench": "FinanceBench (n=150)", "finqa": "FinQA (n=530)"}

    # --- violin per metric ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    fig.suptitle("Lexical Mismatch: FinanceBench vs FinQA", fontsize=14)
    for ax, metric in zip(axes, ALL_METRICS):
        sns.violinplot(
            data=df, x="dataset", y=metric,
            palette=DATASET_PALETTE, inner="box", ax=ax, order=datasets,
        )
        ax.set_xticklabels([labels[d] for d in datasets], rotation=10, ha="right")
        ax.set_xlabel("")
        ax.set_ylabel(metric)
        ax.set_title(METRIC_LABELS[metric].split("(")[0].strip())

        # Mann-Whitney U test
        a = df.loc[df["dataset"] == "financebench", metric].values
        b = df.loc[df["dataset"] == "finqa",        metric].values
        stat, p = mannwhitneyu(a, b, alternative="two-sided")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.set_title(f"{METRIC_LABELS[metric].split('(')[0].strip()}\nMW-U p={p:.3f} {sig}", fontsize=9)

    plt.tight_layout()
    savefig("02_dataset_comparison_violin.png")

    # --- bar chart (mean ± std) ---
    stats = df.groupby("dataset")[ALL_METRICS].agg(["mean", "std"]).round(4)
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle("Mean Lexical Metrics: FinanceBench vs FinQA (± 1 SD)", fontsize=13)
    for ax, metric in zip(axes, ALL_METRICS):
        means = stats[metric]["mean"].reindex(datasets)
        stds  = stats[metric]["std"].reindex(datasets)
        colors = [DATASET_PALETTE[d] for d in datasets]
        ax.bar(
            [labels[d] for d in datasets], means, yerr=stds,
            color=colors, capsize=6, edgecolor="white", linewidth=0.8,
        )
        ax.set_title(METRIC_LABELS[metric].split("(")[0].strip())
        ax.set_xlabel("")
        ax.set_xticklabels([labels[d] for d in datasets], rotation=10, ha="right")
        ax.set_ylim(0, min(1.0, means.max() + stds.max() + 0.1))
    plt.tight_layout()
    savefig("03_dataset_comparison_bar.png")

    # print MW-U summary
    print("\n[Dataset comparison — Mann-Whitney U tests]")
    fb  = df[df["dataset"] == "financebench"]
    fq  = df[df["dataset"] == "finqa"]
    for metric in ALL_METRICS:
        stat, p = mannwhitneyu(fb[metric].values, fq[metric].values, alternative="two-sided")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {metric:12s}: MW-U={stat:.0f}  p={p:.4f}  {sig}  "
              f"FB_mean={fb[metric].mean():.3f}  FQ_mean={fq[metric].mean():.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Per question type (all 4 types including "finqa")
# ═══════════════════════════════════════════════════════════════════════════════
def plot_per_question_type(df):
    qtypes = sorted(df["question_type"].unique())

    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    fig.suptitle("Lexical Mismatch by Question Type (All 4 Types)", fontsize=14)
    for ax, metric in zip(axes, ALL_METRICS):
        sns.violinplot(
            data=df, x="question_type", y=metric,
            palette=QTYPE_PALETTE, inner="box", ax=ax, order=qtypes,
        )
        ax.set_xlabel("")
        ax.set_ylabel(metric)
        ax.set_title(METRIC_LABELS[metric].split("(")[0].strip())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
    plt.tight_layout()
    savefig("04_question_type_violin.png")

    stats = df.groupby("question_type")[ALL_METRICS].agg(["mean", "std"]).round(4)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Mean Lexical Metrics by Question Type (± 1 SD)", fontsize=13)
    for ax, metric in zip(axes, ALL_METRICS):
        means = stats[metric]["mean"]
        stds  = stats[metric]["std"]
        colors = [QTYPE_PALETTE.get(qt, "#888") for qt in means.index]
        ax.bar(means.index, means, yerr=stds, color=colors,
               capsize=5, edgecolor="white", linewidth=0.8)
        ax.set_title(METRIC_LABELS[metric].split("(")[0].strip())
        ax.set_xlabel("")
        ax.set_xticklabels(means.index, rotation=15, ha="right")
        ax.set_ylim(0, min(1.0, means.max() + stds.max() + 0.1))
    plt.tight_layout()
    savefig("05_question_type_bar.png")

    print("\n[Question Type — Kruskal-Wallis tests]")
    for metric in ALL_METRICS:
        groups = [df.loc[df["question_type"] == qt, metric].values for qt in qtypes]
        stat, p = kruskal(*groups)
        print(f"  {metric:12s}: H={stat:.2f}  p={p:.4f}  {'***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'}")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 4 — Per doc type
# ═══════════════════════════════════════════════════════════════════════════════
def plot_per_doc_type(df):
    dtypes = sorted(df["doc_type"].unique())
    pal = dict(zip(dtypes, DOC_PALETTE))

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.suptitle("Lexical Mismatch by Document Type", fontsize=14)
    for ax, metric in zip(axes, ALL_METRICS):
        sns.violinplot(
            data=df, x="doc_type", y=metric,
            palette=pal, inner="box", ax=ax, order=dtypes,
        )
        ax.set_xlabel("")
        ax.set_ylabel(metric)
        ax.set_title(METRIC_LABELS[metric].split("(")[0].strip())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
    plt.tight_layout()
    savefig("06_doc_type_violin.png")

    stats = df.groupby("doc_type")[ALL_METRICS].agg(["mean", "std"]).round(4)
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle("Mean Lexical Metrics by Document Type (± 1 SD)", fontsize=13)
    for ax, metric in zip(axes, ALL_METRICS):
        means = stats[metric]["mean"]
        stds  = stats[metric]["std"]
        colors = [pal.get(dt, "#888") for dt in means.index]
        ax.bar(means.index, means, yerr=stds, color=colors,
               capsize=5, edgecolor="white", linewidth=0.8)
        ax.set_title(METRIC_LABELS[metric].split("(")[0].strip())
        ax.set_xlabel("")
        ax.set_xticklabels(means.index, rotation=15, ha="right")
        ax.set_ylim(0, min(1.0, means.max() + stds.max() + 0.1))
    plt.tight_layout()
    savefig("07_doc_type_bar.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 5 — Cross heatmaps (question_type × doc_type) — full 680q
# ═══════════════════════════════════════════════════════════════════════════════
def plot_cross_heatmaps(df):
    qtypes = sorted(df["question_type"].unique())
    dtypes = sorted(df["doc_type"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Lexical Mismatch: Question Type × Document Type (n=680)", fontsize=14, y=1.01)

    for ax, metric in zip(axes.flat, ALL_METRICS):
        pivot = df.groupby(["question_type", "doc_type"])[metric].mean().unstack(fill_value=np.nan)
        pivot = pivot.reindex(index=qtypes, columns=dtypes)
        cmap  = "Reds" if metric in MISMATCH_METRICS else "Blues"
        vals  = pivot.values[~np.isnan(pivot.values)]
        vmin  = vals.min() * 0.95 if len(vals) else 0
        vmax  = vals.max() * 1.02 if len(vals) else 1
        sns.heatmap(pivot, ax=ax, annot=True, fmt=".3f", cmap=cmap,
                    vmin=vmin, vmax=vmax, linewidths=0.5, linecolor="white",
                    cbar_kws={"shrink": 0.7}, annot_kws={"size": 9})
        direction = "↑ worse" if metric in MISMATCH_METRICS else "↑ better"
        ax.set_title(f"{metric.upper()} ({direction})", fontsize=11)
        ax.set_xlabel("Document Type")
        ax.set_ylabel("Question Type")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    savefig("08_cross_heatmap.png")

    # count heatmap
    fig, ax = plt.subplots(figsize=(10, 5))
    counts = df.groupby(["question_type", "doc_type"]).size().unstack(fill_value=0)
    counts = counts.reindex(index=qtypes, columns=dtypes, fill_value=0)
    sns.heatmap(counts, ax=ax, annot=True, fmt="d", cmap="YlOrBr",
                linewidths=0.5, linecolor="white", cbar_kws={"shrink": 0.7})
    ax.set_title("Sample Count: Question Type × Document Type")
    ax.set_xlabel("Document Type")
    ax.set_ylabel("Question Type")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    savefig("09_cross_count_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 6 — Corpus-level term frequency gap (per dataset)
# ═══════════════════════════════════════════════════════════════════════════════

def build_corpus_counters(raw_rows):
    query_cnt = Counter()
    doc_cnt   = Counter()
    ds_q      = defaultdict(Counter)   # dataset → query counter
    ds_d      = defaultdict(Counter)   # dataset → doc counter
    qt_q      = defaultdict(Counter)   # question_type → query counter

    for r in raw_rows:
        qtoks = tokenise(r["question"])
        query_cnt.update(qtoks)
        ds_q[r["dataset"]].update(qtoks)
        qt_q[r["question_type"]].update(qtoks)

        dtoks = tokenise(" ".join(r["ev_texts"]))
        doc_cnt.update(dtoks)
        ds_d[r["dataset"]].update(dtoks)

    return query_cnt, doc_cnt, ds_q, ds_d, qt_q


def _term_scores(q_cnt, d_cnt):
    total_q = sum(q_cnt.values())
    total_d = sum(d_cnt.values())
    scores  = {}
    vocab   = set(q_cnt) | set(d_cnt)
    for term in vocab:
        pq  = (q_cnt.get(term, 0) + 1) / (total_q + len(vocab))
        pd_ = (d_cnt.get(term, 0) + 1) / (total_d + len(vocab))
        scores[term] = math.log2(pq / pd_)
    return scores


def plot_term_gap_global(query_cnt, doc_cnt, top_n=25):
    scores = _term_scores(query_cnt, doc_cnt)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1])
    doc_dominant   = sorted_scores[:top_n]
    query_dominant = sorted_scores[-top_n:][::-1]

    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    fig.suptitle("Term Frequency Gap: Queries vs Documents (Combined 680q)\n(log₂ ratio, smoothed)", fontsize=13)

    terms_q, scores_q = zip(*query_dominant)
    axes[0].barh(list(reversed(terms_q)), list(reversed(scores_q)), color="#4C72B0", edgecolor="white")
    axes[0].set_title("Query-Dominant Terms\n(overrepresented in queries)", color="#4C72B0")
    axes[0].set_xlabel("log₂(P_query / P_doc)")
    axes[0].axvline(0, color="black", lw=0.8)

    terms_d, scores_d = zip(*doc_dominant)
    scores_d_abs = [-s for s in scores_d]
    axes[1].barh(list(reversed(terms_d)), list(reversed(scores_d_abs)), color="#DD8452", edgecolor="white")
    axes[1].set_title("Document-Dominant Terms\n(overrepresented in documents)", color="#DD8452")
    axes[1].set_xlabel("log₂(P_doc / P_query)")
    axes[1].axvline(0, color="black", lw=0.8)

    plt.tight_layout()
    savefig("10_term_gap_global.png")


def plot_term_gap_by_dataset(ds_q, ds_d, top_n=20):
    """Side-by-side term gap for FinanceBench vs FinQA."""
    datasets = ["financebench", "finqa"]
    titles   = {"financebench": "FinanceBench", "finqa": "FinQA"}

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("Term Frequency Gap by Dataset\n(query-dominant top, doc-dominant bottom)", fontsize=13)

    for col, ds in enumerate(datasets):
        scores = _term_scores(ds_q[ds], ds_d[ds])
        sorted_sc = sorted(scores.items(), key=lambda x: x[1])
        doc_dom   = sorted_sc[:top_n]
        q_dom     = sorted_sc[-top_n:][::-1]
        color     = DATASET_PALETTE[ds]

        # query dominant
        ax = axes[0][col]
        terms_q, scores_q = zip(*q_dom) if q_dom else ([], [])
        ax.barh(list(reversed(terms_q)), list(reversed(scores_q)), color=color, edgecolor="white")
        ax.set_title(f"{titles[ds]} — Query-Dominant Terms", color=color, fontsize=11)
        ax.set_xlabel("log₂(P_query / P_doc)")
        ax.axvline(0, color="black", lw=0.8)

        # doc dominant
        ax = axes[1][col]
        terms_d, scores_d = zip(*doc_dom) if doc_dom else ([], [])
        scores_d_abs = [-s for s in scores_d]
        ax.barh(list(reversed(terms_d)), list(reversed(scores_d_abs)), color="#DD8452", edgecolor="white")
        ax.set_title(f"{titles[ds]} — Document-Dominant Terms", fontsize=11)
        ax.set_xlabel("log₂(P_doc / P_query)")
        ax.axvline(0, color="black", lw=0.8)

    plt.tight_layout()
    savefig("11_term_gap_by_dataset.png")


def plot_query_exclusive_by_qtype(qt_q, doc_cnt, top_n=15):
    """For each question_type, show top query-exclusive terms vs all corpus doc text."""
    qtypes = sorted(qt_q.keys())
    total_d_n = sum(doc_cnt.values())

    fig, axes = plt.subplots(1, len(qtypes), figsize=(6 * len(qtypes), 8))
    if len(qtypes) == 1:
        axes = [axes]
    fig.suptitle("Query-Exclusive Terms by Question Type\n(overrepresented in queries vs all doc text)", fontsize=12)

    for ax, qt in zip(axes, qtypes):
        qc = qt_q[qt]
        total_q_n = sum(qc.values())
        scores = {}
        for term in qc:
            pq  = (qc[term] + 1) / (total_q_n + len(qc))
            pd_ = (doc_cnt.get(term, 0) + 1) / (total_d_n + len(doc_cnt))
            scores[term] = math.log2(pq / pd_)
        top = sorted(scores.items(), key=lambda x: -x[1])[:top_n]
        terms, sc = zip(*top) if top else ([], [])
        color = QTYPE_PALETTE.get(qt, "#888")
        ax.barh(list(reversed(terms)), list(reversed(sc)), color=color, edgecolor="white")
        ax.set_title(qt, color=color, fontsize=10)
        ax.set_xlabel("log₂(P_query / P_doc)")
        ax.axvline(0, color="black", lw=0.8)
        ax.invert_yaxis()

    plt.tight_layout()
    savefig("12_query_exclusive_by_question_type.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Extreme examples
# ═══════════════════════════════════════════════════════════════════════════════
def extreme_examples(df, n=5) -> str:
    lines = ["\n" + "═"*80, "QUALITATIVE: EXTREME MISMATCH EXAMPLES", "═"*80]
    for metric, label, ascending in [
        ("oov_rate", "Highest OOV Rate (most mismatch)", False),
        ("oov_rate", "Lowest OOV Rate (most overlap)",   True),
        ("jsd",      "Highest JSD (most divergence)",    False),
    ]:
        lines.append(f"\n── {label} ──")
        sub = df.sort_values(metric, ascending=ascending).head(n)
        for _, row in sub.iterrows():
            lines.append(f"  [{row['dataset']} | {row['question_type']} | {row['doc_type']} | {row['company']}]")
            lines.append(f"  Q: {row['question'][:120]}")
            lines.append(f"  oov={row['oov_rate']:.3f}  jaccard={row['jaccard']:.3f}  "
                         f"coverage={row['coverage']:.3f}  jsd={row['jsd']:.3f}")
            if row["exclusive_q"]:
                lines.append(f"  Query-exclusive tokens: {row['exclusive_q'][:8]}")
            lines.append("")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Written report
# ═══════════════════════════════════════════════════════════════════════════════
def write_report(df, extreme_text):
    lines = [
        "LEXICAL MISMATCH ANALYSIS REPORT — FinanceBench + FinQA (Benchmark Extension)",
        "="*75,
        f"Total queries analysed: {len(df)}",
        f"  FinanceBench : {(df['dataset']=='financebench').sum()}",
        f"  FinQA        : {(df['dataset']=='finqa').sum()}",
        "",
    ]

    lines.append("── COMBINED OVERALL ──")
    for m in ALL_METRICS:
        lines.append(f"  {m:12s}: mean={df[m].mean():.4f}  median={df[m].median():.4f}  "
                     f"std={df[m].std():.4f}  min={df[m].min():.4f}  max={df[m].max():.4f}")

    lines.append("\n── BY DATASET ──")
    for ds, gdf in df.groupby("dataset"):
        lines.append(f"  [{ds}]  n={len(gdf)}")
        for m in ALL_METRICS:
            lines.append(f"    {m:12s}: mean={gdf[m].mean():.4f}  "
                         f"median={gdf[m].median():.4f}  std={gdf[m].std():.4f}")

    lines.append("\n── DATASET COMPARISON — Mann-Whitney U ──")
    fb = df[df["dataset"] == "financebench"]
    fq = df[df["dataset"] == "finqa"]
    for m in ALL_METRICS:
        stat, p = mannwhitneyu(fb[m].values, fq[m].values, alternative="two-sided")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        lines.append(f"  {m:12s}: MW-U={stat:.0f}  p={p:.4f}  {sig}")

    for groupby, label in [("question_type", "QUESTION TYPE"), ("doc_type", "DOC TYPE")]:
        lines.append(f"\n── BY {label} ──")
        for g, gdf in df.groupby(groupby):
            lines.append(f"  [{g}]  n={len(gdf)}")
            for m in ALL_METRICS:
                lines.append(f"    {m:12s}: mean={gdf[m].mean():.4f}  "
                              f"median={gdf[m].median():.4f}  std={gdf[m].std():.4f}")

    lines.append("\n── BY QUESTION TYPE × DOC TYPE (OOV Rate) ──")
    pivot = df.groupby(["question_type", "doc_type"])["oov_rate"].agg(["mean", "count"])
    lines.append(pivot.to_string())

    lines.append(extreme_text)

    report_path = OUT_DIR / "lexical_mismatch_report.txt"
    report_path.write_text("\n".join(lines))
    print(f"  saved → {report_path.relative_to(BASE)}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("Loading data …")
    fb_rows, doc_info = load_financebench()
    fq_rows           = load_finqa()
    all_rows          = fb_rows + fq_rows
    print(f"  Combined: {len(all_rows)} questions")

    print("\nComputing per-query metrics …")
    df = compute_metrics(all_rows)
    print(f"  {len(df)} valid rows  "
          f"(FB={( df['dataset']=='financebench').sum()}  "
          f"FQ={(df['dataset']=='finqa').sum()})")
    df.to_csv(OUT_DIR / "metrics_per_query.csv", index=False)
    print(f"  saved → analysis/lexical_mismatch_benchmark_extension/metrics_per_query.csv")

    # per-dataset CSVs for easy downstream use
    df[df["dataset"] == "financebench"].to_csv(OUT_DIR / "metrics_financebench.csv", index=False)
    df[df["dataset"] == "finqa"].to_csv(OUT_DIR / "metrics_finqa.csv", index=False)

    print("\nBuilding corpus counters …")
    query_cnt, doc_cnt, ds_q, ds_d, qt_q = build_corpus_counters(all_rows)

    print("\nGenerating plots …")
    plot_overall(df)
    plot_dataset_comparison(df)
    plot_per_question_type(df)
    plot_per_doc_type(df)
    plot_cross_heatmaps(df)
    plot_term_gap_global(query_cnt, doc_cnt)
    plot_term_gap_by_dataset(ds_q, ds_d)
    plot_query_exclusive_by_qtype(qt_q, doc_cnt)

    print("\nWriting report …")
    extreme = extreme_examples(df)
    write_report(df, extreme)

    print("\nDone.")


if __name__ == "__main__":
    main()
