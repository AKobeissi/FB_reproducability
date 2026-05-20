#!/usr/bin/env python3
"""
Lexical / Vocabulary Mismatch Analysis — FinanceBench

Quantitative metrics per query:
  - OOV Rate        : fraction of content query tokens absent from evidence-page vocab
  - Jaccard         : |Q ∩ D| / |Q ∪ D|
  - Term Coverage   : |Q ∩ D| / |Q|  (recall-like)
  - JSD             : Jensen-Shannon divergence between query & doc unigram distributions

Levels of aggregation:
  1. Overall
  2. Per question_type
  3. Per doc_type
  4. question_type × doc_type (heatmap)

Qualitative:
  - Corpus-level term-frequency gap (query-exclusive vs doc-exclusive top terms)
  - Per-slice distinctive terms
  - Example queries at high / low mismatch extremes
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
from scipy.stats import kruskal

# ── paths ──────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE / "data"
OUT_DIR = Path(__file__).resolve().parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

QUERIES_FILE = DATA_DIR / "financebench_open_source.jsonl"
DOC_INFO_FILE = DATA_DIR / "financebench_document_information.jsonl"

# ── style ──────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.15)
PALETTE = {
    "metrics-generated": "#4C72B0",
    "domain-relevant":   "#DD8452",
    "novel-generated":   "#55A868",
}
DOC_PALETTE = sns.color_palette("Set2", 6)

# ── stop-words (compact English + finance-specific function words) ──────────────
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

# ── tokeniser ─────────────────────────────────────────────────────────────────
_NUM_RE = re.compile(r"^\d[\d,.\-/:%$]*$")

def tokenise(text: str, remove_numbers: bool = True) -> list[str]:
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


# ── data loading ───────────────────────────────────────────────────────────────
def load_data():
    queries = [json.loads(l) for l in open(QUERIES_FILE)]
    doc_info = {d["doc_name"]: d for d in (json.loads(l) for l in open(DOC_INFO_FILE))}
    return queries, doc_info


# ── per-query metrics ─────────────────────────────────────────────────────────
def compute_query_metrics(queries, doc_info):
    rows = []
    for q in queries:
        qtoks = tokenise(q["question"])
        if not qtoks:
            continue
        q_set   = set(qtoks)
        q_cnt   = Counter(qtoks)

        # aggregate ALL evidence pages for this query as document text
        doc_texts = []
        for ev in q.get("evidence", []):
            page_text = ev.get("evidence_text_full_page") or ev.get("evidence_text", "")
            doc_texts.append(page_text)
        doc_full = " ".join(doc_texts)
        dtoks = tokenise(doc_full)
        if not dtoks:
            continue
        d_set  = set(dtoks)
        d_cnt  = Counter(dtoks)

        # --- core metrics ---
        intersection = q_set & d_set
        union        = q_set | d_set

        jaccard    = len(intersection) / len(union) if union else 0.0
        coverage   = len(intersection) / len(q_set) if q_set else 0.0
        oov_rate   = 1.0 - coverage
        js_div     = jsd(q_cnt, d_cnt)

        # query-exclusive terms (the mismatch tokens)
        exclusive_q = sorted(q_set - d_set)

        doc_name = q.get("doc_name", "")
        doc_meta = doc_info.get(doc_name, {})

        rows.append({
            "financebench_id": q["financebench_id"],
            "question":        q["question"],
            "question_type":   q.get("question_type", "unknown"),
            "question_reasoning": q.get("question_reasoning", ""),
            "doc_name":        doc_name,
            "doc_type":        doc_meta.get("doc_type", "unknown"),
            "company":         q.get("company", ""),
            "jaccard":         jaccard,
            "coverage":        coverage,
            "oov_rate":        oov_rate,
            "jsd":             js_div,
            "n_query_tokens":  len(qtoks),
            "n_query_unique":  len(q_set),
            "n_doc_tokens":    len(dtoks),
            "n_doc_unique":    len(d_set),
            "n_overlap":       len(intersection),
            "exclusive_q":     exclusive_q,
        })
    return pd.DataFrame(rows)


# ── helpers ───────────────────────────────────────────────────────────────────
METRIC_LABELS = {
    "oov_rate":  "OOV Rate (↑ = more mismatch)",
    "jaccard":   "Jaccard Similarity (↑ = more overlap)",
    "coverage":  "Term Coverage (↑ = more overlap)",
    "jsd":       "Jensen-Shannon Divergence (↑ = more mismatch)",
}
MISMATCH_METRICS = ["oov_rate", "jsd"]
OVERLAP_METRICS  = ["jaccard", "coverage"]
ALL_METRICS      = ["oov_rate", "jaccard", "coverage", "jsd"]


def savefig(name):
    path = OUT_DIR / name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {path.relative_to(BASE)}")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. OVERALL DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════════════════════════
def plot_overall(df):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Overall Lexical Mismatch Distribution (FinanceBench)", fontsize=14, y=1.01)

    for ax, metric in zip(axes.flat, ALL_METRICS):
        sns.histplot(df[metric], bins=30, kde=True, ax=ax, color="#4C72B0", edgecolor="white")
        mu, med = df[metric].mean(), df[metric].median()
        ax.axvline(mu,  color="red",    lw=1.5, ls="--", label=f"mean={mu:.3f}")
        ax.axvline(med, color="orange", lw=1.5, ls=":",  label=f"median={med:.3f}")
        ax.set_xlabel(METRIC_LABELS[metric])
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)

    plt.tight_layout()
    savefig("01_overall_distributions.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PER QUESTION TYPE
# ═══════════════════════════════════════════════════════════════════════════════
def plot_per_question_type(df):
    qtypes = sorted(df["question_type"].unique())

    # --- violin / box ---
    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    fig.suptitle("Lexical Mismatch by Question Type", fontsize=14)
    for ax, metric in zip(axes, ALL_METRICS):
        sns.violinplot(
            data=df, x="question_type", y=metric,
            palette=PALETTE, inner="box", ax=ax, order=qtypes
        )
        ax.set_xlabel("")
        ax.set_ylabel(metric)
        ax.set_title(METRIC_LABELS[metric].split("(")[0].strip())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
    plt.tight_layout()
    savefig("02_question_type_violin.png")

    # --- bar chart (mean ± std) ---
    stats = df.groupby("question_type")[ALL_METRICS].agg(["mean", "std"]).round(4)
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Mean Lexical Metrics by Question Type (± 1 SD)", fontsize=13)
    for ax, metric in zip(axes, ALL_METRICS):
        means = stats[metric]["mean"]
        stds  = stats[metric]["std"]
        colors = [PALETTE.get(qt, "#888") for qt in means.index]
        bars = ax.bar(means.index, means, yerr=stds, color=colors,
                      capsize=5, edgecolor="white", linewidth=0.8)
        ax.set_title(METRIC_LABELS[metric].split("(")[0].strip())
        ax.set_xlabel("")
        ax.set_xticklabels(means.index, rotation=15, ha="right")
        ax.set_ylim(0, min(1.0, means.max() + stds.max() + 0.1))
    plt.tight_layout()
    savefig("03_question_type_bar.png")

    # --- Kruskal-Wallis significance ---
    print("\n[Question Type — Kruskal-Wallis tests]")
    for metric in ALL_METRICS:
        groups = [df.loc[df["question_type"] == qt, metric].values for qt in qtypes]
        stat, p = kruskal(*groups)
        print(f"  {metric:12s}: H={stat:.2f}  p={p:.4f}  {'***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PER DOC TYPE
# ═══════════════════════════════════════════════════════════════════════════════
def plot_per_doc_type(df):
    dtypes = sorted(df["doc_type"].unique())
    pal = dict(zip(dtypes, DOC_PALETTE))

    # --- violin ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    fig.suptitle("Lexical Mismatch by Document Type", fontsize=14)
    for ax, metric in zip(axes, ALL_METRICS):
        sns.violinplot(
            data=df, x="doc_type", y=metric,
            palette=pal, inner="box", ax=ax, order=dtypes
        )
        ax.set_xlabel("")
        ax.set_ylabel(metric)
        ax.set_title(METRIC_LABELS[metric].split("(")[0].strip())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
    plt.tight_layout()
    savefig("04_doc_type_violin.png")

    # --- bar chart ---
    stats = df.groupby("doc_type")[ALL_METRICS].agg(["mean", "std"]).round(4)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
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
    savefig("05_doc_type_bar.png")

    print("\n[Doc Type — Kruskal-Wallis tests]")
    for metric in ALL_METRICS:
        groups = [df.loc[df["doc_type"] == dt, metric].values for dt in dtypes if len(df.loc[df["doc_type"]==dt])>1]
        if len(groups) < 2:
            continue
        stat, p = kruskal(*groups)
        print(f"  {metric:12s}: H={stat:.2f}  p={p:.4f}  {'***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. QUESTION TYPE × DOC TYPE HEATMAPS
# ═══════════════════════════════════════════════════════════════════════════════
def plot_cross_heatmaps(df):
    qtypes = sorted(df["question_type"].unique())
    dtypes = sorted(df["doc_type"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Lexical Mismatch: Question Type × Document Type", fontsize=14, y=1.01)

    cmap_mismatch = "Reds"
    cmap_overlap  = "Blues"

    for ax, metric in zip(axes.flat, ALL_METRICS):
        pivot = df.groupby(["question_type", "doc_type"])[metric].mean().unstack(fill_value=np.nan)
        pivot = pivot.reindex(index=qtypes, columns=dtypes)

        cmap  = cmap_mismatch if metric in MISMATCH_METRICS else cmap_overlap
        vmin  = pivot.values[~np.isnan(pivot.values)].min() * 0.95
        vmax  = pivot.values[~np.isnan(pivot.values)].max() * 1.02

        sns.heatmap(
            pivot, ax=ax, annot=True, fmt=".3f",
            cmap=cmap, vmin=vmin, vmax=vmax,
            linewidths=0.5, linecolor="white",
            cbar_kws={"shrink": 0.7},
            annot_kws={"size": 10},
        )
        direction = "↑ worse" if metric in MISMATCH_METRICS else "↑ better"
        ax.set_title(f"{metric.upper()} ({direction})", fontsize=11)
        ax.set_xlabel("Document Type")
        ax.set_ylabel("Question Type")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    savefig("06_cross_heatmap.png")

    # --- count heatmap (how many examples per cell) ---
    fig, ax = plt.subplots(figsize=(8, 4))
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
    savefig("07_cross_count_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. RADAR / SPIDER CHART — overall profile per question type
# ═══════════════════════════════════════════════════════════════════════════════
def plot_radar(df):
    metrics = ["oov_rate", "jsd", "jaccard", "coverage"]
    labels  = ["OOV Rate", "JSD", "Jaccard", "Coverage"]
    qtypes  = sorted(df["question_type"].unique())
    N = len(metrics)
    angles = [n / N * 2 * math.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    ax.set_title("Mismatch Profile by Question Type\n(normalised to [0,1])", pad=20)

    # normalise each metric to [0,1] across question types for visual comparison
    agg = df.groupby("question_type")[metrics].mean()
    norm = (agg - agg.min()) / (agg.max() - agg.min() + 1e-9)

    for qt in qtypes:
        vals = norm.loc[qt].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, "o-", lw=2, label=qt, color=PALETTE.get(qt))
        ax.fill(angles, vals, alpha=0.12, color=PALETTE.get(qt))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=11)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
    plt.tight_layout()
    savefig("08_radar_question_type.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. QUALITATIVE — corpus-level term frequency gap
# ═══════════════════════════════════════════════════════════════════════════════
def build_corpus_counters(queries, doc_info):
    query_cnt  = Counter()
    doc_cnt    = Counter()
    slice_q    = defaultdict(Counter)   # question_type → Counter
    slice_d    = defaultdict(Counter)   # doc_type → Counter

    for q in queries:
        qtoks = tokenise(q["question"])
        query_cnt.update(qtoks)
        qt = q.get("question_type", "unknown")
        slice_q[qt].update(qtoks)

        doc_name = q.get("doc_name", "")
        doc_meta = doc_info.get(doc_name, {})
        dt = doc_meta.get("doc_type", "unknown")

        for ev in q.get("evidence", []):
            page_text = ev.get("evidence_text_full_page") or ev.get("evidence_text", "")
            dtoks = tokenise(page_text)
            doc_cnt.update(dtoks)
            slice_d[dt].update(dtoks)

    return query_cnt, doc_cnt, slice_q, slice_d


def plot_term_gap(query_cnt, doc_cnt, top_n=25):
    """Bar chart of terms uniquely dominant in queries vs documents."""
    # relative frequency ratio: log(P_query / P_doc)  for terms in queries
    total_q = sum(query_cnt.values())
    total_d = sum(doc_cnt.values())

    scores = {}
    for term in set(query_cnt) | set(doc_cnt):
        pq = (query_cnt.get(term, 0) + 1) / (total_q + len(query_cnt))
        pd_ = (doc_cnt.get(term, 0) + 1) / (total_d + len(doc_cnt))
        scores[term] = math.log2(pq / pd_)

    sorted_scores = sorted(scores.items(), key=lambda x: x[1])

    # doc-dominant = negative score (bottom), query-dominant = positive (top)
    doc_dominant   = sorted_scores[:top_n]
    query_dominant = sorted_scores[-top_n:][::-1]

    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    fig.suptitle("Term Frequency Gap: Queries vs Documents\n(log₂ ratio, smoothed)", fontsize=13)

    # query-exclusive
    terms_q, scores_q = zip(*query_dominant)
    axes[0].barh(list(reversed(terms_q)), list(reversed(scores_q)), color="#4C72B0", edgecolor="white")
    axes[0].set_title("Query-Dominant Terms\n(overrepresented in queries)", color="#4C72B0")
    axes[0].set_xlabel("log₂(P_query / P_doc)")
    axes[0].axvline(0, color="black", lw=0.8)

    # doc-exclusive
    terms_d, scores_d = zip(*doc_dominant)
    scores_d_abs = [-s for s in scores_d]
    axes[1].barh(list(reversed(terms_d)), list(reversed(scores_d_abs)), color="#DD8452", edgecolor="white")
    axes[1].set_title("Document-Dominant Terms\n(overrepresented in documents)", color="#DD8452")
    axes[1].set_xlabel("log₂(P_doc / P_query)")
    axes[1].axvline(0, color="black", lw=0.8)

    plt.tight_layout()
    savefig("09_term_gap_global.png")


def plot_term_gap_by_slice(slice_q, slice_d, top_n=15):
    """For each question_type × doc_type slice, show top query-exclusive terms."""
    qtypes = sorted(slice_q.keys())
    dtypes = sorted(slice_d.keys())

    total_q_all = {qt: sum(c.values()) for qt, c in slice_q.items()}
    total_d_all = {dt: sum(c.values()) for dt, c in slice_d.items()}

    # ---- per question type: query-exclusive terms relative to all doc text ----
    total_d = Counter()
    for dt in slice_d.keys() - {'unknown'}:
        total_d += slice_d[dt]
    total_d_n = sum(total_d.values())

    fig, axes = plt.subplots(1, len(qtypes), figsize=(6 * len(qtypes), 8))
    if len(qtypes) == 1:
        axes = [axes]
    fig.suptitle("Query-Exclusive Terms by Question Type\n(terms most overrepresented in queries vs all doc text)", fontsize=12)

    for ax, qt in zip(axes, qtypes):
        qc = slice_q[qt]
        total_q_n = total_q_all[qt]
        scores = {}
        for term in qc:
            pq  = (qc[term] + 1) / (total_q_n + len(qc))
            pd_ = (total_d.get(term, 0) + 1) / (total_d_n + len(total_d))
            scores[term] = math.log2(pq / pd_)
        top = sorted(scores.items(), key=lambda x: -x[1])[:top_n]
        terms, sc = zip(*top) if top else ([], [])
        color = PALETTE.get(qt, "#888")
        ax.barh(list(reversed(terms)), list(reversed(sc)), color=color, edgecolor="white")
        ax.set_title(qt, color=color, fontsize=10)
        ax.set_xlabel("log₂(P_query / P_doc)")
        ax.axvline(0, color="black", lw=0.8)
        ax.invert_yaxis()

    plt.tight_layout()
    savefig("10_query_exclusive_by_question_type.png")

    # ---- per doc type: doc-exclusive terms relative to all query text ----
    total_q = Counter()
    for qt in slice_q.keys() - {'unknown'}:
        total_q += slice_q[qt]
    total_q_n = sum(total_q.values())

    fig, axes = plt.subplots(1, len(dtypes), figsize=(5 * len(dtypes), 8))
    if len(dtypes) == 1:
        axes = [axes]
    fig.suptitle("Document-Exclusive Terms by Doc Type\n(terms most overrepresented in documents vs all queries)", fontsize=12)
    pal = dict(zip(dtypes, DOC_PALETTE))

    for ax, dt in zip(axes, dtypes):
        dc = slice_d[dt]
        total_d_n2 = total_d_all.get(dt, 1)
        scores = {}
        for term in dc:
            pd_ = (dc[term] + 1) / (total_d_n2 + len(dc))
            pq  = (total_q.get(term, 0) + 1) / (total_q_n + len(total_q))
            scores[term] = math.log2(pd_ / pq)
        top = sorted(scores.items(), key=lambda x: -x[1])[:top_n]
        terms, sc = zip(*top) if top else ([], [])
        color = pal.get(dt, "#888")
        ax.barh(list(reversed(terms)), list(reversed(sc)), color=color, edgecolor="white")
        ax.set_title(dt, color="black", fontsize=10)
        ax.set_xlabel("log₂(P_doc / P_query)")
        ax.axvline(0, color="black", lw=0.8)
        ax.invert_yaxis()

    plt.tight_layout()
    savefig("11_doc_exclusive_by_doc_type.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. EXTREME EXAMPLES  (qualitative)
# ═══════════════════════════════════════════════════════════════════════════════
def print_extreme_examples(df, n=5):
    lines = []
    lines.append("\n" + "═"*80)
    lines.append("QUALITATIVE: EXTREME MISMATCH EXAMPLES")
    lines.append("═"*80)

    for metric, label, ascending in [
        ("oov_rate", "Highest OOV Rate (most mismatch)", False),
        ("oov_rate", "Lowest OOV Rate (most overlap)",  True),
        ("jsd",      "Highest JSD (most distribution divergence)", False),
    ]:
        lines.append(f"\n── {label} ──")
        sub = df.sort_values(metric, ascending=ascending).head(n)
        for _, row in sub.iterrows():
            lines.append(
                f"  [{row['question_type']} | {row['doc_type']} | {row['company']}]"
            )
            lines.append(f"  Q: {row['question'][:120]}...")
            lines.append(
                f"  oov={row['oov_rate']:.3f}  jaccard={row['jaccard']:.3f}  "
                f"coverage={row['coverage']:.3f}  jsd={row['jsd']:.3f}"
            )
            excl = row['exclusive_q'][:8]
            if excl:
                lines.append(f"  Query-exclusive tokens: {excl}")
            lines.append("")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. WRITTEN REPORT
# ═══════════════════════════════════════════════════════════════════════════════
def write_report(df, extreme_text):
    n = len(df)
    lines = []
    lines.append("LEXICAL MISMATCH ANALYSIS REPORT — FinanceBench")
    lines.append("="*60)
    lines.append(f"Total queries analysed: {n}")
    lines.append("")

    lines.append("── OVERALL ──")
    for m in ALL_METRICS:
        lines.append(f"  {m:12s}: mean={df[m].mean():.4f}  median={df[m].median():.4f}  "
                     f"std={df[m].std():.4f}  min={df[m].min():.4f}  max={df[m].max():.4f}")

    for groupby, label in [("question_type", "QUESTION TYPE"), ("doc_type", "DOC TYPE")]:
        lines.append(f"\n── BY {label} ──")
        for g, gdf in df.groupby(groupby):
            lines.append(f"  [{g}]  n={len(gdf)}")
            for m in ALL_METRICS:
                lines.append(f"    {m:12s}: mean={gdf[m].mean():.4f}  "
                              f"median={gdf[m].median():.4f}  std={gdf[m].std():.4f}")

    lines.append("\n── BY QUESTION TYPE × DOC TYPE ──")
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
    queries, doc_info = load_data()
    print(f"  {len(queries)} queries, {len(doc_info)} document metadata entries")

    print("Computing per-query metrics …")
    df = compute_query_metrics(queries, doc_info)
    print(f"  {len(df)} valid query rows")
    df.to_csv(OUT_DIR / "metrics_per_query.csv", index=False)
    print(f"  saved → analysis/lexical_mismatch/metrics_per_query.csv")

    print("\nOverall summary:")
    print(df[ALL_METRICS].describe().round(4).to_string())

    print("\nGenerating plots …")

    print("  [1/8] Overall distributions")
    plot_overall(df)

    print("  [2/8] Per question type")
    plot_per_question_type(df)

    print("  [3/8] Per doc type")
    plot_per_doc_type(df)

    print("  [4/8] Cross heatmaps")
    plot_cross_heatmaps(df)

    print("  [5/8] Radar chart")
    plot_radar(df)

    print("  [6/8] Building corpus counters …")
    query_cnt, doc_cnt, slice_q, slice_d = build_corpus_counters(queries, doc_info)

    print("  [7/8] Term frequency gap plots")
    plot_term_gap(query_cnt, doc_cnt)
    plot_term_gap_by_slice(slice_q, slice_d)

    print("  [8/8] Writing report")
    extreme_text = print_extreme_examples(df)
    print(extreme_text)
    write_report(df, extreme_text)

    print("\nDone. All outputs in analysis/lexical_mismatch/")


if __name__ == "__main__":
    main()
