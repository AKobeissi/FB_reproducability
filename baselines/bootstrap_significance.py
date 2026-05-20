#!/usr/bin/env python3
"""
Paired bootstrap significance tests for retrieval methods.

For each evaluation set (FB-only, Benchmark Extension), computes per-query
PageRec@5 for every method and runs paired bootstrap tests (10,000 resamples)
on all method pairs. Applies Holm-Bonferroni correction for multiple comparisons.

Output is written to baselines/bootstrap_significance/
"""

import json
import numpy as np
from pathlib import Path
from itertools import combinations
import csv
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent
PRED_DIR_FB = REPO_ROOT / "baselines/results/predictions"
PRED_DIR_EXT= REPO_ROOT / "baselines/results/benchmark_extension/predictions"
OUT_DIR     = REPO_ROOT / "baselines/bootstrap_significance"

K           = 5          # PageRec@K
N_BOOTSTRAP = 10_000
RNG_SEED    = 42
ALPHA       = 0.05


# ── Helpers ───────────────────────────────────────────────────────────────────

def _norm(name: str) -> str:
    name = str(name).lower().strip()
    if name.endswith(".pdf"):
        name = name[:-4]
    return name


def per_query_page_rec(items: list, k: int = K) -> np.ndarray:
    """Return binary 0/1 array: 1 if any top-k chunk hits a gold page."""
    scores = []
    for item in items:
        gold_pages = set()
        for seg in item.get("gold_evidence_segments", []):
            doc = seg.get("doc_name") or seg.get("document", "")
            page = seg.get("page") if seg.get("page") is not None else seg.get("evidence_page_num")
            if doc and page is not None:
                gold_pages.add((_norm(doc), str(page).strip()))

        if not gold_pages:
            scores.append(0.0)
            continue

        retrieved = item.get("retrieved_chunks", [])[:k]
        hit = 0
        for chunk in retrieved:
            meta = chunk.get("metadata", {})
            doc  = meta.get("doc_name") or meta.get("source", "")
            page = meta.get("page")
            if page is not None and (_norm(doc), str(page).strip()) in gold_pages:
                hit = 1
                break
        scores.append(float(hit))
    return np.array(scores)


def bootstrap_pvalue(a: np.ndarray, b: np.ndarray, n: int = N_BOOTSTRAP, seed: int = RNG_SEED) -> float:
    """
    Two-sided paired bootstrap p-value for H0: mean(a) == mean(b).

    We test H0: delta=0. Under the null we centre the differences at 0 and
    count how often a bootstrap resample exceeds the observed |delta|.
    """
    rng = np.random.default_rng(seed)
    diff = a - b
    observed = diff.mean()
    # centre at 0 for null distribution
    centred = diff - observed
    boots = rng.choice(centred, size=(n, len(centred)), replace=True).mean(axis=1)
    p = np.mean(np.abs(boots) >= np.abs(observed))
    return float(p)


def holm_bonferroni(pvals: list[float]) -> list[float]:
    """Return adjusted p-values using Holm-Bonferroni step-down correction."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = pvals[idx] * (m - rank)
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def method_name(path: Path) -> str:
    return path.stem.replace("_retrieval", "").replace("_", " ")


# ── Core evaluation ───────────────────────────────────────────────────────────

def _run_group(methods: dict, label: str) -> None:
    """Run bootstrap tests and write report for one same-corpus group."""
    method_list = sorted(methods, key=lambda m: -methods[m].mean())
    pairs = list(combinations(method_list, 2))
    if not pairs:
        print(f"  Skipping {label}: only 1 method")
        return

    raw_pvals = [bootstrap_pvalue(methods[a], methods[b]) for a, b in pairs]
    adj_pvals = holm_bonferroni(raw_pvals)

    safe = label.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "").replace("—", "").replace("-", "").replace("/", "")
    out_dir = OUT_DIR / safe
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "significance_report.txt"
    csv_path    = out_dir / "pairwise_pvalues.csv"

    with open(report_path, "w") as rpt:

        def w(line=""):
            print(line)
            rpt.write(line + "\n")

        w(f"Bootstrap Significance Report — {label}")
        w(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        w(f"Bootstrap resamples: {N_BOOTSTRAP:,}   Seed: {RNG_SEED}")
        w(f"Significance level: α={ALPHA}  (Holm-Bonferroni corrected)")
        w()
        w(f"{'Rank':<5} {'Method':<45} {'PageRec@'+str(K):<14} {'n':>5}")
        w("-" * 72)
        for rank, name in enumerate(method_list, 1):
            scores = methods[name]
            w(f"{rank:<5} {name:<45} {scores.mean():.4f}         {len(scores):>5}")

        w()
        w("─" * 72)
        w("SIGNIFICANT DIFFERENCES (adj. p < 0.05, Holm-Bonferroni)")
        w("─" * 72)
        sig_count = 0
        for (a_name, b_name), raw_p, adj_p in zip(pairs, raw_pvals, adj_pvals):
            if adj_p < ALPHA:
                delta = methods[a_name].mean() - methods[b_name].mean()
                direction = ">" if delta > 0 else "<"
                w(f"  {a_name} {direction} {b_name}")
                w(f"    Δ={delta:+.4f}   raw p={raw_p:.4f}   adj p={adj_p:.4f}")
                sig_count += 1
        if sig_count == 0:
            w("  (none)")

        w()
        w("─" * 72)
        w("BORDERLINE (0.05 ≤ adj. p < 0.10)")
        w("─" * 72)
        border_count = 0
        for (a_name, b_name), raw_p, adj_p in zip(pairs, raw_pvals, adj_pvals):
            if ALPHA <= adj_p < 0.10:
                delta = methods[a_name].mean() - methods[b_name].mean()
                direction = ">" if delta > 0 else "<"
                w(f"  {a_name} {direction} {b_name}")
                w(f"    Δ={delta:+.4f}   raw p={raw_p:.4f}   adj p={adj_p:.4f}")
                border_count += 1
        if border_count == 0:
            w("  (none)")

        w()
        w("─" * 72)
        w("FULL PAIRWISE TABLE (sorted by |Δ|, top 30 pairs)")
        w("─" * 72)
        sorted_pairs = sorted(
            zip(pairs, raw_pvals, adj_pvals),
            key=lambda x: abs(methods[x[0][0]].mean() - methods[x[0][1]].mean()),
            reverse=True,
        )[:30]
        w(f"  {'Method A':<42} {'Method B':<42} {'Δ':>8} {'raw p':>8} {'adj p':>8} {'sig':>5}")
        w("  " + "-" * 120)
        for (a_name, b_name), raw_p, adj_p in sorted_pairs:
            delta = methods[a_name].mean() - methods[b_name].mean()
            sig = "***" if adj_p < 0.001 else ("**" if adj_p < 0.01 else ("*" if adj_p < ALPHA else ("." if adj_p < 0.10 else "")))
            w(f"  {a_name:<42} {b_name:<42} {delta:>+8.4f} {raw_p:>8.4f} {adj_p:>8.4f} {sig:>5}")

    print(f"\n  Report saved → {report_path}")

    with open(csv_path, "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(["method_a", "method_b", "mean_a", "mean_b", "delta", "raw_pval", "adj_pval", "significant"])
        for (a_name, b_name), raw_p, adj_p in zip(pairs, raw_pvals, adj_pvals):
            delta = methods[a_name].mean() - methods[b_name].mean()
            writer.writerow([
                a_name, b_name,
                f"{methods[a_name].mean():.4f}",
                f"{methods[b_name].mean():.4f}",
                f"{delta:+.4f}",
                f"{raw_p:.4f}",
                f"{adj_p:.4f}",
                "yes" if adj_p < ALPHA else "no",
            ])
    print(f"  CSV saved     → {csv_path}")

    npy_path = out_dir / "per_query_scores.npz"
    np.savez(npy_path, **{k.replace(" ", "_"): v for k, v in methods.items()})
    print(f"  Scores saved  → {npy_path}")


def evaluate_directory(pred_dir: Path, label: str) -> None:
    files = sorted(pred_dir.glob("*_retrieval.json"))
    if not files:
        print(f"  No retrieval files found in {pred_dir}")
        return

    # Load all methods and group by corpus size (only compare same-corpus methods)
    from collections import defaultdict
    groups: dict[int, dict[str, np.ndarray]] = defaultdict(dict)
    for f in files:
        with open(f) as fh:
            items = json.load(fh)
        scores = per_query_page_rec(items)
        groups[len(scores)][method_name(f)] = scores

    for corpus_n, methods in sorted(groups.items(), reverse=True):
        sublabel = f"{label} (n={corpus_n})"
        print(f"\n{'='*70}")
        print(f"  {sublabel}  — {len(methods)} methods, K={K}, bootstrap resamples={N_BOOTSTRAP:,}")
        print(f"{'='*70}")
        for name, scores in sorted(methods.items(), key=lambda x: -x[1].mean()):
            print(f"  {name:45s}  PageRec@{K}={scores.mean():.4f}")
        _run_group(methods, sublabel)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    evaluate_directory(PRED_DIR_FB,  "FB-only (150q)")
    evaluate_directory(PRED_DIR_EXT, "Benchmark Extension (680q)")

    print(f"\nAll outputs in: {OUT_DIR}")
    print("Significance codes: *** p<0.001  ** p<0.01  * p<0.05  . p<0.10  (Holm-Bonferroni)")


if __name__ == "__main__":
    main()
