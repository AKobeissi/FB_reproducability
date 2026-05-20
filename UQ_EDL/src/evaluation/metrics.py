"""
All ranking, calibration, and uncertainty metrics.
Implemented from scratch (+ sklearn where appropriate).
"""
import math
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.calibration import calibration_curve


# ─── Ranking Metrics ─────────────────────────────────────────────────────────

def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    hits = sum(1 for r in ranked[:k] if r in gold)
    return hits / len(gold)


def mrr_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    for i, r in enumerate(ranked[:k]):
        if r in gold:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    dcg = 0.0
    for i, r in enumerate(ranked[:k]):
        if r in gold:
            dcg += 1.0 / math.log2(i + 2)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / ideal if ideal > 0 else 0.0


def average_precision_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    hits = 0
    sum_prec = 0.0
    for i, r in enumerate(ranked[:k]):
        if r in gold:
            hits += 1
            sum_prec += hits / (i + 1)
    return sum_prec / len(gold) if gold else 0.0


def compute_ranking_metrics(
    qid_to_ranked: dict[str, list[str]],
    qid_to_gold: dict[str, set[str]],
    ks: list[int] = [1, 3, 5, 10],
    mrr_k: int = 10,
    ndcg_ks: list[int] = [5, 10],
    map_k: int = 10,
) -> dict:
    results = {f"recall@{k}": [] for k in ks}
    results[f"mrr@{mrr_k}"] = []
    for k in ndcg_ks:
        results[f"ndcg@{k}"] = []
    results[f"map@{map_k}"] = []

    for qid, ranked in qid_to_ranked.items():
        gold = qid_to_gold.get(qid, set())
        for k in ks:
            results[f"recall@{k}"].append(recall_at_k(ranked, gold, k))
        results[f"mrr@{mrr_k}"].append(mrr_at_k(ranked, gold, mrr_k))
        for k in ndcg_ks:
            results[f"ndcg@{k}"].append(ndcg_at_k(ranked, gold, k))
        results[f"map@{map_k}"].append(average_precision_at_k(ranked, gold, map_k))

    return {metric: float(np.mean(vals)) for metric, vals in results.items()}


# ─── Calibration ──────────────────────────────────────────────────────────────

def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """ECE: weighted mean of |accuracy - confidence| across bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs - labels) ** 2))


def reliability_diagram_data(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> dict:
    fraction_of_positives, mean_predicted = calibration_curve(labels, probs, n_bins=n_bins, strategy="uniform")
    return {
        "fraction_of_positives": fraction_of_positives.tolist(),
        "mean_predicted": mean_predicted.tolist(),
    }


# ─── Failure Detection ─────────────────────────────────────────────────────────

def compute_failure_detection_metrics(
    uncertainty_scores: np.ndarray,
    failure_labels: np.ndarray,
) -> dict:
    """
    uncertainty_scores: higher = more uncertain
    failure_labels: 1 if retrieval failed (gold not in top-k), 0 if success
    """
    if failure_labels.sum() == 0 or failure_labels.sum() == len(failure_labels):
        return {"auroc": float("nan"), "auprc": float("nan")}
    auroc = roc_auc_score(failure_labels, uncertainty_scores)
    auprc = average_precision_score(failure_labels, uncertainty_scores)
    return {"auroc": float(auroc), "auprc": float(auprc)}


# ─── Selective Evaluation ─────────────────────────────────────────────────────

def selective_recall(
    qid_to_ranked: dict[str, list[str]],
    qid_to_gold: dict[str, set[str]],
    qid_to_uncertainty: dict[str, float],
    coverage_levels: list[float],
    k: int = 5,
) -> list[dict]:
    """
    For each coverage level c, keep the c*100% of questions with lowest uncertainty
    and report Recall@k on that subset.
    """
    all_qids = sorted(qid_to_ranked.keys())
    uncertainties = np.array([qid_to_uncertainty.get(q, 1.0) for q in all_qids])
    sorted_order = np.argsort(uncertainties)  # ascending uncertainty

    results = []
    for coverage in coverage_levels:
        n_keep = max(1, int(len(all_qids) * coverage))
        kept_indices = sorted_order[:n_keep]
        kept_qids = [all_qids[i] for i in kept_indices]
        rec_k = np.mean([
            recall_at_k(qid_to_ranked[q], qid_to_gold.get(q, set()), k)
            for q in kept_qids
        ])
        failure_rate = np.mean([
            1.0 - recall_at_k(qid_to_ranked[q], qid_to_gold.get(q, set()), 1)
            for q in kept_qids
        ])
        results.append({
            "coverage": coverage,
            "n_questions": n_keep,
            f"recall@{k}": float(rec_k),
            "failure_rate@1": float(failure_rate),
        })
    return results
