#!/usr/bin/env python3
"""
Query vs Gold-Evidence Difficulty Analysis (FinanceBench-style).

What it does
------------
For each QA sample, compute:
  1) Lexical overlap (token recall)
  2) BM25 score of query against the sample's gold evidence text
  3) Semantic similarity (cosine) between query embedding and gold evidence embedding

Then, for each retrieval results JSON you provide:
  - Merge difficulty signals with retrieval hit@k (or any hit key)
  - Produce:
      * boxplots of difficulty signals for hit vs miss
      * bucket ("regime") analysis: low/high lexical × low/high semantic
      * correlations / AUC diagnostics
      * CSV outputs per method and a combined summary CSV

Usage
-----
python src/analysis/query_difficulty.py \
  --data_path data/financebench_open_source.jsonl \
  --retrieval_results results/run1.json results/run2.json \
  --hit_key page_hit@5 \
  --embed_model sentence-transformers/all-MiniLM-L6-v2 \
  --device cpu \
  --out_dir analysis/query_difficulty

Notes
-----
- This analysis is meant for *post-hoc diagnosis*, not tuning on the test set.
- If your dataset does not include gold evidence text, this script can't compute signals.
"""

import os
import re
import json
import math
import argparse
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr
except Exception:
    roc_auc_score = None
    spearmanr = None


# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("query_difficulty")


# ---------------------------
# Text utils
# ---------------------------
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[._/-][A-Za-z0-9]+)*")

def tokenize(text: str) -> List[str]:
    """Simple, robust tokenizer: lowercase alnum + internal ._/- kept."""
    if text is None:
        return []
    return [m.group(0).lower() for m in _WORD_RE.finditer(str(text))]

def safe_text(x: Any) -> str:
    return "" if x is None else str(x)


# ---------------------------
# Minimal BM25 (Okapi) implementation
# ---------------------------
class BM25Okapi:
    def __init__(self, corpus_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_tokens = corpus_tokens

        self.N = len(corpus_tokens)
        self.doc_len = np.array([len(doc) for doc in corpus_tokens], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.N > 0 else 0.0

        # df and idf
        df: Dict[str, int] = {}
        for doc in corpus_tokens:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1

        self.idf: Dict[str, float] = {}
        for term, freq in df.items():
            # classic BM25 idf with +0.5 smoothing
            self.idf[term] = math.log(1.0 + (self.N - freq + 0.5) / (freq + 0.5))

        # term frequencies per doc
        self.tf: List[Dict[str, int]] = []
        for doc in corpus_tokens:
            tfd: Dict[str, int] = {}
            for t in doc:
                tfd[t] = tfd.get(t, 0) + 1
            self.tf.append(tfd)

    def score_one(self, query_tokens: List[str], doc_idx: int) -> float:
        if self.N == 0:
            return 0.0
        score = 0.0
        dl = float(self.doc_len[doc_idx]) if self.doc_len[doc_idx] > 0 else 1.0
        denom_const = self.k1 * (1.0 - self.b + self.b * dl / (self.avgdl + 1e-9))
        tfd = self.tf[doc_idx]
        for t in query_tokens:
            if t not in tfd:
                continue
            idf = self.idf.get(t, 0.0)
            tf = float(tfd[t])
            score += idf * (tf * (self.k1 + 1.0)) / (tf + denom_const)
        return float(score)

    def score_query_against_docs(self, query_tokens: List[str]) -> np.ndarray:
        return np.array([self.score_one(query_tokens, i) for i in range(self.N)], dtype=np.float32)


# ---------------------------
# Embedding wrapper (SentenceTransformers preferred)
# ---------------------------
class Embedder:
    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 32):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.backend = None
        self.model = None

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name, device=device)
            self.backend = "sentence_transformers"
            logger.info(f"Loaded SentenceTransformer: {model_name} on {device}")
        except Exception as e:
            logger.warning(f"SentenceTransformer load failed ({e}). Trying langchain_huggingface...")
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
                self.model = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={"device": device}
                )
                self.backend = "langchain_hf"
                logger.info(f"Loaded LangChain HuggingFaceEmbeddings: {model_name} on {device}")
            except Exception as e2:
                raise RuntimeError(
                    "Could not load embeddings backend. Install sentence-transformers or langchain-huggingface."
                ) from e2

    def embed(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        texts = [safe_text(t) for t in texts]
        if self.backend == "sentence_transformers":
            emb = self.model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=normalize
            )
            return emb.astype(np.float32)
        else:
            # langchain backend
            vecs: List[List[float]] = []
            # embed_documents batches internally but not always; keep simple batching
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                batch_vecs = self.model.embed_documents(batch)
                vecs.extend(batch_vecs)
            emb = np.array(vecs, dtype=np.float32)
            if normalize:
                norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
                emb = emb / norms
            return emb


# ---------------------------
# Dataset parsing
# ---------------------------
def extract_query(row: Dict[str, Any]) -> str:
    for key in ["question", "query", "prompt", "input"]:
        if key in row and row[key]:
            return safe_text(row[key])
    return ""

def extract_financebench_id(row: Dict[str, Any]) -> Optional[str]:
    for key in ["financebench_id", "id", "qid", "question_id", "uid"]:
        if key in row and row[key] is not None:
            return safe_text(row[key])
    return None

_EVID_TEXT_KEYS = ["evidence_text", "text", "page_text", "content", "snippet", "gold_text"]

def extract_gold_text(row: Dict[str, Any], max_chars: int = 6000) -> str:
    """
    Tries hard to get a usable gold evidence text from various FinanceBench-like schemas.
    """
    # Common: row["evidence"] is a list of dicts containing evidence_text or similar
    for ev_key in ["evidence", "gold_evidence", "references", "supporting_evidence"]:
        if ev_key not in row or row[ev_key] is None:
            continue
        ev = row[ev_key]
        texts: List[str] = []

        if isinstance(ev, str):
            texts = [ev]
        elif isinstance(ev, dict):
            for k in _EVID_TEXT_KEYS:
                if k in ev and ev[k]:
                    texts.append(safe_text(ev[k]))
        elif isinstance(ev, list):
            for item in ev:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict):
                    for k in _EVID_TEXT_KEYS:
                        if k in item and item[k]:
                            texts.append(safe_text(item[k]))
                            break

        gold = " ".join(t.strip() for t in texts if t and t.strip())
        gold = gold.strip()
        if gold:
            return gold[:max_chars]

    # fallback: sometimes stored as row["answer_context"] or similar
    for key in ["context", "answer_context", "gold_context"]:
        if key in row and row[key]:
            return safe_text(row[key])[:max_chars]

    return ""


# ---------------------------
# Retrieval results parsing
# ---------------------------
def load_results_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)

def parse_results(
    raw: Any,
    hit_key: str,
    fallback_hit_keys: List[str],
) -> pd.DataFrame:
    """
    Accepts JSON structures:
      - {"results": [ ... ]}
      - [ ... ]
    Each item may contain:
      - sample_id (int)
      - financebench_id (str)
      - retrieval_metrics: {...}
    """
    items = raw.get("results", raw) if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        raise ValueError("Unsupported retrieval results format: expected list or dict with 'results'.")

    rows = []
    for item in items:
        if not isinstance(item, dict):
            continue

        sample_id = item.get("sample_id", item.get("idx", item.get("index", None)))
        fb_id = item.get("financebench_id", item.get("id", None))

        metrics = item.get("retrieval_metrics", item.get("metrics", item))
        hit_val = None

        if isinstance(metrics, dict):
            if hit_key in metrics:
                hit_val = metrics.get(hit_key)
            else:
                for k in fallback_hit_keys:
                    if k in metrics:
                        hit_val = metrics.get(k)
                        break

        # Optional: gold rank if present
        gold_rank = None
        for rk in ["gold_rank", "gold_page_rank", "evidence_rank", "rank_gold"]:
            if rk in item and item[rk] is not None:
                gold_rank = item[rk]
                break
            if isinstance(metrics, dict) and rk in metrics and metrics[rk] is not None:
                gold_rank = metrics[rk]
                break

        # normalize hit to 0/1
        hit = 0
        if hit_val is not None:
            try:
                hit = int(float(hit_val) > 0.0)
            except Exception:
                hit = int(bool(hit_val))

        rows.append({
            "sample_index": sample_id,
            "financebench_id": safe_text(fb_id) if fb_id is not None else None,
            "hit": hit,
            "gold_rank": gold_rank
        })

    return pd.DataFrame(rows)


def smart_merge(difficulty_df: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prefer merging on financebench_id if both sides have it and overlap is decent,
    otherwise merge on sample_index.
    """
    if "financebench_id" in difficulty_df.columns and "financebench_id" in results_df.columns:
        left_ids = set(difficulty_df["financebench_id"].dropna().astype(str))
        right_ids = set(results_df["financebench_id"].dropna().astype(str))
        overlap = len(left_ids.intersection(right_ids))
        if overlap >= max(10, int(0.25 * min(len(left_ids), len(right_ids)))):
            merged = pd.merge(
                difficulty_df,
                results_df.drop(columns=["sample_index"], errors="ignore"),
                on="financebench_id",
                how="inner"
            )
            if not merged.empty:
                return merged

    merged = pd.merge(difficulty_df, results_df, on="sample_index", how="inner")
    return merged


# ---------------------------
# Main analyzer
# ---------------------------
@dataclass
class DifficultyConfig:
    embed_model: str
    device: str
    batch_size: int = 32
    max_gold_chars: int = 6000
    max_embed_chars: int = 2500
    bucket_split: str = "median"  # or "quartile"


class QueryDifficultyAnalyzer:
    def __init__(self, cfg: DifficultyConfig):
        self.cfg = cfg
        self.embedder = Embedder(cfg.embed_model, device=cfg.device, batch_size=cfg.batch_size)

    def compute_difficulty_signals(self, data_df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Extracting queries and gold evidence texts...")
        records = []
        for idx, row in data_df.iterrows():
            rowd = row.to_dict()
            q = extract_query(rowd)
            fb_id = extract_financebench_id(rowd)
            gold = extract_gold_text(rowd, max_chars=self.cfg.max_gold_chars)

            records.append({
                "sample_index": int(idx),
                "financebench_id": fb_id,
                "query": q,
                "gold_text": gold,
                "query_len": len(tokenize(q)),
                "gold_len": len(tokenize(gold)),
            })

        df = pd.DataFrame(records)
        missing_gold = (df["gold_text"].str.len() == 0).sum()
        if missing_gold > 0:
            logger.warning(f"{missing_gold}/{len(df)} samples have empty gold_text. Their scores will be 0.")

        # Lexical token recall
        logger.info("Computing lexical token recall...")
        lex_scores = []
        for q, g in tqdm(zip(df["query"], df["gold_text"]), total=len(df), desc="Lexical"):
            qt = set(tokenize(q))
            gt = set(tokenize(g))
            if not qt or not gt:
                lex_scores.append(0.0)
            else:
                lex_scores.append(len(qt.intersection(gt)) / float(len(qt)))
        df["lexical_recall"] = np.array(lex_scores, dtype=np.float32)

        # BM25-to-gold: build BM25 corpus over gold texts (one per sample)
        logger.info("Building BM25 over gold-evidence corpus and scoring each query vs its own gold...")
        corpus_tokens = [tokenize(t) for t in df["gold_text"].tolist()]
        bm25 = BM25Okapi(corpus_tokens)
        bm25_scores = []
        for i, q in tqdm(list(enumerate(df["query"].tolist())), total=len(df), desc="BM25"):
            qtok = tokenize(q)
            bm25_scores.append(bm25.score_one(qtok, i))
        df["bm25_to_gold"] = np.array(bm25_scores, dtype=np.float32)

        # Semantic cosine (normalized embeddings => cosine = dot)
        logger.info("Computing semantic similarity (cosine) query ↔ gold...")
        q_texts = df["query"].fillna("").tolist()
        g_texts = [safe_text(t)[: self.cfg.max_embed_chars] for t in df["gold_text"].tolist()]

        q_emb = self.embedder.embed(q_texts, normalize=True)
        g_emb = self.embedder.embed(g_texts, normalize=True)

        df["semantic_cosine"] = np.sum(q_emb * g_emb, axis=1).astype(np.float32)

        return df

    def bucket_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.cfg.bucket_split == "quartile":
            lex_thr = df["lexical_recall"].quantile(0.5)
            sem_thr = df["semantic_cosine"].quantile(0.5)
        else:
            lex_thr = df["lexical_recall"].median()
            sem_thr = df["semantic_cosine"].median()

        def regime(row: pd.Series) -> str:
            high_l = row["lexical_recall"] >= lex_thr
            high_s = row["semantic_cosine"] >= sem_thr
            if high_l and high_s:
                return "Easy (High Lex, High Sem)"
            if (not high_l) and (not high_s):
                return "Hard (Low Lex, Low Sem)"
            if high_l and (not high_s):
                return "Semantic Gap (High Lex, Low Sem)"
            return "Lexical Gap (Low Lex, High Sem)"

        out = df.copy()
        out["regime"] = out.apply(regime, axis=1)
        out.attrs["lex_thr"] = float(lex_thr)
        out.attrs["sem_thr"] = float(sem_thr)
        return out

    def _boxplot_hit_miss(self, df: pd.DataFrame, col: str, title: str, out_path: str):
        hit = df[df["hit"] == 1][col].dropna().values
        miss = df[df["hit"] == 0][col].dropna().values

        plt.figure(figsize=(7, 5))
        plt.boxplot([miss, hit], labels=["Miss", "Hit"], showfliers=False)
        plt.title(title)
        plt.ylabel(col)
        # jitter points
        for x, arr in enumerate([miss, hit], start=1):
            if len(arr) == 0:
                continue
            xs = np.random.normal(loc=x, scale=0.06, size=len(arr))
            plt.scatter(xs, arr, s=10, alpha=0.35)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

    def analyze_one_method(self, merged: pd.DataFrame, method_name: str, out_dir: str) -> Dict[str, Any]:
        os.makedirs(out_dir, exist_ok=True)
        if merged.empty:
            logger.warning(f"[{method_name}] Empty merged dataframe; skipping.")
            return {}

        # Bucket regimes
        merged_b = self.bucket_regimes(merged)
        lex_thr = merged_b.attrs["lex_thr"]
        sem_thr = merged_b.attrs["sem_thr"]

        # Summary per regime
        summary = merged_b.groupby("regime")["hit"].agg(["count", "mean"]).reset_index()
        summary.rename(columns={"count": "Count", "mean": "HitRate"}, inplace=True)
        summary["HitRate"] = (summary["HitRate"] * 100.0).round(2)

        # Diagnostics: AUC, correlations (if libs available)
        diag: Dict[str, Any] = {
            "method": method_name,
            "n": int(len(merged_b)),
            "lex_thr": float(lex_thr),
            "sem_thr": float(sem_thr),
            "overall_hit_rate": float(merged_b["hit"].mean() * 100.0),
        }

        if roc_auc_score is not None and merged_b["hit"].nunique() == 2:
            try:
                diag["auc_lexical_recall"] = float(roc_auc_score(merged_b["hit"], merged_b["lexical_recall"]))
                diag["auc_bm25_to_gold"] = float(roc_auc_score(merged_b["hit"], merged_b["bm25_to_gold"]))
                diag["auc_semantic_cosine"] = float(roc_auc_score(merged_b["hit"], merged_b["semantic_cosine"]))
            except Exception:
                pass

        if spearmanr is not None and "gold_rank" in merged_b.columns and merged_b["gold_rank"].notna().any():
            # Spearman between similarity and gold_rank (lower rank is better, so expect negative)
            try:
                sub = merged_b.dropna(subset=["gold_rank"])
                for c in ["lexical_recall", "bm25_to_gold", "semantic_cosine"]:
                    rho, p = spearmanr(sub[c].values, sub["gold_rank"].values)
                    diag[f"spearman_{c}_vs_gold_rank_rho"] = float(rho)
                    diag[f"spearman_{c}_vs_gold_rank_p"] = float(p)
            except Exception:
                pass

        # Save per-sample merged CSV
        merged_csv = os.path.join(out_dir, f"{method_name}__merged_samples.csv")
        merged_b.to_csv(merged_csv, index=False)

        # Save regime summary CSV
        summary_csv = os.path.join(out_dir, f"{method_name}__regime_summary.csv")
        summary.to_csv(summary_csv, index=False)

        # Save diagnostics JSON
        diag_path = os.path.join(out_dir, f"{method_name}__diagnostics.json")
        with open(diag_path, "w") as f:
            json.dump(diag, f, indent=2)

        # Plots: hit vs miss boxplots
        self._boxplot_hit_miss(
            merged_b, "lexical_recall",
            f"[{method_name}] Lexical Recall vs Hit",
            os.path.join(out_dir, f"{method_name}__lexical_vs_hit.png")
        )
        self._boxplot_hit_miss(
            merged_b, "bm25_to_gold",
            f"[{method_name}] BM25(query, gold) vs Hit",
            os.path.join(out_dir, f"{method_name}__bm25_vs_hit.png")
        )
        self._boxplot_hit_miss(
            merged_b, "semantic_cosine",
            f"[{method_name}] Semantic Cosine vs Hit",
            os.path.join(out_dir, f"{method_name}__semantic_vs_hit.png")
        )

        return {
            "diagnostics": diag,
            "regime_summary": summary
        }


def method_name_from_path(path: str) -> str:
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    return name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="FinanceBench jsonl with gold evidence text")
    parser.add_argument("--retrieval_results", type=str, nargs="+", required=True, help="One or more scored JSON files")
    parser.add_argument("--hit_key", type=str, default="page_hit@5", help="Metric key in retrieval_metrics used as hit label")
    parser.add_argument(
        "--fallback_hit_keys",
        type=str,
        nargs="*",
        default=["doc_hit@5", "hit@5", "page_hit@5", "doc_hit", "hit"],
        help="Fallback keys to look for if hit_key missing"
    )
    parser.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out_dir", type=str, default="analysis/query_difficulty")
    parser.add_argument("--bucket_split", type=str, choices=["median", "quartile"], default="median")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load dataset
    logger.info(f"Loading dataset: {args.data_path}")
    data_df = pd.read_json(args.data_path, lines=True)

    # Compute difficulty signals once
    cfg = DifficultyConfig(
        embed_model=args.embed_model,
        device=args.device,
        batch_size=args.batch_size,
        bucket_split=args.bucket_split
    )
    analyzer = QueryDifficultyAnalyzer(cfg)
    difficulty_df = analyzer.compute_difficulty_signals(data_df)

    difficulty_path = os.path.join(args.out_dir, "difficulty_signals.csv")
    difficulty_df.to_csv(difficulty_path, index=False)
    logger.info(f"Saved difficulty signals: {difficulty_path}")

    # Analyze each retrieval run
    all_regime_summaries = []
    all_diags = []

    for res_path in args.retrieval_results:
        method = method_name_from_path(res_path)
        logger.info(f"Loading retrieval results: {res_path} (method={method})")
        raw = load_results_json(res_path)
        results_df = parse_results(raw, hit_key=args.hit_key, fallback_hit_keys=args.fallback_hit_keys)

        merged = smart_merge(difficulty_df, results_df)
        if merged.empty:
            logger.warning(f"[{method}] Merge produced 0 rows. Check IDs (sample_id vs financebench_id).")
            continue

        method_out = os.path.join(args.out_dir, method)
        outputs = analyzer.analyze_one_method(merged, method_name=method, out_dir=method_out)
        if not outputs:
            continue

        diag = outputs["diagnostics"]
        summ = outputs["regime_summary"].copy()
        summ.insert(0, "method", method)

        all_diags.append(diag)
        all_regime_summaries.append(summ)

    # Combined summaries
    if all_regime_summaries:
        combined = pd.concat(all_regime_summaries, ignore_index=True)
        combined_path = os.path.join(args.out_dir, "ALL__regime_summary.csv")
        combined.to_csv(combined_path, index=False)
        logger.info(f"Saved combined regime summary: {combined_path}")

    if all_diags:
        diags_df = pd.DataFrame(all_diags)
        diags_path = os.path.join(args.out_dir, "ALL__diagnostics.csv")
        diags_df.to_csv(diags_path, index=False)
        logger.info(f"Saved combined diagnostics: {diags_path}")


if __name__ == "__main__":
    main()
