#!/usr/bin/env python3
"""
Unified-baseline SAE Retrieval Gap Diagnostics (post-hoc)

Purpose
-------
Run an SAE-based *diagnostic* (not reranker) on top of an existing unified pipeline output JSON.
This characterizes which gold pages are retrieved (TP) vs missed (FN) by your unified baseline.

Key properties
--------------
- Independent of finance_adaptive_sweep / learned page scorer
- Uses BGE-M3 base page embeddings
- Leakage-safe:
  * doc-level split
  * SAE fit on train-doc pages only
  * MI feature selection on train only
  * probe trained on train only
- Preserves unified output style (augments results with metadata_sae_diagnostics)

Expected inputs
---------------
1) Unified pipeline results JSON (output of your best baseline run)
2) FinanceBench-like dataset JSON used for that run (same order preferred)
3) Local pdf directory used by unified pipeline (for page text extraction)

Assumptions / schema notes
--------------------------
- unified results contain `retrieved_chunks` with `metadata` holding page information (e.g., page/page_number)
- dataset contains doc_name and evidence/page annotations (parser is robust to a few common variants)
"""

import argparse
import json
import logging
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional, Set

import numpy as np
import torch
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

# Reuse your repo loader used by unified pipeline for local/remote PDF fallback
from src.ingestion.pdf_utils import load_pdf_with_fallback

logger = logging.getLogger("unified_sae_gap_diag")


# -----------------------------
# Utilities
# -----------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_doc_name(name: Optional[str]) -> str:
    if not name:
        return "unknown"
    return str(name).strip()


def safe_json_load(path: str) -> Any:
    """Load either a standard JSON file or a JSONL file.

    For JSONL, returns a list of parsed records (one JSON object per line).
    """
    with open(path, "r", encoding="utf-8") as f:
        if str(path).lower().endswith(".jsonl"):
            rows = []
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSONL at {path}:{line_no}: {e}") from e
            return rows
        return json.load(f)


def safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return None
        if isinstance(x, (int, np.integer)):
            return int(x)
        if isinstance(x, float):
            return int(x)
        s = str(x).strip()
        if s == "":
            return None
        if re.fullmatch(r"-?\d+", s):
            return int(s)
        # handle values like "49.0"
        if re.fullmatch(r"-?\d+(\.\d+)?", s):
            return int(float(s))
        return None
    except Exception:
        return None


def extract_page_from_chunk_metadata(meta: Dict[str, Any]) -> Optional[int]:
    """
    Tries common page keys from LangChain / custom pipelines.
    """
    if not isinstance(meta, dict):
        return None
    candidate_keys = [
        "page", "page_number", "page_num", "source_page", "pdf_page", "page_index"
    ]
    for k in candidate_keys:
        if k in meta:
            p = safe_int(meta.get(k))
            if p is not None:
                return p

    # Sometimes nested metadata
    if "metadata" in meta and isinstance(meta["metadata"], dict):
        p = extract_page_from_chunk_metadata(meta["metadata"])
        if p is not None:
            return p

    return None


def parse_gold_pages_from_sample(sample: Dict[str, Any]) -> List[int]:
    """
    Robust parser for FinanceBench-like samples.
    Tries multiple field conventions. Returns unique sorted pages.
    """
    pages: Set[int] = set()

    # Common explicit fields
    for key in ["gold_pages", "evidence_pages", "page", "pages", "evidence_page"]:
        if key in sample:
            val = sample[key]
            if isinstance(val, list):
                for v in val:
                    p = safe_int(v)
                    if p is not None and p >= 0:
                        pages.add(p)
            else:
                p = safe_int(val)
                if p is not None and p >= 0:
                    pages.add(p)

    # Parse from evidence field if structured
    ev = sample.get("evidence", None)
    if isinstance(ev, dict):
        for k in ["page", "page_number", "pages", "evidence_page"]:
            if k in ev:
                v = ev[k]
                if isinstance(v, list):
                    for vv in v:
                        p = safe_int(vv)
                        if p is not None and p >= 0:
                            pages.add(p)
                else:
                    p = safe_int(v)
                    if p is not None and p >= 0:
                        pages.add(p)
    elif isinstance(ev, list):
        for item in ev:
            if isinstance(item, dict):
                for k in ["page", "page_number", "evidence_page"]:
                    if k in item:
                        p = safe_int(item[k])
                        if p is not None and p >= 0:
                            pages.add(p)

    # Parse from free text evidence if needed ("page 49", "pages 12 and 13")
    if isinstance(ev, str):
        # single / plural page mentions
        for m in re.finditer(r"\bpages?\s*[:#]?\s*([0-9]{1,4})(?:\s*[-,]\s*([0-9]{1,4}))?", ev.lower()):
            p1 = safe_int(m.group(1))
            p2 = safe_int(m.group(2)) if m.group(2) else None
            if p1 is not None:
                pages.add(p1)
            if p1 is not None and p2 is not None and p2 >= p1 and (p2 - p1) <= 5:
                for p in range(p1, p2 + 1):
                    pages.add(p)

    return sorted(pages)


def maybe_normalize_page_numbering(page: int, available_pages: Set[int]) -> Optional[int]:
    """
    Align 0-based vs 1-based numbering if needed.
    """
    if page in available_pages:
        return page
    # try off-by-one
    if (page - 1) in available_pages:
        return page - 1
    if (page + 1) in available_pages:
        return page + 1
    return None


def cosine_recon(a: np.ndarray, b: np.ndarray) -> float:
    an = np.linalg.norm(a, axis=1) + 1e-12
    bn = np.linalg.norm(b, axis=1) + 1e-12
    return float(np.mean(np.sum(a * b, axis=1) / (an * bn)))


def choose_default_results_json(output_dir_or_file: str) -> str:
    if os.path.isfile(output_dir_or_file):
        return output_dir_or_file
    if not os.path.isdir(output_dir_or_file):
        raise FileNotFoundError(output_dir_or_file)

    candidates = []
    for fn in os.listdir(output_dir_or_file):
        if fn.endswith(".json"):
            path = os.path.join(output_dir_or_file, fn)
            try:
                obj = safe_json_load(path)
                if isinstance(obj, list) and obj and isinstance(obj[0], dict) and "retrieved_chunks" in obj[0]:
                    candidates.append(path)
            except Exception:
                continue
    if not candidates:
        raise FileNotFoundError(
            f"No unified results JSON found in {output_dir_or_file}. "
            f"Pass --unified-results explicitly."
        )
    # pick largest (usually final results file, not config)
    candidates.sort(key=lambda p: os.path.getsize(p), reverse=True)
    return candidates[0]


# -----------------------------
# Page corpus extraction
# -----------------------------
@dataclass
class PageRecord:
    global_idx: int
    doc_name: str
    page: int
    text: str
    source: str


def build_page_corpus_from_dataset(
    dataset: List[Dict[str, Any]],
    pdf_dir: str,
) -> Tuple[List[PageRecord], Dict[Tuple[str, int], int], Dict[str, List[int]]]:
    """
    Loads raw PDF pages (not chunks) for all docs referenced in dataset.
    """
    unique_docs: Dict[str, str] = {}
    for s in dataset:
        doc_name = normalize_doc_name(s.get("doc_name"))
        doc_link = s.get("doc_link", "")
        unique_docs.setdefault(doc_name, doc_link)

    page_records: List[PageRecord] = []
    page_index: Dict[Tuple[str, int], int] = {}
    doc_to_pages: Dict[str, List[int]] = {}

    logger.info(f"Loading PDF pages for {len(unique_docs)} unique docs...")

    for doc_name, doc_link in tqdm(unique_docs.items(), desc="Load PDF pages"):
        try:
            pdf_docs, src = load_pdf_with_fallback(doc_name, doc_link, pdf_dir)
        except Exception as e:
            logger.warning(f"[SKIP] Failed to load PDF for {doc_name}: {e}")
            continue

        if not pdf_docs:
            logger.warning(f"[SKIP] No PDF pages returned for {doc_name}")
            continue

        pages_for_doc: List[int] = []
        for i, d in enumerate(pdf_docs):
            text = getattr(d, "page_content", "") or ""
            meta = getattr(d, "metadata", {}) or {}

            # LangChain PDF loaders often use 0-based "page"; fallback to sequence index
            p = extract_page_from_chunk_metadata(meta)
            if p is None:
                p = i  # fallback page index

            key = (doc_name, p)
            if key in page_index:
                # Some loaders can duplicate; keep longest page text
                prev_idx = page_index[key]
                if len(text) > len(page_records[prev_idx].text):
                    page_records[prev_idx].text = text
                continue

            global_idx = len(page_records)
            rec = PageRecord(
                global_idx=global_idx,
                doc_name=doc_name,
                page=int(p),
                text=text,
                source=str(src),
            )
            page_records.append(rec)
            page_index[key] = global_idx
            pages_for_doc.append(int(p))

        doc_to_pages[doc_name] = sorted(set(pages_for_doc))

    logger.info(f"Built page corpus: {len(page_records)} pages across {len(doc_to_pages)} docs")
    return page_records, page_index, doc_to_pages


# -----------------------------
# SAE model
# -----------------------------
class TopKSAE(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, topk: int):
        super().__init__()
        self.encoder = torch.nn.Linear(input_dim, hidden_dim)
        self.decoder = torch.nn.Linear(hidden_dim, input_dim, bias=False)
        self.topk = topk

        # Small init
        torch.nn.init.xavier_uniform_(self.encoder.weight)
        torch.nn.init.zeros_(self.encoder.bias)
        torch.nn.init.xavier_uniform_(self.decoder.weight)

    def encode_dense(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.encoder(x))

    def topk_sparse(self, z_dense: torch.Tensor) -> torch.Tensor:
        if self.topk <= 0 or self.topk >= z_dense.shape[1]:
            return z_dense
        vals, idx = torch.topk(z_dense, k=self.topk, dim=1)
        z = torch.zeros_like(z_dense)
        z.scatter_(1, idx, vals)
        return z

    def forward(self, x: torch.Tensor):
        z_dense = self.encode_dense(x)
        z_sparse = self.topk_sparse(z_dense)
        x_hat = self.decoder(z_sparse)
        return x_hat, z_dense, z_sparse


@torch.no_grad()
def encode_sae_sparse(model: TopKSAE, X: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    model.eval()
    outs = []
    for i in range(0, len(X), batch_size):
        xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32, device=device)
        _, _, z = model(xb)
        outs.append(z.detach().cpu().numpy())
    return np.vstack(outs) if outs else np.zeros((0, model.encoder.out_features), dtype=np.float32)


def train_sae(
    X_train: np.ndarray,
    input_dim: int,
    hidden_dim: int,
    topk: int,
    epochs: int,
    batch_size: int,
    lr: float,
    l1_lambda: float,
    device: str,
) -> Tuple[TopKSAE, Dict[str, Any]]:
    model = TopKSAE(input_dim=input_dim, hidden_dim=hidden_dim, topk=topk).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    X = X_train.astype(np.float32)
    n = X.shape[0]
    idxs = np.arange(n)

    history = []
    model.train()
    for ep in range(epochs):
        np.random.shuffle(idxs)
        total_loss = 0.0
        total_recon = 0.0
        total_l1 = 0.0
        total_count = 0

        for start in range(0, n, batch_size):
            batch_idx = idxs[start:start+batch_size]
            xb = torch.tensor(X[batch_idx], dtype=torch.float32, device=device)

            x_hat, z_dense, _ = model(xb)
            recon = torch.nn.functional.mse_loss(x_hat, xb)
            l1 = z_dense.abs().mean()
            loss = recon + l1_lambda * l1

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            bs = xb.size(0)
            total_loss += float(loss.item()) * bs
            total_recon += float(recon.item()) * bs
            total_l1 += float(l1.item()) * bs
            total_count += bs

        epoch_stats = {
            "epoch": ep + 1,
            "loss": total_loss / max(1, total_count),
            "recon_mse": total_recon / max(1, total_count),
            "l1_mean": total_l1 / max(1, total_count),
        }
        history.append(epoch_stats)
        logger.info(
            f"SAE epoch {ep+1}/{epochs} | "
            f"loss={epoch_stats['loss']:.6f} recon={epoch_stats['recon_mse']:.6f} l1={epoch_stats['l1_mean']:.6f}"
        )

    # Reconstruction diagnostic (train only)
    with torch.no_grad():
        X_hat = []
        for i in range(0, n, batch_size):
            xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32, device=device)
            x_hat, _, _ = model(xb)
            X_hat.append(x_hat.cpu().numpy())
        X_hat = np.vstack(X_hat)
    recon_cos = cosine_recon(X, X_hat)

    info = {
        "train_rows": int(n),
        "input_dim": int(input_dim),
        "hidden_dim": int(hidden_dim),
        "topk": int(topk),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "l1_lambda": float(l1_lambda),
        "reconstruction_cosine_train": recon_cos,
        "history_last": history[-1] if history else {},
    }
    return model, info


# -----------------------------
# Diagnostic row construction
# -----------------------------
@dataclass
class GoldPageRow:
    sample_idx: int
    question: str
    doc_name: str
    page: int
    page_global_idx: int
    label_fn: int   # 1 = false negative (missed gold page), 0 = true positive
    retrieved_pages: List[int]
    gold_pages_raw: List[int]


def build_gold_tp_fn_rows(
    dataset: List[Dict[str, Any]],
    unified_results: List[Dict[str, Any]],
    page_index: Dict[Tuple[str, int], int],
    doc_to_pages: Dict[str, List[int]],
) -> Tuple[List[GoldPageRow], Dict[str, Any]]:
    """
    Build diagnostic rows restricted to gold pages only:
      y = 1 if gold page missed (FN), else 0 if retrieved (TP)
    """
    n = min(len(dataset), len(unified_results))
    rows: List[GoldPageRow] = []

    samples_with_no_gold = 0
    samples_missing_doc_in_page_corpus = 0
    gold_pages_unresolved = 0

    for i in range(n):
        sample = dataset[i]
        result = unified_results[i]

        doc_name = normalize_doc_name(sample.get("doc_name") or result.get("doc_name"))
        question = str(sample.get("question") or result.get("question") or "")

        gold_pages_raw = parse_gold_pages_from_sample(sample)
        if not gold_pages_raw:
            samples_with_no_gold += 1
            continue

        available_pages = set(doc_to_pages.get(doc_name, []))
        if not available_pages:
            samples_missing_doc_in_page_corpus += 1
            continue

        # Normalize gold page numbering against actual loaded pages
        gold_pages: List[int] = []
        for gp in gold_pages_raw:
            p = maybe_normalize_page_numbering(gp, available_pages)
            if p is None:
                gold_pages_unresolved += 1
                continue
            gold_pages.append(int(p))
        gold_pages = sorted(set(gold_pages))
        if not gold_pages:
            continue

        # Extract retrieved pages from unified result retrieved_chunks metadata
        retrieved_pages: Set[int] = set()
        for ch in result.get("retrieved_chunks", []) or []:
            meta = ch.get("metadata", {}) if isinstance(ch, dict) else {}
            p = extract_page_from_chunk_metadata(meta)
            if p is None:
                continue
            p_norm = maybe_normalize_page_numbering(p, available_pages)
            if p_norm is not None:
                retrieved_pages.add(int(p_norm))

        for gp in gold_pages:
            gidx = page_index.get((doc_name, gp))
            if gidx is None:
                # page parser mismatch / missing page
                continue
            label_fn = 0 if gp in retrieved_pages else 1
            rows.append(
                GoldPageRow(
                    sample_idx=i,
                    question=question,
                    doc_name=doc_name,
                    page=gp,
                    page_global_idx=gidx,
                    label_fn=label_fn,
                    retrieved_pages=sorted(retrieved_pages),
                    gold_pages_raw=gold_pages_raw,
                )
            )

    stats = {
        "num_samples_considered": int(n),
        "num_rows_gold_pages": int(len(rows)),
        "samples_with_no_gold_pages": int(samples_with_no_gold),
        "samples_missing_doc_in_page_corpus": int(samples_missing_doc_in_page_corpus),
        "gold_pages_unresolved_after_page_alignment": int(gold_pages_unresolved),
        "num_tp_rows": int(sum(1 for r in rows if r.label_fn == 0)),
        "num_fn_rows": int(sum(1 for r in rows if r.label_fn == 1)),
    }
    return rows, stats


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unified-results", required=True,
                    help="Path to unified output JSON (list of per-sample results) OR directory containing it.")
    ap.add_argument("--dataset-json", required=True,
                    help="Path to dataset JSON used in the unified run (FinanceBench-like open-source subset).")
    ap.add_argument("--pdf-dir", required=True,
                    help="Local PDF directory used by unified pipeline (same corpus).")
    ap.add_argument("--embedding-model", default="BAAI/bge-m3",
                    help="SentenceTransformer model id/path for page embeddings (default: BGE-M3 base).")
    ap.add_argument("--cache-dir", default="outputs/cache_sae_gap_diag",
                    help="Directory for cached page embeddings.")
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--train-doc-frac", type=float, default=0.8,
                    help="Fraction of unique docs used for train split (doc-level split).")
    ap.add_argument("--sae-hidden-dim", type=int, default=4096)
    ap.add_argument("--sae-topk", type=int, default=64)
    ap.add_argument("--sae-epochs", type=int, default=12)
    ap.add_argument("--sae-batch-size", type=int, default=256)
    ap.add_argument("--sae-lr", type=float, default=1e-3)
    ap.add_argument("--sae-l1", type=float, default=1e-3)
    ap.add_argument("--mi-topk", type=int, default=128,
                    help="Top MI features selected on train only for logistic probe.")
    ap.add_argument("--probe-c", type=float, default=0.2,
                    help="Inverse regularization for L1 logistic probe.")
    ap.add_argument("--feature-cards-k", type=int, default=12,
                    help="How many globally top activated pages to store per selected feature.")
    ap.add_argument("--out-prefix", required=True,
                    help="Prefix for output files (augmented unified JSON, diagnostics JSON, feature cards JSON).")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    seed_everything(args.split_seed)
    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    unified_path = choose_default_results_json(args.unified_results)
    logger.info(f"Using unified results JSON: {unified_path}")

    unified_results = safe_json_load(unified_path)
    dataset = safe_json_load(args.dataset_json)

    if not isinstance(unified_results, list) or not isinstance(dataset, list):
        raise ValueError("Both unified results and dataset JSONs must be lists of samples/results.")

    logger.info(f"Loaded dataset={len(dataset)} samples | unified_results={len(unified_results)} rows")

    # 1) Build page corpus (raw PDF pages)
    page_records, page_index, doc_to_pages = build_page_corpus_from_dataset(dataset, args.pdf_dir)
    if not page_records:
        raise RuntimeError("No pages loaded from PDFs. Check --pdf-dir and dataset doc_name/doc_link values.")

    # 2) Embed all pages with BGE-M3 base (or provided embedding model)
    cache_model_slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", args.embedding_model)
    emb_cache_path = os.path.join(args.cache_dir, f"page_embeds_{cache_model_slug}.npz")

    if os.path.exists(emb_cache_path):
        logger.info(f"Loading cached page embeddings from {emb_cache_path}")
        npz = np.load(emb_cache_path, allow_pickle=True)
        page_embeddings = npz["embeddings"].astype(np.float32)
        cache_doc_names = npz["doc_names"].tolist()
        cache_pages = [int(x) for x in npz["pages"].tolist()]
        # Basic sanity check
        if len(cache_doc_names) != len(page_records) or len(cache_pages) != len(page_records):
            logger.warning("Cache size mismatch with current page corpus; rebuilding embeddings.")
            os.remove(emb_cache_path)
            page_embeddings = None
        else:
            # verify exact ordering
            same = True
            for rec, dn, pg in zip(page_records, cache_doc_names, cache_pages):
                if rec.doc_name != dn or rec.page != pg:
                    same = False
                    break
            if not same:
                logger.warning("Cache ordering mismatch with current page corpus; rebuilding embeddings.")
                os.remove(emb_cache_path)
                page_embeddings = None
    else:
        page_embeddings = None

    if page_embeddings is None:
        logger.info(f"Encoding {len(page_records)} pages with {args.embedding_model} ...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(args.embedding_model, device=device)

        texts = [rec.text if rec.text and rec.text.strip() else " " for rec in page_records]
        page_embeddings = model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype(np.float32)

        np.savez_compressed(
            emb_cache_path,
            embeddings=page_embeddings,
            doc_names=np.array([rec.doc_name for rec in page_records], dtype=object),
            pages=np.array([rec.page for rec in page_records], dtype=np.int32),
        )
        logger.info(f"Saved page embedding cache -> {emb_cache_path}")

    input_dim = int(page_embeddings.shape[1])

    # 3) Build diagnostic rows (gold page TP/FN) from unified baseline outputs
    rows, row_stats = build_gold_tp_fn_rows(dataset, unified_results, page_index, doc_to_pages)
    logger.info(f"Gold-page diagnostic rows: {row_stats}")

    if len(rows) < 20:
        raise RuntimeError("Too few diagnostic rows after parsing gold pages. Check dataset schema / page parsing.")

    # 4) Leakage-safe doc split (doc-level)
    unique_docs = sorted({r.doc_name for r in rows})
    rnd = random.Random(args.split_seed)
    rnd.shuffle(unique_docs)
    n_train_docs = max(1, int(round(len(unique_docs) * args.train_doc_frac)))
    n_train_docs = min(n_train_docs, len(unique_docs) - 1) if len(unique_docs) > 1 else 1
    train_docs = set(unique_docs[:n_train_docs])
    test_docs = set(unique_docs[n_train_docs:]) if len(unique_docs) > 1 else set()

    if not test_docs:
        # Fallback leave-one-out style if tiny set
        test_docs = {unique_docs[-1]}
        train_docs = set(unique_docs[:-1])

    train_rows = [r for r in rows if r.doc_name in train_docs]
    test_rows = [r for r in rows if r.doc_name in test_docs]

    # Split page embeddings for SAE training (train docs only)
    train_page_indices = [rec.global_idx for rec in page_records if rec.doc_name in train_docs]
    test_page_indices = [rec.global_idx for rec in page_records if rec.doc_name in test_docs]

    if len(train_page_indices) < 50 or len(test_page_indices) < 1:
        raise RuntimeError("Doc split produced too few train/test pages. Adjust --train-doc-frac or check corpus.")

    X_pages_train = page_embeddings[train_page_indices]

    logger.info(
        f"Doc split | train_docs={len(train_docs)} test_docs={len(test_docs)} | "
        f"train_rows={len(train_rows)} test_rows={len(test_rows)} | "
        f"train_pages={len(train_page_indices)} test_pages={len(test_page_indices)}"
    )

    # 5) Train SAE on train-doc pages only
    sae_device = "cuda" if torch.cuda.is_available() else "cpu"
    sae, sae_info = train_sae(
        X_train=X_pages_train,
        input_dim=input_dim,
        hidden_dim=args.sae_hidden_dim,
        topk=args.sae_topk,
        epochs=args.sae_epochs,
        batch_size=args.sae_batch_size,
        lr=args.sae_lr,
        l1_lambda=args.sae_l1,
        device=sae_device,
    )

    # 6) Encode all pages with trained SAE (allowed; inference only)
    Z_all = encode_sae_sparse(sae, page_embeddings, batch_size=args.sae_batch_size, device=sae_device).astype(np.float32)

    # 7) Build diagnostic train/test matrices (page-only sparse features)
    def rows_to_matrix(rs: List[GoldPageRow]) -> Tuple[np.ndarray, np.ndarray]:
        if not rs:
            return np.zeros((0, Z_all.shape[1]), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        X = np.vstack([Z_all[r.page_global_idx] for r in rs]).astype(np.float32)
        y = np.array([r.label_fn for r in rs], dtype=np.int64)
        return X, y

    X_train_diag, y_train = rows_to_matrix(train_rows)
    X_test_diag, y_test = rows_to_matrix(test_rows)

    # Need both classes in train and test for AUROC etc.
    if len(np.unique(y_train)) < 2:
        raise RuntimeError("Train diagnostic labels have one class only (all TP or all FN). Cannot fit probe.")
    if len(np.unique(y_test)) < 2:
        logger.warning("Test diagnostic labels have one class only. AUROC/AUPRC may be undefined.")

    # 8) MI feature selection on TRAIN only
    # Treat as continuous features; sparse many zeros is fine
    mi = mutual_info_classif(X_train_diag, y_train, discrete_features=False, random_state=args.split_seed)
    mi = np.nan_to_num(mi, nan=0.0, posinf=0.0, neginf=0.0)
    mi_order = np.argsort(mi)[::-1]
    sel_k = min(args.mi_topk, X_train_diag.shape[1])
    selected = mi_order[:sel_k]

    Xtr = X_train_diag[:, selected]
    Xte = X_test_diag[:, selected]

    # 9) L1 logistic probe (diagnostic only)
    # class_weight balanced helps if FN/TP imbalance is strong
    probe = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=args.probe_c,
        class_weight="balanced",
        max_iter=2000,
        random_state=args.split_seed,
    )
    probe.fit(Xtr, y_train)

    p_train = probe.predict_proba(Xtr)[:, 1]
    p_test = probe.predict_proba(Xte)[:, 1] if len(Xte) else np.array([])

    yhat_train = (p_train >= 0.5).astype(int)
    yhat_test = (p_test >= 0.5).astype(int) if len(Xte) else np.array([], dtype=int)

    def safe_metric(fn, *vals):
        try:
            return float(fn(*vals))
        except Exception:
            return None

    probe_metrics = {
        "train_accuracy": safe_metric(accuracy_score, y_train, yhat_train),
        "train_auroc": safe_metric(roc_auc_score, y_train, p_train) if len(np.unique(y_train)) > 1 else None,
        "train_auprc": safe_metric(average_precision_score, y_train, p_train) if len(np.unique(y_train)) > 1 else None,
        "test_accuracy": safe_metric(accuracy_score, y_test, yhat_test) if len(Xte) else None,
        "test_auroc": safe_metric(roc_auc_score, y_test, p_test) if len(Xte) and len(np.unique(y_test)) > 1 else None,
        "test_auprc": safe_metric(average_precision_score, y_test, p_test) if len(Xte) and len(np.unique(y_test)) > 1 else None,
        "num_selected_features": int(sel_k),
        "num_nonzero_probe_weights": int(np.sum(np.abs(probe.coef_[0]) > 1e-12)),
    }

    # 10) Feature enrichment summary on test set (or train if test empty)
    X_eval = X_test_diag if len(X_test_diag) else X_train_diag
    y_eval = y_test if len(y_test) else y_train
    eval_split_name = "test" if len(X_test_diag) else "train"

    feature_enrichment = []
    if len(X_eval):
        for rank_j, feat_id in enumerate(selected):
            col = X_eval[:, rank_j]
            fn_mask = (y_eval == 1)
            tp_mask = (y_eval == 0)

            fn_vals = col[fn_mask] if np.any(fn_mask) else np.array([])
            tp_vals = col[tp_mask] if np.any(tp_mask) else np.array([])

            entry = {
                "feature_id": int(feat_id),
                "mi_train": float(mi[feat_id]),
                "probe_weight": float(probe.coef_[0][rank_j]),
                "mean_activation_fn": float(fn_vals.mean()) if len(fn_vals) else 0.0,
                "mean_activation_tp": float(tp_vals.mean()) if len(tp_vals) else 0.0,
                "delta_mean_fn_minus_tp": float((fn_vals.mean() if len(fn_vals) else 0.0) - (tp_vals.mean() if len(tp_vals) else 0.0)),
                "activation_rate_fn": float((fn_vals > 0).mean()) if len(fn_vals) else 0.0,
                "activation_rate_tp": float((tp_vals > 0).mean()) if len(tp_vals) else 0.0,
                "delta_act_rate_fn_minus_tp": float(((fn_vals > 0).mean() if len(fn_vals) else 0.0) - ((tp_vals > 0).mean() if len(tp_vals) else 0.0)),
            }
            feature_enrichment.append(entry)

    # Sort primarily by |probe weight| then MI
    feature_enrichment.sort(key=lambda d: (abs(d["probe_weight"]), d["mi_train"]), reverse=True)

    # 11) Feature cards for manual interpretation (global top activations)
    # Also store top FN rows / TP rows activations for each feature
    selected_set = set(int(x) for x in selected.tolist())
    feature_cards = {}

    # Pre-index rows by page for quick lookup
    page_to_row_labels = {}
    for r in rows:
        page_to_row_labels.setdefault(r.page_global_idx, []).append(r.label_fn)

    for feat_id in selected.tolist():
        feat_id = int(feat_id)
        activ = Z_all[:, feat_id]
        top_idx = np.argsort(activ)[::-1][:args.feature_cards_k]

        card = {
            "feature_id": feat_id,
            "mi_train": float(mi[feat_id]),
            "global_top_pages": [],
            "top_fn_gold_pages": [],
            "top_tp_gold_pages": [],
        }

        for gidx in top_idx:
            rec = page_records[int(gidx)]
            text_preview = (rec.text or "").replace("\n", " ").strip()
            text_preview = text_preview[:400]
            card["global_top_pages"].append({
                "doc_name": rec.doc_name,
                "page": rec.page,
                "activation": float(activ[gidx]),
                "text_preview": text_preview,
            })

        # Gold-page rows split by FN/TP for this feature
        fn_rows_sorted = sorted(
            [r for r in rows if r.label_fn == 1],
            key=lambda r: float(Z_all[r.page_global_idx, feat_id]),
            reverse=True,
        )[:args.feature_cards_k]
        tp_rows_sorted = sorted(
            [r for r in rows if r.label_fn == 0],
            key=lambda r: float(Z_all[r.page_global_idx, feat_id]),
            reverse=True,
        )[:args.feature_cards_k]

        for r in fn_rows_sorted:
            rec = page_records[r.page_global_idx]
            card["top_fn_gold_pages"].append({
                "sample_idx": r.sample_idx,
                "doc_name": r.doc_name,
                "page": r.page,
                "activation": float(Z_all[r.page_global_idx, feat_id]),
                "question": r.question,
                "text_preview": (rec.text or "").replace("\n", " ").strip()[:300],
            })
        for r in tp_rows_sorted:
            rec = page_records[r.page_global_idx]
            card["top_tp_gold_pages"].append({
                "sample_idx": r.sample_idx,
                "doc_name": r.doc_name,
                "page": r.page,
                "activation": float(Z_all[r.page_global_idx, feat_id]),
                "question": r.question,
                "text_preview": (rec.text or "").replace("\n", " ").strip()[:300],
            })

        feature_cards[str(feat_id)] = card

    # 12) Per-sample diagnostics metadata augmentation (preserve unified style)
    rows_by_sample: Dict[int, List[GoldPageRow]] = {}
    for r in rows:
        rows_by_sample.setdefault(r.sample_idx, []).append(r)

    # Map selected features to page activations for quick inspect
    top_selected_by_absweight = [int(selected[i]) for i in np.argsort(np.abs(probe.coef_[0]))[::-1][:10]]

    augmented_results = []
    n_out = len(unified_results)
    for i in range(n_out):
        out = dict(unified_results[i])  # shallow copy

        sample_rows = rows_by_sample.get(i, [])
        if sample_rows:
            sample_diag = {
                "sae_gap_diag_version": "v1_page_only_posthoc",
                "label_definition": "Among gold pages only: FN=missed gold page by unified retrieval, TP=retrieved gold page",
                "split": "train" if sample_rows[0].doc_name in train_docs else "test",
                "doc_name": sample_rows[0].doc_name,
                "num_gold_pages_evaluated": len(sample_rows),
                "gold_pages": sorted({int(r.page) for r in sample_rows}),
                "retrieved_pages_union": sorted({p for r in sample_rows for p in r.retrieved_pages}),
                "tp_pages": sorted([int(r.page) for r in sample_rows if r.label_fn == 0]),
                "fn_pages": sorted([int(r.page) for r in sample_rows if r.label_fn == 1]),
                "page_feature_activations_top_probe_features": [],
            }

            for r in sample_rows:
                # show activations on each gold page for top features
                feat_vals = []
                z = Z_all[r.page_global_idx]
                for fid in top_selected_by_absweight:
                    feat_vals.append({
                        "feature_id": int(fid),
                        "activation": float(z[fid]),
                    })
                feat_vals.sort(key=lambda d: d["activation"], reverse=True)
                sample_diag["page_feature_activations_top_probe_features"].append({
                    "page": int(r.page),
                    "label": "FN" if r.label_fn == 1 else "TP",
                    "top_feature_activations": feat_vals[:5],
                })
        else:
            sample_diag = {
                "sae_gap_diag_version": "v1_page_only_posthoc",
                "status": "no_gold_page_rows_parsed_for_this_sample",
            }

        out["metadata_sae_diagnostics"] = sample_diag
        augmented_results.append(out)

    # 13) Save outputs
    diagnostics = {
        "experiment_name": "unified_sae_gap_diagnostics_posthoc",
        "baseline_source": {
            "unified_results_json": unified_path,
            "dataset_json": args.dataset_json,
            "pdf_dir": args.pdf_dir,
            "embedding_model": args.embedding_model,
        },
        "leakage_controls": {
            "split_level": "doc_name",
            "sae_fit_on_train_docs_only": True,
            "feature_selection_train_only": True,
            "probe_train_only": True,
            "posthoc_inference_on_test_pages_only": True,
        },
        "split_info": {
            "split_seed": args.split_seed,
            "train_doc_frac": args.train_doc_frac,
            "num_unique_docs_with_rows": len(unique_docs),
            "num_train_docs": len(train_docs),
            "num_test_docs": len(test_docs),
            "train_docs": sorted(list(train_docs)),
            "test_docs": sorted(list(test_docs)),
        },
        "page_corpus": {
            "num_pages": len(page_records),
            "num_docs": len(doc_to_pages),
        },
        "row_stats": row_stats,
        "sae_info": sae_info,
        "probe_metrics": probe_metrics,
        "feature_selection": {
            "mi_topk_requested": args.mi_topk,
            "selected_feature_ids": [int(x) for x in selected.tolist()],
            "top_20_by_mi": [
                {"feature_id": int(fid), "mi": float(mi[fid])}
                for fid in mi_order[:20]
            ],
            "top_20_by_abs_probe_weight": [
                {
                    "feature_id": int(selected[idx]),
                    "probe_weight": float(probe.coef_[0][idx]),
                    "mi_train": float(mi[int(selected[idx])]),
                }
                for idx in np.argsort(np.abs(probe.coef_[0]))[::-1][:20]
            ],
        },
        "feature_enrichment_eval_split": eval_split_name,
        "feature_enrichment_top": feature_enrichment[:100],
    }

    augmented_path = f"{args.out_prefix}_unified_augmented.json"
    diag_path = f"{args.out_prefix}_diagnostics.json"
    cards_path = f"{args.out_prefix}_feature_cards.json"

    with open(augmented_path, "w", encoding="utf-8") as f:
        json.dump(augmented_results, f, indent=2, ensure_ascii=False)
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2, ensure_ascii=False)
    with open(cards_path, "w", encoding="utf-8") as f:
        json.dump(feature_cards, f, indent=2, ensure_ascii=False)

    logger.info("Done.")
    logger.info(f"Augmented unified output: {augmented_path}")
    logger.info(f"Diagnostics summary:     {diag_path}")
    logger.info(f"Feature cards:           {cards_path}")


if __name__ == "__main__":
    main()