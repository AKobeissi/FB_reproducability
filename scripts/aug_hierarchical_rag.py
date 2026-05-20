#!/usr/bin/env python3
"""
Augmented Fine-tuning + Hierarchical RAG
=========================================

Addresses the key failure mode of the original hierarchical_rag.py:
  - Original FT model trained only on FinQA (metrics-style questions)
  - FinanceBench domain/novel questions need MD&A and narrative pages
  - Result: PageRec@5 = 0.19 for domain-relevant, 0.32 for novel-generated

This script:
  1. Builds augmented training data (query diversification + MDA injection)
  2. Fine-tunes a new BGE-M3 model on augmented data → models/fin_adapted_biencoder_aug/
  3. Runs the SAME hierarchical evaluation variants as hierarchical_rag.py
  4. Compares aug model vs original FT model
  5. Produces all the same plots (recall@k, bar charts, heatmaps) plus comparison plots

Usage:
  python aug_hierarchical_rag.py                    # train + eval + plot
  python aug_hierarchical_rag.py --skip-train       # eval only (use existing aug model)
  python aug_hierarchical_rag.py --variants hier_ft_page_chunk
"""

import argparse
import copy
import csv
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("aug_hier_rag")

# ─── Paths & constants ────────────────────────────────────────────────────────

ORIG_FT_MODEL_PATH = str(PROJECT_ROOT / "models" / "fin_adapted_biencoder_bge_m3")
AUG_FT_MODEL_PATH  = str(PROJECT_ROOT / "models" / "fin_adapted_biencoder_aug")
BASE_MODEL         = "BAAI/bge-m3"

FINQA_TEST_GOLD    = str(PROJECT_ROOT / "data"    / "finqa_test_gold_pages.jsonl")
FINQA_TRAIN_JSON   = str(PROJECT_ROOT / "finqa"   / "train.json")
FINQA_PDF_DIR      = str(PROJECT_ROOT / "Final-PDF")
FB_DATA_PATH       = str(PROJECT_ROOT / "data"    / "financebench_open_source.jsonl")
FB_DOC_INFO_PATH   = str(PROJECT_ROOT / "data"    / "financebench_document_information.jsonl")
FB_PDF_DIR         = str(PROJECT_ROOT / "pdfs")
RESULTS_DIR        = str(PROJECT_ROOT / "hierarchical_rag_aug" / "results")
VS_DIR             = str(PROJECT_ROOT / "hierarchical_rag_aug" / "vector_store")
ORIG_RESULTS_DIR   = str(PROJECT_ROOT / "hierarchical_rag"     / "results")

EMBED_MODEL    = "BAAI/bge-m3"
PAGE_MAX_TOKENS = 2048
CHUNK_SIZE      = 512
CHUNK_OVERLAP   = 64
K_VALUES        = [1, 3, 5, 10, 20]
MAIN_K          = 5
M_PAGES         = 20
N_BM25          = 50

QUESTION_TYPES  = ["metrics-generated", "domain-relevant", "novel-generated"]

VARIANT_LABELS = {
    "hier_ft_page_chunk":        "Aug: FT Page → Chunk",
    "hier_bm25_ft_rerank_chunk": "Aug: BM25 + FT Rerank → Chunk",
}

# Original model labels for comparison (loaded from existing results)
ORIG_VARIANT_LABELS = {
    "hier_ft_page_chunk":        "Orig: FT Page → Chunk",
    "hier_bm25_ft_rerank_chunk": "Orig: BM25 + FT Rerank → Chunk",
}


# ─── 1. Augmented training ────────────────────────────────────────────────────

def train_augmented_model(output_path: str, force: bool = False) -> None:
    """
    Build augmented training data and fine-tune BGE-M3.

    Augmentation strategies:
      1. Query style transformation (domain/novel variants of FinQA questions)
      2. FinQA train.json expansion (~1700 more examples)
      3. MDA page injection (narrative pages with synthetic queries)
      4. Cross-type hard negatives (table pages as negs for domain queries)
    """
    model_file = os.path.join(output_path, "model.safetensors")
    if os.path.exists(model_file) and not force:
        logger.info(f"Augmented model already exists at {output_path}, skipping training.")
        return

    import torch
    from sentence_transformers import SentenceTransformer, losses
    from torch.utils.data import DataLoader

    from domain_adapted_retrieval.augmented_data_prep import (
        build_augmented_training_pairs,
        to_sentence_transformer_examples,
    )
    from domain_adapted_retrieval.train_biencoder import (
        freeze_except_last_n_layers,
        build_ir_evaluator,
        set_seed,
    )

    set_seed(42)

    logger.info("=== Building augmented training data ===")
    all_pairs, train_pairs, val_pairs = build_augmented_training_pairs(
        finqa_test_gold_path=FINQA_TEST_GOLD,
        finqa_train_json_path=FINQA_TRAIN_JSON,
        pdf_dir=FINQA_PDF_DIR,
        n_domain_variants=2,
        n_novel_variants=1,
        n_mda_templates=3,
        include_mda_injection=True,
        include_train_expansion=True,
        seed=42,
    )

    train_examples = to_sentence_transformer_examples(train_pairs)
    logger.info(f"Training on {len(train_examples)} InputExamples")

    logger.info(f"Loading base model: {BASE_MODEL}")
    model = SentenceTransformer(BASE_MODEL)
    model.max_seq_length = 512

    # Freeze all but last 3 layers to prevent catastrophic forgetting
    freeze_except_last_n_layers(model, n=3)

    train_dataloader = DataLoader(
        train_examples, shuffle=True, batch_size=16, drop_last=True
    )
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    evaluator  = build_ir_evaluator(val_pairs)

    steps_per_epoch = len(train_dataloader)
    num_epochs = 5
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * 0.1)

    logger.info(
        f"Training: {num_epochs} epochs × {steps_per_epoch} steps = "
        f"{total_steps} total | warmup: {warmup_steps}"
    )

    os.makedirs(output_path, exist_ok=True)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=num_epochs,
        evaluation_steps=steps_per_epoch,
        warmup_steps=warmup_steps,
        output_path=output_path,
        save_best_model=True,
        show_progress_bar=True,
        optimizer_params={"lr": 5e-6},
        use_amp=torch.cuda.is_available(),
        checkpoint_path=os.path.join(output_path, "checkpoints"),
        checkpoint_save_steps=steps_per_epoch,
        checkpoint_save_total_limit=2,
    )

    # Save metadata
    from collections import Counter
    type_counts = Counter(p["augmentation_type"] for p in all_pairs)
    meta = {
        "base_model": BASE_MODEL,
        "output_path": output_path,
        "num_epochs": num_epochs,
        "batch_size": 16,
        "learning_rate": 5e-6,
        "trainable_layers": 3,
        "total_pairs": len(all_pairs),
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "train_examples": len(train_examples),
        "augmentation_type_counts": dict(type_counts),
    }
    with open(os.path.join(output_path, "training_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Augmented model saved to: {output_path}")


# ─── 2. Data loading (shared with hierarchical_rag.py) ───────────────────────

def load_financebench_samples(fb_path: str, doc_info_path: str) -> Tuple[List[Dict], Dict]:
    samples = []
    with open(fb_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            gold_segs = []
            for ev in raw.get("evidence", []):
                gold_segs.append({
                    "text":     ev.get("evidence_text", ""),
                    "doc_name": ev.get("doc_name", raw.get("doc_name", "")),
                    "page":     ev.get("evidence_page_num", -1),
                })
            samples.append({
                "financebench_id":       raw.get("financebench_id", ""),
                "question":              raw.get("question", ""),
                "reference_answer":      raw.get("answer", ""),
                "question_type":         raw.get("question_type", "unknown"),
                "doc_name":              raw.get("doc_name", ""),
                "doc_link":              raw.get("doc_link", ""),
                "gold_evidence_segments": gold_segs,
                "retrieved_chunks":      [],
                "generated_answer":      "",
            })

    doc_info: Dict[str, Dict] = {}
    if doc_info_path and os.path.exists(doc_info_path):
        with open(doc_info_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                doc_info[d["doc_name"]] = d

    counts = {qt: sum(1 for s in samples if s["question_type"] == qt)
              for qt in QUESTION_TYPES}
    logger.info(f"Loaded {len(samples)} FinanceBench samples — {counts}")
    return samples, doc_info


def load_all_pages(samples: List[Dict], pdf_dir: str) -> Dict[str, List[Dict]]:
    from src.ingestion.pdf_utils import load_pdf_with_fallback

    doc_pages: Dict[str, List[Dict]] = {}
    unique_docs = {(s["doc_name"], s["doc_link"]) for s in samples}

    for doc_name, doc_link in tqdm(unique_docs, desc="Loading PDFs"):
        if doc_name in doc_pages:
            continue
        try:
            pages, _ = load_pdf_with_fallback(doc_name, doc_link, pdf_dir)
            doc_pages[doc_name] = [
                {
                    "text":     p.page_content,
                    "page_num": p.metadata.get("page", i),
                    "doc_name": doc_name,
                }
                for i, p in enumerate(pages or [])
            ]
        except Exception as e:
            logger.warning(f"Could not load {doc_name}: {e}")
            doc_pages[doc_name] = []

    total_pages = sum(len(v) for v in doc_pages.values())
    logger.info(f"Loaded {len(doc_pages)} docs, {total_pages:,} pages total")
    return doc_pages


# ─── 3. Tokenisation & chunking ───────────────────────────────────────────────

_tokenizer_cache = {}

def _get_tokenizer(model_name: str = EMBED_MODEL):
    if model_name not in _tokenizer_cache:
        from transformers import AutoTokenizer
        _tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(
            model_name, use_fast=True
        )
    return _tokenizer_cache[model_name]


def chunk_text_tokens(text: str, chunk_size: int = CHUNK_SIZE,
                      chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    tokenizer  = _get_tokenizer()
    token_ids  = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return []
    stride = max(1, chunk_size - chunk_overlap)
    chunks = []
    start  = 0
    while start < len(token_ids):
        end   = min(start + chunk_size, len(token_ids))
        chunk = tokenizer.decode(token_ids[start:end], skip_special_tokens=True).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(token_ids):
            break
        start += stride
    return chunks


def pages_to_chunks(page_dicts: List[Dict]) -> List[Dict]:
    chunks = []
    for page in page_dicts:
        for chunk_text in chunk_text_tokens(page["text"]):
            chunks.append({
                "text":     chunk_text,
                "metadata": {"doc_name": page["doc_name"], "page": page["page_num"]},
            })
    return chunks


# ─── 4. Index building ────────────────────────────────────────────────────────

def build_bm25_index(doc_pages: Dict[str, List[Dict]]):
    from rank_bm25 import BM25Okapi

    flat_pages = [p for pages in doc_pages.values() for p in pages]

    def _tokenize(text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^\w\s$.,%-]", " ", text)
        return text.split()

    corpus_tokens = [_tokenize(p["text"]) for p in flat_pages]
    index = BM25Okapi(corpus_tokens)
    logger.info(f"BM25 index built over {len(flat_pages):,} pages")
    return index, flat_pages, _tokenize


def build_page_chroma_index(
    doc_pages: Dict[str, List[Dict]],
    model_path: str,
    vs_dir: str,
    collection_name: str,
):
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    os.makedirs(vs_dir, exist_ok=True)
    ef = SentenceTransformerEmbeddingFunction(
        model_name=model_path, device="cuda", trust_remote_code=True
    )
    client     = chromadb.PersistentClient(path=vs_dir)
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    existing_ids = set(collection.get(include=[])["ids"])
    tokenizer    = _get_tokenizer()
    docs_add, metas_add, ids_add = [], [], []

    for doc_name, pages in tqdm(doc_pages.items(), desc="Indexing pages"):
        for page in pages:
            pid = f"{doc_name}__page_{page['page_num']}"
            if pid in existing_ids:
                continue
            toks     = tokenizer.encode(page["text"], add_special_tokens=False)
            truncated = tokenizer.decode(toks[:PAGE_MAX_TOKENS], skip_special_tokens=True)
            docs_add.append(truncated)
            metas_add.append({"doc_name": doc_name, "page": int(page["page_num"])})
            ids_add.append(pid)
            if len(docs_add) >= 512:
                collection.add(documents=docs_add, metadatas=metas_add, ids=ids_add)
                docs_add, metas_add, ids_add = [], [], []

    if docs_add:
        collection.add(documents=docs_add, metadatas=metas_add, ids=ids_add)

    logger.info(f"Page index '{collection_name}': {collection.count():,} pages")
    return collection


# ─── 5. Embedding helpers ─────────────────────────────────────────────────────

_embed_model_cache = {}

def load_embed_model(model_path: str):
    if model_path not in _embed_model_cache:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {model_path}")
        _embed_model_cache[model_path] = SentenceTransformer(
            model_path, device="cuda", trust_remote_code=True
        )
    return _embed_model_cache[model_path]


def embed_texts(model, texts: List[str], batch_size: int = 64) -> np.ndarray:
    return model.encode(
        texts, normalize_embeddings=True, batch_size=batch_size,
        show_progress_bar=False, convert_to_numpy=True,
    )


# ─── 6. Retrieval algorithms ──────────────────────────────────────────────────

def retrieve_hier_ft_page_chunk(sample, embed_model, page_collection,
                                 doc_pages, M=M_PAGES, k=MAIN_K):
    q_emb  = embed_texts(embed_model, [sample["question"]])[0]
    n      = min(M * 2, page_collection.count())
    res    = page_collection.query(
        query_embeddings=[q_emb.tolist()], n_results=n,
        include=["documents", "metadatas", "distances"],
    )
    seen_pages: set = set()
    top_pages: List[Dict] = []
    for text, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        pid = (meta["doc_name"], meta["page"])
        if pid in seen_pages:
            continue
        seen_pages.add(pid)
        full_text = text
        for p in doc_pages.get(meta["doc_name"], []):
            if int(p["page_num"]) == int(meta["page"]):
                full_text = p["text"]
                break
        top_pages.append({
            "text": full_text, "page_num": int(meta["page"]),
            "doc_name": meta["doc_name"],
        })
        if len(top_pages) >= M:
            break

    if not top_pages:
        return []
    chunks      = pages_to_chunks(top_pages)
    if not chunks:
        return []
    chunk_texts = [c["text"] for c in chunks]
    chunk_embs  = embed_texts(embed_model, chunk_texts)
    scores      = chunk_embs @ q_emb
    top_idx     = np.argsort(scores)[::-1][:k]
    results = []
    for rank, idx in enumerate(top_idx, start=1):
        c = dict(chunks[idx])
        c["rank"]   = rank
        c["_score"] = float(scores[idx])
        results.append(c)
    return results


def retrieve_hier_bm25_ft_rerank_chunk(sample, embed_model, bm25_index,
                                        bm25_pages, bm25_tokenize,
                                        doc_pages, N=N_BM25, M=M_PAGES, k=MAIN_K):
    query_tokens    = bm25_tokenize(sample["question"])
    bm25_scores     = bm25_index.get_scores(query_tokens)
    top_n_idx       = np.argsort(bm25_scores)[::-1][:N]
    bm25_candidates = [bm25_pages[int(i)] for i in top_n_idx]
    if not bm25_candidates:
        return []

    q_emb       = embed_texts(embed_model, [sample["question"]])[0]
    page_texts  = [p["text"][:4000] for p in bm25_candidates]
    page_embs   = embed_texts(embed_model, page_texts)
    page_scores = page_embs @ q_emb
    top_m_idx   = np.argsort(page_scores)[::-1][:M]
    top_pages   = [bm25_candidates[int(i)] for i in top_m_idx]

    chunks      = pages_to_chunks(top_pages)
    if not chunks:
        return []
    chunk_texts = [c["text"] for c in chunks]
    chunk_embs  = embed_texts(embed_model, chunk_texts)
    scores      = chunk_embs @ q_emb
    top_idx     = np.argsort(scores)[::-1][:k]
    results = []
    for rank, idx in enumerate(top_idx, start=1):
        c = dict(chunks[idx])
        c["rank"]   = rank
        c["_score"] = float(scores[idx])
        results.append(c)
    return results


# ─── 7. Evaluation ────────────────────────────────────────────────────────────

def compute_retrieval_metrics(samples: List[Dict], k_values: List[int] = K_VALUES) -> Dict:
    from src.evaluation.retrieval_evaluator import RetrievalEvaluator
    return RetrievalEvaluator().compute_metrics(samples, k_values=k_values)


def compute_generative_metrics(samples: List[Dict]) -> Dict:
    try:
        from rouge_score import rouge_scorer as rs_module
        scorer = rs_module.RougeScorer(["rougeL"], use_stemmer=True)
    except ImportError:
        scorer = None

    rougeL_scores, numeric_matches = [], []
    for s in samples:
        gen = s.get("generated_answer", "")
        ref = s.get("reference_answer", "")
        if gen and ref and scorer:
            score = scorer.score(ref, gen)["rougeL"].fmeasure
        else:
            score = 0.0
        rougeL_scores.append(score)
        if s.get("question_type") == "metrics-generated":
            def _extract(t):
                t = re.sub(r"[$,€£%]", "", t)
                ms = re.findall(r"-?\d[\d,]*\.?\d*", t)
                return float(ms[0].replace(",", "")) if ms else None
            pn, rn = _extract(gen), _extract(ref)
            if pn is not None and rn is not None and rn != 0:
                numeric_matches.append(1.0 if abs(pn - rn) / abs(rn) <= 0.03 else 0.0)
            else:
                numeric_matches.append(0.0)

    return {
        "answer_rougeL": float(np.mean(rougeL_scores)) if rougeL_scores else 0.0,
        "numeric_match":  float(np.mean(numeric_matches)) if numeric_matches else 0.0,
        "n_samples":      len(samples),
    }


def aggregate_by_group(samples: List[Dict], key_fn, k_values=K_VALUES) -> Dict:
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for s in samples:
        groups[key_fn(s)].append(s)
    return {
        grp: {**compute_retrieval_metrics(gs, k_values), **compute_generative_metrics(gs)}
        for grp, gs in groups.items()
    }


# ─── 8. Saving results ────────────────────────────────────────────────────────

def save_predictions(samples: List[Dict], variant: str, results_dir: str) -> None:
    pred_dir = os.path.join(results_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    path = os.path.join(pred_dir, f"{variant}_retrieval.json")
    with open(path, "w") as f:
        json.dump(samples, f, indent=2)
    logger.info(f"Predictions saved: {path}")


def save_metrics(metrics: Dict, variant: str, results_dir: str) -> None:
    met_dir = os.path.join(results_dir, "metrics")
    os.makedirs(met_dir, exist_ok=True)
    path = os.path.join(met_dir, f"{variant}_metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved: {path}")


def save_tables(all_results: Dict, by_type_all: Dict, results_dir: str) -> None:
    met_dir = os.path.join(results_dir, "metrics")
    os.makedirs(met_dir, exist_ok=True)

    # all_variants JSON
    with open(os.path.join(met_dir, "all_variants_metrics.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # CSV
    headers = (
        ["Method"]
        + [f"DocRec@{k}" for k in K_VALUES]
        + [f"PageRec@{k}" for k in K_VALUES]
        + [f"BLEU@{MAIN_K}", f"ROUGE-L@{MAIN_K}", "AnsROUGE-L", "NumericMatch"]
    )
    csv_path = os.path.join(met_dir, "aug_hier_table.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for name, label in VARIANT_LABELS.items():
            if name not in all_results:
                continue
            r = all_results[name]
            row = [label]
            row += [f"{r.get(f'doc_recall@{k}', float('nan')):.3f}" for k in K_VALUES]
            row += [f"{r.get(f'page_recall@{k}', float('nan')):.3f}" for k in K_VALUES]
            row += [
                f"{r.get(f'context_bleu@{MAIN_K}', float('nan')):.3f}",
                f"{r.get(f'context_rougeL@{MAIN_K}', float('nan')):.3f}",
                f"{r.get('answer_rougeL', float('nan')):.3f}",
                f"{r.get('numeric_match', float('nan')):.3f}",
            ]
            w.writerow(row)
    logger.info(f"CSV: {csv_path}")

    # by-type CSV
    by_type_path = os.path.join(met_dir, "by_question_type.csv")
    with open(by_type_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Method", "QuestionType", "N",
                    f"DocRec@{MAIN_K}", f"PageRec@{MAIN_K}",
                    f"BLEU@{MAIN_K}", f"ROUGE-L@{MAIN_K}",
                    "AnsROUGE-L", "NumericMatch"])
        for name, label in VARIANT_LABELS.items():
            if name not in by_type_all:
                continue
            for qt, m in by_type_all[name].items():
                w.writerow([
                    label, qt, m.get("n_samples", 0),
                    f"{m.get(f'doc_recall@{MAIN_K}', float('nan')):.3f}",
                    f"{m.get(f'page_recall@{MAIN_K}', float('nan')):.3f}",
                    f"{m.get(f'context_bleu@{MAIN_K}', float('nan')):.3f}",
                    f"{m.get(f'context_rougeL@{MAIN_K}', float('nan')):.3f}",
                    f"{m.get('answer_rougeL', float('nan')):.3f}",
                    f"{m.get('numeric_match', float('nan')):.3f}",
                ])
    logger.info(f"By-type CSV: {by_type_path}")

    # LaTeX
    tex_path = os.path.join(met_dir, "aug_hier_table_k5.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{tabular}{lcccccc}\n\\toprule\n")
        f.write("\\textbf{Method} & \\textbf{DocRec@5} & \\textbf{PageRec@5} "
                "& \\textbf{BLEU@5} & \\textbf{ROUGE-L@5} "
                "& \\textbf{AnsROUGE-L} & \\textbf{NumMatch} \\\\\n\\midrule\n")
        for name, label in VARIANT_LABELS.items():
            if name not in all_results:
                continue
            r = all_results[name]
            f.write(
                f"{label} & "
                f"{r.get('doc_recall@5', 0):.3f} & "
                f"{r.get('page_recall@5', 0):.3f} & "
                f"{r.get('context_bleu@5', 0):.3f} & "
                f"{r.get('context_rougeL@5', 0):.3f} & "
                f"{r.get('answer_rougeL', 0):.3f} & "
                f"{r.get('numeric_match', 0):.3f} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n")
    logger.info(f"LaTeX: {tex_path}")


# ─── 9. Plots ─────────────────────────────────────────────────────────────────

def _save_fig(fig, path: str) -> None:
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"Plot: {path}")


def plot_recall_at_k(aug_results: Dict, orig_results: Optional[Dict],
                     results_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    plots_dir = os.path.join(results_dir, "plots")

    # Build combined label→results mapping (aug + orig side by side)
    combined: Dict[str, Tuple[str, Dict, str]] = {}
    colors = cm.Set1(np.linspace(0, 1, len(VARIANT_LABELS) * 2))
    ci = 0
    for name, label in VARIANT_LABELS.items():
        if name in aug_results:
            combined[f"aug_{name}"] = (f"Aug: {label.split(': ')[1]}", aug_results[name], colors[ci])
            ci += 1
    if orig_results:
        for name, label in ORIG_VARIANT_LABELS.items():
            if name in orig_results:
                combined[f"orig_{name}"] = (label, orig_results[name], colors[ci])
                ci += 1

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric_prefix, ylabel in [
        (axes[0], "page_recall", "PageRec@k"),
        (axes[1], "doc_recall",  "DocRec@k"),
    ]:
        for key, (label, res, color) in combined.items():
            ls = "-" if key.startswith("aug") else "--"
            vals = [res.get(f"{metric_prefix}@{k}", 0) for k in K_VALUES]
            ax.plot(K_VALUES, vals, marker="o", label=label, color=color,
                    linewidth=2, linestyle=ls)
        ax.set_xlabel("k")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.set_xticks(K_VALUES)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Augmented vs Original FT — Recall Curves", fontsize=11)
    fig.tight_layout()
    _save_fig(fig, os.path.join(plots_dir, "recall_at_k_curves.pdf"))


def plot_bar_k5(aug_results: Dict, orig_results: Optional[Dict],
                results_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = os.path.join(results_dir, "plots")

    # Build rows: label, result_dict, hatching
    rows = []
    for name, label in VARIANT_LABELS.items():
        if name in aug_results:
            rows.append((label, aug_results[name], ""))
    if orig_results:
        for name, label in ORIG_VARIANT_LABELS.items():
            if name in orig_results:
                rows.append((label, orig_results[name], "//"))

    if not rows:
        return

    metric_map = {
        f"doc_recall@{MAIN_K}":     "DocRec@5",
        f"page_recall@{MAIN_K}":    "PageRec@5",
        f"context_bleu@{MAIN_K}":   "BLEU@5",
        f"context_rougeL@{MAIN_K}": "ROUGE-L@5",
    }

    x      = np.arange(len(rows))
    width  = 0.20
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 2.5), 5))
    for i, (mk, ml) in enumerate(metric_map.items()):
        vals    = [r[1].get(mk, 0) for r in rows]
        hatches = [r[2] for r in rows]
        bars = ax.bar(x + (i - 1.5) * width, vals, width,
                      label=ml, color=colors[i], alpha=0.85)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)

    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title(f"Augmented vs Original FT @ k={MAIN_K}\n(hatched = original FT)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, os.path.join(plots_dir, "retrieval_bar_k5.pdf"))


def plot_heatmap_by_type(aug_by_type: Dict, orig_by_type: Optional[Dict],
                         results_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir  = os.path.join(results_dir, "plots")
    types      = QUESTION_TYPES
    type_labels = ["Metrics", "Domain", "Novel"]

    # Collect method rows (aug + orig)
    method_rows = []
    for name, label in VARIANT_LABELS.items():
        if name in aug_by_type:
            method_rows.append((f"Aug: {label.split(': ')[1]}", aug_by_type[name]))
    if orig_by_type:
        for name, label in ORIG_VARIANT_LABELS.items():
            if name in orig_by_type:
                method_rows.append((f"Orig: {label.split(': ')[1]}", orig_by_type[name]))

    if not method_rows:
        return

    matrix = np.zeros((len(method_rows), len(types)))
    for i, (_, by_type) in enumerate(method_rows):
        for j, qt in enumerate(types):
            matrix[i, j] = by_type.get(qt, {}).get(f"page_recall@{MAIN_K}", 0)

    fig, ax = plt.subplots(figsize=(9, max(3, len(method_rows) * 0.9)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels(type_labels, fontsize=10)
    ax.set_yticks(range(len(method_rows)))
    ax.set_yticklabels([r[0] for r in method_rows], fontsize=9)
    for i in range(len(method_rows)):
        for j in range(len(types)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=10,
                    color="black" if matrix[i, j] < 0.6 else "white")
    plt.colorbar(im, ax=ax, label="PageRec@5")
    ax.set_title(f"PageRec@{MAIN_K} by Question Type  (Aug vs Original FT)")
    fig.tight_layout()
    _save_fig(fig, os.path.join(plots_dir, "heatmap_by_question_type.pdf"))


def plot_improvement_by_type(aug_by_type: Dict, orig_by_type: Dict,
                              results_dir: str) -> None:
    """Bar chart showing per-question-type improvement of Aug over Original FT."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir   = os.path.join(results_dir, "plots")
    types       = QUESTION_TYPES
    type_labels = ["Metrics", "Domain-Relevant", "Novel-Generated"]

    # Use the best aug variant vs best orig variant (hier_ft_page_chunk)
    aug_name  = "hier_ft_page_chunk"
    orig_name = "hier_ft_page_chunk"

    if aug_name not in aug_by_type or orig_name not in orig_by_type:
        logger.warning("Cannot make improvement plot — missing results for hier_ft_page_chunk")
        return

    aug_by_qt  = aug_by_type[aug_name]
    orig_by_qt = orig_by_type[orig_name]

    metrics = [
        (f"page_recall@{MAIN_K}",    "PageRec@5"),
        (f"doc_recall@{MAIN_K}",     "DocRec@5"),
        (f"context_rougeL@{MAIN_K}", "ROUGE-L@5"),
    ]

    x     = np.arange(len(types))
    width = 0.25
    colors = ["#1565C0", "#2E7D32", "#C62828"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5), sharey=False)

    for ax, (mk, ml) in zip(axes, metrics):
        aug_vals  = [aug_by_qt.get(qt,  {}).get(mk, 0) for qt in types]
        orig_vals = [orig_by_qt.get(qt, {}).get(mk, 0) for qt in types]
        deltas    = [a - o for a, o in zip(aug_vals, orig_vals)]

        bar_colors = ["#4CAF50" if d >= 0 else "#F44336" for d in deltas]
        bars = ax.bar(x, deltas, color=bar_colors, alpha=0.85, edgecolor="white")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(type_labels, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel(f"Δ {ml}")
        ax.set_title(f"Improvement in {ml}")
        ax.grid(axis="y", alpha=0.3)
        for bar, delta in zip(bars, deltas):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.005 if delta >= 0 else -0.015),
                f"{delta:+.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

    fig.suptitle("Augmented FT vs Original FT — Improvement per Question Type\n"
                 "(variant: Hier FT Page → Chunk)", fontsize=11)
    fig.tight_layout()
    _save_fig(fig, os.path.join(plots_dir, "improvement_by_question_type.pdf"))


# ─── 10. Main ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Augmented Fine-tuning + Hierarchical RAG")
    p.add_argument("--skip-train",   action="store_true",
                   help="Skip training; use existing aug model")
    p.add_argument("--force-train",  action="store_true",
                   help="Re-train even if aug model exists")
    p.add_argument("--variants",     nargs="+",
                   default=list(VARIANT_LABELS.keys()),
                   choices=list(VARIANT_LABELS.keys()))
    p.add_argument("--ft-model",     default=AUG_FT_MODEL_PATH,
                   help="Path to augmented fine-tuned model")
    p.add_argument("--results-dir",  default=RESULTS_DIR)
    p.add_argument("--vs-dir",       default=VS_DIR)
    p.add_argument("--orig-results", default=ORIG_RESULTS_DIR,
                   help="Path to original hierarchical_rag results for comparison")
    p.add_argument("--k",  type=int, default=MAIN_K)
    p.add_argument("--M",  type=int, default=M_PAGES)
    p.add_argument("--N",  type=int, default=N_BM25)
    return p.parse_args()


def main():
    args = parse_args()
    t0   = time.time()

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.vs_dir, exist_ok=True)

    ft_model_path = args.ft_model

    # ── Phase 0: Augmented training ───────────────────────────────────────────
    if not args.skip_train:
        logger.info("=== Phase 0: Augmented fine-tuning ===")
        train_augmented_model(ft_model_path, force=args.force_train)
    else:
        logger.info(f"=== Phase 0: Skipping training — using {ft_model_path} ===")

    # ── Phase 1: Load data & PDFs ─────────────────────────────────────────────
    logger.info("=== Phase 1: Loading data & PDFs ===")
    samples, doc_info = load_financebench_samples(FB_DATA_PATH, FB_DOC_INFO_PATH)
    doc_pages         = load_all_pages(samples, FB_PDF_DIR)

    # ── Phase 2: Build indices ────────────────────────────────────────────────
    logger.info("=== Phase 2: Building indices ===")
    bm25_index, bm25_pages, bm25_tokenize = build_bm25_index(doc_pages)

    collection_name = "aug_hier_ft_pages_v1"
    page_collection = build_page_chroma_index(
        doc_pages=doc_pages,
        model_path=ft_model_path,
        vs_dir=args.vs_dir,
        collection_name=collection_name,
    )
    embed_model = load_embed_model(ft_model_path)

    # ── Phase 3: Retrieval ────────────────────────────────────────────────────
    logger.info("=== Phase 3: Retrieval ===")
    all_results:  Dict[str, Dict] = {}
    by_type_all:  Dict[str, Dict] = {}

    for variant in args.variants:
        logger.info(f"--- Running variant: {variant} ---")
        var_samples = copy.deepcopy(samples)

        if variant == "hier_ft_page_chunk":
            for s in tqdm(var_samples, desc=variant):
                s["retrieved_chunks"] = retrieve_hier_ft_page_chunk(
                    s, embed_model, page_collection, doc_pages,
                    M=args.M, k=args.k,
                )
        elif variant == "hier_bm25_ft_rerank_chunk":
            for s in tqdm(var_samples, desc=variant):
                s["retrieved_chunks"] = retrieve_hier_bm25_ft_rerank_chunk(
                    s, embed_model, bm25_index, bm25_pages, bm25_tokenize, doc_pages,
                    N=args.N, M=args.M, k=args.k,
                )

        save_predictions(var_samples, variant, args.results_dir)

        ret_metrics = compute_retrieval_metrics(var_samples)
        gen_metrics = compute_generative_metrics(var_samples)
        all_results[variant] = {**ret_metrics, **gen_metrics}
        save_metrics(all_results[variant], variant, args.results_dir)

        by_type = aggregate_by_group(
            var_samples,
            key_fn=lambda s: s["question_type"],
        )
        by_type_all[variant] = by_type

        pr5 = all_results[variant].get(f"page_recall@{MAIN_K}", 0)
        dr5 = all_results[variant].get(f"doc_recall@{MAIN_K}", 0)
        logger.info(
            f"{VARIANT_LABELS[variant]}: "
            f"page_recall@{MAIN_K}={pr5:.3f}  doc_recall@{MAIN_K}={dr5:.3f}"
        )
        for qt in QUESTION_TYPES:
            pr = by_type.get(qt, {}).get(f"page_recall@{MAIN_K}", 0)
            logger.info(f"  [{qt}] PageRec@{MAIN_K}={pr:.3f}")

    # ── Phase 4: Load original results for comparison ─────────────────────────
    orig_results:  Optional[Dict] = None
    orig_by_type:  Optional[Dict] = None
    orig_metrics_path = os.path.join(args.orig_results, "metrics", "all_variants_metrics.json")
    orig_by_type_path  = os.path.join(args.orig_results, "metrics", "by_question_type.csv")

    if os.path.exists(orig_metrics_path):
        with open(orig_metrics_path) as f:
            orig_results = json.load(f)
        logger.info(f"Loaded original results from {orig_metrics_path}")

    if os.path.exists(orig_by_type_path):
        # Parse the by_question_type CSV from original run
        orig_by_type = defaultdict(lambda: defaultdict(dict))
        with open(orig_by_type_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Map original method labels back to internal variant names
                method_key = None
                if "BM25" in row["Method"]:
                    method_key = "hier_bm25_ft_rerank_chunk"
                elif "FT Page" in row["Method"] or "Page" in row["Method"]:
                    method_key = "hier_ft_page_chunk"
                if method_key is None:
                    continue
                qt = row["QuestionType"]
                orig_by_type[method_key][qt] = {
                    f"page_recall@{MAIN_K}":    float(row.get(f"PageRec@{MAIN_K}", 0)),
                    f"doc_recall@{MAIN_K}":     float(row.get(f"DocRec@{MAIN_K}", 0)),
                    f"context_bleu@{MAIN_K}":   float(row.get(f"BLEU@{MAIN_K}", 0)),
                    f"context_rougeL@{MAIN_K}": float(row.get(f"ROUGE-L@{MAIN_K}", 0)),
                    "answer_rougeL":             float(row.get("AnsROUGE-L", 0)),
                    "n_samples":                 int(row.get("N", 0)),
                }
        orig_by_type = dict(orig_by_type)
        logger.info("Parsed original by-type results for comparison plots")

    # ── Phase 5: Tables & plots ───────────────────────────────────────────────
    logger.info("=== Phase 5: Saving tables & plots ===")
    save_tables(all_results, by_type_all, args.results_dir)
    plot_recall_at_k(all_results, orig_results, args.results_dir)
    plot_bar_k5(all_results, orig_results, args.results_dir)
    plot_heatmap_by_type(by_type_all, orig_by_type, args.results_dir)
    if orig_by_type:
        plot_improvement_by_type(by_type_all, orig_by_type, args.results_dir)

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    logger.info(f"Done in {elapsed / 60:.1f} min")

    print("\n" + "=" * 70)
    print("AUGMENTED HIERARCHICAL RAG — RESULTS SUMMARY")
    print("=" * 70)
    hdr = f"{'Method':<35} {'DocRec@5':>9} {'PageRec@5':>9} {'BLEU@5':>7} {'ROUGE-L@5':>9}"
    print(hdr)
    print("-" * 70)
    for name, label in VARIANT_LABELS.items():
        if name not in all_results:
            continue
        r = all_results[name]
        print(
            f"{label:<35} "
            f"{r.get('doc_recall@5', 0):>9.3f} "
            f"{r.get('page_recall@5', 0):>9.3f} "
            f"{r.get('context_bleu@5', 0):>7.3f} "
            f"{r.get('context_rougeL@5', 0):>9.3f}"
        )

    if orig_results:
        print("\n--- Original FT (for comparison) ---")
        for name, label in ORIG_VARIANT_LABELS.items():
            if name not in orig_results:
                continue
            r = orig_results[name]
            print(
                f"{label:<35} "
                f"{r.get('doc_recall@5', 0):>9.3f} "
                f"{r.get('page_recall@5', 0):>9.3f} "
                f"{r.get('context_bleu@5', 0):>7.3f} "
                f"{r.get('context_rougeL@5', 0):>9.3f}"
            )

    print("\n--- Per question type (Aug Hier FT Page → Chunk) ---")
    aug_name = "hier_ft_page_chunk"
    if aug_name in by_type_all:
        for qt in QUESTION_TYPES:
            pr = by_type_all[aug_name].get(qt, {}).get(f"page_recall@{MAIN_K}", 0)
            orig_pr = orig_by_type.get(aug_name, {}).get(qt, {}).get(f"page_recall@{MAIN_K}", 0) if orig_by_type else None
            if orig_pr is not None:
                print(f"  {qt:<22}: PageRec@5 = {pr:.3f}  (orig: {orig_pr:.3f},  Δ={pr-orig_pr:+.3f})")
            else:
                print(f"  {qt:<22}: PageRec@5 = {pr:.3f}")

    print(f"\nResults saved in: {args.results_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
