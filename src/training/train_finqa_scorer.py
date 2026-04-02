"""
src/training/train_finqa_scorer.py

Train an embedding model on FinQA financial QA data, then evaluate on:
  1. FinQA test set   – in-document passage retrieval (Hit@K, Recall@K, MRR)
  2. FinanceBench 150 – PDF page retrieval using the same trained model

Training strategy mirrors train_k_fold.py:
  - Use MultipleNegativesRankingLoss (MNRL)
  - Positive = gold supporting passages from qa.gold_inds
  - In-batch negatives = other passages in the same batch

FinQA passage layout per entry
  text_N  -> pre_text[0..] then post_text[0..], merged and indexed sequentially
  table_N -> table row N rendered as "col1 | col2 | ..."

NO MODEL WEIGHTS ARE SAVED.
  The trained model lives only in GPU memory for the duration of the job.
  Every run retrains from scratch. Only lightweight artefacts are persisted:
    $SCRATCH/finqa_scorer/<timestamp>/config.json
    $SCRATCH/finqa_scorer/<timestamp>/final_results.json
    $SCRATCH/finqa_scorer/<timestamp>/summary.txt
    $SCRATCH/finqa_scorer/<timestamp>/training_curves.png

Usage (always via sbatch – see scripts/run_finqa_scorer.sh)
-----
sbatch scripts/run_finqa_scorer.sh

Or for a quick interactive test:
  python -m src.training.train_finqa_scorer \
      --finqa-dir finqa \
      --pdf-dir pdfs \
      --base-model sentence-transformers/all-mpnet-base-v2 \
      --epochs 10 \
      --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses

# Local imports (mirror train_k_fold.py)
from src.ingestion.data_loader import FinanceBenchLoader
from src.ingestion.page_processor import extract_pages_from_pdf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150


# =============================================================================
# Scratch directory helper
# =============================================================================


def _resolve_output_dir(config_scratch_dir: str, timestamp: str) -> Path:
    """
    Resolve where lightweight run artefacts (JSON, plots) are written.

    Priority:
      1. config.scratch_dir  if explicitly set via --scratch-dir
      2. $SCRATCH env var    set automatically by SLURM on most clusters
      3. ./scratch           local fallback for interactive runs

    Model weights are NEVER written anywhere.
    """
    if config_scratch_dir:
        base = Path(config_scratch_dir)
    elif "SCRATCH" in os.environ:
        base = Path(os.environ["SCRATCH"]) / "finqa_scorer"
    else:
        base = Path("scratch") / "finqa_scorer"

    out = base / timestamp
    out.mkdir(parents=True, exist_ok=True)
    return out


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class FinQATrainConfig:
    """All hyper-parameters for FinQA-based training."""

    # Data paths
    finqa_dir: str = "finqa"          # Contains train.json, dev.json, test.json
    pdf_dir: str = "pdfs"             # FinanceBench PDFs

    # Model
    base_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    max_seq_length: int = 512

    # Training
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    eval_every_n_epochs: int = 2
    patience: int = 3
    examples_per_question: int = 1     # Positive pairs per QA item

    # FinQA passage settings
    max_passage_chars: int = 1000      # Truncate each passage to this length
    max_table_cols: int = 20           # Max table columns to render

    # Page scorer settings (for FinanceBench eval)
    max_page_chars: int = 2000
    page_k: int = 5                    # Top-K pages for FB eval

    # Retrieval eval (FinQA)
    finqa_top_k: int = 5               # Top-K passages for FinQA retrieval eval

    # Output — lightweight artefacts only (NO model weights)
    # Uses $SCRATCH env var if set, otherwise falls back to ./scratch
    scratch_dir: str = ""              # Resolved at runtime; see _resolve_scratch_dir()
    random_seed: int = 42


# =============================================================================
# FinQA Data Model
# =============================================================================


@dataclass
class FinQAPassage:
    """One retrievable passage from a FinQA entry."""
    entry_id: str
    passage_key: str    # e.g. "text_0", "table_2"
    text: str           # Rendered text


@dataclass
class FinQAEntry:
    """Parsed FinQA entry ready for training / eval."""
    entry_id: str
    question: str
    all_passages: List[FinQAPassage]   # All candidate passages
    gold_keys: Set[str]               # e.g. {"text_0", "table_1"}
    gold_passages: List[FinQAPassage]  # Passage objects for gold_keys
    exe_ans: Optional[str] = None


# =============================================================================
# FinQA Loading & Parsing
# =============================================================================


def _table_row_to_str(row: List[Any], max_cols: int = 20) -> str:
    """Render one table row as a pipe-delimited string."""
    cells = [str(c).strip() for c in row[:max_cols]]
    return " | ".join(cells)


def _normalize_passage(text: str, max_chars: int = 1000) -> str:
    """Light normalisation + truncation for a passage string."""
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text[:max_chars]


def parse_finqa_entry(
    raw: Dict[str, Any],
    max_passage_chars: int = 1000,
    max_table_cols: int = 20,
) -> Optional[FinQAEntry]:
    """
    Convert one raw FinQA JSON entry into a FinQAEntry.

    Passage indexing:
      text_0 .. text_{len(pre_text)-1}         -> pre_text items
      text_{len(pre_text)} .. text_{N-1}        -> post_text items
      table_0 .. table_{len(table)-1}           -> table rows
    """
    entry_id = raw.get("id", "")
    pre_text: List[str] = raw.get("pre_text", [])
    post_text: List[str] = raw.get("post_text", [])
    table: List[List[Any]] = raw.get("table", [])
    qa: Dict[str, Any] = raw.get("qa", {})

    question = qa.get("question", "").strip()
    if not question:
        return None

    # ---- Build all passages ----
    all_passages: List[FinQAPassage] = []

    # Text passages: pre then post, unified text_N index
    all_texts = pre_text + post_text
    for i, txt in enumerate(all_texts):
        norm = _normalize_passage(txt, max_passage_chars)
        if norm:
            all_passages.append(
                FinQAPassage(
                    entry_id=entry_id,
                    passage_key=f"text_{i}",
                    text=norm,
                )
            )

    # Table row passages
    for i, row in enumerate(table):
        row_str = _table_row_to_str(row, max_table_cols)
        norm = _normalize_passage(row_str, max_passage_chars)
        if norm:
            all_passages.append(
                FinQAPassage(
                    entry_id=entry_id,
                    passage_key=f"table_{i}",
                    text=norm,
                )
            )

    if not all_passages:
        return None

    # ---- Identify gold passages ----
    gold_inds: Dict[str, str] = qa.get("gold_inds", {})
    gold_keys: Set[str] = set(gold_inds.keys())

    passage_map = {p.passage_key: p for p in all_passages}
    gold_passages = [passage_map[k] for k in gold_keys if k in passage_map]

    # If gold_inds refers to keys we couldn't resolve, skip entry
    if not gold_passages:
        logger.debug(f"Entry {entry_id}: no resolvable gold passages (gold_inds={gold_inds})")
        return None

    return FinQAEntry(
        entry_id=entry_id,
        question=question,
        all_passages=all_passages,
        gold_keys=gold_keys,
        gold_passages=gold_passages,
        exe_ans=str(qa.get("exe_ans", "")),
    )


def load_finqa_file(path: Path, config: FinQATrainConfig) -> List[FinQAEntry]:
    """Load and parse a FinQA JSON file (train / dev / test)."""
    logger.info(f"Loading FinQA data from {path} ...")
    with open(path, "r", encoding="utf-8") as f:
        raw_entries = json.load(f)

    entries: List[FinQAEntry] = []
    skipped = 0
    for raw in raw_entries:
        entry = parse_finqa_entry(
            raw,
            max_passage_chars=config.max_passage_chars,
            max_table_cols=config.max_table_cols,
        )
        if entry is not None:
            entries.append(entry)
        else:
            skipped += 1

    logger.info(
        f"  Parsed {len(entries)} valid entries, skipped {skipped} "
        f"from {path.name}"
    )
    return entries


# =============================================================================
# Training Example Construction
# =============================================================================


def create_finqa_training_examples(
    entries: List[FinQAEntry],
    examples_per_question: int = 1,
    seed: int = 42,
) -> List[InputExample]:
    """
    Build MNRL InputExamples: (question, positive_passage).

    For each QA entry we emit `examples_per_question` pairs,
    each using a randomly chosen gold passage as the positive.
    In-batch passages from other entries act as negatives automatically.
    """
    random.seed(seed)
    train_examples: List[InputExample] = []
    skipped = 0

    for entry in entries:
        if not entry.gold_passages:
            skipped += 1
            continue

        for _ in range(examples_per_question):
            pos = random.choice(entry.gold_passages)
            train_examples.append(
                InputExample(texts=[entry.question, pos.text])
            )

    logger.info(
        f"Created {len(train_examples)} training pairs "
        f"({skipped} entries skipped)"
    )
    return train_examples


# =============================================================================
# FinanceBench Page Record (mirrors train_k_fold.py)
# =============================================================================


@dataclass
class PageRecord:
    """Represents a single PDF page from FinanceBench."""

    doc_name: str
    page_num: int
    page_text: str
    page_id: str = field(init=False)

    def __post_init__(self):
        self.page_id = f"{self.doc_name}_p{self.page_num}"
        self.page_text = self._normalize(self.page_text)

    @staticmethod
    def _normalize(text: str, max_chars: int = 2000) -> str:
        if not text:
            return " "
        text = text.replace("\x00", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()[:max_chars] or " "


def build_page_records(
    pdf_dir: Path,
    doc_names: Set[str],
    max_chars: int = 2000,
) -> Dict[str, List[PageRecord]]:
    """Extract pages from PDFs and wrap in PageRecord objects."""
    pages_by_doc: Dict[str, List[PageRecord]] = {}

    for doc_name in tqdm(doc_names, desc="Loading PDFs"):
        pdf_path = pdf_dir / f"{doc_name}.pdf"
        if not pdf_path.exists():
            logger.warning(f"PDF not found: {pdf_path}")
            continue
        try:
            raw_pages = extract_pages_from_pdf(pdf_path, doc_name)
            pages_by_doc[doc_name] = [
                PageRecord(
                    doc_name=doc_name,
                    page_num=p["page"],
                    page_text=p["text"],
                )
                for p in raw_pages
            ]
        except Exception as exc:
            logger.warning(f"Failed to load {pdf_path}: {exc}")

    total = sum(len(v) for v in pages_by_doc.values())
    logger.info(f"Loaded {total} pages from {len(pages_by_doc)} documents")
    return pages_by_doc


# =============================================================================
# Evaluation – FinQA test set
# =============================================================================


def evaluate_finqa_retrieval(
    model: SentenceTransformer,
    entries: List[FinQAEntry],
    top_k: int = 5,
    split_name: str = "test",
) -> Dict[str, float]:
    """
    For each FinQA entry, embed all its passages and retrieve top-K.
    Metrics: Hit@K (any gold in top-K), Recall@K (fraction of golds in top-K), MRR.

    This is *in-document* retrieval: we only rank passages within the same entry.
    """
    model.eval()

    hit_list: List[float] = []
    recall_list: List[float] = []
    mrr_list: List[float] = []

    skipped = 0
    for entry in tqdm(entries, desc=f"Eval FinQA {split_name}"):
        if not entry.gold_passages or len(entry.all_passages) < 2:
            skipped += 1
            continue

        passage_texts = [p.text for p in entry.all_passages]
        passage_keys = [p.passage_key for p in entry.all_passages]

        # Embed
        with torch.no_grad():
            q_emb = model.encode(entry.question, convert_to_tensor=True)
            p_embs = model.encode(passage_texts, convert_to_tensor=True)

        scores = torch.nn.functional.cosine_similarity(q_emb.unsqueeze(0), p_embs)
        ranked_indices = torch.argsort(scores, descending=True).cpu().tolist()

        top_k_keys = {passage_keys[i] for i in ranked_indices[:top_k]}
        gold_keys = entry.gold_keys & set(passage_keys)  # only keys that resolved

        if not gold_keys:
            skipped += 1
            continue

        # Hit@K
        hit = 1.0 if top_k_keys & gold_keys else 0.0
        hit_list.append(hit)

        # Recall@K
        recall = len(top_k_keys & gold_keys) / len(gold_keys)
        recall_list.append(recall)

        # MRR (rank of first gold passage)
        first_gold_rank = None
        for rank, idx in enumerate(ranked_indices, start=1):
            if passage_keys[idx] in gold_keys:
                first_gold_rank = rank
                break
        mrr_list.append(1.0 / first_gold_rank if first_gold_rank else 0.0)

    n = len(hit_list)
    logger.info(
        f"FinQA {split_name}: evaluated {n} entries, skipped {skipped}"
    )

    if n == 0:
        return {f"finqa_{split_name}_hit@{top_k}": 0.0,
                f"finqa_{split_name}_recall@{top_k}": 0.0,
                f"finqa_{split_name}_mrr": 0.0,
                "n_evaluated": 0}

    return {
        f"finqa_{split_name}_hit@{top_k}": float(np.mean(hit_list)),
        f"finqa_{split_name}_recall@{top_k}": float(np.mean(recall_list)),
        f"finqa_{split_name}_mrr": float(np.mean(mrr_list)),
        "n_evaluated": n,
    }


# =============================================================================
# Evaluation – FinanceBench 150
# =============================================================================


# =============================================================================
# Evaluation – FinanceBench 150
# =============================================================================


def _fb_row_to_gold_segments(row) -> List[Dict[str, Any]]:
    """Extract gold evidence segments from a FinanceBench DataFrame row.

    Fixes applied vs original:
      1. HuggingFace `to_pandas()` stores list-of-dict columns as numpy object
         arrays, not Python lists → `isinstance(x, list)` was always False and
         silently zeroed out every evidence list.  Now we coerce any iterable
         to a plain Python list before proceeding.
      2. Falsy-zero bug: `ev.get("evidence_page_num") or ...` evaluates page 0
         as falsy and falls through to None.  Replaced with explicit None checks.
    """
    evidence_list = row.get("evidence", [])
    # Coerce numpy arrays / other iterables produced by to_pandas()
    if evidence_list is None:
        evidence_list = []
    elif not isinstance(evidence_list, list):
        try:
            evidence_list = list(evidence_list)
        except Exception:
            evidence_list = []

    segments = []
    for ev in evidence_list:
        if not hasattr(ev, "get"):
            # Guard against non-dict items (e.g. scalars in malformed rows)
            continue
        # Fix: use explicit None check so that page 0 is not treated as falsy
        p = ev.get("evidence_page_num")
        if p is None:
            p = ev.get("page_ix")
        if p is None:
            p = ev.get("page")
        segments.append({
            "doc_name": row["doc_name"],
            "page": int(p) if p is not None else None,
            "text": ev.get("evidence_text", "") or "",
        })
    return segments


def evaluate_financebench_retrieval(
    model: SentenceTransformer,
    df,
    pages_by_doc: Dict[str, List[PageRecord]],
    top_k: int = 5,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Per-document retrieval: for each FB question, rank pages within its own PDF.

    Returns
    -------
    predictions : list of sample dicts in the unified all_predictions.json format
    metrics     : output of RetrievalEvaluator.compute_metrics(predictions)
    """
    from src.evaluation.retrieval_evaluator import RetrievalEvaluator

    model.eval()
    predictions: List[Dict[str, Any]] = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Eval FB per-doc"):
        doc_name = row["doc_name"]
        if doc_name not in pages_by_doc:
            continue

        question = str(row.get("question", ""))
        gold_segments = _fb_row_to_gold_segments(row)

        # Filter to pages that actually exist
        valid_gold_pages = {
            seg["page"] for seg in gold_segments
            if seg["page"] is not None and seg["page"] < len(pages_by_doc[doc_name])
        }
        if not valid_gold_pages:
            continue

        doc_pages = pages_by_doc[doc_name]

        with torch.no_grad():
            q_emb = model.encode(question, convert_to_tensor=True)
            page_texts = [p.page_text for p in doc_pages]
            p_embs = model.encode(page_texts, convert_to_tensor=True)

        scores = torch.nn.functional.cosine_similarity(q_emb.unsqueeze(0), p_embs)
        ranked = torch.argsort(scores, descending=True).cpu().tolist()

        # Build retrieved_chunks in unified format (top_k pages as "chunks")
        retrieved_chunks = [
            {
                "text": doc_pages[i].page_text,
                "metadata": {"doc_name": doc_name, "page": doc_pages[i].page_num},
                "score": float(scores[i]),
            }
            for i in ranked[:top_k]
        ]

        context_text = "\n\n".join(c["text"] for c in retrieved_chunks)
        predictions.append({
            "sample_id": idx,
            "doc_name": doc_name,
            "question": question,
            "reference_answer": str(row.get("answer", "")),
            "question_type": str(row.get("question_type", "")),
            "question_reasoning": str(row.get("question_reasoning", "")),
            "gold_evidence": "\n\n".join(s["text"] for s in gold_segments),
            "gold_evidence_segments": gold_segments,
            "retrieved_chunks": retrieved_chunks,
            "num_retrieved": len(retrieved_chunks),
            "context_length": len(context_text),
            "generated_answer": "",   # no generation in this script
            "generation_length": 0,
            "experiment_type": "finqa_scorer_perdoc",
            "retrieval_mode": "per_document",
        })

    evaluator = RetrievalEvaluator()
    metrics = evaluator.compute_metrics(predictions)
    return predictions, metrics


# =============================================================================
# Evaluation – FinanceBench 150, CORPUS-LEVEL (all 84 docs as one pool)
# =============================================================================


@dataclass
class CorpusIndex:
    """
    Pre-computed embedding matrix for every page across all 84 FB documents.

    page_ids[i]  = (doc_name, page_num)  for the i-th row of embeddings
    embeddings   = float32 tensor of shape (N_total_pages, hidden_dim)
                   already L2-normalised for fast cosine via matmul
    """
    page_ids: List[Tuple[str, int]]
    embeddings: torch.Tensor            # (N, D), normalised, on CPU


def build_corpus_index(
    model: SentenceTransformer,
    pages_by_doc: Dict[str, List[PageRecord]],
    batch_size: int = 256,
) -> CorpusIndex:
    """
    Embed every page from every document in pages_by_doc.

    All pages are concatenated into one flat list, embedded in batches,
    then L2-normalised so that cosine similarity reduces to a dot product.
    The result is kept on CPU to avoid OOM when querying 150 questions.
    """
    model.eval()

    page_ids: List[Tuple[str, int]] = []
    all_texts: List[str] = []

    for doc_name, pages in sorted(pages_by_doc.items()):
        for page in pages:
            page_ids.append((doc_name, page.page_num))
            all_texts.append(page.page_text)

    logger.info(
        f"Building corpus index: {len(all_texts)} pages "
        f"from {len(pages_by_doc)} documents ..."
    )

    # Encode in batches, collect on CPU
    all_vecs: List[torch.Tensor] = []
    with torch.no_grad():
        for start in tqdm(
            range(0, len(all_texts), batch_size),
            desc="Encoding corpus pages",
        ):
            batch = all_texts[start : start + batch_size]
            vecs = model.encode(
                batch,
                convert_to_tensor=True,
                show_progress_bar=False,
            ).cpu()
            all_vecs.append(vecs)

    embeddings = torch.cat(all_vecs, dim=0)  # (N, D)

    # L2-normalise so cosine sim = dot product
    norms = embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
    embeddings = embeddings / norms

    logger.info(f"Corpus index built: {embeddings.shape}")
    return CorpusIndex(page_ids=page_ids, embeddings=embeddings)


def evaluate_financebench_corpus_retrieval(
    model: SentenceTransformer,
    df,
    pages_by_doc: Dict[str, List[PageRecord]],
    corpus_index: CorpusIndex,
    top_k: int = 5,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Open-corpus retrieval: rank all pages from all 84 docs simultaneously.

    Returns
    -------
    predictions : list of sample dicts in the unified all_predictions.json format
    metrics     : output of RetrievalEvaluator.compute_metrics(predictions)
    """
    from src.evaluation.retrieval_evaluator import RetrievalEvaluator

    model.eval()
    predictions: List[Dict[str, Any]] = []

    device = next(model.parameters()).device
    corpus_emb = corpus_index.embeddings.to(device)  # (N, D), already normalised

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Eval FB corpus"):
        doc_name = row["doc_name"]
        if doc_name not in pages_by_doc:
            continue

        question = str(row.get("question", ""))
        gold_segments = _fb_row_to_gold_segments(row)

        valid_gold_pairs = {
            (doc_name, seg["page"])
            for seg in gold_segments
            if seg["page"] is not None
               and seg["page"] < len(pages_by_doc[doc_name])
        }
        if not valid_gold_pairs:
            continue

        with torch.no_grad():
            q_emb = model.encode(
                question, convert_to_tensor=True, show_progress_bar=False
            ).to(device)
            q_emb = q_emb / q_emb.norm().clamp(min=1e-8)

        scores = corpus_emb @ q_emb          # (N,)
        ranked = torch.argsort(scores, descending=True).cpu().tolist()

        # Build retrieved_chunks from the global ranked list
        retrieved_chunks = []
        for global_idx in ranked[:top_k]:
            c_doc, c_page = corpus_index.page_ids[global_idx]
            c_text = pages_by_doc[c_doc][c_page].page_text if (
                c_doc in pages_by_doc and c_page < len(pages_by_doc[c_doc])
            ) else ""
            retrieved_chunks.append({
                "text": c_text,
                "metadata": {"doc_name": c_doc, "page": c_page},
                "score": float(scores[global_idx]),
            })

        context_text = "\n\n".join(c["text"] for c in retrieved_chunks)
        predictions.append({
            "sample_id": idx,
            "doc_name": doc_name,
            "question": question,
            "reference_answer": str(row.get("answer", "")),
            "question_type": str(row.get("question_type", "")),
            "question_reasoning": str(row.get("question_reasoning", "")),
            "gold_evidence": "\n\n".join(s["text"] for s in gold_segments),
            "gold_evidence_segments": gold_segments,
            "retrieved_chunks": retrieved_chunks,
            "num_retrieved": len(retrieved_chunks),
            "context_length": len(context_text),
            "generated_answer": "",   # no generation in this script
            "generation_length": 0,
            "experiment_type": "finqa_scorer_corpus",
            "retrieval_mode": "full_corpus",
        })

    evaluator = RetrievalEvaluator()
    metrics = evaluator.compute_metrics(predictions)
    return predictions, metrics


# =============================================================================
# Training Loop
# =============================================================================


def train_model(
    train_examples: List[InputExample],
    config: FinQATrainConfig,
    # For in-training eval on FinQA dev
    dev_entries: Optional[List[FinQAEntry]] = None,
    # For in-training eval on FinanceBench (per-doc)
    fb_df=None,
    fb_pages_by_doc: Optional[Dict[str, List[PageRecord]]] = None,
    # For in-training eval on FinanceBench (full corpus)
    fb_corpus_index: Optional[CorpusIndex] = None,
) -> Tuple[SentenceTransformer, Dict[str, Any]]:
    """
    Train the embedding model on FinQA training pairs.

    Returns trained model + training history dict.
    """
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_seed)

    logger.info("=" * 80)
    logger.info("TRAINING ON FinQA")
    logger.info(f"  Base model : {config.base_model_name}")
    logger.info(f"  Examples   : {len(train_examples)}")
    logger.info(f"  Epochs     : {config.epochs}")
    logger.info(f"  Batch size : {config.batch_size}")
    logger.info(f"  LR         : {config.learning_rate}")
    logger.info("=" * 80)

    model = SentenceTransformer(config.base_model_name)
    model.max_seq_length = config.max_seq_length

    train_dataloader = DataLoader(
        train_examples, shuffle=True, batch_size=config.batch_size
    )
    train_loss = losses.MultipleNegativesRankingLoss(model)

    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)

    logger.info(f"Steps / epoch: {steps_per_epoch}, Total: {total_steps}, Warmup: {warmup_steps}")

    history: Dict[str, Any] = {
        "epoch_eval": [],
        "best_epoch": 0,
        "best_finqa_dev_hit": 0.0,
    }

    best_score = 0.0
    best_state = None
    epochs_no_improve = 0
    start_time = time.time()

    # ---- Baseline (epoch 0) ----
    eval_result = _eval_checkpoint(
        model, dev_entries, fb_df, fb_pages_by_doc, fb_corpus_index, config, epoch=0
    )
    history["epoch_eval"].append(eval_result)
    logger.info(f"Baseline: {eval_result}")

    # ---- Training loop ----
    for epoch in range(config.epochs):
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            warmup_steps=warmup_steps if epoch == 0 else 0,
            show_progress_bar=True,
            output_path=None,
            use_amp=True,
            optimizer_params={"lr": config.learning_rate},
        )

        if (epoch + 1) % config.eval_every_n_epochs == 0 or epoch == config.epochs - 1:
            eval_result = _eval_checkpoint(
                model, dev_entries, fb_df, fb_pages_by_doc, fb_corpus_index, config, epoch=epoch + 1
            )
            history["epoch_eval"].append(eval_result)
            logger.info(f"Epoch {epoch + 1}: {eval_result}")

            # Use FinQA dev Hit@K as the early-stopping signal
            dev_hit_key = f"finqa_dev_hit@{config.finqa_top_k}"
            current_score = eval_result.get(dev_hit_key, 0.0)

            if current_score > best_score:
                best_score = current_score
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                history["best_epoch"] = epoch + 1
                history["best_finqa_dev_hit"] = best_score
                epochs_no_improve = 0
                logger.info(f"  ✓ New best model ({dev_hit_key} = {best_score:.4f})")
            else:
                epochs_no_improve += 1
                logger.info(f"  No improvement for {epochs_no_improve} eval checks")
                if epochs_no_improve >= config.patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info(
            f"Restored best model from epoch {history['best_epoch']} "
            f"({dev_hit_key} = {history['best_finqa_dev_hit']:.4f})"
        )

    history["training_time_seconds"] = time.time() - start_time
    logger.info(f"Training complete in {history['training_time_seconds']:.1f}s")

    return model, history


def _eval_checkpoint(
    model,
    dev_entries,
    fb_df,
    fb_pages_by_doc,
    fb_corpus_index,
    config: FinQATrainConfig,
    epoch: int,
) -> Dict[str, float]:
    """
    Run all evaluations at one training checkpoint and merge results.

    During training we run:
      - FinQA dev   : full RetrievalEvaluator metrics (cheap — small passage pools)
      - FB per-doc  : full RetrievalEvaluator metrics (doc_hit, page_hit, mrr, …)
      - FB corpus   : skipped during training — corpus index must be rebuilt after
                      each weight update to be valid, which is prohibitively expensive.
                      Corpus eval only runs once after training completes.
    """
    result: Dict[str, float] = {"epoch": epoch}

    if dev_entries is not None:
        finqa_metrics = evaluate_finqa_retrieval(
            model, dev_entries, top_k=config.finqa_top_k, split_name="dev"
        )
        result.update(finqa_metrics)

    if fb_df is not None and fb_pages_by_doc is not None:
        # Per-document: returns (predictions, metrics); we only need the metrics here
        _, fb_metrics = evaluate_financebench_retrieval(
            model, fb_df, fb_pages_by_doc, top_k=config.page_k
        )
        # Prefix keys so they don't clash with corpus keys later
        result.update({f"fb_perdoc_{k}": v for k, v in fb_metrics.items()})

    return result


# =============================================================================
# Plotting
# =============================================================================


def plot_training_curves(history: Dict[str, Any], output_dir: Path) -> None:
    """Save training-curve plots."""
    epoch_evals = history.get("epoch_eval", [])
    if not epoch_evals:
        return

    epochs = [e["epoch"] for e in epoch_evals]

    finqa_keys = [k for k in epoch_evals[0] if "finqa_dev_hit" in k or "finqa_dev_recall" in k]
    fb_perdoc_keys = [k for k in epoch_evals[0] if k.startswith("fb_perdoc_page_hit") or k.startswith("fb_perdoc_page_recall") or k.startswith("fb_perdoc_doc_hit")]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.set_title("FinQA Dev Retrieval over Training")
    for key in finqa_keys:
        vals = [e.get(key, 0) for e in epoch_evals]
        ax.plot(epochs, vals, marker="o", label=key)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.legend(fontsize=8)
    ax.grid(True)

    ax = axes[1]
    ax.set_title("FinanceBench 150 — per-document pool")
    for key in fb_perdoc_keys:
        vals = [e.get(key, 0) for e in epoch_evals]
        ax.plot(epochs, vals, marker="o", label=key)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.legend(fontsize=8)
    ax.grid(True)

    plt.tight_layout()
    path = output_dir / "training_curves.png"
    plt.savefig(path)
    plt.close()
    logger.info(f"Training curves saved to {path}")


# =============================================================================
# Main Entry Point
# =============================================================================


def main(config: Optional[FinQATrainConfig] = None) -> Dict[str, Any]:
    if config is None:
        config = FinQATrainConfig()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = _resolve_output_dir(config.scratch_dir, timestamp)

    logger.info("=" * 80)
    logger.info("FinQA Embedding Model Training  (no weights saved)")
    logger.info(f"Artefacts → {output_dir}")
    logger.info("=" * 80)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    # ---- Load FinQA data ----
    finqa_dir = Path(config.finqa_dir)
    train_entries = load_finqa_file(finqa_dir / "train.json", config)
    dev_entries = load_finqa_file(finqa_dir / "dev.json", config)
    test_entries = load_finqa_file(finqa_dir / "test.json", config)

    logger.info(
        f"\nFinQA split sizes: "
        f"train={len(train_entries)}, dev={len(dev_entries)}, test={len(test_entries)}"
    )

    # ---- Load FinanceBench 150 ----
    logger.info("\nLoading FinanceBench 150 ...")
    fb_loader = FinanceBenchLoader()
    fb_df = fb_loader.load_data()
    all_fb_docs: Set[str] = set(fb_df["doc_name"].unique())

    # Diagnostic: verify evidence lists are parseable (catches numpy-array coercion issues)
    n_with_evidence = 0
    n_with_page = 0
    for _, _row in fb_df.head(10).iterrows():
        segs = _fb_row_to_gold_segments(_row)
        if segs:
            n_with_evidence += 1
        if any(s["page"] is not None for s in segs):
            n_with_page += 1
    logger.info(
        f"Evidence diagnostic (first 10 rows): "
        f"{n_with_evidence}/10 have evidence, {n_with_page}/10 have valid page numbers. "
        f"(Expected: 10/10 and 10/10 — if lower, evidence parsing is still broken)"
    )

    pdf_dir = Path(config.pdf_dir)
    fb_pages_by_doc = build_page_records(pdf_dir, all_fb_docs, config.max_page_chars)

    # ---- Build training examples ----
    train_examples = create_finqa_training_examples(
        train_entries,
        examples_per_question=config.examples_per_question,
        seed=config.random_seed,
    )

    # ---- Train (model lives only in memory) ----
    # corpus_index is NOT passed here — it's built after training with the
    # final weights so we don't have to rebuild it at every eval checkpoint.
    model, history = train_model(
        train_examples=train_examples,
        config=config,
        dev_entries=dev_entries,
        fb_df=fb_df,
        fb_pages_by_doc=fb_pages_by_doc,
        fb_corpus_index=None,   # corpus eval only at the end
    )

    # ---- Build corpus index once with the final trained model ----
    logger.info("\n" + "=" * 80)
    logger.info("BUILDING FULL-CORPUS INDEX WITH FINAL MODEL")
    logger.info(f"  Pool: all pages of all {len(fb_pages_by_doc)} FB documents")
    logger.info("=" * 80)
    fb_corpus_index = build_corpus_index(model, fb_pages_by_doc)

    # ---- Final evaluation on FinQA test ----
    logger.info("\n" + "=" * 80)
    logger.info("FINAL EVALUATION ON FinQA TEST SET")
    logger.info("=" * 80)
    test_metrics = evaluate_finqa_retrieval(
        model, test_entries, top_k=config.finqa_top_k, split_name="test"
    )
    logger.info(f"FinQA test: {test_metrics}")

    # ---- Final evaluation on FinanceBench – per-document ----
    logger.info("\n" + "=" * 80)
    logger.info("FINAL EVALUATION: FinanceBench 150 — per-document pool")
    logger.info("=" * 80)
    fb_predictions, fb_metrics = evaluate_financebench_retrieval(
        model, fb_df, fb_pages_by_doc, top_k=config.page_k
    )
    logger.info(f"FinanceBench per-doc: {fb_metrics}")

    # ---- Final evaluation on FinanceBench – full corpus ----
    logger.info("\n" + "=" * 80)
    logger.info("FINAL EVALUATION: FinanceBench 150 — full corpus pool (all 84 docs)")
    logger.info("=" * 80)
    fb_corpus_predictions, fb_corpus_metrics = evaluate_financebench_corpus_retrieval(
        model, fb_df, fb_pages_by_doc, fb_corpus_index, top_k=config.page_k
    )
    logger.info(f"FinanceBench corpus: {fb_corpus_metrics}")

    # Model is no longer needed — free GPU memory explicitly
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Model weights freed from memory.")

    # ---- Save all_predictions.json files (unified format, comparable to other expts) ----
    with open(output_dir / "all_predictions_fb_perdoc.json", "w") as f:
        json.dump(fb_predictions, f, indent=2, default=str)
    logger.info(f"Saved {len(fb_predictions)} per-doc predictions")

    with open(output_dir / "all_predictions_fb_corpus.json", "w") as f:
        json.dump(fb_corpus_predictions, f, indent=2, default=str)
    logger.info(f"Saved {len(fb_corpus_predictions)} corpus predictions")

    # ---- Persist lightweight artefacts only ----
    final_results = {
        "config": asdict(config),
        "timestamp": timestamp,
        "artefacts_dir": str(output_dir),
        "finqa_sizes": {
            "train": len(train_entries),
            "dev": len(dev_entries),
            "test": len(test_entries),
        },
        "n_training_examples": len(train_examples),
        "training_history": history,
        "finqa_test_metrics": test_metrics,
        "financebench_per_doc_metrics": fb_metrics,
        "financebench_corpus_metrics": fb_corpus_metrics,
    }

    with open(output_dir / "final_results.json", "w") as f:
        json.dump(final_results, f, indent=2, default=str)

    # ---- Summary ----
    summary_lines = [
        "=" * 70,
        "FinQA SCORER – FINAL RESULTS",
        "=" * 70,
        f"Base model   : {config.base_model_name}",
        f"Best epoch   : {history.get('best_epoch', '?')} / {config.epochs}",
        f"Training time: {history.get('training_time_seconds', 0):.1f}s",
        f"Artefacts    : {output_dir}",
        "",
        f"FinQA TEST (top-{config.finqa_top_k} in-document retrieval):",
    ]
    for k, v in test_metrics.items():
        if k != "n_evaluated":
            summary_lines.append(f"  {k}: {v:.4f}")
    summary_lines.append(f"  n_evaluated: {test_metrics.get('n_evaluated', 0)}")
    summary_lines.append("")
    summary_lines.append(f"FinanceBench 150 — per-document pool (top-{config.page_k}):")
    for k, v in sorted(fb_metrics.items()):
        if isinstance(v, float):
            summary_lines.append(f"  {k}: {v:.4f}")
        else:
            summary_lines.append(f"  {k}: {v}")
    summary_lines.append("")
    summary_lines.append(
        f"FinanceBench 150 — FULL CORPUS pool (all 84 docs, top-{config.page_k}):"
    )
    for k, v in sorted(fb_corpus_metrics.items()):
        if isinstance(v, float):
            summary_lines.append(f"  {k}: {v:.4f}")
        else:
            summary_lines.append(f"  {k}: {v}")
    summary_lines.append("=" * 70)

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    with open(output_dir / "summary.txt", "w") as f:
        f.write(summary_text)

    # ---- Plots ----
    plot_training_curves(history, output_dir)

    logger.info(f"\nArtefacts saved to: {output_dir}")
    logger.info("(No model weights were written to disk.)")
    return final_results


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Train embedding model on FinQA, eval on FinQA test + FinanceBench 150.\n"
            "NO model weights are saved. Run via sbatch for cluster use."
        )
    )
    parser.add_argument("--finqa-dir", default="finqa",
                        help="Directory containing train.json, dev.json, test.json")
    parser.add_argument("--pdf-dir", default="pdfs",
                        help="Directory with FinanceBench PDFs")
    parser.add_argument("--base-model", default="sentence-transformers/all-mpnet-base-v2",
                        help="Base sentence-transformer model name")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--page-k", type=int, default=5,
                        help="Top-K for FinanceBench page retrieval eval")
    parser.add_argument("--finqa-top-k", type=int, default=5,
                        help="Top-K for FinQA passage retrieval eval")
    parser.add_argument("--scratch-dir", default="",
                        help=(
                            "Where to write lightweight artefacts (JSON, plots). "
                            "Defaults to $SCRATCH/finqa_scorer if $SCRATCH is set, "
                            "else ./scratch/finqa_scorer"
                        ))
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    cfg = FinQATrainConfig(
        finqa_dir=args.finqa_dir,
        pdf_dir=args.pdf_dir,
        base_model_name=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        page_k=args.page_k,
        finqa_top_k=args.finqa_top_k,
        scratch_dir=args.scratch_dir,
        random_seed=args.seed,
    )

    main(cfg)