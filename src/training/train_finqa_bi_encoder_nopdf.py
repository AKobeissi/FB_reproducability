#!/usr/bin/env python3
"""
src/training/train_finqa_bi_encoder_nopdf.py
============================================
Fine-tunes BGE-M3 as a page-level bi-encoder on FinQA train.json
without requiring any PDFs.

Gold passage = pre_text + table + post_text (same signal used by the
fine-tuned cross-encoder reranker).  Negatives = other documents' gold
page texts, sampled/mined from the same training pool.

Training objective: MultipleNegativesRankingLoss (MNR) with LoRA.
  - In-batch negatives: all other samples' gold pages in the same batch
  - Explicit hard negatives: N_hard additional passages per row,
    optionally BM25-mined (--use-bm25-hard-negs)

No same-doc negatives: without PDFs we can't get non-gold pages from
the same document, but cross-doc negatives match the FinanceBench
inference setting (global index across all docs) better anyway.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

try:
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
        losses,
    )
except ImportError:
    logger.error("sentence-transformers not found (need v3.0+)"); sys.exit(1)

try:
    from datasets import Dataset
except ImportError:
    logger.error("datasets not found — pip install datasets"); sys.exit(1)

try:
    from peft import LoraConfig, get_peft_model, TaskType
except ImportError:
    logger.error("peft not found — pip install peft"); sys.exit(1)

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

BGE_MAX_SEQ_LENGTH = 2048

_LORA_BASELINE = ["query", "key", "value"]
_LORA_EXTENDED = ["query", "key", "value", "attention.output.dense", "intermediate.dense"]


# ─────────────────────────────────────────────────────────────────────────────
# 1.  FinQA helpers (same as reranker)
# ─────────────────────────────────────────────────────────────────────────────

def flatten_table(table: List[List]) -> str:
    rows = [" | ".join(str(cell).strip() for cell in row) for row in table]
    return "\n".join(rows)


def build_page_text(sample: Dict) -> str:
    pre  = " ".join(str(t).strip() for t in sample.get("pre_text",  []))
    tbl  = flatten_table(sample.get("table", []))
    post = " ".join(str(t).strip() for t in sample.get("post_text", []))
    return f"{pre}\n{tbl}\n{post}".strip()


def get_doc_key(sample: Dict) -> str:
    """e.g. 'ADI/2009/page_49.pdf' → 'ADI/2009'"""
    return sample.get("filename", "").rsplit("/page_", 1)[0]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Hard negative mining
# ─────────────────────────────────────────────────────────────────────────────

def mine_hard_negatives(
    query: str,
    pool: List[str],
    n: int,
    use_bm25: bool,
    rng: random.Random,
) -> List[str]:
    if not pool:
        return []
    if not use_bm25 or not HAS_BM25:
        return rng.sample(pool, min(n, len(pool)))
    tokenized = [p.lower().split() for p in pool]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(range(len(pool)), key=lambda i: -scores[i])
    return [pool[i] for i in ranked[:n]]


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Dataset builder
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(
    train_json: Path,
    hard_negs_per_positive: int,
    use_bm25: bool,
    min_passage_len: int,
    seed: int,
    query_prefix: str,
) -> Dataset:
    rng = random.Random(seed)
    np.random.seed(seed)

    logger.info(f"Loading {train_json} ...")
    samples = json.load(open(train_json, encoding="utf-8"))
    logger.info(f"  {len(samples)} entries loaded")

    # Build per-sample gold texts upfront
    all_texts: List[str] = [build_page_text(s) for s in samples]
    all_docs:  List[str] = [get_doc_key(s) for s in samples]
    all_qs:    List[str] = [s["qa"]["question"].strip() for s in samples]

    # Filter unusable entries
    valid_idx = [
        i for i in range(len(samples))
        if all_qs[i] and len(all_texts[i]) >= min_passage_len
    ]
    logger.info(f"  {len(valid_idx)} usable entries after length filter")

    # For BM25 mining we need a cross-doc pool per query; for random we just sample.
    # Build a doc → list-of-indices map so we can exclude same-doc entries.
    doc_to_indices: Dict[str, List[int]] = {}
    for i in valid_idx:
        doc_to_indices.setdefault(all_docs[i], []).append(i)

    rows: List[List[str]] = []
    n_cols = 2 + hard_negs_per_positive  # query, gold, neg_1..N

    for i in valid_idx:
        query    = f"{query_prefix}{all_qs[i]}" if query_prefix else all_qs[i]
        gold     = all_texts[i]
        src_doc  = all_docs[i]

        # Negative pool: other documents' gold passages
        other_idx = [j for j in valid_idx if all_docs[j] != src_doc]
        neg_pool  = [all_texts[j] for j in other_idx]

        negs = mine_hard_negatives(query, neg_pool, hard_negs_per_positive, use_bm25, rng)
        # Pad to exactly hard_negs_per_positive (edge case: too few other docs)
        while len(negs) < hard_negs_per_positive:
            negs.append("")

        row = [query, gold] + negs[:hard_negs_per_positive]
        rows.append(row)

    logger.info(f"  Training rows built: {len(rows)}")
    logger.info(f"  Negatives per row  : {hard_negs_per_positive} "
                f"({'BM25-mined' if use_bm25 and HAS_BM25 else 'random'})")
    logger.info(f"  Dataset columns    : {n_cols}")

    data_dict = {f"sentence_{i}": [r[i] for r in rows] for i in range(n_cols)}
    return Dataset.from_dict(data_dict)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  LoRA
# ─────────────────────────────────────────────────────────────────────────────

def apply_lora(
    model: SentenceTransformer,
    r: int,
    lora_alpha: int,
    lora_dropout: float,
    extended: bool,
) -> SentenceTransformer:
    target_modules = _LORA_EXTENDED if extended else _LORA_BASELINE
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    hf_model = model[0].auto_model
    peft_model = get_peft_model(hf_model, lora_config)
    peft_model.enable_input_require_grads()
    model[0].auto_model = peft_model
    trainable, total = peft_model.get_nb_trainable_parameters()
    logger.info(
        f"LoRA applied — trainable: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)"
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Train
# ─────────────────────────────────────────────────────────────────────────────

def train(
    train_json:           Path,
    output_path:          Path,
    base_model:           str   = "BAAI/bge-m3",
    epochs:               int   = 10,
    batch_size:           int   = 8,
    grad_accumulation:    int   = 4,
    lr:                   float = 2e-5,
    warmup_frac:          float = 0.1,
    lora_r:               int   = 16,
    lora_alpha:           int   = 32,
    lora_dropout:         float = 0.1,
    hard_negatives:       int   = 3,
    loss_scale:           float = 50.0,
    use_bm25_hard_negs:   bool  = False,
    extended_lora:        bool  = False,
    min_passage_len:      int   = 50,
    seed:                 int   = 42,
    query_prefix:         str   = "",
):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    logger.info(f"\n{'='*60}")
    logger.info(f"Training config (no PDF):")
    logger.info(f"  train_json        : {train_json}")
    logger.info(f"  base_model        : {base_model}")
    logger.info(f"  hard_negatives    : {hard_negatives} ({'BM25' if use_bm25_hard_negs else 'random'})")
    logger.info(f"  loss_scale        : {loss_scale}")
    logger.info(f"  batch_size        : {batch_size} × {grad_accumulation} = {batch_size*grad_accumulation} effective")
    logger.info(f"{'='*60}")

    logger.info(f"Loading base model: {base_model} ...")
    model = SentenceTransformer(base_model, device=device)
    model.max_seq_length = BGE_MAX_SEQ_LENGTH

    model = apply_lora(model, r=lora_r, lora_alpha=lora_alpha,
                       lora_dropout=lora_dropout, extended=extended_lora)

    train_dataset = build_dataset(
        train_json,
        hard_negs_per_positive=hard_negatives,
        use_bm25=use_bm25_hard_negs,
        min_passage_len=min_passage_len,
        seed=seed,
        query_prefix=query_prefix,
    )

    train_loss = losses.MultipleNegativesRankingLoss(model, scale=loss_scale)

    n_examples  = len(train_dataset)
    total_steps = max(1, n_examples // batch_size) * epochs
    warmup_steps = int(total_steps * warmup_frac)

    logger.info(f"\nTraining: {n_examples} examples, {total_steps} total steps, "
                f"{warmup_steps} warmup, lr={lr}")

    checkpoint_dir = output_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    training_args = SentenceTransformerTrainingArguments(
        output_dir                  = str(checkpoint_dir),
        num_train_epochs            = epochs,
        per_device_train_batch_size = batch_size,
        gradient_accumulation_steps = grad_accumulation,
        learning_rate               = lr,
        warmup_steps                = warmup_steps,
        fp16                        = (device == "cuda"),
        logging_steps               = 10,
        save_strategy               = "no",
        seed                        = seed,
        dataloader_drop_last        = False,
        remove_unused_columns       = False,
        gradient_checkpointing      = True,
    )

    trainer = SentenceTransformerTrainer(
        model         = model,
        args          = training_args,
        train_dataset = train_dataset,
        loss          = train_loss,
    )

    logger.info("Starting training ...")
    trainer.train()
    logger.info("Training complete.")

    output_path.mkdir(parents=True, exist_ok=True)
    adapter_path = output_path / "adapter"
    adapter_path.mkdir(exist_ok=True)
    model[0].auto_model.save_pretrained(str(adapter_path))
    model.tokenizer.save_pretrained(str(output_path / "tokenizer"))

    meta = {
        "base_model":          base_model,
        "training_data":       str(train_json),
        "data_source":         "finqa_train_json_nopdf",
        "gold_text":           "pre_text + table + post_text",
        "negative_source":     "cross_doc",
        "max_seq_length":      BGE_MAX_SEQ_LENGTH,
        "epochs":              epochs,
        "batch_size":          batch_size,
        "grad_accumulation":   grad_accumulation,
        "effective_batch":     batch_size * grad_accumulation,
        "lr":                  lr,
        "lora_r":              lora_r,
        "lora_alpha":          lora_alpha,
        "hard_negatives":      hard_negatives,
        "loss_scale":          loss_scale,
        "n_training_examples": n_examples,
        "ablation": {
            "use_bm25_hard_negs": use_bm25_hard_negs,
            "extended_lora":      extended_lora,
        },
    }
    with open(output_path / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    total_bytes = sum(p.stat().st_size for p in adapter_path.rglob("*") if p.is_file())
    logger.info(f"\nLoRA adapter saved to: {adapter_path} ({total_bytes/1e6:.1f} MB)")
    logger.info(f"Metadata: {output_path / 'training_meta.json'}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Fine-tune BGE-M3 bi-encoder on FinQA train.json (no PDFs needed)"
    )
    p.add_argument("--train-json",        default="finqa/train.json")
    p.add_argument("--output",            default="models/finqa_train_bi_encoder_nopdf")
    p.add_argument("--base-model",        default="BAAI/bge-m3")
    p.add_argument("--epochs",            type=int,   default=10)
    p.add_argument("--batch-size",        type=int,   default=8)
    p.add_argument("--grad-accum",        type=int,   default=4)
    p.add_argument("--lr",                type=float, default=2e-5)
    p.add_argument("--lora-r",            type=int,   default=16)
    p.add_argument("--lora-alpha",        type=int,   default=32)
    p.add_argument("--hard-negatives",    type=int,   default=3,
                   help="Hard negatives per positive, sampled from other documents")
    p.add_argument("--loss-scale",        type=float, default=50.0)
    p.add_argument("--min-passage-len",   type=int,   default=50,
                   help="Minimum character length to include a passage")
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--use-bm25-hard-negs", action="store_true",
                   help="Mine hard negatives with BM25 instead of random sampling")
    p.add_argument("--extended-lora",     action="store_true",
                   help="Extend LoRA to attn-output + FFN-intermediate projections")
    p.add_argument("--query-prefix",      type=str,   default="",
                   help="Instruction prefix prepended to queries before encoding")

    args = p.parse_args()

    train_json = Path(args.train_json)
    if not train_json.exists():
        logger.error(f"--train-json not found: {train_json}"); sys.exit(1)

    train(
        train_json          = train_json,
        output_path         = Path(args.output),
        base_model          = args.base_model,
        epochs              = args.epochs,
        batch_size          = args.batch_size,
        grad_accumulation   = args.grad_accum,
        lr                  = args.lr,
        lora_r              = args.lora_r,
        lora_alpha          = args.lora_alpha,
        hard_negatives      = args.hard_negatives,
        loss_scale          = args.loss_scale,
        use_bm25_hard_negs  = args.use_bm25_hard_negs,
        extended_lora       = args.extended_lora,
        min_passage_len     = args.min_passage_len,
        seed                = args.seed,
        query_prefix        = args.query_prefix,
    )


if __name__ == "__main__":
    main()
