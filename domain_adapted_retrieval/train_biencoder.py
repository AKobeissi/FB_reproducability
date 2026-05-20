"""
Fine-tune a bi-encoder for financial document page retrieval.

Model:   BAAI/bge-m3  (same as the baseline — fine-tuning adapts the model to
         financial domain terminology and within-doc page discrimination, starting
         from the same strong representations the baseline already uses)

Loss:    MultipleNegativesRankingLoss — treats all non-matching in-batch positives
         as negatives, plus explicitly pushes hard negatives in triplet examples.

Key design choices:
 • Layer freezing: all transformer layers except the last `trainable_layers` are
   frozen.  This prevents catastrophic forgetting on the small (~450 example)
   FinQA dataset while still allowing the top layers to specialize.  Prior run
   (full fine-tune of bge-base on 2718 examples) caused DocRec@5 to collapse
   from 0.71 → 0.51 — a clear sign of forgetting.
 • Raw page text at training time (no doc/page prefix) — matching the index
   format used at inference time.
 • Intra-document hard negatives teach the model to rank pages of the same filing,
   which is the bottleneck task (within-doc page recall, not cross-doc routing).
"""

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Layer freezing
# ---------------------------------------------------------------------------

def freeze_except_last_n_layers(model, n: int = 3) -> None:
    """
    Freeze all transformer layers except the last n, plus keep the pooling
    module trainable.

    This prevents catastrophic forgetting when fine-tuning a large model on a
    small dataset.  With n=3 and bge-m3 (24 layers), ~87% of parameters are
    frozen; only the top 3 layers + pooling adapt to the financial domain.

    Args:
        model: SentenceTransformer instance.
        n:     Number of transformer layers to keep trainable (from the top).
    """
    if n <= 0:
        logger.info("Layer freezing disabled (trainable_layers=0). Training all parameters.")
        return

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Find the transformer encoder layers
    transformer = model[0].auto_model
    encoder_layers = None

    if hasattr(transformer, "encoder") and hasattr(transformer.encoder, "layer"):
        encoder_layers = transformer.encoder.layer          # BERT / XLM-R
    elif hasattr(transformer, "layers"):
        encoder_layers = transformer.layers                 # some other architectures

    if encoder_layers is None:
        logger.warning(
            "Could not find transformer encoder layers to freeze. "
            "All parameters will be trained."
        )
        for param in model.parameters():
            param.requires_grad = True
        return

    num_layers = len(encoder_layers)
    n_freeze = max(0, num_layers - n)
    logger.info(
        f"Layer freezing: {n_freeze} layers frozen, "
        f"last {min(n, num_layers)} layers trainable "
        f"(total encoder layers: {num_layers})"
    )

    # Unfreeze the last n layers
    for i in range(n_freeze, num_layers):
        for param in encoder_layers[i].parameters():
            param.requires_grad = True

    # Always unfreeze the pooling module (sentence-transformers index 1)
    if len(model) > 1:
        for param in model[1].parameters():
            param.requires_grad = True

    # Also unfreeze the final linear/normalisation if present (index -1)
    if len(model) > 2:
        for param in model[-1].parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable parameters: {trainable:,} / {total:,} "
        f"({100.0 * trainable / total:.1f}%)"
    )


# ---------------------------------------------------------------------------
# Evaluator construction
# ---------------------------------------------------------------------------

def build_ir_evaluator(val_pairs: List[Dict], name: str = "finqa_val"):
    """
    Build an InformationRetrievalEvaluator from held-out validation pairs.
    The corpus includes all positives + hard negatives from the validation set.
    """
    from sentence_transformers.evaluation import InformationRetrievalEvaluator

    queries: Dict[str, str] = {}
    corpus: Dict[str, str] = {}
    relevant_docs: Dict[str, set] = {}

    for idx, pair in enumerate(val_pairs):
        q_id = f"q_{idx}"
        d_id = f"d_{idx}"
        queries[q_id] = pair["question"]
        corpus[d_id] = pair["positive_page_text"]
        relevant_docs[q_id] = {d_id}

    neg_count = 0
    for idx, pair in enumerate(val_pairs):
        for neg_text in pair.get("hard_negatives", []):
            neg_id = f"neg_{abs(hash(neg_text[:80]))}"
            if neg_id not in corpus:
                corpus[neg_id] = neg_text
                neg_count += 1

    logger.info(
        f"IR evaluator: {len(queries)} queries | corpus size: {len(corpus)} "
        f"(includes {neg_count} hard-negative pages)"
    )

    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=name,
        show_progress_bar=False,
        ndcg_at_k=[1, 3, 10],
        mrr_at_k=[10],
        map_at_k=[10],
    )


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train_biencoder(
    config,
    train_pairs: List[Dict],
    val_pairs: List[Dict],
) -> "SentenceTransformer":
    """
    Fine-tune the bi-encoder with layer freezing and save the best checkpoint.

    Returns the best SentenceTransformer model loaded from disk.
    """
    from sentence_transformers import SentenceTransformer, losses
    from torch.utils.data import DataLoader

    set_seed(config.seed)
    tc = config.training
    output_path = tc.output_model_path
    os.makedirs(output_path, exist_ok=True)

    # ---- Load base model ----
    logger.info(f"Loading base model: {tc.base_model}")
    model = SentenceTransformer(tc.base_model)
    model.max_seq_length = tc.max_seq_length

    # ---- Layer freezing ----
    freeze_except_last_n_layers(model, n=tc.trainable_layers)

    # ---- Build training examples ----
    from domain_adapted_retrieval.data_prep import to_sentence_transformer_examples
    train_examples = to_sentence_transformer_examples(train_pairs)
    logger.info(f"Training on {len(train_examples)} InputExamples")

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=tc.batch_size,
        drop_last=True,
    )

    # ---- Loss ----
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # ---- Evaluator ----
    evaluator = build_ir_evaluator(val_pairs)

    # ---- Schedule ----
    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * tc.num_epochs
    warmup_steps = int(total_steps * tc.warmup_ratio)

    logger.info(
        f"Training plan: {tc.num_epochs} epochs × {steps_per_epoch} steps = "
        f"{total_steps} total steps | warmup: {warmup_steps} | lr: {tc.learning_rate}"
    )

    # ---- Train ----
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=tc.num_epochs,
        evaluation_steps=steps_per_epoch,
        warmup_steps=warmup_steps,
        output_path=output_path,
        save_best_model=True,
        show_progress_bar=True,
        optimizer_params={"lr": tc.learning_rate},
        use_amp=torch.cuda.is_available(),
        checkpoint_path=os.path.join(output_path, "checkpoints"),
        checkpoint_save_steps=steps_per_epoch,
        checkpoint_save_total_limit=2,
    )

    # ---- Load best model ----
    logger.info(f"Loading best checkpoint from {output_path}")
    best_model = SentenceTransformer(output_path)

    # ---- Save training metadata ----
    metadata = {
        "base_model": tc.base_model,
        "output_path": output_path,
        "num_epochs": tc.num_epochs,
        "batch_size": tc.batch_size,
        "learning_rate": tc.learning_rate,
        "trainable_layers": tc.trainable_layers,
        "max_seq_length": tc.max_seq_length,
        "num_train_examples": len(train_examples),
        "num_train_pairs": len(train_pairs),
        "num_val_pairs": len(val_pairs),
        "warmup_steps": warmup_steps,
        "total_steps": total_steps,
    }
    with open(os.path.join(output_path, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Training complete. Model saved to: {output_path}")
    return best_model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from domain_adapted_retrieval.config import ExperimentConfig
    from domain_adapted_retrieval.data_prep import build_training_pairs

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--trainable-layers", type=int, default=None)
    args = parser.parse_args()

    cfg = ExperimentConfig()
    if args.epochs:
        cfg.training.num_epochs = args.epochs
    if args.batch_size:
        cfg.training.batch_size = args.batch_size
    if args.lr:
        cfg.training.learning_rate = args.lr
    if args.base_model:
        cfg.training.base_model = args.base_model
    if args.output_path:
        cfg.training.output_model_path = args.output_path
    if args.trainable_layers is not None:
        cfg.training.trainable_layers = args.trainable_layers

    all_pairs, train_pairs, val_pairs = build_training_pairs(cfg)
    trained_model = train_biencoder(cfg, train_pairs, val_pairs)
    print(f"\nFine-tuned model saved to: {cfg.training.output_model_path}")
