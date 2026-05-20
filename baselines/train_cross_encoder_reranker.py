#!/usr/bin/env python3
"""
train_cross_encoder_reranker.py
================================
Fine-tunes BAAI/bge-reranker-v2-m3 on FinQA for financial-domain passage
reranking.

Training data (FinQA train.json)
---------------------------------
Each FinQA sample contains:
  - question          : the query
  - pre_text / table / post_text : text surrounding the relevant table on the
                                   gold page — we concatenate these as the
                                   positive passage.
  - text_retrieved_all / table_retrieved_all : TF-IDF-retrieved passages ranked
                                   by score; passages NOT in gold_inds serve as
                                   hard negatives.

For each question we create:
  1 positive  : (question, full_page_text,  label=1.0)
  up to 2 hard negatives from retrieved passages not in gold_inds
  1 random negative sampled from another question's positive page

Evaluation (FinQA dev.json) uses CERerankingEvaluator (MAP / MRR).

Outputs
-------
  checkpoints/ft_cross_encoder/   ← best checkpoint (highest dev MAP)
"""

import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_reranker")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FINQA_TRAIN  = PROJECT_ROOT / "finqa/train.json"
FINQA_DEV    = PROJECT_ROOT / "finqa/dev.json"
CHECKPOINT   = PROJECT_ROOT / "checkpoints/ft_cross_encoder"

BASE_MODEL   = "BAAI/bge-reranker-v2-m3"
MAX_LENGTH   = 512
EPOCHS       = 3
BATCH_SIZE   = 32   # 32 × 512 tokens in FP16 ≈ 18 GB; L40S (46 GB) has plenty of headroom
LR           = 2e-5
WARMUP_RATIO = 0.1
SEED         = 42

RANDOM_NEGS_PER_SAMPLE = 3   # random negatives from other questions' full pages
# NOTE: We deliberately do NOT use text_retrieved_all as hard negatives.
# Those entries are sentence-level passages retrieved from within the SAME page
# as the gold evidence (intra-document negatives).  Our retrieval task is
# inter-page: ranking different pages against each other.  Training on
# intra-page negatives would teach the model the wrong skill (within-page
# sentence discrimination) and bias it away from the actual inference setting.


# ---------------------------------------------------------------------------
# FinQA helpers
# ---------------------------------------------------------------------------

def flatten_table(table: List[List]) -> str:
    """Convert a FinQA table (list of rows) to a readable string."""
    rows = [" | ".join(str(cell).strip() for cell in row) for row in table]
    return "\n".join(rows)


def build_page_text(sample: Dict) -> str:
    """Reconstruct the full page text from pre_text + table + post_text."""
    pre  = " ".join(str(t).strip() for t in sample.get("pre_text",  []))
    tbl  = flatten_table(sample.get("table", []))
    post = " ".join(str(t).strip() for t in sample.get("post_text", []))
    return f"{pre}\n{tbl}\n{post}".strip()


def get_passage_by_ind(sample: Dict, ind: str) -> Optional[str]:
    """
    Retrieve the passage text for a given FinQA evidence index.
    Indices like 'text_3'  → (pre_text + post_text)[3]
                 'table_1' → table row 1 formatted as a string
    """
    if ind.startswith("text_"):
        idx = int(ind.split("_")[1])
        all_texts = sample.get("pre_text", []) + sample.get("post_text", [])
        if idx < len(all_texts):
            return str(all_texts[idx]).strip()
    elif ind.startswith("table_"):
        idx = int(ind.split("_")[1])
        table = sample.get("table", [])
        if idx < len(table):
            return " | ".join(str(c).strip() for c in table[idx])
    return None


def build_training_pairs(
    samples: List[Dict],
    random_negs_per_sample: int = RANDOM_NEGS_PER_SAMPLE,
    seed: int = SEED,
) -> List[Tuple[str, str, float]]:
    """
    Returns a list of (query, passage, label) tuples.

    Positive  : (question, full_page_text, 1.0)
                full_page_text = pre_text + table + post_text for the gold page.

    Negatives : other questions' full pages, sampled randomly.
                These are inter-document negatives — a different company's
                financial page for a different question.  This directly matches
                the inference setting where the reranker must distinguish the
                gold page from other pages in the same annual report.
    """
    rng = random.Random(seed)
    all_positives = [build_page_text(s) for s in samples]

    # Group by source document so we can optionally prefer same-corpus negatives
    # (a page from the same company/year is a harder negative than a random one)
    doc_to_indices: Dict[str, List[int]] = {}
    for i, s in enumerate(samples):
        doc = s.get("filename", "").split("/page_")[0]  # e.g. "ADI/2009"
        doc_to_indices.setdefault(doc, []).append(i)

    pairs: List[Tuple[str, str, float]] = []

    for i, sample in enumerate(samples):
        question = sample["qa"]["question"]
        pos_text = all_positives[i]
        src_doc  = sample.get("filename", "").split("/page_")[0]

        # ── Positive ──────────────────────────────────────────────────────
        pairs.append((question, pos_text, 1.0))

        # ── Negatives: prefer pages from a different document ──────────────
        # We want pages that look plausibly financial but answer different
        # questions, so we sample from other documents first.
        other_indices = [j for j in range(len(samples))
                         if j != i and
                         samples[j].get("filename", "").split("/page_")[0] != src_doc]

        if len(other_indices) < random_negs_per_sample:
            # Fallback: allow same-document pages if pool is too small
            other_indices = [j for j in range(len(samples)) if j != i]

        chosen = rng.sample(other_indices, k=min(random_negs_per_sample, len(other_indices)))
        for j in chosen:
            pairs.append((question, all_positives[j], 0.0))

    n_pos = sum(1 for p in pairs if p[2] == 1.0)
    n_neg = sum(1 for p in pairs if p[2] == 0.0)
    logger.info(
        f"Built {len(pairs)} training pairs from {len(samples)} samples "
        f"({n_pos} pos, {n_neg} neg, avg {n_neg/max(n_pos,1):.1f} negs/pos)"
    )
    return pairs


def build_dev_reranking_samples(samples: List[Dict]) -> List[Dict]:
    """
    Format for CERerankingEvaluator:
      [{'query': ..., 'positive': [...], 'negative': [...]}, ...]
    """
    dev_samples = []
    for sample in samples:
        question  = sample["qa"]["question"]
        gold_inds = set(sample["qa"].get("gold_inds", {}).keys())
        pos_text  = build_page_text(sample)

        negatives = []
        for entry in sample.get("text_retrieved_all", []) + sample.get("table_retrieved_all", []):
            ind = entry.get("ind", "")
            if ind in gold_inds:
                continue
            text = get_passage_by_ind(sample, ind)
            if text and len(text.split()) >= 5:
                negatives.append(text)

        if negatives:   # skip if no negatives available
            dev_samples.append({
                "query":    question,
                "positive": [pos_text],
                "negative": negatives[:5],  # cap at 5 to keep eval fast
            })
    return dev_samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(SEED)
    np.random.seed(SEED)

    # ── Load FinQA data ───────────────────────────────────────────────────
    logger.info(f"Loading FinQA train from {FINQA_TRAIN}")
    train_samples = json.load(open(FINQA_TRAIN))
    dev_samples   = json.load(open(FINQA_DEV))
    logger.info(f"Train: {len(train_samples)}, Dev: {len(dev_samples)}")

    # ── Build training pairs ──────────────────────────────────────────────
    train_pairs = build_training_pairs(train_samples)
    dev_reranking = build_dev_reranking_samples(dev_samples)
    logger.info(f"Dev reranking samples: {len(dev_reranking)}")

    # ── Load dependencies (sentence-transformers) ─────────────────────────
    try:
        from sentence_transformers.cross_encoder import CrossEncoder
        from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
        from sentence_transformers import InputExample
        from torch.utils.data import DataLoader
    except ImportError as e:
        logger.error(f"sentence-transformers not available: {e}")
        sys.exit(1)

    # ── Initialise model ──────────────────────────────────────────────────
    logger.info(f"Loading base model: {BASE_MODEL}")
    model = CrossEncoder(
        BASE_MODEL,
        num_labels=1,
        max_length=MAX_LENGTH,
    )

    # ── Prepare DataLoader ────────────────────────────────────────────────
    train_examples = [
        InputExample(texts=[q, p], label=float(label))
        for q, p, label in train_pairs
    ]
    train_loader = DataLoader(
        train_examples, shuffle=True, batch_size=BATCH_SIZE
    )

    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    logger.info(
        f"Training: {total_steps} steps total, {warmup_steps} warmup, "
        f"lr={LR}, batch={BATCH_SIZE}, epochs={EPOCHS}"
    )

    # ── Evaluator (MAP / MRR on dev) ──────────────────────────────────────
    evaluator = CERerankingEvaluator(
        dev_reranking,
        name="finqa_dev",
        write_csv=True,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    CHECKPOINT.mkdir(parents=True, exist_ok=True)
    model.fit(
        train_dataloader=train_loader,
        evaluator=evaluator,
        epochs=EPOCHS,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": LR},
        output_path=str(CHECKPOINT),
        save_best_model=True,
        show_progress_bar=True,
        use_amp=True,   # FP16 mixed-precision — halves activation memory, fits on 24 GB GPUs
    )

    # sentence-transformers v4 CrossEncoder.fit() uses SaveModelCallback which
    # may not fire when the evaluator callback triggers on_evaluate without
    # passing `model` through the HF Trainer callback chain.  Explicitly save
    # the final trained model here to guarantee the checkpoint exists.
    logger.info("Explicitly saving final model to: %s", CHECKPOINT)
    model.save(str(CHECKPOINT))

    # Ensure the saved config.json has a `model_type` key that AutoConfig can
    # recognise.  sentence-transformers may write a minimal config that omits
    # this field; patch it in from the base model when necessary.
    _config_path = CHECKPOINT / "config.json"
    if _config_path.exists():
        import json as _json
        _cfg = _json.loads(_config_path.read_text())
        if not _cfg.get("model_type"):
            from transformers import AutoConfig as _AC
            _base_type = _AC.from_pretrained(BASE_MODEL).model_type
            _cfg["model_type"] = _base_type
            _config_path.write_text(_json.dumps(_cfg, indent=2))
            logger.info("Patched checkpoint config.json with model_type=%s", _base_type)
        else:
            logger.info("Checkpoint config.json already has model_type=%s", _cfg["model_type"])
    else:
        logger.warning("No config.json found at %s after save — Stage 2 may fail", _config_path)


if __name__ == "__main__":
    main()
