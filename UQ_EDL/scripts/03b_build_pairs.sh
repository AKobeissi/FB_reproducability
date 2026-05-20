#!/usr/bin/env bash
# Stage 2b: Build training pairs with hard negatives
set -euo pipefail
cd "$(dirname "$0")/.."

# Generate finance-aware hard negatives
python src/data/generate_hard_negatives.py \
    --finqa_pages data/processed/pages/finqa_pages.jsonl \
    --splits data/processed/splits/finqa_splits.json \
    --out data/processed/pairs/finqa_hard_negatives.jsonl \
    --max_per_type 2 \
    --seed 42

# Build training pairs (uses hard negatives + easy negatives)
python src/data/make_pairs.py \
    --pages data/processed/pages/finqa_pages.jsonl \
    --splits data/processed/splits/finqa_splits.json \
    --candidates data/processed/candidates/finqa_train_bm25_top100.jsonl \
    --hard_negatives data/processed/pairs/finqa_hard_negatives.jsonl \
    --out_dir data/processed/pairs \
    --negative_ratio 6 \
    --hard_neg_ratio 3 \
    --seed 42

echo "Training pairs done."
