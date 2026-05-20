#!/usr/bin/env bash
# Stage 1a: Extract FinQA pages from PDFs
set -euo pipefail
cd "$(dirname "$0")/.."

python src/data/build_finqa_pages.py \
    --gold_pages ../data/finqa_test_gold_pages.jsonl \
    --pdf_dirs ../pdfs-extended-v4 ../pdfs \
    --out_dir data/processed/pages \
    --min_chars 50

python src/data/make_splits.py \
    --gold_pages ../data/finqa_test_gold_pages.jsonl \
    --out_dir data/processed/splits \
    --train_ratio 0.70 \
    --val_ratio 0.15 \
    --seed 42

echo "FinQA pages and splits done."
