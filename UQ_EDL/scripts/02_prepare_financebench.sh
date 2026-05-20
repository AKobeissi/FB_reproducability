#!/usr/bin/env bash
# Stage 1b: Extract FinanceBench pages and align evidence
set -euo pipefail
cd "$(dirname "$0")/.."

python src/data/build_financebench_pages.py \
    --fb_path ../data/financebench_open_source.jsonl \
    --docinfo_path ../data/financebench_document_information.jsonl \
    --pdf_dir ../pdfs \
    --out_dir data/processed/pages \
    --align_threshold 0.35 \
    --min_chars 50

echo "FinanceBench pages done."
