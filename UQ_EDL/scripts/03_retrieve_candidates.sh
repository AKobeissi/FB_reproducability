#!/usr/bin/env bash
# Stage 2: First-stage retrieval — BM25, BGE-M3, RRF
set -euo pipefail
cd "$(dirname "$0")/.."

# FinQA retrieval (within-document, train split)
python src/retrieval/retrieve_candidates.py \
    --questions data/processed/splits/finqa_train.jsonl \
    --pages data/processed/pages/finqa_pages.jsonl \
    --out_dir data/processed/candidates \
    --retrievers bm25 bge_m3 rrf \
    --top_k 100 \
    --index_dir data/processed/candidates/indices \
    --dataset finqa_train \
    --per_doc

# FinQA val/test
for split in val test; do
    python src/retrieval/retrieve_candidates.py \
        --questions data/processed/splits/finqa_${split}.jsonl \
        --pages data/processed/pages/finqa_pages.jsonl \
        --out_dir data/processed/candidates \
        --retrievers bm25 bge_m3 rrf \
        --top_k 100 \
        --index_dir data/processed/candidates/indices \
        --dataset finqa_${split} \
        --per_doc
done

# FinanceBench retrieval
python src/retrieval/retrieve_candidates.py \
    --questions ../data/financebench_open_source.jsonl \
    --pages data/processed/pages/financebench_pages.jsonl \
    --out_dir data/processed/candidates \
    --retrievers bm25 bge_m3 rrf \
    --top_k 100 \
    --index_dir data/processed/candidates/indices_fb \
    --dataset financebench \
    --per_doc

echo "Candidate retrieval done."
