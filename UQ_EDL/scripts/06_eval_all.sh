#!/usr/bin/env bash
# Stage 5: Evaluate all models on FinanceBench and FinQA test
# Usage: bash scripts/06_eval_all.sh <ce_model_dir> <edl_model_dir>
set -euo pipefail
cd "$(dirname "$0")/.."

CE_MODEL="${1:-results/runs/ce_best/best_model}"
EDL_MODEL="${2:-results/runs/edl_best/best_model}"
BACKBONE="${3:-BAAI/bge-reranker-v2-m3}"

evaluate_model() {
    local model_dir="$1"
    local model_type="$2"
    local pairs="$3"
    local tag="$4"
    local beta="${5:-0.25}"

    python src/evaluation/evaluate_ranking.py \
        --model_dir "$model_dir" \
        --pairs "$pairs" \
        --backbone "$BACKBONE" \
        --model_type "$model_type" \
        --batch_size 32 \
        --beta "$beta" \
        --out_dir "results" \
        --tag "$tag"

    python src/evaluation/evaluate_uncertainty.py \
        --scored_pairs "results/${tag}_scored_pairs.jsonl" \
        --out_dir "results" \
        --tag "$tag"

    python src/evaluation/evaluate_selective.py \
        --scored_pairs "results/${tag}_scored_pairs.jsonl" \
        --out_dir "results/tables" \
        --tag "$tag"
}

# FinQA internal test
evaluate_model "$CE_MODEL" "ce" \
    "data/processed/pairs/finqa_test_pairs.jsonl" \
    "ce_finqa_test"

evaluate_model "$EDL_MODEL" "edl" \
    "data/processed/pairs/finqa_test_pairs.jsonl" \
    "edl_finqa_test"

# FinanceBench external test
evaluate_model "$CE_MODEL" "ce" \
    "data/processed/pairs/financebench_eval_pairs.jsonl" \
    "ce_financebench"

evaluate_model "$EDL_MODEL" "edl" \
    "data/processed/pairs/financebench_eval_pairs.jsonl" \
    "edl_financebench"

# Beta ablation for EDL
for beta in 0.0 0.1 0.25 0.5 1.0; do
    python src/evaluation/evaluate_ranking.py \
        --model_dir "$EDL_MODEL" \
        --pairs "data/processed/pairs/financebench_eval_pairs.jsonl" \
        --model_type edl \
        --beta "$beta" \
        --out_dir results \
        --tag "edl_fb_beta${beta}"
done

echo "Evaluation done."
