#!/usr/bin/env bash
# Run evaluation on both datasets + generate all paper outputs
# Usage: bash scripts/run_eval_and_analysis.sh <ce_run_dir> <edl_run_dir>
# Example: bash scripts/run_eval_and_analysis.sh results/runs/ce_20260519_120000 results/runs/edl_20260519_140000
set -euo pipefail
export HF_HOME=/data/rech/kobeissa/hf
cd "$(dirname "$0")/.."

source ../venv/bin/activate

CE_RUN="${1:-}"
EDL_RUN="${2:-}"

if [ -z "$CE_RUN" ]; then
    CE_RUN=$(ls -dt results/runs/ce_* 2>/dev/null | head -1)
fi
if [ -z "$EDL_RUN" ]; then
    EDL_RUN=$(ls -dt results/runs/edl_* 2>/dev/null | head -1)
fi

CE_MODEL="$CE_RUN/best_model"
EDL_MODEL="$EDL_RUN/best_model"
echo "CE model: $CE_MODEL"
echo "EDL model: $EDL_MODEL"

run_eval() {
    local model_dir="$1"
    local model_type="$2"
    local pairs="$3"
    local tag="$4"
    local beta="${5:-0.0}"

    python src/evaluation/evaluate_ranking.py \
        --model_dir "$model_dir" \
        --pairs "$pairs" \
        --backbone BAAI/bge-reranker-v2-m3 \
        --model_type "$model_type" \
        --batch_size 32 \
        --beta "$beta" \
        --out_dir results \
        --tag "$tag"

    python src/evaluation/evaluate_uncertainty.py \
        --scored_pairs "results/${tag}_scored_pairs.jsonl" \
        --out_dir results \
        --tag "$tag"

    python src/evaluation/evaluate_selective.py \
        --scored_pairs "results/${tag}_scored_pairs.jsonl" \
        --out_dir results/tables \
        --tag "$tag"
}

# CE evaluations
run_eval "$CE_MODEL" "ce" "data/processed/pairs/finqa_test_eval_pairs.jsonl" "ce_finqa_test"
run_eval "$CE_MODEL" "ce" "data/processed/pairs/financebench_eval_pairs.jsonl" "ce_financebench"

# EDL evaluations — pure p_relevant (beta=0)
run_eval "$EDL_MODEL" "edl" "data/processed/pairs/finqa_test_eval_pairs.jsonl" "edl_finqa_test" "0.0"
run_eval "$EDL_MODEL" "edl" "data/processed/pairs/financebench_eval_pairs.jsonl" "edl_financebench" "0.0"

# EDL uncertainty-penalized (beta=0.25)
run_eval "$EDL_MODEL" "edl" "data/processed/pairs/financebench_eval_pairs.jsonl" "edl_fb_beta025" "0.25"

# Beta ablation
for beta in 0.0 0.1 0.25 0.5 1.0; do
    python src/evaluation/evaluate_ranking.py \
        --model_dir "$EDL_MODEL" \
        --pairs data/processed/pairs/financebench_eval_pairs.jsonl \
        --model_type edl \
        --beta "$beta" \
        --out_dir results \
        --tag "edl_fb_beta${beta//./_}"
done

# Error analysis
python src/analysis/error_analysis.py \
    --scored_pairs results/edl_financebench_scored_pairs.jsonl \
    --fb_questions ../data/financebench_open_source.jsonl \
    --out_dir results/qualitative \
    --top_k 5

# Tables + plots + summaries
python src/analysis/make_tables.py --results_dir results --out_dir results/tables
python src/analysis/make_plots.py --results_dir results --out_dir results/plots
python src/analysis/write_paper_summaries.py --results_dir results --out_dir results/paper_summaries

echo ""
echo "=== Evaluation complete ==="
echo "Tables: results/tables/"
echo "Plots:  results/plots/"
echo "Summaries: results/paper_summaries/"
