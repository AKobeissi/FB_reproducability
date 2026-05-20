#!/usr/bin/env bash
# Stage 6: Generate all paper tables and plots
set -euo pipefail
cd "$(dirname "$0")/.."

python src/analysis/make_tables.py \
    --results_dir results \
    --out_dir results/tables

python src/analysis/make_plots.py \
    --results_dir results \
    --out_dir results/plots

python src/analysis/write_paper_summaries.py \
    --results_dir results \
    --out_dir results/paper_summaries

# Error analysis (FinanceBench)
if [ -f "results/edl_financebench_scored_pairs.jsonl" ]; then
    python src/analysis/error_analysis.py \
        --scored_pairs results/edl_financebench_scored_pairs.jsonl \
        --fb_questions ../data/financebench_open_source.jsonl \
        --out_dir results/qualitative \
        --top_k 5
fi

echo "Tables and plots done. See results/tables/ and results/plots/"
