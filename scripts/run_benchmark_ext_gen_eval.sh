#!/bin/bash
#SBATCH --job-name=bench_ext_gen_eval
#SBATCH --partition=rali
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=bench_ext_gen_eval_%j.log
#SBATCH --error=bench_ext_gen_eval_%j.log

# Generates answers (Qwen2.5-7B-Instruct) for all 18 benchmark-extension
# retrieval variants across 680 questions (FB + FinQA), then computes
# ROUGE-L, BERTScore F1, and NumericMatch globally.
#
# Resume-safe: already-generated answers are skipped.
# Expected runtime: ~12-16h on a single L40S (18 variants × 680q).
#
# Outputs → baselines/results/benchmark_extension/gen_eval/
#   gen_eval_report.txt   — human-readable table
#   gen_eval_results.csv  — machine-readable
#   gen_eval_results.json — full per-variant metrics

set -euo pipefail

SUBMIT_DIR=$SLURM_SUBMIT_DIR
VENV_PATH="$SUBMIT_DIR/venv"
SCRATCH_DIR=/Tmp/$(whoami)/${SLURM_JOB_ID}

export HF_HOME=/data/rech/kobeissa/hf
export HF_HUB_CACHE=/data/rech/kobeissa/hf/hub
export HF_DATASETS_CACHE=/data/rech/kobeissa/hf/datasets
export TRANSFORMERS_CACHE=/data/rech/kobeissa/hf
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "========================================================="
echo "Job ID        : ${SLURM_JOB_ID:-local}"
echo "Node          : ${SLURMD_NODENAME:-$(hostname)}"
echo "GPU           : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
echo "Start time    : $(date)"
echo "========================================================="

mkdir -p "$SCRATCH_DIR"

echo "Copying repository to scratch..."
rsync -a --quiet \
    --exclude 'venv' \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.log' \
    --exclude 'vector_stores' \
    --exclude 'pdfs' \
    --exclude 'Final-PDF' \
    "$SUBMIT_DIR/" "$SCRATCH_DIR/"

# Copy prediction files (we update them in-place during generation)
mkdir -p "$SCRATCH_DIR/baselines/results/benchmark_extension/predictions"
rsync -a "$SUBMIT_DIR/baselines/results/benchmark_extension/predictions/" \
         "$SCRATCH_DIR/baselines/results/benchmark_extension/predictions/"

cd "$SCRATCH_DIR"
source "$VENV_PATH/bin/activate"

echo ""
echo ">>> Generating answers + computing generative metrics"
echo "========================================================="

python baselines/run_benchmark_ext_gen_eval.py \
    --model Qwen/Qwen2.5-7B-Instruct

EXIT_CODE=$?

echo ""
echo ">>> Syncing results back..."
rsync -a "$SCRATCH_DIR/baselines/results/benchmark_extension/predictions/" \
         "$SUBMIT_DIR/baselines/results/benchmark_extension/predictions/"
rsync -a "$SCRATCH_DIR/baselines/results/benchmark_extension/gen_eval/" \
         "$SUBMIT_DIR/baselines/results/benchmark_extension/gen_eval/"

if [ $EXIT_CODE -ne 0 ]; then
    echo "[ERROR] Generation/eval failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo ""
echo "========================================================="
echo "Done!   End time : $(date)"
echo "========================================================="

# Print the report inline in the log
echo ""
echo ">>> Final report:"
cat "$SUBMIT_DIR/baselines/results/benchmark_extension/gen_eval/gen_eval_report.txt"
