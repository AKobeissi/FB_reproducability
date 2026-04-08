#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=chunk_analysis
#SBATCH --output=%x_%j.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00

# ─── Chunk Property Analysis (Phase 2 only) ───────────────────────────────────
# Re-runs the analysis script on already-saved Phase 1 outputs.
# No GPU needed — CPU only.
# Submit: sbatch scripts/run_chunk_analysis.sh
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SUBMIT_DIR=$SLURM_SUBMIT_DIR
VENV_PATH="$SUBMIT_DIR/venv"

echo "================================================="
echo "Job:    $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "Start:  $(date)"
echo "================================================="

source "$VENV_PATH/bin/activate"

cd "$SUBMIT_DIR"

python src/experiments/chunk_property_analysis.py \
  --chunk-dir         outputs/chunking_sweep/chunk_data \
  --results-dir       outputs/chunking_sweep \
  --financebench-json data/financebench_open_source.jsonl \
  --output-dir        outputs/chunking_sweep/analysis \
  --embedding-model   BAAI/bge-m3

echo "================================================="
echo "Finished: $(date)"
echo "================================================="
