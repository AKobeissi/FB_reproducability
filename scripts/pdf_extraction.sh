#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=pdf_collect
#SBATCH --output=pdf_collect_%j.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00

# =============================================================================
# run_pdf_collect.sh — SLURM job for downloading/collecting PDFs from a JSONL
# (CPU-only: no --gres=gpu)
# =============================================================================

set -euo pipefail

SUBMIT_DIR=$SLURM_SUBMIT_DIR
VENV_PATH="$SUBMIT_DIR/venv"
SCRATCH_DIR=/Tmp/$(whoami)/${SLURM_JOB_ID}

echo "========================================================="
echo "Job:         $SLURM_JOB_ID"
echo "Node:        $SLURMD_NODENAME"
echo "Submit dir:  $SUBMIT_DIR"
echo "Scratch dir: $SCRATCH_DIR"
echo "========================================================="

mkdir -p "$SCRATCH_DIR"

echo "Copying repository to scratch..."
rsync -a --quiet \
  --exclude 'venv' \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '*.log' \
  "$SUBMIT_DIR/" "$SCRATCH_DIR/"

cd "$SCRATCH_DIR"

echo "Activating venv: $VENV_PATH"
source "$VENV_PATH/bin/activate"

# Optional (helps avoid home quota if anything caches):
export HF_HOME="/data/rech/$(whoami)/hf"
export HF_DATASETS_CACHE="/data/rech/$(whoami)/hf_datasets"
export TRANSFORMERS_CACHE="/data/rech/$(whoami)/hf_transformers"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

# ---- Inputs / Outputs ----
JSONL_REL="data/secqa_test.jsonl"
OUT_REL="pdfs-secqa"

if [ ! -f "$JSONL_REL" ]; then
  echo "ERROR: JSONL not found at $SCRATCH_DIR/$JSONL_REL"
  exit 1
fi

mkdir -p "$OUT_REL"

echo "Running PDF collection..."
python -u scripts/pdf_ex.py \
  --jsonl "$JSONL_REL" \
  --out_dir "$OUT_REL"

EXIT_CODE=$?

# Copy PDFs back to your submit dir (repo)
FINAL_DEST="$SUBMIT_DIR/$OUT_REL"
echo "Copying PDFs to: $FINAL_DEST"
mkdir -p "$FINAL_DEST"
rsync -a "$SCRATCH_DIR/$OUT_REL/" "$FINAL_DEST/"

# Cleanup
rm -rf "$SCRATCH_DIR"

echo "Done (exit code: $EXIT_CODE)"
exit $EXIT_CODE