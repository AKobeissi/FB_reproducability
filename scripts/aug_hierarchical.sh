#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=aug_hier_rag
#SBATCH --output=%x_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:ls40:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00

# ─── Augmented Fine-tuning + Hierarchical RAG ─────────────────────────────────
#
# Trains a new BGE-M3 model on augmented FinQA data (query style diversification
# + MDA page injection + FinQA train expansion) then evaluates with the same
# hierarchical retrieval pipeline as hierarchical_rag.py.
#
# Submit: sbatch scripts/aug_hierarchical.sh
# Skip training (eval only): sbatch scripts/aug_hierarchical.sh --skip-train
# ──────────────────────────────────────────────────────────────────────────────

SUBMIT_DIR=$SLURM_SUBMIT_DIR
VENV_PATH="$SUBMIT_DIR/venv"
SCRATCH_DIR=/Tmp/$(whoami)/${SLURM_JOB_ID}

EXTRA_ARGS="${@:-}"   # pass through any CLI flags, e.g. --skip-train

echo "================================================="
echo "Job:         $SLURM_JOB_ID"
echo "Node:        $SLURMD_NODENAME"
echo "GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start:       $(date)"
echo "Submit dir:  $SUBMIT_DIR"
echo "Scratch dir: $SCRATCH_DIR"
echo "Extra args:  ${EXTRA_ARGS:-none}"
echo "================================================="

mkdir -p "$SCRATCH_DIR"

echo "Copying repository to scratch..."
rsync -a --quiet \
  --exclude 'venv' \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude 'outputs' \
  --exclude 'vector_stores' \
  --exclude '*.log' \
  "$SUBMIT_DIR/" "$SCRATCH_DIR/"

cd "$SCRATCH_DIR"

echo "Activating venv: $VENV_PATH"
source "$VENV_PATH/bin/activate"

echo "Python: $(which python)"
echo "Torch CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Run augmented training + hierarchical evaluation
python aug_hierarchical_rag.py \
  --results-dir hierarchical_rag_aug/results \
  --vs-dir      hierarchical_rag_aug/vector_store \
  --orig-results hierarchical_rag/results \
  $EXTRA_ARGS

STATUS=$?

# Copy results back to submit directory
echo "Copying results back to $SUBMIT_DIR ..."
rsync -a --quiet \
  "$SCRATCH_DIR/hierarchical_rag_aug/" \
  "$SUBMIT_DIR/hierarchical_rag_aug/"

rsync -a --quiet \
  "$SCRATCH_DIR/models/fin_adapted_biencoder_aug/" \
  "$SUBMIT_DIR/models/fin_adapted_biencoder_aug/" 2>/dev/null || true

rm -rf "$SCRATCH_DIR"

echo "================================================="
echo "Finished: $(date)"
echo "Exit status: $STATUS"
echo "================================================="

exit $STATUS
