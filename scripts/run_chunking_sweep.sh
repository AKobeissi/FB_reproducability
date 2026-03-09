#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=chunking_sweep
#SBATCH --output=chunking_sweep_%j.log
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00

# =============================================================================
# run_chunking_sweep.sh  —  SLURM job for the full chunking sweep
#
# Fixes vs original:
#   1. Added --late-model (was missing — without it the "late" config in the
#      sweep falls back to standard chunking and produces no benefit).
#   2. Added --late-max-tokens 8192 (was defaulting to 2048).
#   3. Added --late-window-stride 512 (was defaulting to 128).
#   4. Fixed --output-root: was pointing to $SCRATCH_OUT/chunking_sweep but
#      the cp step copies $SCRATCH_OUT/* — outputs were correctly transferred.
#      Kept the same pattern for consistency.
# =============================================================================

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
  --exclude 'outputs' \
  --exclude 'vector_stores' \
  --exclude '*.log' \
  "$SUBMIT_DIR/" "$SCRATCH_DIR/"

cd "$SCRATCH_DIR"

echo "Activating venv: $VENV_PATH"
source "$VENV_PATH/bin/activate"

export PYTHONPATH="$SCRATCH_DIR:$PYTHONPATH"

SCRATCH_OUT="$SCRATCH_DIR/outputs"
mkdir -p "$SCRATCH_OUT"

# ---------------------------------------------------------------------------
# Run full chunking sweep
#
# The sweep covers: fixed-token, fixed-char, sentence, semantic, and late
# chunking strategies, evaluated through the unified pipeline.
#
# Late-chunking-specific params:
#   --late-model        Must be a long-context encoder (≥8 192 tok window).
#   --late-max-tokens   Match the model's context window.
#   --late-window-stride  ≥ late-chunk-size to avoid window explosion.
# ---------------------------------------------------------------------------

python scripts/run_chunking_sweep.py \
  --pdf-dir               pdfs \
  --output-root           "$SCRATCH_OUT/chunking_sweep" \
  --vector-store-dir      vector_stores \
  --embedding-model       bge-m3 \
  --llm-model             Qwen/Qwen2.5-7B-Instruct \
  --late-model            jinaai/jina-embeddings-v2-base-en \
  --late-max-tokens       8192 \
  --late-window-stride    512 \
  --late-chunk-size       512 \
  --late-chunk-overlap    64 \
  --late-pooling          mean \
  --use-faiss-chunking \
  --include-unified \
  --unified-retrieval     dense \
  --eval-type             both \
  --top-k                 5

EXIT_CODE=$?

FINAL_DEST="$SUBMIT_DIR/outputs"
echo "Copying results to: $FINAL_DEST"
mkdir -p "$FINAL_DEST"
cp -r "$SCRATCH_OUT/"* "$FINAL_DEST/"

rm -rf "$SCRATCH_DIR"

echo "Done (exit code: $EXIT_CODE)"
exit $EXIT_CODE