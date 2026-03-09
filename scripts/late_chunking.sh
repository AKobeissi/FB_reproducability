#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=late_chunking
#SBATCH --output=late_chunking_%j.log
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00

# =============================================================================
# late_chunking.sh  —  SLURM job for the late-chunking-only experiment
#
# Fixes vs original:
#   1. Added --late-model (was missing entirely — this was the primary cause of
#      bad results; without it late_model=None and the implementation falls
#      back to standard BGE-M3 with no long-context benefit).
#   2. Added --late-max-tokens 8192 (was defaulting to 2048, which only covers
#      ~4 chunks of FinanceBench docs that can be 80-200 pages).
#   3. Added --late-window-stride 512 (was defaulting to 128, creating 16×
#      redundant windows per chunk and diluting embeddings).
#   4. Job name corrected from "chunking_sweep" to "late_chunking".
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
# Run late-chunking experiment
#
# Key parameters:
#   --late-model        Long-context encoder required for late chunking.
#                       jina-embeddings-v2-base-en has an 8 192-token window.
#                       Alternatives: nomic-ai/nomic-embed-text-v1
#   --late-max-tokens   Match the model's context window (8192 for Jina).
#   --late-window-stride  Must be ≥ late-chunk-size (512) to avoid redundant
#                       overlapping context windows.
#   --late-chunk-size   Final retrieval chunk size in tokens.
#   --late-chunk-overlap  Token overlap between adjacent chunks.
# ---------------------------------------------------------------------------

python scripts/run_late_chunking_only.py \
  --pdf-dir             pdfs \
  --output-root         "$SCRATCH_OUT/chunking_sweep" \
  --vector-store-dir    vector_stores \
  --embedding-model     bge-m3 \
  --llm-model           Qwen/Qwen2.5-7B-Instruct \
  --late-model          BAAI/bge-m3 \
  --late-max-tokens     8192 \
  --late-window-stride  512 \
  --late-chunk-size     512 \
  --late-chunk-overlap  64 \
  --late-pooling        mean \
  --use-faiss-chunking \
  --unified-retrieval   dense \
  --eval-type           both \
  --top-k               5

EXIT_CODE=$?

FINAL_DEST="$SUBMIT_DIR/outputs"
echo "Copying results to: $FINAL_DEST"
mkdir -p "$FINAL_DEST"
cp -r "$SCRATCH_OUT/"* "$FINAL_DEST/"

rm -rf "$SCRATCH_DIR"

echo "Done (exit code: $EXIT_CODE)"
exit $EXIT_CODE