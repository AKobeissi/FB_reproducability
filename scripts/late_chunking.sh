#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=late_chunking
#SBATCH --output=late_chunking_%j.log
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:ls40:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00

# =============================================================================
# late_chunking.sh  —  SLURM job for the late-chunking-only experiment
#
# Key change:  --late-model is now BAAI/bge-m3  (same model used for
# retrieval).  The rewritten late_chunking.py uses SentenceTransformer's
# encode() for queries (CLS pooling) and raw last_hidden_state for document
# chunks.  Both live in the same embedding space because they share the
# same BGE-M3 backbone.
#
# Previous runs with jinaai/jina-embeddings-v3 produced doc_recall@5 ≈ 12%
# because raw hidden-state mean-pooling bypassed Jina's LoRA adapters and
# projection head.
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
# IMPORTANT: Delete any cached late-chunk indices from previous runs.
# The old Jina-v3-based indices are incompatible with the new BGE-M3
# implementation (different embedding dimensions / spaces).
# ---------------------------------------------------------------------------
echo "Clearing stale late-chunk caches..."
rm -rf "$SCRATCH_DIR/vector_stores/late_chunks"

# ---------------------------------------------------------------------------
# Run late-chunking experiment
#
# Key parameters:
#   --late-model BAAI/bge-m3
#       Same model as --embedding-model.  This ensures document chunk
#       embeddings (mean-pooled hidden states) and query embeddings
#       (SentenceTransformer CLS pooling) live in the same space.
#
#   --late-max-tokens 8192
#       BGE-M3's max position embeddings.  The code internally reserves
#       2 tokens for [CLS]/[SEP], so each window encodes up to 8190
#       content tokens.
#
#   --late-window-stride 512
#       Overlap between consecutive windows.  Equal to chunk_size so
#       every chunk is fully contained in at least one window.
#
#   --late-chunk-size 512
#       Retrieval chunk size in tokens.
#
#   --late-chunk-overlap 64
#       Token overlap between adjacent chunks.
# ---------------------------------------------------------------------------

python scripts/run_late_chunking_only.py \
  --pdf-dir             pdfs \
  --output-root         "$SCRATCH_OUT/chunking_sweep" \
  --vector-store-dir    vector_stores \
  --embedding-model     BAAI/bge-m3 \
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