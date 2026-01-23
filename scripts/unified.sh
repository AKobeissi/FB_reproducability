#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=unified_rag_pipeline
#SBATCH --output=unified_rag_pipeline_%j.log
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00

# --- 1. SETUP PATHS ---
# Assuming sbatch is launched from the repo root
SUBMIT_DIR=$SLURM_SUBMIT_DIR

# Path to the virtual environment
VENV_PATH="$SUBMIT_DIR/venv"

# Define Scratch Space
SCRATCH_DIR=/Tmp/$(whoami)/${SLURM_JOB_ID}
echo "Working in scratch: $SCRATCH_DIR"
mkdir -p $SCRATCH_DIR

# --- 2. COPY CODE TO SCRATCH ---
echo "Copying repository content from $SUBMIT_DIR to scratch..."
rsync -av \
    --exclude 'venv' \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude 'outputs' \
    --exclude 'vector_stores' \
    --exclude '*.log' \
    "$SUBMIT_DIR/" "$SCRATCH_DIR/"

# Move into the scratch directory
cd $SCRATCH_DIR

# --- 3. ACTIVATE ENVIRONMENT ---
echo "Activating venv from: $VENV_PATH"
source "$VENV_PATH/bin/activate"

# Ensure the scratch directory is in PYTHONPATH
export PYTHONPATH="$SCRATCH_DIR:$PYTHONPATH"

# --- 4. RUN EXPERIMENTS ---
# Output directory within scratch
SCRATCH_OUT="$SCRATCH_DIR/outputs"
mkdir -p "$SCRATCH_OUT"

# Common Settings:
# - Model: 'both' (Llama 3.2 3B + Qwen 2.5 7B)
# - HyDE Generations: 4 (Multi-HyDE)
# - Chunking: 1024 tokens (Standard baseline)

echo "========================================================"
echo "Experiment 1: Multi-HyDE + Hybrid (Dense+Sparse) + Reranker"
echo "========================================================"
python -m src.core.rag_experiments qwen \
  -e unified \
  --chunking-unit tokens \
  --chunk-size 1024 \
  --chunk-overlap 128 \
  --pdf-dir pdfs \
  --embedding-model bge-m3 \
  --unified-hyde \
  --unified-hyde-k 4 \
  --unified-retrieval dense \
  --unified-rerank \
  --output-dir "$SCRATCH_OUT/unified_hybrid_rerank"

echo "========================================================"
echo "Experiment 2: Multi-HyDE + Dense (BGE-M3) + Reranker"
echo "========================================================"
python -m src.core.rag_experiments both \
  -e unified \
  --chunking-unit tokens \
  --chunk-size 1024 \
  --chunk-overlap 128 \
  --pdf-dir pdfs \
  --embedding-model bge-m3 \
  --unified-hyde \
  --unified-hyde-k 4 \
  --unified-retrieval hybrid \
  --unified-rerank \
  --output-dir "$SCRATCH_OUT/unified_dense_rerank"

echo "========================================================"
echo "Experiment 3: Multi-HyDE + Sparse (BM25) + Reranker"
echo "========================================================"
# Note: In 'sparse' mode, HyDE generates hypotheticals (overhead)
# but BM25 is strictly lexical on the query. 
# This serves as a baseline to see if dense components are necessary.
python -m src.core.rag_experiments both \
  -e unified \
  --chunking-unit tokens \
  --chunk-size 1024 \
  --chunk-overlap 128 \
  --pdf-dir pdfs \
  --unified-hyde \
  --unified-hyde-k 4 \
  --unified-retrieval sparse \
  --unified-rerank \
  --output-dir "$SCRATCH_OUT/unified_sparse_rerank"

# --- 5. SAVE RESULTS ---
# Copy results back to the original submission directory
FINAL_DEST="$SUBMIT_DIR/outputs"

echo "Copying results back to: $FINAL_DEST"
mkdir -p "$FINAL_DEST"
cp -r "$SCRATCH_OUT/"* "$FINAL_DEST/"

# Cleanup
rm -rf $SCRATCH_DIR

echo "Done!"