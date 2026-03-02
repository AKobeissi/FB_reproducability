#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=kfold_page_reranker
#SBATCH --output=kfold_page_reranker_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=48:00:00

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
    --exclude 'results' \
    --exclude '*.log' \
    "$SUBMIT_DIR/" "$SCRATCH_DIR/"

# Move into the scratch directory
cd $SCRATCH_DIR

# --- 3. ACTIVATE ENVIRONMENT ---
echo "Activating venv from: $VENV_PATH"
source "$VENV_PATH/bin/activate"

# Ensure the scratch directory is in PYTHONPATH
export PYTHONPATH="$SCRATCH_DIR:$PYTHONPATH"

# Set environment variables
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# --- 4. RUN EXPERIMENT ---
# Output directory within scratch
SCRATCH_OUT="$SCRATCH_DIR/results"
mkdir -p "$SCRATCH_OUT"

echo "=========================================="
echo "Starting K-Fold Page Scorer + Reranker"
echo "Date: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

echo ""
echo "==================== EXPERIMENT: PAGE+CHUNK+RERANK ===================="necho "Model: BAAI/bge-m3"
echo "Page-k: 20 | Chunk-k: 5 | Reranker: BAAI/bge-reranker-v2-m3"
echo "======================================================================="

python train_k_fold2.py \
  --page-k 20 \
  --chunk-k 5 \
  --n-folds 5 \
  --epochs 15 \
  --batch-size 16 \
  --lr 2e-5 \
  --seed 42 \
  --use-reranker \
  --base-model "BAAI/bge-m3" \
  --reranker-model "BAAI/bge-reranker-v2-m3" \
  --reranker-candidates 100 \
  --reranker-top-k 5 \
  --output-dir "$SCRATCH_OUT/kfold_page_scorer_reranker" \
  --pdf-dir "pdfs"

if [ $? -eq 0 ]; then
    echo "✓ Experiment (Page+Chunk+Rerank) completed successfully"
else
    echo "✗ Experiment (Page+Chunk+Rerank) failed"
fi

# --- 5. SAVE RESULTS ---
# Copy results back to the original submission directory
FINAL_DEST="$SUBMIT_DIR/results"

echo ""
echo "=========================================="
echo "All experiments completed"
echo "Date: $(date)"
echo "=========================================="

echo "Copying results back to: $FINAL_DEST"
mkdir -p "$FINAL_DEST"
cp -r "$SCRATCH_OUT/"* "$FINAL_DEST/"

# Cleanup
rm -rf $SCRATCH_DIR

echo "Done!"
