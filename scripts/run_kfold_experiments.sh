#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=kfold_page_scorer
#SBATCH --output=kfold_page_scorer_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=48:00:00

# =============================================================================
# K-FOLD PAGE-SCORER EXPERIMENTS (BGE-M3)
#
# Runs train_k_fold2.py for multiple page budgets:
#   1) page-k=20
#   2) page-k=10
#   3) page-k=5
# =============================================================================

# --- 1. SETUP PATHS ---
SUBMIT_DIR=$SLURM_SUBMIT_DIR
VENV_PATH="$SUBMIT_DIR/venv"

SCRATCH_DIR=/Tmp/$(whoami)/${SLURM_JOB_ID}
echo "Working in scratch: $SCRATCH_DIR"
mkdir -p "$SCRATCH_DIR"

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

cd "$SCRATCH_DIR"

# --- 3. ACTIVATE ENVIRONMENT ---
echo "Activating venv from: $VENV_PATH"
source "$VENV_PATH/bin/activate"

export PYTHONPATH="$SCRATCH_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# --- 4. RUN EXPERIMENTS ---
SCRATCH_OUT="$SCRATCH_DIR/results"
mkdir -p "$SCRATCH_OUT"

COMMON_ARGS="--n-folds 5 --epochs 15 --batch-size 16 --lr 2e-5 --chunk-k 5 --base-model BAAI/bge-m3 --pdf-dir pdfs --seed 42"

echo "=========================================="
echo "Starting K-Fold Experiments (BGE-M3)"
echo "Date: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# =============================================================================
# EXPERIMENT 1: BGE-M3 | page-k=20
# =============================================================================
echo ""
echo "==================== EXPERIMENT 1: BGE-M3 P=20 ===================="
echo "Model: BAAI/bge-m3 | chunk-k=5"
echo "=============================================================="

python train_k_fold2.py \
    $COMMON_ARGS \
    --page-k 20 \
    --output-dir "$SCRATCH_OUT/kfold_page_scorer_bge_m3_p20" \
    

if [ $? -eq 0 ]; then
    echo "✓ Experiment 1 (BGE-M3 P=20) completed successfully"
else
    echo "✗ Experiment 1 (BGE-M3 P=20) failed"
fi

# =============================================================================
# EXPERIMENT 2: BGE-M3 | page-k=10
# =============================================================================
echo ""
echo "==================== EXPERIMENT 2: BGE-M3 P=10 ===================="
echo "Model: BAAI/bge-m3 | chunk-k=5"
echo "====================================================================="

python train_k_fold2.py \
    $COMMON_ARGS \
    --page-k 10 \
    --output-dir "$SCRATCH_OUT/kfold_page_scorer_bge_m3_p10" \


if [ $? -eq 0 ]; then
    echo "✓ Experiment 2 (BGE-M3 P=10) completed successfully"
else
    echo "✗ Experiment 2 (BGE-M3 P=10) failed"
fi

# =============================================================================
# EXPERIMENT 3: BGE-M3 | page-k=5
# =============================================================================
echo ""
echo "==================== EXPERIMENT 3: BGE-M3 P=5 ===================="
echo "Model: BAAI/bge-m3 | chunk-k=5"
echo "==========================================================================="

python train_k_fold2.py \
    $COMMON_ARGS \
    --page-k 5 \
    --output-dir "$SCRATCH_OUT/kfold_page_scorer_bge_m3_p5" \


if [ $? -eq 0 ]; then
    echo "✓ Experiment 3 (BGE-M3 P=5) completed successfully"
else
    echo "✗ Experiment 3 (BGE-M3 P=5) failed"
fi

# --- 5. SAVE RESULTS ---
FINAL_DEST="$SUBMIT_DIR/results"

echo ""
echo "=========================================="
echo "All experiments completed"
echo "Date: $(date)"
echo "=========================================="

echo "Copying results back to: $FINAL_DEST"
mkdir -p "$FINAL_DEST"
cp -r "$SCRATCH_OUT/"* "$FINAL_DEST/" 2>/dev/null || true

# Cleanup
rm -rf "$SCRATCH_DIR"

echo "Done!"
