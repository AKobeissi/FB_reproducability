#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=kfold_all_optimized
#SBATCH --output=kfold_all_optimized_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=96:00:00

# =============================================================================
# K-FOLD ALL EXPERIMENTS (Optimized for FinanceBench page statistics)
#
# Parameters rationale:
#   base-model : BAAI/bge-m3  — supports 8192 tokens; best general bi-encoder
#   max-seq-length (baked into defaults): 1536 — covers 95th percentile (~6000 chars / ~1500 tokens)
#   batch-size (baked into defaults): 32   — 2× more in-batch negatives for MNRL
#   chunk-size (baked into defaults): 384  — median page (~840 tokens) → 2–3 chunks; meaningful two-stage retrieval
#   chunk-overlap (baked into defaults): 64 — ~17% overlap
#
#   Baseline  page-k=5  : focused; 5 pages × ~2–3 chunks = ~10–15 candidates → top-5
#   Reranker  page-k=10 : wider recall; 10 pages × ~2–3 chunks = ~20–30 candidates
#             reranker-candidates=50 covers all chunks from those pages
#             chunk-k=5 : final output after cross-encoder re-ranking
#
# Experiments:
#   1. train_k_fold2.py            – Baseline          (global chunk index)
#   2. train_k_fold2.py            – + Reranker        (global chunk index)
#   3. train_k_fold2_filtered_chunks.py – Baseline     (page-filtered chunk pool)
#   4. train_k_fold2_filtered_chunks.py – + Reranker   (page-filtered chunk pool)
# =============================================================================

# --- 1. SETUP PATHS ---
SUBMIT_DIR=$SLURM_SUBMIT_DIR
VENV_PATH="$SUBMIT_DIR/venv"

SCRATCH_DIR=/Tmp/$(whoami)/${SLURM_JOB_ID}
echo "Working in scratch: $SCRATCH_DIR"
mkdir -p "$SCRATCH_DIR"

# --- 2. COPY CODE TO SCRATCH ---
echo "Copying repository to scratch..."
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
echo "Activating venv: $VENV_PATH"
source "$VENV_PATH/bin/activate"

export PYTHONPATH="$SCRATCH_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# --- 4. SHARED PARAMETERS ---
SCRATCH_OUT="$SCRATCH_DIR/results"
mkdir -p "$SCRATCH_OUT"

COMMON_ARGS="--n-folds 5 --epochs 15 --batch-size 32 --lr 2e-5 --seed 42 --pdf-dir pdfs"
# --batch-size 32: 2× more in-batch negatives for MNRL vs the old default of 16
# --base-model BAAI/bge-m3 and chunk_size=384 are baked into the script defaults.

RERANKER_ARGS="--use-reranker \
  --reranker-model BAAI/bge-reranker-v2-m3 \
  --reranker-candidates 50 \
  --reranker-top-k 5 \
  --reranker-batch-size 16"

echo ""
echo "=============================================================="
echo "  K-FOLD ALL OPTIMIZED EXPERIMENTS"
echo "  Date  : $(date)"
echo "  GPU   : $CUDA_VISIBLE_DEVICES"
echo "  Scratch: $SCRATCH_DIR"
echo "=============================================================="

# =============================================================================
# EXPERIMENT 1: train_k_fold2.py — Baseline (global chunk index)
# =============================================================================
echo ""
echo "=============================================================="
echo " EXP 1/4  |  train_k_fold2  |  Baseline  |  page-k=20 chunk-k=5"
echo "           Global chunk index — no reranker"
echo "=============================================================="

python train_k_fold2.py \
    $COMMON_ARGS \
    --page-k 20 \
    --chunk-k 5 \
    --output-dir "$SCRATCH_OUT/kfold_global_baseline"

if [ $? -eq 0 ]; then
    echo "✓ EXP 1 completed"
else
    echo "✗ EXP 1 FAILED — continuing with remaining experiments"
fi

# =============================================================================
# EXPERIMENT 2: train_k_fold2.py — + Reranker (global chunk index)
# =============================================================================
echo ""
echo "=============================================================="
echo " EXP 2/4  |  train_k_fold2  |  + Reranker  |  page-k=20 chunk-k=5"
echo "           Global chunk index — with BGE reranker"
echo "=============================================================="

python train_k_fold2.py \
    $COMMON_ARGS \
    --page-k 20 \
    --chunk-k 5 \
    $RERANKER_ARGS \
    --output-dir "$SCRATCH_OUT/kfold_global_reranker"

if [ $? -eq 0 ]; then
    echo "✓ EXP 2 completed"
else
    echo "✗ EXP 2 FAILED — continuing with remaining experiments"
fi

# =============================================================================
# EXPERIMENT 3: train_k_fold2_filtered_chunks.py — Baseline (filtered pool)
# =============================================================================
echo ""
echo "=============================================================="
echo " EXP 3/4  |  train_k_fold2_filtered_chunks  |  Baseline  |  page-k=20 chunk-k=5"
echo "           Page-filtered chunk pool — no reranker"
echo "=============================================================="

python train_k_fold2_filtered_chunks.py \
    $COMMON_ARGS \
    --page-k 20 \
    --chunk-k 5 \
    --output-dir "$SCRATCH_OUT/kfold_filtered_baseline"

if [ $? -eq 0 ]; then
    echo "✓ EXP 3 completed"
else
    echo "✗ EXP 3 FAILED — continuing with remaining experiments"
fi

# =============================================================================
# EXPERIMENT 4: train_k_fold2_filtered_chunks.py — + Reranker (filtered pool)
# =============================================================================
echo ""
echo "=============================================================="
echo " EXP 4/4  |  train_k_fold2_filtered_chunks  |  + Reranker  |  page-k=20 chunk-k=5"
echo "           Page-filtered chunk pool — with BGE reranker"
echo "=============================================================="

python train_k_fold2_filtered_chunks.py \
    $COMMON_ARGS \
    --page-k 20 \
    --chunk-k 5 \
    $RERANKER_ARGS \
    --output-dir "$SCRATCH_OUT/kfold_filtered_reranker"

if [ $? -eq 0 ]; then
    echo "✓ EXP 4 completed"
else
    echo "✗ EXP 4 FAILED"
fi

# --- 5. SAVE RESULTS BACK ---
FINAL_DEST="$SUBMIT_DIR/results"
mkdir -p "$FINAL_DEST"

echo ""
echo "=============================================================="
echo "  All experiments finished — $(date)"
echo "  Copying results to: $FINAL_DEST"
echo "=============================================================="

cp -r "$SCRATCH_OUT/"* "$FINAL_DEST/"

echo ""
echo "Results saved:"
for dir in "$FINAL_DEST"/kfold_*/; do
    if [ -f "$dir/summary.txt" ]; then
        echo ""
        echo "--- $(basename $dir) ---"
        cat "$dir/summary.txt"
    fi
done

# Cleanup scratch
rm -rf "$SCRATCH_DIR"

echo ""
echo "Done!"
