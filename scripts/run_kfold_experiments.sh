#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=kfold_page_scorer
#SBATCH --output=kfold_page_scorer_%j.log
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

# --- 4. RUN EXPERIMENTS ---
# Output directory within scratch
SCRATCH_OUT="$SCRATCH_DIR/results"
mkdir -p "$SCRATCH_OUT"

echo "=========================================="
echo "Starting K-Fold Experiments"
echo "Date: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# # Experiment 1: MPNet (all-mpnet-base-v2)
# echo ""
# echo "==================== EXPERIMENT 1: MPNet ===================="
# echo "Model: sentence-transformers/all-mpnet-base-v2"
# echo "Chunk size: 1024 tokens, Overlap: 128 tokens"
# echo "=============================================================="

# python train_k_fold.py \
#     --n-folds 5 \
#     --epochs 15 \
#     --batch-size 16 \
#     --lr 2e-5 \
#     --page-k 5 \
#     --chunk-k 5 \
#     --base-model "sentence-transformers/all-mpnet-base-v2" \
#     --output-dir "$SCRATCH_OUT/kfold_page_scorer_mpnet" \
#     --pdf-dir "pdfs" \
#     --seed 42

# if [ $? -eq 0 ]; then
#     echo "✓ Experiment 1 (MPNet) completed successfully"
# else
#     echo "✗ Experiment 1 (MPNet) failed"
# fi

# Experiment 2: BGE-M3
echo ""
echo "==================== EXPERIMENT 1: BGE-M3 P=20 ===================="
echo "Model: BAAI/bge-m3"
echo "Chunk size: 1024 tokens, Overlap: 128 tokens"
echo "=============================================================="

python train_k_fold2.py \
    --n-folds 5 \
    --epochs 15 \
    --batch-size 16 \
    --lr 2e-5 \
    --page-k 20 \
    --chunk-k 5 \
    --base-model "BAAI/bge-m3" \
    --output-dir "$SCRATCH_OUT/kfold_page_scorer_bge_m3_p20" \
    --pdf-dir "pdfs" \
    --seed 42

if [ $? -eq 0 ]; then
    echo "✓ Experiment 1 (BGE-M3) completed successfully"
else
    echo "✗ Experiment 1 (BGE-M3) failed"
fi

# # Experiment 3: MPNet + HyDE (single)
# echo ""
# echo "==================== EXPERIMENT 3: MPNet + HyDE ===================="
# echo "Model: sentence-transformers/all-mpnet-base-v2"
# echo "Chunk size: 1024 tokens, Overlap: 128 tokens"
# echo "HyDE: Single generation with Qwen 2.5 7B"
# echo "====================================================================="

# python train_k_fold.py \
#     --n-folds 5 \
#     --epochs 15 \
#     --batch-size 16 \
#     --lr 2e-5 \
#     --page-k 5 \
#     --chunk-k 5 \
#     --base-model "sentence-transformers/all-mpnet-base-v2" \
#     --output-dir "$SCRATCH_OUT/kfold_page_scorer_mpnet_hyde" \
#     --pdf-dir "pdfs" \
#     --seed 42 \
#     --use-hyde \
#     --hyde-num-generations 1

# if [ $? -eq 0 ]; then
#     echo "✓ Experiment 3 (MPNet + HyDE) completed successfully"
# else
#     echo "✗ Experiment 3 (MPNet + HyDE) failed"
# fi

# # Experiment 4: MPNet + Multi-HyDE (k=4)
# echo ""
# echo "==================== EXPERIMENT 4: MPNet + Multi-HyDE ===================="
# echo "Model: sentence-transformers/all-mpnet-base-v2"
# echo "Chunk size: 1024 tokens, Overlap: 128 tokens"
# echo "Multi-HyDE: 4 generations with Qwen 2.5 7B, mean aggregation"
# echo "=========================================================================="

# python train_k_fold.py \
#     --n-folds 5 \
#     --epochs 15 \
#     --batch-size 16 \
#     --lr 2e-5 \
#     --page-k 5 \
#     --chunk-k 5 \
#     --base-model "sentence-transformers/all-mpnet-base-v2" \
#     --output-dir "$SCRATCH_OUT/kfold_page_scorer_mpnet_multihyde" \
#     --pdf-dir "pdfs" \
#     --seed 42 \
#     --use-hyde \
#     --hyde-num-generations 4 \
#     --hyde-aggregate mean

# if [ $? -eq 0 ]; then
#     echo "✓ Experiment 4 (MPNet + Multi-HyDE) completed successfully"
# else
#     echo "✗ Experiment 4 (MPNet + Multi-HyDE) failed"
# fi

# Experiment 1: BGE-M3 + HyDE (single)
echo ""
echo "==================== EXPERIMENT 2: BGE-M3 + P=5 ===================="
echo "Model: BAAI/bge-m3"
echo "Chunk size: 1024 tokens, Overlap: 128 tokens"
echo "HyDE: Single generation with Qwen 2.5 7B"
echo "====================================================================="

python train_k_fold2.py \
    --n-folds 5 \
    --epochs 15 \
    --batch-size 16 \
    --lr 2e-5 \
    --page-k 5 \
    --chunk-k 5 \
    --base-model "BAAI/bge-m3" \
    --output-dir "$SCRATCH_OUT/kfold_page_scorer_bge_m3_p5_base_embed" \
    --pdf-dir "pdfs" \
    --seed 42 \
 

if [ $? -eq 0 ]; then
    echo "✓ Experiment 2 (BGE-M3 + HyDE) completed successfully"
else
    echo "✗ Experiment 2 (BGE-M3 + HyDE) failed"
fi

# Experiment 2: BGE-M3 + Multi-HyDE (k=4)
echo ""
echo "==================== EXPERIMENT 3: BGE-M3 + P=10 ===================="
echo "Model: BAAI/bge-m3"
echo "Chunk size: 1024 tokens, Overlap: 128 tokens"
echo "Multi-HyDE: 4 generations with Qwen 2.5 7B, mean aggregation"
echo "==========================================================================="

python train_k_fold2.py \
    --n-folds 5 \
    --epochs 15 \
    --batch-size 16 \
    --lr 2e-5 \
    --page-k 10 \
    --chunk-k 5 \
    --base-model "BAAI/bge-m3" \
    --output-dir "$SCRATCH_OUT/kfold_page_scorer_bge_m3_p10" \
    --pdf-dir "pdfs" \
    --seed 42 

if [ $? -eq 0 ]; then
    echo "✓ Experiment 3 (BGE-M3 + Multi-HyDE) completed successfully"
else
    echo "✗ Experiment 3 (BGE-M3 + Multi-HyDE) failed"
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
