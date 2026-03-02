#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=reranker_experiments
#SBATCH --output=reranker_experiments_%j.log
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
echo "Starting Reranker Experiments"
echo "Date: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# Experiment 1: Cross-Encoder Reranker
echo ""
echo "==================== EXPERIMENT 1: CROSS-ENCODER ===================="
echo "Embedding: BAAI/bge-m3"
echo "Reranker: BAAI/bge-reranker-v2-m3"
echo "Initial K: 100, Top-K: 5"
echo "====================================================================="

# Recreate vector store for this experiment
rm -rf "vector_stores"

# python -m src.core.rag_experiments qwen \
#     -e reranking \
#     --embedding-model "BAAI/bge-m3" \
#     --reranker-model "BAAI/bge-reranker-v2-m3" \
#     --reranker-style "cross_encoder" \
#     --k-cand 100 \
#     --top-k 5 \
#     --chunk-size 1024 \
#     --chunk-overlap 128 \
#     --chunking-unit "tokens" \
#     --output-dir "$SCRATCH_OUT/reranking_cross_encoder" \
#     --pdf-dir "pdfs" 

if [ $? -eq 0 ]; then
    echo "✓ Experiment 1 (Cross-Encoder) completed successfully"
else
    echo "✗ Experiment 1 (Cross-Encoder) failed"
fi

# Experiment 2: Late-Interaction Reranker (ColBERT-style)
echo ""
echo "==================== EXPERIMENT 2: LATE-INTERACTION ===================="
echo "Embedding: BAAI/bge-m3"
echo "Reranker: colbert-ir/colbertv2.0"
echo "Initial K: 100, Top-K: 5"
echo "=========================================================================="

# Recreate vector store for this experiment
rm -rf "vector_stores"

python -m src.core.rag_experiments qwen \
    -e reranking \
    --embedding-model "BAAI/bge-m3" \
    --reranker-style "late_interaction" \
    --late-interaction-model "colbert-ir/colbertv2.0" \
    --late-interaction-query-max-len 128 \
    --late-interaction-doc-max-len 512 \
    --k-cand 100 \
    --top-k 5 \
    --chunk-size 512 \
    --chunk-overlap 64 \
    --chunking-unit "tokens" \
    --output-dir "$SCRATCH_OUT/reranking_late_interaction1" \
    --pdf-dir "pdfs" 

if [ $? -eq 0 ]; then
    echo "✓ Experiment 2 (Late-Interaction) completed successfully"
else
    echo "✗ Experiment 2 (Late-Interaction) failed"
fi

# Experiment 3: Slot-Coverage Reranking (Set-Cover)
echo ""
echo "==================== EXPERIMENT 3: SLOT-COVERAGE ===================="
echo "Embedding: BAAI/bge-m3"
echo "Selection: Greedy set-cover with redundancy penalty"
echo "Initial K: 100, Top-K: 5"
echo "Evaluation: Retrieval + Numeric Match"
echo "======================================================================"

# Recreate vector store for this experiment
rm -rf "vector_stores"

python -m src.core.rag_experiments qwen \
    -e slot_coverage \
    --embedding-model "BAAI/bge-m3" \
    --k-cand 100 \
    --top-k 5 \
    --chunk-size 1024 \
    --chunk-overlap 128 \
    --chunking-unit "tokens" \
    --eval-type retrieval \
    --output-dir "$SCRATCH_OUT/slot_coverage_reranking" \
    --pdf-dir "pdfs" 

if [ $? -eq 0 ]; then
    echo "✓ Experiment 3 (Slot-Coverage) completed successfully"
else
    echo "✗ Experiment 3 (Slot-Coverage) failed"
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
