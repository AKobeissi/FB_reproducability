#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=rag_embeddings
#SBATCH --output=rag_embeddings_%j.log
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00

# --- 1. SETUP PATHS ---
# SLURM_SUBMIT_DIR = /u/kobeissa/Documents/thesis/experiments
SUBMIT_DIR=$SLURM_SUBMIT_DIR

# The specific project folder name
PROJECT_NAME="FB_reproducability"

VENV_PATH="$SUBMIT_DIR/$PROJECT_NAME/venv"

# Define Scratch Space
SCRATCH_DIR=/Tmp/$(whoami)/${SLURM_JOB_ID}
echo "Working in scratch: $SCRATCH_DIR"
mkdir -p $SCRATCH_DIR

# --- 2. COPY CODE TO SCRATCH ---
echo "Copying $PROJECT_NAME to scratch..."

rsync -av \
    --exclude 'venv' \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude 'outputs' \
    --exclude 'vector_stores' \
    "$SUBMIT_DIR/$PROJECT_NAME" "$SCRATCH_DIR/"

# Move into the scratch directory
cd $SCRATCH_DIR

# --- 3. ACTIVATE ENVIRONMENT ---
echo "Activating venv from: $VENV_PATH"
source "$VENV_PATH/bin/activate"

# --- 4. RUN EXPERIMENTS ---
# We use the new --embedding-model flag with the aliases we added.
# We'll use a fixed chunk size of 1024 tokens for a fair comparison.

SCRATCH_OUT="$SCRATCH_DIR/$PROJECT_NAME/outputs"
mkdir -p "$SCRATCH_OUT"

echo "--- Starting Experiment 1: BGE-M3 ---"
python -m FB_reproducability.rag_experiments qwen \
  -e shared \
  --embedding-model bge-m3 \
  --chunking-unit tokens \
  --chunk-size 512 \
  --chunk-overlap 64 \
  --output-dir "$SCRATCH_OUT/bge_m3_512"

echo "--- Starting Experiment 2: FinanceMTEB (Fin-E5) ---"
python -m FB_reproducability.rag_experiments qwen \
  -e shared \
  --embedding-model financemteb \
  --chunking-unit tokens \
  --chunk-size 512 \
  --chunk-overlap 64 \
  --output-dir "$SCRATCH_OUT/financemteb_512"

# --- 5. SAVE RESULTS ---
FINAL_DEST="$SUBMIT_DIR/$PROJECT_NAME/outputs"

echo "Copying results back to: $FINAL_DEST"
mkdir -p "$FINAL_DEST"
cp -r "$SCRATCH_OUT/"* "$FINAL_DEST/"

# Cleanup
rm -rf $SCRATCH_DIR

echo "Done!"