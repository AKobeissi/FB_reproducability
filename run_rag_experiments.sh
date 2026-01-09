#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=rag_chunking
#SBATCH --output=rag_chunking_%j.log
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
# Want to copy the 'FB_reproducability' folder
# into the scratch directory.
echo "Copying $PROJECT_NAME to scratch..."

# rsync source: $SUBMIT_DIR/$PROJECT_NAME (the folder itself)
# rsync dest:   $SCRATCH_DIR/             (puts the folder inside scratch)
rsync -av \
    --exclude 'venv' \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude 'outputs' \
    --exclude 'vector_stores' \
    "$SUBMIT_DIR/$PROJECT_NAME" "$SCRATCH_DIR/"

# Move into the scratch directory so we are sitting right outside the module
cd $SCRATCH_DIR

# --- 3. ACTIVATE ENVIRONMENT ---
# We activate the venv from its original permanent location
echo "Activating venv from: $VENV_PATH"
source "$VENV_PATH/bin/activate"

# --- 4. RUN EXPERIMENTS ---
# Since we are in $SCRATCH_DIR, and 'FB_reproducability' is a folder here,
# we can run 'python -m FB_reproducability.rag_experiments' directly.

# Define local output directory in scratch
SCRATCH_OUT="$SCRATCH_DIR/$PROJECT_NAME/outputs"

echo "--- Starting 128 Token Experiment ---"
python -m FB_reproducability.rag_experiments qwen \
  -e shared \
  --chunking-unit tokens \
  --chunk-size 128 \
  --chunk-overlap 16 \
  --output-dir "$SCRATCH_OUT/tokens_128"

echo "--- Starting 256 Token Experiment ---"
python -m FB_reproducability.rag_experiments qwen \
  -e shared \
  --chunking-unit tokens \
  --chunk-size 256 \
  --chunk-overlap 32 \
  --output-dir "$SCRATCH_OUT/tokens_256"

echo "--- Starting 512 Token Experiment ---"
python -m FB_reproducability.rag_experiments qwen \
  -e shared \
  --chunking-unit tokens \
  --chunk-size 512 \
  --chunk-overlap 64 \
  --output-dir "$SCRATCH_OUT/tokens_512"

echo "--- Starting 1024 Token Experiment ---"
python -m FB_reproducability.rag_experiments qwen \
  -e shared \
  --chunking-unit tokens \
  --chunk-size 1024 \
  --chunk-overlap 128 \
  --output-dir "$SCRATCH_OUT/tokens_1024"

# --- 5. SAVE RESULTS ---
# Copy results back to permanent storage
# We save them to experiments/FB_reproducability/outputs (or wherever you prefer)
FINAL_DEST="$SUBMIT_DIR/$PROJECT_NAME/outputs"

echo "Copying results back to: $FINAL_DEST"
mkdir -p "$FINAL_DEST"
cp -r "$SCRATCH_OUT/"* "$FINAL_DEST/"

# Cleanup (optional)
rm -rf $SCRATCH_DIR

echo "Done!"