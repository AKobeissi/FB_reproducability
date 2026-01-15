#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=HyDE_exp
#SBATCH --output=HyDE_%j.log
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# --- 1. SETUP PATHS ---
# SLURM_SUBMIT_DIR is the directory where you ran sbatch
SUBMIT_DIR=$SLURM_SUBMIT_DIR

# The specific project folder name
PROJECT_NAME="FB_reproducability"

VENV_PATH="$SUBMIT_DIR/$PROJECT_NAME/venv"

# Define Scratch Space
SCRATCH_DIR=/Tmp/$(whoami)/${SLURM_JOB_ID}
echo "Working in scratch: $SCRATCH_DIR"
mkdir -p $SCRATCH_DIR

# --- 2. COPY CODE TO SCRATCH ---
# Copy the 'FB_reproducability' folder into the scratch directory.
echo "Copying $PROJECT_NAME to scratch..."

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
echo "Activating venv from: $VENV_PATH"
source "$VENV_PATH/bin/activate"

# --- 4. RUN EXPERIMENTS ---
# Define local output directory in scratch
SCRATCH_OUT="$SCRATCH_DIR/$PROJECT_NAME/outputs"

# IMPORTANT: Added --use-all-pdfs because HyDE experiments are 'shared' index types.

# --- Set 1: Baseline Context (1024 Tokens) ---

echo "--- Starting Single HyDE (1024 Tokens) ---"
python -m FB_reproducability.rag_experiments qwen \
  -e hyde_shared \
  --chunking-unit tokens \
  --chunk-size 1024 \
  --chunk-overlap 128 \
  --output-dir "$SCRATCH_OUT/hyde_1024"

echo "--- Starting Multi-HyDE (1024 Tokens) ---"
python -m FB_reproducability.rag_experiments qwen \
  -e multi_hyde_shared \
  --chunking-unit tokens \
  --chunk-size 1024 \
  --chunk-overlap 128 \
  --output-dir "$SCRATCH_OUT/multi_hyde_1024"

# --- Set 2: Granular Context (512 Tokens) ---
# Smaller chunks often work better for dense vector retrieval (like HyDE)

echo "--- Starting Single HyDE (512 Tokens) ---"
python -m FB_reproducability.rag_experiments qwen \
  -e hyde_shared \
  --chunking-unit tokens \
  --chunk-size 512 \
  --chunk-overlap 64 \
  --output-dir "$SCRATCH_OUT/hyde_512"

echo "--- Starting Multi-HyDE (512 Tokens) ---"
python -m FB_reproducability.rag_experiments qwen \
  -e multi_hyde_shared \
  --chunking-unit tokens \
  --chunk-size 512 \
  --chunk-overlap 64 \
  --output-dir "$SCRATCH_OUT/multi_hyde_512"


# --- 5. SAVE RESULTS ---
# Copy results back to permanent storage
FINAL_DEST="$SUBMIT_DIR/$PROJECT_NAME/outputs"

echo "Copying results back to: $FINAL_DEST"
mkdir -p "$FINAL_DEST"
cp -r "$SCRATCH_OUT/"* "$FINAL_DEST/"

# Cleanup
rm -rf $SCRATCH_DIR

echo "Done!"