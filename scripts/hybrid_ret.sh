#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=hybrid_splade_retrieval
#SBATCH --output=hybrid_splade_retrieval_%j.log
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# --- 1. SETUP PATHS ---
# Assuming sbatch is launched from the repo root (FB_reproducability)
SUBMIT_DIR=$SLURM_SUBMIT_DIR

# Path to the virtual environment (inside the repo)
VENV_PATH="$SUBMIT_DIR/venv"

# Define Scratch Space
SCRATCH_DIR=/Tmp/$(whoami)/${SLURM_JOB_ID}
echo "Working in scratch: $SCRATCH_DIR"
mkdir -p $SCRATCH_DIR

# --- 2. COPY CODE TO SCRATCH ---
echo "Copying repository content from $SUBMIT_DIR to scratch..."

# rsync content of the current directory to scratch
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

# Ensure the scratch directory is in PYTHONPATH so python -m src... works
export PYTHONPATH="$SCRATCH_DIR:$PYTHONPATH"

# --- 4. RUN EXPERIMENTS ---
# Output directory within scratch
SCRATCH_OUT="$SCRATCH_DIR/outputs"
mkdir -p "$SCRATCH_OUT"

# Common parameters: 1024 tokens, 128 overlap, Qwen model

# ---------------- SPLADE EXPERIMENT ----------------
# SPLADE uses its own specific model (naver/splade...), so embedding arg doesn't apply here.
echo "--- Starting SPLADE (1024 tokens / 128 overlap) ---"
python -m src.core.rag_experiments qwen \
  -e splade \
  --chunking-unit tokens \
  --chunk-size 1024 \
  --chunk-overlap 128 \
  --output-dir "$SCRATCH_OUT/splade_1024"

echo "--- Starting SPLADE (512 tokens / 64 overlap) ---"
python -m src.core.rag_experiments qwen \
  -e splade \
  --chunking-unit tokens \
  --chunk-size 512 \
  --chunk-overlap 64 \
  --output-dir "$SCRATCH_OUT/splade_512"
  
# ---------------- HYBRID EXPERIMENTS ----------------

# 1. Hybrid with Base Embedding (Default: all-mpnet-base-v2)
echo "--- Starting Hybrid [Base/MPNet] (1024 tokens / 128 overlap) ---"
python -m src.core.rag_experiments qwen \
  -e hybrid \
  --chunking-unit tokens \
  --chunk-size 1024 \
  --chunk-overlap 128 \
  --embedding-model mpnet \
  --output-dir "$SCRATCH_OUT/hybrid_base_1024"

# 2. Hybrid with BGE Embedding (BGE-M3)
echo "--- Starting Hybrid [BGE-M3] (1024 tokens / 128 overlap) ---"
python -m src.core.rag_experiments qwen \
  -e hybrid \
  --chunking-unit tokens \
  --chunk-size 1024 \
  --chunk-overlap 128 \
  --embedding-model bge-m3 \
  --output-dir "$SCRATCH_OUT/hybrid_bge_1024"

# --- 5. SAVE RESULTS ---
# Copy results back to the original submission directory's output folder
FINAL_DEST="$SUBMIT_DIR/outputs"

echo "Copying results back to: $FINAL_DEST"
mkdir -p "$FINAL_DEST"
cp -r "$SCRATCH_OUT/"* "$FINAL_DEST/"

# Cleanup
rm -rf $SCRATCH_DIR

echo "Done!"