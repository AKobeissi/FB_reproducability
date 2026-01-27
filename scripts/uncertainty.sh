#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=uncertainty_quantification
#SBATCH --output=uncertainty_quantification_%j.log
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

echo "========================================================"
echo "Experiment: Uncertainty Quantification (MC-Dropout)"
echo "Settings: T=10, L_mc=30, Alpha=1.0, Model=Qwen"
echo "========================================================"

# Run Uncertainty Experiment
# We use 'qwen' here as the base generator, but you can change to 'llama' or 'both'
python -m src.core.rag_experiments qwen \
  -e uncertainty \
  --pdf-dir pdfs \
  --chunking-unit tokens \
  --chunk-size 1024 \
  --chunk-overlap 128 \
  --embedding-model "BAAI/bge-m3" \
  --reranker-model "BAAI/bge-reranker-v2-m3" \
  --mc-t 10 \
  --mc-l 30 \
  --mc-alpha 1.0 \
  --k-cand 100 \
  --output-dir "$SCRATCH_OUT/uncertainty_experiment"

# --- 5. SAVE RESULTS ---
# Copy results back to the original submission directory
FINAL_DEST="$SUBMIT_DIR/outputs"

echo "Copying results back to: $FINAL_DEST"
mkdir -p "$FINAL_DEST"
cp -r "$SCRATCH_OUT/"* "$FINAL_DEST/"

# Cleanup
rm -rf $SCRATCH_DIR

echo "Done!"s