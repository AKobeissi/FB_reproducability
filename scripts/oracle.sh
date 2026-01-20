#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=oracle_retrieval
#SBATCH --output=oracle_retrieval_%j.log
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00

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

# Ensure the scratch directory is in PYTHONPATH
export PYTHONPATH="$SCRATCH_DIR:$PYTHONPATH"

# --- 4. RUN EXPERIMENTS ---
# Output directory within scratch
SCRATCH_OUT="$SCRATCH_DIR/outputs"
mkdir -p "$SCRATCH_OUT"

# ---------------- ORACLE DOCUMENT EXPERIMENT ----------------
# Restricts retrieval to the Gold Document (Cross-doc vs Within-doc)
echo "--- Starting Oracle Document Experiment (Llama + Qwen) ---"
python -m src.core.rag_experiments qwen \
  -e oracle_doc \
  --pdf-dir pdfs \
  --output-dir "$SCRATCH_OUT/oracle_doc"

# ---------------- ORACLE PAGE EXPERIMENT ----------------
# Restricts retrieval to the Gold Page (Upper Bound / Perfect Retrieval)
echo "--- Starting Oracle Page Experiment (Llama + Qwen) ---"
python -m src.core.rag_experiments qwen \
  -e oracle_page \
  --pdf-dir pdfs \
  --output-dir "$SCRATCH_OUT/oracle_page"
  
# --- 5. SAVE RESULTS ---
# Copy results back to the original submission directory's output folder
FINAL_DEST="$SUBMIT_DIR/outputs"

echo "Copying results back to: $FINAL_DEST"
mkdir -p "$FINAL_DEST"
cp -r "$SCRATCH_OUT/"* "$FINAL_DEST/"

# Cleanup
rm -rf $SCRATCH_DIR

echo "Done!"