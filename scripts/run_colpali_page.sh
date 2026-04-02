#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=colqwen2
#SBATCH --output=colqwen2_%j.log
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:ls40:1

#SBATCH --mem=48G
#SBATCH --time=24:00:00

# =============================================================================
# run_colqwen2_page.sh  —  SLURM job for the ColQwen2 page retrieval experiment
# =============================================================================

SUBMIT_DIR=$SLURM_SUBMIT_DIR
VENV_PATH="$SUBMIT_DIR/venv"
SCRATCH_DIR=/Tmp/$(whoami)/${SLURM_JOB_ID}

echo "========================================================="
echo "Job:         $SLURM_JOB_ID"
echo "Node:        $SLURMD_NODENAME"
echo "Submit dir:  $SUBMIT_DIR"
echo "Scratch dir: $SCRATCH_DIR"
echo "========================================================="

mkdir -p "$SCRATCH_DIR"

echo "Copying repository to scratch..."
rsync -a --quiet \
  --exclude 'venv' \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude 'outputs' \
  --exclude 'vector_stores' \
  --exclude '*.log' \
  "$SUBMIT_DIR/" "$SCRATCH_DIR/"

cd "$SCRATCH_DIR"

echo "Activating venv: $VENV_PATH"
source "$VENV_PATH/bin/activate"

python - <<'PY'
missing = []

try:
    from transformers import ColQwen2ForRetrieval, ColQwen2Processor
except Exception:
    missing.append("transformers[ColQwen2]")

for name in ("PIL", "fitz"):
    try:
        __import__(name)
    except Exception:
        missing.append(name)

if missing:
    raise SystemExit("Missing: " + ", ".join(missing))
PY

if [ $? -ne 0 ]; then
  echo "Installing missing ColQwen2 dependencies in venv..."
  pip install --upgrade pip
  pip install -r requirements.txt
  pip install -U transformers accelerate pillow pymupdf
fi

export PYTHONPATH="$SCRATCH_DIR:$PYTHONPATH"

SCRATCH_OUT="$SCRATCH_DIR/outputs"
mkdir -p "$SCRATCH_OUT"

python src/core/rag_experiments.py qwen --experiment colpali_page \
  --pdf-dir ./pdfs \
  --output-dir ./outputs \
  --vector-store-dir ./vector_stores \
  --colpali-model vidore/colqwen2-v1.0-hf \
  --colpali-dpi 150 \
  --top-k 5 \
  --max-context-chars 16000 \
  --colqwen2-first-stage-k 50 \
  --colqwen2-top-docs 3 \
  --colqwen2-pages-per-doc 8 \
  --colqwen2-neighbor-window 1 \
  --colqwen2-visual-weight 0.50 \
  --colqwen2-text-weight 0.40 \
  --colqwen2-doc-bonus-weight 0.10 \
  --colqwen2-page-batch-size 4 \
  --colqwen2-query-batch-size 1 \
  --colqwen2-score-batch-size 64

EXIT_CODE=$?

FINAL_DEST="$SUBMIT_DIR/outputs"
echo "Copying results to: $FINAL_DEST"
mkdir -p "$FINAL_DEST"
cp -r "$SCRATCH_OUT/"* "$FINAL_DEST/"

SCRATCH_RESULTS="$SCRATCH_DIR/results"
if [ -d "$SCRATCH_RESULTS" ]; then
  FINAL_RESULTS="$SUBMIT_DIR/results"
  mkdir -p "$FINAL_RESULTS"
  cp -r "$SCRATCH_RESULTS/"* "$FINAL_RESULTS/"
  echo "Scored results copied to: $FINAL_RESULTS"
fi

rm -rf "$SCRATCH_DIR"

echo "Done (exit code: $EXIT_CODE)"
exit $EXIT_CODE
