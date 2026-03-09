#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=colpali
#SBATCH --output=colpali_%j.log
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00

# =============================================================================
# run_colpali_page.sh  —  SLURM job for the ColPali page retrieval experiment
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
import importlib
missing = []
for name in ("colpali_engine", "PIL"):
  try:
    importlib.import_module(name)
  except Exception:
    missing.append(name)
if missing:
  raise SystemExit("Missing: " + ", ".join(missing))
PY

if [ $? -ne 0 ]; then
  echo "Installing missing ColPali dependencies in venv..."
  pip install --upgrade pip
  pip install -r requirements.txt
  pip install "colpali-engine>=0.3.0,<0.4.0"
fi

export PYTHONPATH="$SCRATCH_DIR:$PYTHONPATH"

SCRATCH_OUT="$SCRATCH_DIR/outputs"
mkdir -p "$SCRATCH_OUT"

python src/core/rag_experiments.py qwen --experiment colpali_page \
  --pdf-dir ./pdfs \
  --output-dir ./outputs \
  --vector-store-dir ./vector_stores \
  --colpali-model vidore/colpali-v1.2 \
  --colpali-dpi 150

EXIT_CODE=$?

FINAL_DEST="$SUBMIT_DIR/outputs"
echo "Copying results to: $FINAL_DEST"
mkdir -p "$FINAL_DEST"
cp -r "$SCRATCH_OUT/"* "$FINAL_DEST/"

# Also copy the results/ dir (where scored JSONs land)
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