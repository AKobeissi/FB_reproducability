#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=colpali
#SBATCH --output=colpali_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:ls40:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00


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
SCRATCH_OUT="${SCRATCH_DIR}/outputs"
SCRATCH_RES="${SCRATCH_DIR}/results"

echo "Activating venv: $VENV_PATH"
source "$VENV_PATH/bin/activate"

export HF_HOME=/data/rech/kobeissa/hf
export HF_HUB_CACHE=/data/rech/kobeissa/hf/hub
export HF_DATASETS_CACHE=/data/rech/kobeissa/hf/datasets

# ── Dependency check ──────────────────────────────────────────────────────────
python - <<'EOF'
import importlib, sys
missing = []
for pkg in ["colpali_engine", "PIL", "fitz"]:
    try:
        importlib.import_module(pkg)
    except ImportError:
        missing.append(pkg)
if missing:
    print(f"[WARN] Missing packages: {missing}. Installing …")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                           "colpali-engine>=0.3.0,<0.4.0", "Pillow", "pymupdf"])
else:
    print("[OK] All dependencies present.")
# pdfplumber optional but recommended
try:
    import pdfplumber
    print("[OK] pdfplumber available (structured table extraction enabled).")
except ImportError:
    print("[WARN] pdfplumber not found — install for better table recall: pip install pdfplumber")
EOF

export PYTHONPATH="${SCRATCH_DIR}:${PYTHONPATH:-}"

# ── Run experiment ────────────────────────────────────────────────────────────
echo "Starting colpali_rerank at $(date)"

python src/core/rag_experiments.py qwen \
  --experiment        colpali_rerank \
  --pdf-dir           ./pdfs \
  --output-dir        ./outputs \
  --vector-store-dir  ./vector_stores \
  --embedding-model   bge-m3 \
  --colpali-model     vidore/colpali-v1.2 \
  --colpali-dpi       150 \
  --colpali-top-m     20 \
  --colpali-alpha     0.35 \
  --top-k             5 \
  --max-new-tokens    512 \
  --max-context-chars 16000 \
  --eval-type         both \
  --eval-mode         static

EXIT_CODE=$?
echo "Experiment finished (exit code: ${EXIT_CODE}) at $(date)"

# ── Copy outputs AND results back ─────────────────────────────────────────────
FINAL_OUTPUTS="${SUBMIT_DIR}/outputs"
FINAL_RESULTS="${SUBMIT_DIR}/results"
mkdir -p "${FINAL_OUTPUTS}" "${FINAL_RESULTS}"

echo "Copying outputs to: ${FINAL_OUTPUTS}"
if [ -d "${SCRATCH_OUT}" ] && [ "$(ls -A "${SCRATCH_OUT}" 2>/dev/null)" ]; then
    cp -r "${SCRATCH_OUT}/"* "${FINAL_OUTPUTS}/"
else
    echo "[WARN] outputs/ is empty — check experiment logs."
fi

echo "Copying scored results to: ${FINAL_RESULTS}"
if [ -d "${SCRATCH_RES}" ] && [ "$(ls -A "${SCRATCH_RES}" 2>/dev/null)" ]; then
    cp -r "${SCRATCH_RES}/"* "${FINAL_RESULTS}/"
else
    echo "[WARN] results/ is empty — scored JSON may be missing (evaluation may have failed)."
fi

# ── Cleanup ───────────────────────────────────────────────────────────────────
rm -rf "${SCRATCH_DIR}"

echo "Done. Exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
