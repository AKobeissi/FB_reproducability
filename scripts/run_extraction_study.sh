#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=fb_extraction_study
#SBATCH --output=logs/extraction_study_%j.log
#SBATCH --error=logs/extraction_study_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00

# ── Robust Environment Setup (inspired by chunking_exp.sh) ────────────────────
SUBMIT_DIR=$SLURM_SUBMIT_DIR
VENV_PATH="$SUBMIT_DIR/venv"
SCRATCH_DIR=/Tmp/$(whoami)/${SLURM_JOB_ID}

echo "========================================================="
echo "Job:         $SLURM_JOB_ID"
echo "Node:        $SLURMD_NODENAME"
echo "Submit dir:  $SUBMIT_DIR"
echo "Scratch dir: $SCRATCH_DIR"
echo "Start time:  $(date)"
echo "========================================================="

mkdir -p "$SCRATCH_DIR"
mkdir -p "$SUBMIT_DIR/logs"

echo "Copying repository to scratch..."
rsync -a --quiet \
  --exclude 'venv' \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude 'outputs' \
  --exclude 'vector_stores' \
  --exclude '*.log' \
  --exclude 'models' \
  --exclude 'hierarchical_rag*' \
  --exclude 'Final-PDF' \
  --exclude 'PDF-Opus*' \
  --exclude 'pdfff' \
  --exclude 'pdfs-extended-v4' \
  --exclude 'pdf-ext-v3' \
  --exclude 'pdf2' \
  "$SUBMIT_DIR/" "$SCRATCH_DIR/"

cd "$SCRATCH_DIR"
SCRATCH_OUT="${SCRATCH_DIR}/outputs"
mkdir -p "$SCRATCH_OUT"

echo "Activating venv: $VENV_PATH"
source "$VENV_PATH/bin/activate"

# Hugging Face Cache
export HF_HOME=/data/rech/kobeissa/hf
export HF_HUB_CACHE=/data/rech/kobeissa/hf/hub
export HF_DATASETS_CACHE=/data/rech/kobeissa/hf/datasets

# ── Dependency check & Quick Fix ──────────────────────────────────────────────
python - <<'EOF'
import importlib, sys
pkgs = ["torch", "transformers", "langchain", "langchain_community", "faiss", "fitz", "pdfplumber", "PyPDF2", "pypdf"]
missing = []
for pkg in pkgs:
    try:
        importlib.import_module(pkg)
    except ImportError:
        missing.append(pkg)

if missing:
    print(f"[WARN] Missing packages: {missing}. Installing …")
    import subprocess
    # Note: installing to the activated venv
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                           "torch", "transformers", "langchain", "langchain-community", 
                           "langchain-huggingface", "faiss-gpu", "pymupdf", "pdfplumber", "PyPDF2", "pypdf"])
else:
    print("[OK] All core dependencies present.")
EOF

# Set PYTHONPATH to include scratch directories
export PYTHONPATH="${SCRATCH_DIR}:${SCRATCH_DIR}/src:${PYTHONPATH:-}"

# ── Run the Experiment ────────────────────────────────────────────────────────
echo ""
echo "Running extraction study at $(date)"

# We use the script we created in baselines/
python3 baselines/extraction_study.py

EXIT_CODE=$?
echo "Experiment finished (exit code: ${EXIT_CODE}) at $(date)"

# ── Copy outputs back ─────────────────────────────────────────────────────────
FINAL_OUTPUTS="${SUBMIT_DIR}/outputs/extraction_impact"
mkdir -p "${FINAL_OUTPUTS}"

echo "Copying outputs back to: ${FINAL_OUTPUTS}"
if [ -d "${SCRATCH_OUT}/extraction_impact" ]; then
    cp -r "${SCRATCH_OUT}/extraction_impact/"* "${FINAL_OUTPUTS}/"
else
    echo "[WARN] No extraction_impact outputs found in scratch — check logs."
fi

# ── Cleanup ───────────────────────────────────────────────────────────────────
echo "Cleaning up scratch..."
rm -rf "${SCRATCH_DIR}"

echo "Done. Exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
