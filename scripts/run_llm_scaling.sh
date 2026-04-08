#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=fb_llm_scaling
#SBATCH --output=logs/llm_scaling_%j.log
#SBATCH --error=logs/llm_scaling_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00

# =============================================================================
# LLM Scaling Study — FinanceBench
#
# Holds retrieval fixed (dense BGE-M3 top-5, pre-computed) and varies the
# generator: Qwen2.5-3B, 7B, 14B (all 4-bit quantised).
#
# Requires:
#   baselines/results/predictions/dense_bge_m3_retrieval.json  (already done)
#
# Memory budget (L40S 46 GB):
#   3B  4-bit  ≈  2.5 GB
#   7B  4-bit  ≈  4.5 GB
#   14B 4-bit  ≈  8.5 GB   ← peak; loaded one at a time, freed after
#
# Outputs → outputs/llm_scaling/
#   summary.json            overall metrics per model
#   summary.csv             same as CSV
#   breakdown_by_qtype.json per-question-type breakdown
#   predictions/            raw generated answers per model (for inspection)
#   plots/                  3 publication-quality figures
# =============================================================================

SUBMIT_DIR=$SLURM_SUBMIT_DIR
VENV_PATH="$SUBMIT_DIR/venv"
SCRATCH_DIR=/Tmp/$(whoami)/${SLURM_JOB_ID}

export HF_HOME=/data/rech/kobeissa/hf
export HF_HUB_CACHE=/data/rech/kobeissa/hf/hub
export HF_DATASETS_CACHE=/data/rech/kobeissa/hf/datasets
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "========================================================="
echo "Job ID     : ${SLURM_JOB_ID:-local}"
echo "Node       : ${SLURMD_NODENAME:-$(hostname)}"
echo "GPU        : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
echo "Start time : $(date)"
echo "========================================================="

mkdir -p "$SCRATCH_DIR"
mkdir -p "$SUBMIT_DIR/logs"

echo "Copying repository to scratch…"
rsync -a --quiet \
    --exclude 'venv' \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.log' \
    --exclude 'outputs' \
    --exclude 'vector_stores' \
    --exclude 'pdfs' \
    --exclude 'Final-PDF' \
    --exclude 'PDF-Opus*' \
    "$SUBMIT_DIR/" "$SCRATCH_DIR/"

# Copy the pre-computed retrieval file (needed by the script)
mkdir -p "$SCRATCH_DIR/baselines/results/predictions"
cp "$SUBMIT_DIR/baselines/results/predictions/dense_bge_m3_retrieval.json" \
   "$SCRATCH_DIR/baselines/results/predictions/"

cd "$SCRATCH_DIR"

echo "Activating venv: $VENV_PATH"
source "$VENV_PATH/bin/activate"

# Ensure nltk punkt is available for BLEU-4
python - <<'EOF'
try:
    import nltk
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
except Exception:
    pass
EOF

export PYTHONPATH="${SCRATCH_DIR}:${SCRATCH_DIR}/src:${PYTHONPATH:-}"

echo ""
echo ">>> Running LLM scaling study"
echo "========================================================="

python3 baselines/llm_scaling_study.py

EXIT_CODE=$?
echo "Script finished (exit code: ${EXIT_CODE}) at $(date)"

# Sync results back
FINAL_OUT="${SUBMIT_DIR}/outputs/llm_scaling"
mkdir -p "$FINAL_OUT"

echo "Syncing outputs back to: $FINAL_OUT"
rsync -a "${SCRATCH_DIR}/outputs/llm_scaling/" "${FINAL_OUT}/"

rm -rf "$SCRATCH_DIR"

echo "========================================================="
echo "Done!  End time: $(date)"
echo ""
echo "Results:"
echo "  outputs/llm_scaling/summary.json"
echo "  outputs/llm_scaling/summary.csv"
echo "  outputs/llm_scaling/breakdown_by_qtype.json"
echo "  outputs/llm_scaling/plots/scaling_overall.*"
echo "  outputs/llm_scaling/plots/scaling_by_qtype.*"
echo "  outputs/llm_scaling/plots/scaling_numeric_match.*"
echo "========================================================="

exit ${EXIT_CODE}
