#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=fb_llm_scaling_top
#SBATCH --output=logs/llm_scaling_top_%j.log
#SBATCH --error=logs/llm_scaling_top_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00

# =============================================================================
# LLM Scaling Study — DENSE+MultiHyDE+ReRANKER (top baseline)
#
# Same generator sweep as run_llm_scaling.sh (Qwen2.5-3B/7B/14B, 4-bit),
# but retrieval is fixed to multi_hyde_reranker instead of plain dense BGE-M3.
#
# Outputs → outputs/llm_scaling_multi_hyde_reranker/
# =============================================================================

SUBMIT_DIR=$SLURM_SUBMIT_DIR
VENV_PATH="$SUBMIT_DIR/venv"
SCRATCH_DIR=/Tmp/$(whoami)/${SLURM_JOB_ID}

export HF_HOME=/data/rech/kobeissa/hf
export HF_HUB_CACHE=/data/rech/kobeissa/hf/hub
export HF_DATASETS_CACHE=/data/rech/kobeissa/hf/datasets
export PYTORCH_ALLOC_CONF=expandable_segments:True

RETRIEVAL_FILE="baselines/results/predictions/multi_hyde_reranker_retrieval.json"
OUTPUT_DIR="outputs/llm_scaling_multi_hyde_reranker"

echo "========================================================="
echo "Job ID     : ${SLURM_JOB_ID:-local}"
echo "Node       : ${SLURMD_NODENAME:-$(hostname)}"
echo "GPU        : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
echo "Retrieval  : ${RETRIEVAL_FILE}"
echo "Output dir : ${OUTPUT_DIR}"
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

# Copy the pre-computed retrieval file
mkdir -p "$SCRATCH_DIR/baselines/results/predictions"
cp "$SUBMIT_DIR/${RETRIEVAL_FILE}" \
   "$SCRATCH_DIR/${RETRIEVAL_FILE}"

cd "$SCRATCH_DIR"

echo "Activating venv: $VENV_PATH"
source "$VENV_PATH/bin/activate"

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
echo ">>> Running LLM scaling study (MultiHyDE+ReRanker retrieval)"
echo "========================================================="

python3 baselines/llm_scaling_study.py \
    --retrieval-file "${RETRIEVAL_FILE}" \
    --output-dir     "${OUTPUT_DIR}"

EXIT_CODE=$?
echo "Script finished (exit code: ${EXIT_CODE}) at $(date)"

# Sync results back
FINAL_OUT="${SUBMIT_DIR}/${OUTPUT_DIR}"
mkdir -p "$FINAL_OUT"

echo "Syncing outputs back to: $FINAL_OUT"
rsync -a "${SCRATCH_DIR}/${OUTPUT_DIR}/" "${FINAL_OUT}/"

rm -rf "$SCRATCH_DIR"

echo "========================================================="
echo "Done!  End time: $(date)"
echo ""
echo "Results:"
echo "  ${OUTPUT_DIR}/summary.json"
echo "  ${OUTPUT_DIR}/summary.csv"
echo "  ${OUTPUT_DIR}/breakdown_by_qtype.json"
echo "  ${OUTPUT_DIR}/plots/"
echo "========================================================="

exit ${EXIT_CODE}
