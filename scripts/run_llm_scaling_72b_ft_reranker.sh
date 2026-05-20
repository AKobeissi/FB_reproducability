#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=fb_llm_72b_ft
#SBATCH --output=logs/llm_scaling_72b_ft_%j.log
#SBATCH --error=logs/llm_scaling_72b_ft_%j.log
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:ls40:1
# #SBATCH --gres=gpu:rtx3090:2
# #SBATCH --gres=gpu:rtx_a5000:2
#SBATCH --mem=96G
#SBATCH --time=24:00:00

# =============================================================================
# LLM Scaling Study — Qwen2.5-72B-Instruct (4-bit NF4)
#                     Retrieval: MultiHyDE + FT-ReRanker (PageRec@5 = 0.56)
#
# Adds the 72B data point to the existing scaling study.  The 3B/7B/14B
# predictions from outputs/llm_scaling_ft_reranker/ are copied to scratch so
# they are treated as cached — only the 72B model is actually loaded and run.
# The final summary.json / summary.csv will contain all four model sizes.
#
# Memory budget (L40S 48 GB):
#   72B 4-bit NF4 weights  ≈ 36–38 GB
#   Activations + KV cache ≈  6–10 GB
#   Total                  ≈ 42–48 GB  (tight; if OOM try --gres=gpu:rtx3090:2)
#
# Requires:
#   baselines/results/predictions/multi_hyde_ft_reranker_retrieval.json
#   outputs/llm_scaling_ft_reranker/predictions/{3B,7B,14B}.json   (cached)
#   outputs/llm_scaling_ft_reranker/oracle_predictions/            (cached)
#
# Outputs → outputs/llm_scaling_ft_reranker/  (merged with existing results)
#   summary.json / summary.csv                 (all 4 model sizes)
#   breakdown_by_qtype.json
#   predictions/Qwen2.5-72B.json
#   oracle_predictions/Qwen2.5-72B_oracle.json
#   plots/                                     (regenerated with all 4 sizes)
# =============================================================================

SUBMIT_DIR=$SLURM_SUBMIT_DIR
VENV_PATH="$SUBMIT_DIR/venv"
SCRATCH_DIR=/Tmp/$(whoami)/${SLURM_JOB_ID}

export HF_HOME=/data/rech/kobeissa/hf
export HF_HUB_CACHE=/data/rech/kobeissa/hf/hub
export HF_DATASETS_CACHE=/data/rech/kobeissa/hf/datasets
export PYTORCH_ALLOC_CONF=expandable_segments:True

RETRIEVAL_FILE="baselines/results/predictions/multi_hyde_ft_reranker_retrieval.json"
OUTPUT_DIR="outputs/llm_scaling_ft_reranker"

echo "========================================================="
echo "Job ID     : ${SLURM_JOB_ID:-local}"
echo "Node       : ${SLURMD_NODENAME:-$(hostname)}"
echo "GPU        : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
echo "Retrieval  : ${RETRIEVAL_FILE}  [MultiHyDE + FT-ReRanker]"
echo "Generator  : Qwen/Qwen2.5-72B-Instruct  (4-bit NF4)"
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

# Copy the pre-computed FT-reranker retrieval file
mkdir -p "$SCRATCH_DIR/baselines/results/predictions"
cp "$SUBMIT_DIR/${RETRIEVAL_FILE}" \
   "$SCRATCH_DIR/${RETRIEVAL_FILE}"

# Copy cached 3B/7B/14B predictions so the script skips generation for them.
# Only Qwen2.5-72B will actually be generated (--model-filter below).
EXISTING_PREDS="$SUBMIT_DIR/${OUTPUT_DIR}"
if [ -d "${EXISTING_PREDS}/predictions" ]; then
    mkdir -p "$SCRATCH_DIR/${OUTPUT_DIR}/predictions"
    cp "${EXISTING_PREDS}/predictions/"*.json \
       "$SCRATCH_DIR/${OUTPUT_DIR}/predictions/" 2>/dev/null && \
        echo "Copied existing standard predictions to scratch." || \
        echo "Warning: no existing standard predictions found — all models will be generated."
fi
if [ -d "${EXISTING_PREDS}/oracle_predictions" ]; then
    mkdir -p "$SCRATCH_DIR/${OUTPUT_DIR}/oracle_predictions"
    cp "${EXISTING_PREDS}/oracle_predictions/"*.json \
       "$SCRATCH_DIR/${OUTPUT_DIR}/oracle_predictions/" 2>/dev/null && \
        echo "Copied existing oracle predictions to scratch." || \
        echo "Warning: no existing oracle predictions found — all models will be generated (oracle)."
fi

cd "$SCRATCH_DIR"

echo "Activating venv: $VENV_PATH"
source "$VENV_PATH/bin/activate"

python3 - <<'EOF'
try:
    import nltk
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
except Exception:
    pass
EOF

export PYTHONPATH="${SCRATCH_DIR}:${SCRATCH_DIR}/src:${PYTHONPATH:-}"

echo ""
echo ">>> Running LLM scaling study — 72B only (cached: 3B, 7B, 14B)"
echo "========================================================="

python3 baselines/llm_scaling_study.py \
    --retrieval-file "${RETRIEVAL_FILE}" \
    --output-dir     "${OUTPUT_DIR}" \
    --model-filter   "Qwen2.5-72B"

EXIT_CODE=$?
echo "Script finished (exit code: ${EXIT_CODE}) at $(date)"

# Sync all results back (predictions + oracle + plots + summaries)
FINAL_OUT="${SUBMIT_DIR}/${OUTPUT_DIR}"
mkdir -p "$FINAL_OUT"

echo "Syncing outputs back to: $FINAL_OUT"
rsync -a "${SCRATCH_DIR}/${OUTPUT_DIR}/" "${FINAL_OUT}/"

rm -rf "$SCRATCH_DIR"

echo "========================================================="
echo "Done!  End time: $(date)"
echo ""
echo "Results (merged — all 4 model sizes):"
echo "  ${OUTPUT_DIR}/summary.json"
echo "  ${OUTPUT_DIR}/summary.csv"
echo "  ${OUTPUT_DIR}/breakdown_by_qtype.json"
echo "  ${OUTPUT_DIR}/oracle_summary.json"
echo "  ${OUTPUT_DIR}/predictions/Qwen2.5-72B.json"
echo "  ${OUTPUT_DIR}/oracle_predictions/Qwen2.5-72B_oracle.json"
echo "  ${OUTPUT_DIR}/plots/"
echo "========================================================="

exit ${EXIT_CODE}
