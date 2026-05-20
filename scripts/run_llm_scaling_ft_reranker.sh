#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=fb_llm_scaling_ft
#SBATCH --output=logs/llm_scaling_ft_%j.log
#SBATCH --error=logs/llm_scaling_ft_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:ls40:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00

# =============================================================================
# LLM Scaling Study — DENSE+MultiHyDE+FT-ReRANKER
#
# Same generator sweep as run_llm_scaling.sh (Qwen2.5-3B/7B/14B, 4-bit),
# but retrieval is fixed to multi_hyde_ft_reranker (fine-tuned on FinQA).
#
# Requires: baselines/results/predictions/multi_hyde_ft_reranker_retrieval.json
#           (produced by scripts/run_ft_reranker.sh)
#
# Outputs → outputs/llm_scaling_ft_reranker/
#   summary.json / summary.csv
#   breakdown_by_qtype.json
#   predictions/Qwen2.5-{3B,7B,14B}.json   ← per-model generated answers
#   oracle_predictions/                      ← gold-context upper bound
#   plots/
#
# After this job completes, run analyze_top_baseline.py to build the HTML
# reports in the same style as top_baseline_analysis/:
#
#   source venv/bin/activate
#   for MODEL in Qwen2.5-3B Qwen2.5-7B Qwen2.5-14B; do
#     python3 analyze_top_baseline.py \
#       --pred   "outputs/llm_scaling_ft_reranker/predictions/${MODEL}.json" \
#       --out    "outputs/ft_reranker_analysis/report_${MODEL}.html" \
#       --model-label "${MODEL}-Instruct (4-bit) | Retrieval: MultiHyDE + FT-ReRanker"
#   done
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

# Copy the pre-computed FT reranker retrieval file
mkdir -p "$SCRATCH_DIR/baselines/results/predictions"
cp "$SUBMIT_DIR/${RETRIEVAL_FILE}" \
   "$SCRATCH_DIR/${RETRIEVAL_FILE}"

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
echo ">>> Running LLM scaling study (MultiHyDE + FT-ReRanker retrieval)"
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

# Generate HTML reports in top_baseline_analysis style
if [ ${EXIT_CODE} -eq 0 ]; then
    echo ""
    echo ">>> Generating HTML reports (top_baseline_analysis style)"
    echo "========================================================="
    source "$VENV_PATH/bin/activate"
    mkdir -p "${SUBMIT_DIR}/outputs/ft_reranker_analysis"
    cd "$SUBMIT_DIR"
    for MODEL in Qwen2.5-3B Qwen2.5-7B Qwen2.5-14B; do
        python3 analyze_top_baseline.py \
            --pred   "${OUTPUT_DIR}/predictions/${MODEL}.json" \
            --out    "outputs/ft_reranker_analysis/report_${MODEL}.html" \
            --model-label "${MODEL}-Instruct (4-bit) | Retrieval: MultiHyDE + FT-ReRanker"
        echo "  → outputs/ft_reranker_analysis/report_${MODEL}.html"
    done
fi

echo "========================================================="
echo "Done!  End time: $(date)"
echo ""
echo "Results:"
echo "  ${OUTPUT_DIR}/summary.json"
echo "  ${OUTPUT_DIR}/summary.csv"
echo "  ${OUTPUT_DIR}/plots/"
echo "  outputs/ft_reranker_analysis/report_Qwen2.5-{3B,7B,14B}.html"
echo "========================================================="

exit ${EXIT_CODE}
