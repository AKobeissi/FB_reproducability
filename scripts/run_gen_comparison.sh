#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=fb_gen_compare
#SBATCH --output=logs/gen_compare_%j.log
#SBATCH --error=logs/gen_compare_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:ls40:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00

# =============================================================================
# Generative Comparison — Qwen2.5-7B-Instruct vs Llama-3.1-8B-Instruct
#
# Retrieval is held fixed at the best pipeline (MultiHyDE + FT-ReRanker).
# Each model generates answers for all 150 FinanceBench questions.
#
# Metrics
#   ROUGE-L              vs reference answer
#   BERTScore F1         (roberta-large, rescaled baseline)
#   NumericMatch         ±3% tolerance
#     — metrics-generated only (50 questions)
#     — all question types     (150 questions)
#
# Breakdowns
#   Overall / by question type / by document type
#
# Requires:
#   baselines/results/predictions/multi_hyde_ft_reranker_retrieval.json
#   data/financebench_open_source.jsonl
#   data/financebench_document_information.jsonl
#
# Outputs → outputs/gen_comparison/
#   summary.json / summary.csv
#   by_question_type.json / .csv
#   by_doc_type.json / .csv
#   predictions/{model}.json
#   plots/
# =============================================================================

SUBMIT_DIR=$SLURM_SUBMIT_DIR
VENV_PATH="$SUBMIT_DIR/venv"
SCRATCH_DIR=/Tmp/$(whoami)/${SLURM_JOB_ID}

export HF_HOME=/data/rech/kobeissa/hf
export HF_HUB_CACHE=/data/rech/kobeissa/hf/hub
export HF_DATASETS_CACHE=/data/rech/kobeissa/hf/datasets
export PYTORCH_ALLOC_CONF=expandable_segments:True

OUTPUT_DIR="outputs/gen_comparison"

echo "========================================================="
echo "Job ID     : ${SLURM_JOB_ID:-local}"
echo "Node       : ${SLURMD_NODENAME:-$(hostname)}"
echo "GPU        : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
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

# Copy pre-computed retrieval file (MultiHyDE + FT-ReRanker — best pipeline)
RETRIEVAL_SRC="$SUBMIT_DIR/baselines/results/predictions/multi_hyde_ft_reranker_retrieval.json"
if [ ! -f "$RETRIEVAL_SRC" ]; then
    echo "ERROR: Retrieval file not found at $RETRIEVAL_SRC"
    exit 1
fi
echo "Copying retrieval predictions…"
mkdir -p "$SCRATCH_DIR/baselines/results/predictions"
cp "$RETRIEVAL_SRC" "$SCRATCH_DIR/baselines/results/predictions/"

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

export PYTHONPATH="${SCRATCH_DIR}:${SCRATCH_DIR}/src:${SCRATCH_DIR}/baselines:${PYTHONPATH:-}"

echo ""
echo ">>> Running generative comparison (Qwen2.5-7B vs Llama-3.1-8B)"
echo "========================================================="

python3 baselines/run_gen_comparison.py \
    --retrieval-file "baselines/results/predictions/multi_hyde_ft_reranker_retrieval.json" \
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
echo "  ${OUTPUT_DIR}/by_question_type.json"
echo "  ${OUTPUT_DIR}/by_doc_type.json"
echo "  ${OUTPUT_DIR}/predictions/"
echo "  ${OUTPUT_DIR}/plots/"
echo "========================================================="

exit ${EXIT_CODE}
