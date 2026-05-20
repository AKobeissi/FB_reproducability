#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=fb_colqwen2
#SBATCH --output=logs/colqwen2_baseline_%j.log
#SBATCH --error=logs/colqwen2_baseline_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:ls40:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00

# =============================================================================
# FinanceBench ColQwen2 Visual Page Baseline
# =============================================================================
#
# Runs the page-level ColQwen2 retriever inside the main FinanceBench baseline
# pipeline so the outputs land in the same metrics tables and plots as the
# chunk-based methods.
#
# Retrieval:
#   1. Render PDF pages to images
#   2. Embed pages with ColQwen2
#   3. Select top documents from a broad first-stage page search
#   4. Expand high-scoring pages with neighbors
#   5. Rerank candidate pages with visual score + lightweight text overlap
#
# Generation / evaluation:
#   Uses the same text-only generation and reporting path as the other
#   FinanceBench baselines for a fair retrieval comparison.
# =============================================================================

SUBMIT_DIR=$SLURM_SUBMIT_DIR
VENV_PATH="$SUBMIT_DIR/venv"
SCRATCH_DIR=/Tmp/$(whoami)/${SLURM_JOB_ID}

export HF_HOME=/data/rech/kobeissa/hf
export HF_HUB_CACHE=/data/rech/kobeissa/hf/hub
export HF_DATASETS_CACHE=/data/rech/kobeissa/hf/datasets
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

echo "========================================================="
echo "Job ID        : ${SLURM_JOB_ID:-local}"
echo "Node          : ${SLURMD_NODENAME:-$(hostname)}"
echo "GPU           : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
echo "Start time    : $(date)"
echo "========================================================="

mkdir -p "$SCRATCH_DIR"
mkdir -p "$SUBMIT_DIR/logs"

echo "Copying repository to scratch..."
rsync -a --quiet \
    --exclude 'venv' \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.log' \
    "$SUBMIT_DIR/" "$SCRATCH_DIR/"

cd "$SCRATCH_DIR"

echo "Activating venv: $VENV_PATH"
source "$VENV_PATH/bin/activate"

EXTRA_ARGS="$@"

echo ""
echo ">>> Running ColQwen2 visual baseline"
echo "    Extra args: ${EXTRA_ARGS:-none}"
echo "========================================================="

python baselines/run_baselines.py \
    --data-path     data/financebench_open_source.jsonl \
    --doc-info-path data/financebench_document_information.jsonl \
    --pdf-dir       pdfs \
    --vs-dir        vector_stores/baselines \
    --output-dir    baselines/results \
    --hyde-cache    baselines/hyde_cache.json \
    --variants      colqwen2_visual \
    $EXTRA_ARGS

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "[ERROR] ColQwen2 baseline failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo ""
echo ">>> Syncing results back to submit dir..."
rsync -a "${SCRATCH_DIR}/baselines/results/" "${SUBMIT_DIR}/baselines/results/"
rsync -a "${SCRATCH_DIR}/vector_stores/baselines/" "${SUBMIT_DIR}/vector_stores/baselines/"

echo ""
echo "========================================================="
echo "Done!   End time : $(date)"
echo ""
echo "Results:"
echo "  baselines/results/metrics/colqwen2_visual_metrics.json"
echo "  baselines/results/predictions/colqwen2_visual_retrieval.json"
echo "  baselines/results/predictions/colqwen2_visual_generated.json"
echo "========================================================="

exit $EXIT_CODE
