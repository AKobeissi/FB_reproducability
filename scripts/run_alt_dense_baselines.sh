#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=fb_alt_dense
#SBATCH --output=logs/alt_dense_%j.log
#SBATCH --error=logs/alt_dense_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=06:00:00

# =============================================================================
# Alternative Dense Bi-Encoder Baselines
# =============================================================================
#
# Runs two new dense retrieval variants against the same FinanceBench setup
# as the main BGE-M3 baseline (same chunking, same pipeline, same eval):
#
#   dense_bge_base      — BAAI/bge-base-en-v1.5   (lighter general-purpose model)
#   dense_investopedia  — FinLang/investopedia_embedding  (finance-domain adapted)
#
# Each model gets its own ChromaDB collection so the BGE-M3 index is untouched.
# Results are written alongside the other baselines in baselines/results/.
#
# Usage:
#   sbatch scripts/run_alt_dense_baselines.sh
#
#   Re-run only one variant (e.g. after a fix):
#   sbatch scripts/run_alt_dense_baselines.sh --variants dense_investopedia
#
#   Resume a partial run without rebuilding the index:
#   sbatch scripts/run_alt_dense_baselines.sh --resume
# =============================================================================

SUBMIT_DIR=$SLURM_SUBMIT_DIR
VENV_PATH="$SUBMIT_DIR/venv"
SCRATCH_DIR=/Tmp/$(whoami)/${SLURM_JOB_ID}

export HF_HOME=/data/rech/kobeissa/hf
export HF_HUB_CACHE=/data/rech/kobeissa/hf/hub
export HF_DATASETS_CACHE=/data/rech/kobeissa/hf/datasets
export PYTORCH_ALLOC_CONF=expandable_segments:True

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
echo ">>> Running alt dense baselines: dense_bge_base, dense_investopedia"
echo "    Extra args: ${EXTRA_ARGS:-none}"
echo "========================================================="

python baselines/run_baselines.py \
    --data-path     data/financebench_open_source.jsonl \
    --doc-info-path data/financebench_document_information.jsonl \
    --pdf-dir       pdfs \
    --vs-dir        vector_stores/baselines \
    --output-dir    baselines/results \
    --hyde-cache    baselines/hyde_cache.json \
    --skip-generation \
    --skip-plots \
    --variants      dense_bge_base dense_investopedia \
    $EXTRA_ARGS

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "[ERROR] Runner failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo ""
echo ">>> Syncing results back to submit dir..."
rsync -a "${SCRATCH_DIR}/baselines/results/"         "${SUBMIT_DIR}/baselines/results/"
rsync -a "${SCRATCH_DIR}/vector_stores/baselines/"   "${SUBMIT_DIR}/vector_stores/baselines/"

echo ""
echo "========================================================="
echo "Done!   End time : $(date)"
echo ""
echo "Results written to baselines/results/:"
echo "  metrics/dense_bge_base_metrics.json"
echo "  metrics/dense_investopedia_metrics.json"
echo "  predictions/dense_bge_base_retrieval.json"
echo "  predictions/dense_investopedia_retrieval.json"
echo "========================================================="
