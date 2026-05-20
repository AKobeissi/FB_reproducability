#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=fb_colbert
#SBATCH --output=logs/colbert_%j.log
#SBATCH --error=logs/colbert_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=12:00:00

# =============================================================================
# ColBERT / Late-Interaction Baselines — three variants
# =============================================================================
#
#  colbert_retriever     — ColBERT first-stage retriever (MaxSim over token store)
#  colbert_reranker      — BGE-M3 dense retrieval → ColBERT MaxSim reranker
#  colbert_hyde_reranker — Multi-HyDE → BGE-M3 → ColBERT reranker  (best variant)
#
# Model: colbert-ir/colbertv2.0 (loaded via AutoModel; MaxSim over last_hidden_state)
# Chunking: identical to all other baselines (1024 tok, 128 overlap)
# Evaluation: retrieval-only (--skip-generation) to isolate retrieval quality
#
# Memory budget:
#   BGE-M3 embed   ~2.0 GB
#   ColBERT model  ~0.4 GB
#   Token store    ~6-12 GB (depends on corpus size; stored on CPU, searched on GPU)
#   80 GB node provides ample headroom
#
# Usage:
#   sbatch scripts/run_colbert_baselines.sh
#   sbatch scripts/run_colbert_baselines.sh --resume
#   sbatch scripts/run_colbert_baselines.sh --variants colbert_reranker
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
echo ">>> Running ColBERT baselines"
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
    --variants      colbert_retriever colbert_reranker colbert_hyde_reranker \
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
echo "  metrics/colbert_retriever_metrics.json"
echo "  metrics/colbert_reranker_metrics.json"
echo "  metrics/colbert_hyde_reranker_metrics.json"
echo "========================================================="
