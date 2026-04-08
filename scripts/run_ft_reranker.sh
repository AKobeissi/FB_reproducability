#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=fb_ft_reranker
#SBATCH --output=logs/ft_reranker_%j.log
#SBATCH --error=logs/ft_reranker_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00

# =============================================================================
# Fine-tune BAAI/bge-reranker-v2-m3 on FinQA, then evaluate on FinanceBench.
#
# Stage 1 — Training  (train_cross_encoder_reranker.py)
#   Input : finqa/train.json + finqa/dev.json
#   Output: checkpoints/ft_cross_encoder/   (best dev-MAP checkpoint)
#
# Stage 2 — Inference + Evaluation  (run_ft_reranker_baselines.py)
#   Input : baselines/results/predictions/dense_bge_m3_retrieval.json
#           baselines/results/predictions/multi_hyde_retrieval.json
#   Output: baselines/results/predictions/dense_bge_m3_ft_reranker_retrieval.json
#           baselines/results/predictions/multi_hyde_ft_reranker_retrieval.json
#           baselines/results/metrics/*_ft_reranker_metrics.json
#           baselines/results/plots/ft_reranker_*.{pdf,png}
#
# Memory budget (L40S 46 GB):
#   bge-reranker-v2-m3 fp16  ≈  1.1 GB
#   training batch=32, seqlen=512 ≈  4-6 GB activations
#   inference batch=64  ≈  3-4 GB
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
echo "Job ID     : ${SLURM_JOB_ID:-local}"
echo "Node       : ${SLURMD_NODENAME:-$(hostname)}"
echo "GPU        : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
echo "Start time : $(date)"
echo "========================================================="

mkdir -p "$SCRATCH_DIR"
mkdir -p "$SUBMIT_DIR/logs"
mkdir -p "$SUBMIT_DIR/checkpoints"

# Copy repo to scratch (fast local I/O during training)
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

# Copy required input files
mkdir -p "$SCRATCH_DIR/baselines/results/predictions"
mkdir -p "$SCRATCH_DIR/baselines/results/metrics"
mkdir -p "$SCRATCH_DIR/finqa"

cp "$SUBMIT_DIR/finqa/train.json" "$SCRATCH_DIR/finqa/"
cp "$SUBMIT_DIR/finqa/dev.json"   "$SCRATCH_DIR/finqa/"

cp "$SUBMIT_DIR/baselines/results/predictions/dense_bge_m3_retrieval.json" \
   "$SCRATCH_DIR/baselines/results/predictions/"
cp "$SUBMIT_DIR/baselines/results/predictions/multi_hyde_retrieval.json" \
   "$SCRATCH_DIR/baselines/results/predictions/"

# Copy existing baseline metrics (needed for comparison plots)
for f in dense_bge_m3 bge_reranker multi_hyde; do
    src="$SUBMIT_DIR/baselines/results/metrics/${f}_metrics.json"
    if [ -f "$src" ]; then
        cp "$src" "$SCRATCH_DIR/baselines/results/metrics/"
    fi
done

cd "$SCRATCH_DIR"

echo "Activating venv: $VENV_PATH"
source "$VENV_PATH/bin/activate"

# Ensure NLTK data is available
python3 - <<'EOF'
try:
    import nltk
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
except Exception:
    pass
EOF

export PYTHONPATH="${SCRATCH_DIR}:${SCRATCH_DIR}/src:${PYTHONPATH:-}"

# =============================================================================
# Stage 1 — Training
# =============================================================================
echo ""
echo ">>> Stage 1: Training cross-encoder reranker on FinQA"
echo "========================================================="

python3 baselines/train_cross_encoder_reranker.py

TRAIN_EXIT=$?
echo "Training finished (exit code: ${TRAIN_EXIT}) at $(date)"

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "ERROR: Training failed — aborting."
    exit $TRAIN_EXIT
fi

# Sync checkpoint back immediately (in case Stage 2 crashes)
echo "Syncing checkpoint back…"
rsync -a "${SCRATCH_DIR}/checkpoints/ft_cross_encoder/" \
         "${SUBMIT_DIR}/checkpoints/ft_cross_encoder/"

# =============================================================================
# Stage 2 — Inference + Evaluation on FinanceBench
# =============================================================================
echo ""
echo ">>> Stage 2: Re-ranking and evaluating on FinanceBench"
echo "========================================================="

python3 baselines/run_ft_reranker_baselines.py

EVAL_EXIT=$?
echo "Evaluation finished (exit code: ${EVAL_EXIT}) at $(date)"

# =============================================================================
# Sync all outputs back
# =============================================================================
echo "Syncing outputs back…"

rsync -a "${SCRATCH_DIR}/baselines/results/predictions/" \
         "${SUBMIT_DIR}/baselines/results/predictions/"

rsync -a "${SCRATCH_DIR}/baselines/results/metrics/" \
         "${SUBMIT_DIR}/baselines/results/metrics/"

rsync -a "${SCRATCH_DIR}/baselines/results/plots/" \
         "${SUBMIT_DIR}/baselines/results/plots/"

rsync -a "${SCRATCH_DIR}/checkpoints/ft_cross_encoder/" \
         "${SUBMIT_DIR}/checkpoints/ft_cross_encoder/"

rm -rf "$SCRATCH_DIR"

echo "========================================================="
echo "Done!  End time: $(date)"
echo ""
echo "Outputs:"
echo "  checkpoints/ft_cross_encoder/          ← trained model"
echo "  baselines/results/predictions/*ft_reranker*"
echo "  baselines/results/metrics/*ft_reranker*"
echo "  baselines/results/plots/ft_reranker_*"
echo "========================================================="

exit ${EVAL_EXIT}
