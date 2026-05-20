#!/bin/bash -l
#SBATCH --job-name=benchmark_ext
#SBATCH --partition=rali
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=12
#SBATCH --time=72:00:00
#SBATCH --output=slurm-%j-benchmark_ext.out
#SBATCH --error=slurm-%j-benchmark_ext.err

# ──────────────────────────────────────────────────────────────────────────────
# Benchmark Extension: FinanceBench + FinQA
#
# Builds a unified global index and runs:
#   - 14 standard baselines (dense/bm25/splade/hybrid/hyde/reranker/…)
#   - dense_bge_m3_ft_reranker
#   - multi_hyde_ft_reranker  ← top performer on FB (0.56 PageRec@5)
#   - oracle_doc, oracle_page
#
# Results reported at three levels:
#   global        — all 680 questions over unified search space
#   financebench  — 150 FB questions (subset of global)
#   finqa         — 530 FinQA questions (subset of global)
# ──────────────────────────────────────────────────────────────────────────────

SUBMIT_DIR=$SLURM_SUBMIT_DIR
VENV_PATH="$SUBMIT_DIR/venv"
SCRATCH_DIR=/Tmp/$(whoami)/${SLURM_JOB_ID}

export HF_HOME=/data/rech/kobeissa/hf
export HF_HUB_CACHE=/data/rech/kobeissa/hf/hub
export HF_DATASETS_CACHE=/data/rech/kobeissa/hf/datasets
export TRANSFORMERS_CACHE=/data/rech/kobeissa/hf
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

echo ""
echo ">>> Running benchmark extension"
echo "========================================================="

python baselines/run_benchmark_extension.py \
    --fb-pdf-dir   pdfs \
    --finqa-pdf-dir Final-PDF \
    --vs-dir        vector_stores/benchmark_extension \
    --output-dir    baselines/results/benchmark_extension \
    --hyde-cache    baselines/benchmark_ext_hyde_cache.json \
    --skip-generation \
    --resume

EXIT_CODE=$?

echo ""
echo ">>> Syncing results back to submit dir..."
rsync -a "${SCRATCH_DIR}/baselines/results/benchmark_extension/"  "${SUBMIT_DIR}/baselines/results/benchmark_extension/"
rsync -a "${SCRATCH_DIR}/baselines/benchmark_ext_hyde_cache.json" "${SUBMIT_DIR}/baselines/benchmark_ext_hyde_cache.json" 2>/dev/null || true
rsync -a "${SCRATCH_DIR}/vector_stores/benchmark_extension/"      "${SUBMIT_DIR}/vector_stores/benchmark_extension/"

if [ $EXIT_CODE -ne 0 ]; then
    echo "[ERROR] Benchmark extension failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo ""
echo "========================================================="
echo "Done!   End time : $(date)"
echo "========================================================="
