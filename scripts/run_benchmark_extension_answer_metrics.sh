#!/bin/bash -l
#SBATCH --job-name=bench_ext_gen
#SBATCH --partition=rali
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j-bench_ext_gen.out
#SBATCH --error=slurm-%j-bench_ext_gen.err

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

echo "Copying repository to scratch..."
rsync -a --quiet \
    --exclude 'venv' \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.log' \
    "$SUBMIT_DIR/" "$SCRATCH_DIR/"

cd "$SCRATCH_DIR"
source "$VENV_PATH/bin/activate"

echo ""
echo ">>> Generating benchmark-extension answers and recomputing answer metrics"
echo "========================================================="

python baselines/compute_benchmark_extension_answer_metrics.py \
    --variants multi_hyde_ft_reranker \
    --dataset-filter financebench \
    --generate-missing \
    --report-variant multi_hyde_ft_reranker \
    --report-level financebench \
    "$@"

EXIT_CODE=$?

echo ""
echo ">>> Syncing results back to submit dir..."
rsync -a "${SCRATCH_DIR}/baselines/results/benchmark_extension/" "${SUBMIT_DIR}/baselines/results/benchmark_extension/"

if [ $EXIT_CODE -ne 0 ]; then
    echo "[ERROR] Benchmark-extension answer metrics failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo ""
echo "========================================================="
echo "Done!   End time : $(date)"
echo "========================================================="
