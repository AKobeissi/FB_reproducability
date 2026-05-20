#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=colqwen2_ablation
#SBATCH --output=colqwen2_ablation_%j.log
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:ls40:1
#SBATCH --mem=80G
#SBATCH --time=96:00:00

# =============================================================================
# ColQwen2 → CrossEncoder reranking ablation (no doc-level aggregation)
#
# 4 variants from a single ColQwen2 embedding pass:
#   colqwen2_top20_bge_rerank   — top-20 pages → BAAI/bge-reranker-v2-m3
#   colqwen2_top100_bge_rerank  — top-100 pages → BAAI/bge-reranker-v2-m3
#   colqwen2_top20_ft_rerank    — top-20 pages → checkpoints/ft_cross_encoder
#   colqwen2_top100_ft_rerank   — top-100 pages → checkpoints/ft_cross_encoder
#
# Results: results/colqwen2_rerank_ablation/<variant>/<variant>_<ts>_scored.json
# =============================================================================

SUBMIT_DIR=$SLURM_SUBMIT_DIR
VENV_PATH="$SUBMIT_DIR/venv"
SCRATCH_DIR=/Tmp/$(whoami)/${SLURM_JOB_ID}

echo "========================================================="
echo "Job:         $SLURM_JOB_ID"
echo "Node:        $SLURMD_NODENAME"
echo "GPU:         $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
echo "Submit dir:  $SUBMIT_DIR"
echo "Scratch dir: $SCRATCH_DIR"
echo "Start:       $(date)"
echo "========================================================="

export HF_HOME=/data/rech/kobeissa/hf
export HF_HUB_CACHE=/data/rech/kobeissa/hf/hub
export HF_DATASETS_CACHE=/data/rech/kobeissa/hf/datasets
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

mkdir -p "$SCRATCH_DIR"

echo "Copying repository to scratch..."
rsync -a --quiet \
  --exclude 'venv' \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude 'outputs' \
  --exclude 'vector_stores' \
  --exclude '*.log' \
  "$SUBMIT_DIR/" "$SCRATCH_DIR/"

cd "$SCRATCH_DIR"

echo "Activating venv: $VENV_PATH"
source "$VENV_PATH/bin/activate"

# Verify key dependencies
python - <<'PY'
missing = []
for mod in ("transformers", "sentence_transformers", "PIL", "fitz"):
    try:
        __import__(mod)
    except ImportError:
        missing.append(mod)
try:
    from transformers import ColQwen2ForRetrieval, ColQwen2Processor
except Exception:
    missing.append("transformers[ColQwen2]")
if missing:
    raise SystemExit("Missing deps: " + ", ".join(missing))
print("All dependencies OK")
PY

if [ $? -ne 0 ]; then
    echo "Installing missing dependencies..."
    pip install -U transformers accelerate pillow pymupdf sentence-transformers
fi

export PYTHONPATH="$SCRATCH_DIR:$PYTHONPATH"

python baselines/run_colqwen2_rerank_ablation.py

EXIT_CODE=$?

# Copy results back
FINAL_RESULTS="$SUBMIT_DIR/results/colqwen2_rerank_ablation"
if [ -d "$SCRATCH_DIR/results/colqwen2_rerank_ablation" ]; then
    mkdir -p "$FINAL_RESULTS"
    cp -r "$SCRATCH_DIR/results/colqwen2_rerank_ablation/"* "$FINAL_RESULTS/"
    echo "Results copied to: $FINAL_RESULTS"
fi

rm -rf "$SCRATCH_DIR"
echo "Done (exit code: $EXIT_CODE) — $(date)"
exit $EXIT_CODE
