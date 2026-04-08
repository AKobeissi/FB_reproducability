#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=fb_baselines_ner
#SBATCH --output=logs/baselines_ner_%j.log
#SBATCH --error=logs/baselines_ner_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=36:00:00

# =============================================================================
# FinanceBench Baselines  —  NER Doc-Filter Ablation
# =============================================================================
#
# Runs all 12 baseline retrieval methods TWICE:
#   1. Without NER filter (global search, as before)
#   2. With NER filter    (entity recognition predicts target document)
#
# The NER filter extracts company name, fiscal year, and document type from
# the question text, then fuzzy-matches against the document corpus.
# No oracle doc_name label is used.
#
# Produces DocRec@k / PageRec@k for k = 1, 3, 5, 10, 20 for all variants.
#
# Outputs
# -------
#   baselines/results/metrics/all_variants_metrics.json   <- all methods
#   baselines/results/metrics/all_k_summary.csv           <- @k table
#   baselines/results/metrics/baseline_table.csv          <- @5 table
#   baselines/results/metrics/by_question_type.csv        <- type breakdown
#   baselines/results/plots/                              <- figures
#
# Submit:
#   sbatch scripts/run_baselines_ner.sh
#
# NER variants only (skip non-NER baselines):
#   sbatch scripts/run_baselines_ner.sh --ner-only
#
# Specific variant pairs:
#   sbatch scripts/run_baselines_ner.sh --variants dense_bge_m3 bge_reranker
#
# Resume a partial run:
#   sbatch scripts/run_baselines_ner.sh --resume
#
# Skip generation (retrieval metrics only, much faster):
#   sbatch scripts/run_baselines_ner.sh --skip-generation
# =============================================================================

SUBMIT_DIR=$SLURM_SUBMIT_DIR
VENV_PATH="$SUBMIT_DIR/venv"
SCRATCH_DIR=/Tmp/$(whoami)/${SLURM_JOB_ID}

# HuggingFace cache
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

echo "Python: $(which python3)"
echo "Torch CUDA: $(python3 -c 'import torch; print(torch.cuda.is_available(), torch.version.cuda)')"

# ── Ensure spaCy and its English model are available ──────────────────────────
python3 -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null || {
  echo "Installing spaCy + English model for NER doc filter..."
  pip install spacy --quiet
  python3 -m spacy download en_core_web_lg --quiet 2>/dev/null || true
  python3 -m spacy download en_core_web_sm --quiet 2>/dev/null || true
}

# Pass any extra CLI args from sbatch command line
EXTRA_ARGS="$@"

echo ""
echo ">>> Running baseline experiments with NER doc-filter ablation"
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
    --ner-filter \
    --ner-top-k 3 \
    $EXTRA_ARGS

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "[ERROR] Baseline runner failed with exit code $EXIT_CODE"
fi

echo ""
echo ">>> Syncing results back to submit dir..."
rsync -a "${SCRATCH_DIR}/baselines/results/"  "${SUBMIT_DIR}/baselines/results/"
rsync -a "${SCRATCH_DIR}/baselines/hyde_cache.json" "${SUBMIT_DIR}/baselines/hyde_cache.json" 2>/dev/null || true
rsync -a "${SCRATCH_DIR}/vector_stores/baselines/"  "${SUBMIT_DIR}/vector_stores/baselines/" 2>/dev/null || true

rm -rf "$SCRATCH_DIR"

echo ""
echo "========================================================="
echo "Done!   End time : $(date)"
echo "Exit status : $EXIT_CODE"
echo ""
echo "Results:"
echo "  baselines/results/metrics/all_k_summary.csv         <- @k=1,3,5,10,20 for all variants + NER variants"
echo "  baselines/results/metrics/baseline_table.csv        <- @5 comparison"
echo "  baselines/results/metrics/all_variants_metrics.json <- full JSON"
echo "  baselines/results/metrics/by_question_type.csv      <- type breakdown"
echo "  baselines/results/plots/                            <- figures"
echo "========================================================="

exit $EXIT_CODE
