#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=fb_baselines
#SBATCH --output=logs/baselines_%j.log
#SBATCH --error=logs/baselines_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00

# =============================================================================
# FinanceBench Baseline Methods — Full Experiment Suite
# =============================================================================
#
# Runs 12 retrieval methods and produces:
#   - DocRec@k / PageRec@k  for k = 1, 3, 5, 10, 20
#   - Context BLEU@5 and ROUGE-L@5  (vs gold evidence)
#   - Generated answer ROUGE-L  (vs reference answer)
#   - Numeric match  (metrics-generated questions only)
#   - Breakdowns by question_type and doc_type
#   - 8 publication-quality plots
#
# Chunking: RecursiveCharacterTextSplitter, size=1024, overlap=30
#
# Memory strategy  (L40S 46 GB):
#   Phase 1  Index building   BGE-M3 + SPLADE  (~3 GB)
#   Phase 2  HyDE pre-gen     Qwen 7B 4-bit only (~4.5 GB)  → freed
#   Phase 3  Retrieval        BGE-M3 + reranker (~4.6 GB)   → freed
#   Phase 4  Generation       Qwen 7B 4-bit only (~4.5 GB)  → freed
#   Phase 5  Eval + plots     CPU only
#
# To re-run only specific variants (e.g. after fixing one method):
#   sbatch scripts/run_baselines.sh --variants dense_bge_m3 bge_reranker
#
# To skip answer generation (retrieval metrics only, faster):
#   sbatch scripts/run_baselines.sh --skip-generation
#
# To resume a partially completed run without rebuilding indexes:
#   sbatch scripts/run_baselines.sh --resume
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

# Pass any extra CLI args from sbatch command line (e.g. --skip-generation)
EXTRA_ARGS="$@"

echo ""
echo ">>> Running baseline experiments"
echo "    Extra args: ${EXTRA_ARGS:-none}"
echo "========================================================="

python baselines/run_baselines.py \
    --data-path     data/financebench_open_source.jsonl \
    --doc-info-path data/financebench_document_information.jsonl \
    --pdf-dir       pdfs \
    --vs-dir        vector_stores/baselines \
    --output-dir    baselines/results \
    --hyde-cache    baselines/hyde_cache.json \
    $EXTRA_ARGS

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "[ERROR] Baseline runner failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo ""
echo ">>> Syncing results back to submit dir..."
rsync -a "${SCRATCH_DIR}/baselines/results/"  "${SUBMIT_DIR}/baselines/results/"
rsync -a "${SCRATCH_DIR}/baselines/hyde_cache.json" "${SUBMIT_DIR}/baselines/hyde_cache.json" 2>/dev/null || true
rsync -a "${SCRATCH_DIR}/vector_stores/baselines/"  "${SUBMIT_DIR}/vector_stores/baselines/"

echo ""
echo "========================================================="
echo "Done!   End time : $(date)"
echo ""
echo "Results:"
echo "  baselines/results/metrics/baseline_table.csv     <- main table"
echo "  baselines/results/metrics/baseline_table_k5.tex  <- LaTeX snippet"
echo "  baselines/results/metrics/by_question_type.csv   <- per-type breakdown"
echo "  baselines/results/metrics/by_doc_type.csv        <- per-doc-type"
echo "  baselines/results/plots/                         <- all figures"
echo "========================================================="
