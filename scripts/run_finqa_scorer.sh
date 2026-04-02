#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=finqa_scorer
#SBATCH --output=logs/finqa_scorer_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:ls40:1
#SBATCH --mem=48G
#SBATCH --time=08:00:00

# =============================================================================
# FinQA Embedding Scorer Training
# =============================================================================
#
# Trains an embedding model on FinQA financial QA pairs (MNRL loss),
# then evaluates on:
#   1. FinQA test set    – in-document passage retrieval (Hit@K, Recall@K, MRR)
#   2. FinanceBench 150  – PDF page retrieval
#
# MODEL WEIGHTS ARE NEVER SAVED.
# Every sbatch run retrains from scratch.
# Only lightweight artefacts (JSON + plots) are written to $SCRATCH.
#
# Output location:
#   $SCRATCH/finqa_scorer/<timestamp>/
#     config.json, final_results.json, summary.txt, training_curves.png
# =============================================================================

set -euo pipefail

source venv/bin/activate

mkdir -p logs

# ---- Paths ----
FINQA_DIR="finqa"       # train.json / dev.json / test.json live here
PDF_DIR="pdfs"          # FinanceBench PDFs

# ---- Model ----
#BASE_MODEL="sentence-transformers/all-mpnet-base-v2"
# Stronger alternative:
BASE_MODEL="BAAI/bge-m3"

# ---- Hyperparameters ----
EPOCHS=10
BATCH_SIZE=32
LR=2e-5
PAGE_K=5        # Top-K for FinanceBench page-retrieval eval
FINQA_TOP_K=5   # Top-K for FinQA in-document passage-retrieval eval
SEED=42

echo "=========================================================="
echo " FinQA Embedding Scorer  (no weights saved)"
echo "=========================================================="
echo " Base model  : $BASE_MODEL"
echo " FinQA dir   : $FINQA_DIR"
echo " PDF dir     : $PDF_DIR"
echo " Epochs      : $EPOCHS"
echo " Batch size  : $BATCH_SIZE"
echo " LR          : $LR"
echo " Page K (FB) : $PAGE_K"
echo " Passage K   : $FINQA_TOP_K"
echo " Artefacts → : \$SCRATCH/finqa_scorer/<timestamp>/"
echo "=========================================================="
echo ""

# $SCRATCH is set automatically by SLURM on this cluster.
# The Python script reads it via os.environ["SCRATCH"].
# Pass --scratch-dir only if you want to override the default.

python -m src.training.train_finqa_scorer \
    --finqa-dir   "$FINQA_DIR"   \
    --pdf-dir     "$PDF_DIR"     \
    --base-model  "$BASE_MODEL"  \
    --epochs      "$EPOCHS"      \
    --batch-size  "$BATCH_SIZE"  \
    --lr          "$LR"          \
    --page-k      "$PAGE_K"      \
    --finqa-top-k "$FINQA_TOP_K" \
    --seed        "$SEED"
# Note: no --scratch-dir → auto-resolves to $SCRATCH/finqa_scorer/<timestamp>

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Done. Artefacts in: \$SCRATCH/finqa_scorer/<timestamp>/"
    echo "  - summary.txt"
    echo "  - final_results.json"
    echo "  - training_curves.png"
    echo "  No model weights were saved."
else
    echo "✗ Failed (exit $EXIT_CODE). Check logs/finqa_scorer_${SLURM_JOB_ID}.log"
fi

exit $EXIT_CODE