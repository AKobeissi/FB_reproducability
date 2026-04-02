#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=domain_adapted_rag
#SBATCH --output=logs/domain_adapted_rag_%j.log
#SBATCH --error=logs/domain_adapted_rag_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00

# =============================================================================
# Domain-Adapted Financial RAG  —  Experiment 8 (v3, global search)
# =============================================================================
#
# Key changes over v2 (job 6027):
#   • GLOBAL SEARCH only — no oracle document filtering.
#     v2 used FinanceBench doc_name to filter ChromaDB per question (oracle mode).
#     v3 searches all pages across the full corpus (~12K pages) per query.
#   • Embed format: PDF-filename-derived prefix (TICKER YEAR DOCTYPE) prepended
#     to each page/chunk's embed_text. Derived from PDF file path only — no
#     FinanceBench metadata used. Provides document-level context for global
#     retrieval without oracle leakage.
#   • New v3 collection names — force-reindex rebuilds with new embed format.
#   • Layer freezing, LR, epochs unchanged from v2.
#
# Ablation variants (all global, no oracle filtering):
#   1. baseline_global          (BGE-M3, global)
#   2. ft_global                (FT BGE-M3, global)
#   3. ft_global_hyde           (FT + HyDE)
#   4. ft_global_rerank         (FT + rerank)
#   5. ft_global_hier           (FT + hier + rerank)
#   6. ft_global_hyde_rerank    (FULL: FT + HyDE + rerank, no oracle)
#
# =============================================================================

SUBMIT_DIR=$SLURM_SUBMIT_DIR
VENV_PATH="$SUBMIT_DIR/venv"
SCRATCH_DIR=/Tmp/$(whoami)/${SLURM_JOB_ID}

# HuggingFace cache (reuse existing)
export HF_HOME=/data/rech/kobeissa/hf
export HF_HUB_CACHE=/data/rech/kobeissa/hf/hub
export HF_DATASETS_CACHE=/data/rech/kobeissa/hf/datasets
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Settings
BASE_MODEL="BAAI/bge-m3"
QWEN_MODEL="Qwen/Qwen2.5-7B-Instruct"
EPOCHS=5
LR=5e-6
TRAINABLE_LAYERS=3
BATCH_SIZE=16
CANDIDATE_PAGES=20
NUM_HYPOTHETICALS=3

echo "========================================================="
echo "Job ID        : ${SLURM_JOB_ID:-local}"
echo "Node          : ${SLURMD_NODENAME:-$(hostname)}"
echo "GPU           : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Start time    : $(date)"
echo "========================================================="
echo "Settings:"
echo "  Base model       : ${BASE_MODEL}"
echo "  Trainable layers : ${TRAINABLE_LAYERS} (last N transformer layers)"
echo "  Epochs / LR      : ${EPOCHS} / ${LR}"
echo "  Batch size       : ${BATCH_SIZE}"
echo "  Qwen model       : ${QWEN_MODEL}"
echo "  Doc-filter       : DISABLED (global search, no oracle)"
echo "  Embed prefix     : PDF-filename-derived (TICKER YEAR DOCTYPE)"
echo "  Hierarchical     : enabled (chunk-level index)"
echo "========================================================="

# Setup scratch
mkdir -p "$SCRATCH_DIR"
mkdir -p "$SUBMIT_DIR/logs"

echo "Copying repository to scratch..."
rsync -a --quiet \
    --exclude 'venv' \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude 'vector_stores' \
    --exclude '*.log' \
    "$SUBMIT_DIR/" "$SCRATCH_DIR/"

cd "$SCRATCH_DIR"

echo "Activating venv: $VENV_PATH"
source "$VENV_PATH/bin/activate"

if [ -z "$HF_TOKEN" ]; then
    echo "[WARNING] HF_TOKEN not set — Qwen download may fail if not cached."
fi

# =============================================================================
# Stage 1: Fine-tune bi-encoder (bge-m3 + layer freezing)
# =============================================================================
echo ""
echo ">>> STAGE 1: Fine-tuning ${BASE_MODEL} on FinQA page pairs"
echo "            (last ${TRAINABLE_LAYERS} layers trainable, LR=${LR})"
echo "========================================================="

python domain_adapted_retrieval/run_experiment.py \
    --mode train \
    --base-model "${BASE_MODEL}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --trainable-layers "${TRAINABLE_LAYERS}"

if [ $? -ne 0 ]; then
    echo "[ERROR] Fine-tuning failed — aborting."
    exit 1
fi

echo ">>> Syncing fine-tuned model to submit dir..."
rsync -a "${SCRATCH_DIR}/models/fin_adapted_biencoder_bge_m3/" \
         "${SUBMIT_DIR}/models/fin_adapted_biencoder_bge_m3/"
echo ">>> Model synced."

# =============================================================================
# Stage 2: Build ChromaDB indexes (page-level + chunk-level + baseline)
# =============================================================================
echo ""
echo ">>> STAGE 2: Building ChromaDB indexes (page + chunk + baseline BGE-M3)"
echo "========================================================="

python domain_adapted_retrieval/run_experiment.py \
    --mode index \
    --force-reindex \
    --no-doc-filter

if [ $? -ne 0 ]; then
    echo "[ERROR] Index build failed — aborting."
    exit 1
fi

# =============================================================================
# Stage 3: Ablation evaluation + per-type metrics + plots
# =============================================================================
echo ""
echo ">>> STAGE 3: Running ablation (7 variants + per-question-type breakdown)"
echo "========================================================="

python domain_adapted_retrieval/run_experiment.py \
    --mode eval \
    --qwen-model "${QWEN_MODEL}" \
    --candidate-pages "${CANDIDATE_PAGES}" \
    --num-hypotheticals "${NUM_HYPOTHETICALS}" \
    --no-doc-filter \
    --no-generation

if [ $? -ne 0 ]; then
    echo "[ERROR] Evaluation failed."
    exit 1
fi

# Sync results back
echo "Syncing results back to submit dir..."
rsync -a "${SCRATCH_DIR}/domain_adapted_retrieval/results/" \
         "${SUBMIT_DIR}/domain_adapted_retrieval/results/"

echo ""
echo "========================================================="
echo "Done!   End time : $(date)"
echo "Results : ${SUBMIT_DIR}/domain_adapted_retrieval/results/"
echo "  metrics/ablation_table.csv      <- overall table (LaTeX-ready)"
echo "  metrics/ablation_by_type.csv    <- per-question-type breakdown"
echo "  plots/by_question_type_k5.pdf   <- per-type bar chart"
echo "  plots/ablation_bar_chart_k5.pdf <- DocRec + PageRec comparison"
echo "  plots/recall_at_k_curves.pdf    <- PageRec@k curves"
echo "  plots/combined_panel_k5.pdf     <- 2x2 all-metrics panel"
echo ""
echo "NOTE: All variants use GLOBAL search (no oracle document filtering)."
echo "      Embed texts are prefixed with PDF-filename-derived context."
echo "========================================================="
