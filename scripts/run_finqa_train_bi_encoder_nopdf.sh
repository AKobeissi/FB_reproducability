#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=finqa_bienc_nopdf
#SBATCH --output=logs/finqa_bienc_nopdf_%j.log
#SBATCH --error=logs/finqa_bienc_nopdf_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00

# =============================================================================
# FinQA-train → BGE-M3 bi-encoder fine-tuning (NO PDFs needed)
# =============================================================================
#
# Gold passage = pre_text + table + post_text directly from finqa/train.json.
# Negatives    = other documents' gold passages (cross-doc, same pool).
# No PDF extraction required — identical pipeline to the cross-encoder reranker.
#
# Stage 1: LoRA fine-tune BGE-M3 bi-encoder
# Stage 2: evaluate on FinanceBench 150
# =============================================================================

SUBMIT_DIR=$SLURM_SUBMIT_DIR
VENV_PATH="$SUBMIT_DIR/venv"
SCRATCH_DIR=/Tmp/$(whoami)/${SLURM_JOB_ID}

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_JSON="finqa/train.json"
PDF_DIR_FB="pdfs"
MODEL_OUT="models/finqa_train_bi_encoder_nopdf"
EVAL_OUT="outputs/finqa_train_bi_encoder_nopdf"
LLM="Qwen/Qwen2.5-7B-Instruct"

# ─────────────────────────────────────────────────────────────────────────────
# Training hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
EPOCHS=10
BATCH_SIZE=8
GRAD_ACCUM=4          # effective batch = 32
LR=2e-5
LORA_R=16
LORA_ALPHA=32
HARD_NEGS=3           # cross-doc hard negatives per positive (BM25-mined)
LOSS_SCALE=50.0

# ─────────────────────────────────────────────────────────────────────────────
# Inference hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
PAGE_K=100
RERANK_K=20
CHUNK_K=5
CHUNK_TOKENS=1024
OVERLAP_TOKENS=128
INDEX_BATCH=64

echo "========================================================="
echo "Job ID        : ${SLURM_JOB_ID:-local}"
echo "Node          : ${SLURMD_NODENAME:-$(hostname)}"
echo "GPU           : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "GPU memory    : $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)"
echo "Start time    : $(date)"
echo "========================================================="
echo "Training config (no PDFs):"
echo "  Gold text     : pre_text + table + post_text from train.json"
echo "  Negatives     : BM25-mined cross-doc passages from other train samples"
echo "  hard_negs     : ${HARD_NEGS}"
echo "  loss_scale    : ${LOSS_SCALE}"
echo "  batch_size    : ${BATCH_SIZE} micro × ${GRAD_ACCUM} = $((BATCH_SIZE*GRAD_ACCUM)) effective"
echo "========================================================="

mkdir -p "$SCRATCH_DIR"
mkdir -p "$SUBMIT_DIR/logs"

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

export HF_HOME=/data/rech/kobeissa/hf
export HF_HUB_CACHE=/data/rech/kobeissa/hf/hub
export HF_DATASETS_CACHE=/data/rech/kobeissa/hf/datasets
export PYTORCH_ALLOC_CONF=expandable_segments:True

if [ -z "$HF_TOKEN" ]; then
    echo "[WARNING] HF_TOKEN not set — gated models (Qwen) may fail in Stage 2."
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — LoRA fine-tuning
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo ">>> STAGE 1: LoRA fine-tuning (no PDFs)"
echo "    Input: ${TRAIN_JSON}  (~6,251 samples)"
echo "========================================================="

python src/training/train_finqa_bi_encoder_nopdf.py \
    --train-json       "${TRAIN_JSON}" \
    --output           "${MODEL_OUT}" \
    --epochs           "${EPOCHS}" \
    --batch-size       "${BATCH_SIZE}" \
    --grad-accum       "${GRAD_ACCUM}" \
    --lr               "${LR}" \
    --lora-r           "${LORA_R}" \
    --lora-alpha       "${LORA_ALPHA}" \
    --hard-negatives   "${HARD_NEGS}" \
    --loss-scale       "${LOSS_SCALE}" \
    --use-bm25-hard-negs

if [ $? -ne 0 ]; then
    echo "[ERROR] Training failed — aborting."
    exit 1
fi

echo ""
echo ">>> Stage 1 complete. Syncing adapter back to submit dir..."
rsync -a "${SCRATCH_DIR}/${MODEL_OUT}/" "${SUBMIT_DIR}/${MODEL_OUT}/"
echo ">>> Adapter persisted to ${SUBMIT_DIR}/${MODEL_OUT}"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — FinanceBench evaluation
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo ">>> STAGE 2: FinanceBench 150 evaluation"
echo "    NOTE: delete ${EVAL_OUT}/index_cache/ to force index rebuild."
echo "========================================================="

python scripts/run_finqa_page_scorer_fb.py \
    --model-path        "${MODEL_OUT}" \
    --pdf-dir           "${PDF_DIR_FB}" \
    --output-dir        "${EVAL_OUT}" \
    --page-k            "${PAGE_K}" \
    --rerank-k          "${RERANK_K}" \
    --use-reranker \
    --reranker-model    "BAAI/bge-reranker-v2-m3" \
    --chunk-k           "${CHUNK_K}" \
    --chunk-tokens      "${CHUNK_TOKENS}" \
    --overlap-tokens    "${OVERLAP_TOKENS}" \
    --index-batch-size  "${INDEX_BATCH}" \
    --llm               "${LLM}"

if [ $? -ne 0 ]; then
    echo "[ERROR] Evaluation failed."
    exit 1
fi

echo "Syncing eval results back to submit dir..."
rsync -a "${SCRATCH_DIR}/${EVAL_OUT}/" "${SUBMIT_DIR}/${EVAL_OUT}/"

echo ""
echo "========================================================="
echo "Done!  End time : $(date)"
echo "Adapter         : ${SUBMIT_DIR}/${MODEL_OUT}"
echo "Results         : ${SUBMIT_DIR}/${EVAL_OUT}/"
echo "========================================================="
