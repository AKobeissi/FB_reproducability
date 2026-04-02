#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=finqa_page_scorer
#SBATCH --output=logs/finqa_page_scorer_%j.log
#SBATCH --error=logs/finqa_page_scorer_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# ─────────────────────────────────────────────────────────────────────────────
# Cluster environment
# ─────────────────────────────────────────────────────────────────────────────
SUBMIT_DIR=$SLURM_SUBMIT_DIR
VENV_PATH="$SUBMIT_DIR/venv"
SCRATCH_DIR=/Tmp/$(whoami)/${SLURM_JOB_ID}
 
# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
JSONL="data/finqa_test_gold_pages.jsonl"
PDF_DIR_FINQA="Final-PDF"
PDF_DIR_FB="pdfs"
MODEL_OUT="models/finqa_page_scorer_lora"
EVAL_OUT="outputs/finqa_page_scorer"
LLM="Qwen/Qwen2.5-7B-Instruct"
 
# ─────────────────────────────────────────────────────────────────────────────
# Training hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
EPOCHS=10
BATCH_SIZE=8
GRAD_ACCUM=4
LR=2e-5
LORA_R=16
LORA_ALPHA=32
HARD_NEGS=3        # same-document hard negatives per positive
CROSS_DOC_NEGS=0   # cross-doc negs disabled — baseline reproduces original behaviour
LOSS_SCALE=20.0    # original scale
 
# ─────────────────────────────────────────────────────────────────────────────
# Inference hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
PAGE_K=100
RERANK_K=20
CHUNK_K=5
CHUNK_TOKENS=1024
OVERLAP_TOKENS=128
INDEX_BATCH=64      # reduce to 32 if OOM during page index build
 
# ─────────────────────────────────────────────────────────────────────────────
echo "========================================================="
echo "Job ID        : ${SLURM_JOB_ID:-local}"
echo "Node          : ${SLURMD_NODENAME:-$(hostname)}"
echo "GPU           : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "GPU memory    : $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)"
echo "Start time    : $(date)"
echo "========================================================="
echo "Key settings (MNR negative maximization):"
echo "  max_seq_length : 2048"
echo "  batch_size     : ${BATCH_SIZE} micro-batch (=${BATCH_SIZE}-1 in-batch MNR negatives)"
echo "  hard_negatives : ${HARD_NEGS}"
echo "  texts per step : $((BATCH_SIZE * (2 + HARD_NEGS))) at 2048 tok ≈ 15-20 GB GPU RAM"
echo "  trainer        : SentenceTransformerTrainer (grad_accum properly applied)"
echo "========================================================="
 
# ── Setup scratch ─────────────────────────────────────────────────────────────
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
 
# ── Activate venv ─────────────────────────────────────────────────────────────
echo "Activating venv: $VENV_PATH"
source "$VENV_PATH/bin/activate"
 
# ── HuggingFace cache ─────────────────────────────────────────────────────────
export HF_HOME=/data/rech/kobeissa/hf
export HF_HUB_CACHE=/data/rech/kobeissa/hf/hub
export HF_DATASETS_CACHE=/data/rech/kobeissa/hf/datasets
 
# Helps with memory fragmentation on long jobs
export PYTORCH_ALLOC_CONF=expandable_segments:True
 
# ─────────────────────────────────────────────────────────────────────────────
# HF_TOKEN — required for Qwen2.5-7B-Instruct (gated model) in Stage 2.
# BGE-M3 is NOT gated, so Stage 1 training works without it.
# Add to your .bashrc:  export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxx
# To get a token: https://huggingface.co/settings/tokens
# Then accept the Qwen2.5-7B-Instruct license at:
#   https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
# ─────────────────────────────────────────────────────────────────────────────
if [ -z "$HF_TOKEN" ]; then
    echo "[WARNING] HF_TOKEN not set — gated models (Qwen) may fail in Stage 2."
    echo "          Stage 1 (BGE-M3 training) will still work."
fi
 
# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — LoRA fine-tuning
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo ">>> STAGE 1: LoRA fine-tuning"
echo "    max_seq=2048, batch=${BATCH_SIZE}×${GRAD_ACCUM}=effective $((BATCH_SIZE*GRAD_ACCUM)), hard_negs=${HARD_NEGS}"
echo "========================================================="
 
python src/training/train_finqa_page_scorer.py \
    --jsonl            "${JSONL}" \
    --pdf-dir          "${PDF_DIR_FINQA}" \
    --output           "${MODEL_OUT}" \
    --epochs           "${EPOCHS}" \
    --batch-size       "${BATCH_SIZE}" \
    --grad-accum       "${GRAD_ACCUM}" \
    --lr               "${LR}" \
    --lora-r           "${LORA_R}" \
    --lora-alpha       "${LORA_ALPHA}" \
    --hard-negatives   "${HARD_NEGS}" \
    --cross-doc-negs   "${CROSS_DOC_NEGS}" \
    --loss-scale       "${LOSS_SCALE}"
 
if [ $? -ne 0 ]; then
    echo "[ERROR] Training failed — aborting."
    exit 1
fi
 
echo ""
echo ">>> Training complete. Syncing adapter back to submit dir..."
rsync -a "${SCRATCH_DIR}/${MODEL_OUT}/" "${SUBMIT_DIR}/${MODEL_OUT}/"
echo ">>> Adapter persisted to ${SUBMIT_DIR}/${MODEL_OUT}"
 
# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — FinanceBench evaluation
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo ">>> STAGE 2: FinanceBench 150 evaluation"
echo "    Scorer: BGE-M3 max_seq=2048 | Reranker: BGE-reranker-v2-m3"
echo "    NOTE: delete outputs/finqa_page_scorer/index_cache/ if re-running"
echo "          with a new model to force index rebuild."
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
 