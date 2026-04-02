#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=exp5_cross_doc
#SBATCH --output=logs/exp5_cross_doc_%j.log
#SBATCH --error=logs/exp5_cross_doc_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# =============================================================================
# Experiment 5 — Cross-document hard negatives + higher loss scale
#
# Root cause identified in exps 1-4:
#   All four experiments used only same-document negatives.
#   At eval time, the model retrieves from a global FAISS index of ALL 84
#   FinanceBench docs (7,000+ pages).  80.7% of top-5 retrieved results mix
#   pages from multiple docs.  The model was never trained to discriminate
#   against cross-doc confusors, which is the primary failure mode.
#
# Fixes applied here:
#   1. --cross-doc-negs 2
#      Adds 2 pages from OTHER documents as additional negatives per training
#      example, alongside the 3 same-doc random negatives (5 total).
#      This directly simulates the cross-doc interference seen at eval time.
#
#   2. --loss-scale 50.0  (was 20.0)
#      BGE-M3 cosine similarities cluster near 1.0 (top-20 scores: 0.55–0.99,
#      top-10 often all > 0.9).  Increasing the temperature scale from 20→50
#      sharpens the softmax so the model receives a larger gradient signal
#      when gold and negative scores differ by only ~0.001.
#
# Negatives per row  : 3 same-doc (random) + 2 cross-doc (random) = 5 total
# Dataset columns    : 7  (query, page_pos, 5×neg)
# Batch memory       : 8 × 7 = 56 texts/step  (fits in 40GB with max_seq=2048)
# =============================================================================

SUBMIT_DIR=$SLURM_SUBMIT_DIR
VENV_PATH="$SUBMIT_DIR/venv"
SCRATCH_DIR=/Tmp/$(whoami)/${SLURM_JOB_ID}

# ── Paths ──────────────────────────────────────────────────────────────────
JSONL="data/finqa_test_gold_pages.jsonl"
PDF_DIR_FINQA="Final-PDF"
PDF_DIR_FB="pdfs"
MODEL_OUT="models/exp5_cross_doc"
EVAL_OUT="outputs/exp5_cross_doc"
LLM="Qwen/Qwen2.5-7B-Instruct"

# ── Training hyperparameters ───────────────────────────────────────────────
EPOCHS=10
BATCH_SIZE=8
GRAD_ACCUM=4         # effective batch = 8×4 = 32
LR=2e-5
LORA_R=16
LORA_ALPHA=32
HARD_NEGS=3          # same-doc random negatives per positive
CROSS_DOC_NEGS=2     # cross-doc random negatives per positive  ← KEY FIX
LOSS_SCALE=50.0      # raised from 20.0  ← KEY FIX

# ── Inference hyperparameters (unchanged from exp 1-4) ─────────────────────
PAGE_K=30
RERANK_K=20
CHUNK_K=5
CHUNK_TOKENS_EVAL=1024
OVERLAP_TOKENS=128
INDEX_BATCH=64

echo "========================================================="
echo "Experiment     : 5 — Cross-doc negs + higher loss scale"
echo "Job ID         : ${SLURM_JOB_ID:-local}"
echo "Node           : ${SLURMD_NODENAME:-$(hostname)}"
echo "GPU            : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "GPU memory     : $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)"
echo "Start time     : $(date)"
echo "========================================================="
echo "Ablation flags : --cross-doc-negs ${CROSS_DOC_NEGS}  --loss-scale ${LOSS_SCALE}"
echo "Loss           : standard page-level MNR, scale=${LOSS_SCALE}"
echo "LoRA targets   : q/k/v only (baseline)"
echo "Negatives      : ${HARD_NEGS} same-doc random + ${CROSS_DOC_NEGS} cross-doc random = $((HARD_NEGS + CROSS_DOC_NEGS)) total"
echo "Dataset cols   : $((2 + HARD_NEGS + CROSS_DOC_NEGS))  (query, page_pos, $((HARD_NEGS + CROSS_DOC_NEGS))×neg)"
echo "batch_size     : ${BATCH_SIZE} × grad_accum ${GRAD_ACCUM} = effective $(( BATCH_SIZE * GRAD_ACCUM ))"
echo "encoder calls  : $((BATCH_SIZE * (2 + HARD_NEGS + CROSS_DOC_NEGS))) texts/step"
echo "========================================================="

mkdir -p "$SCRATCH_DIR"
mkdir -p "$SUBMIT_DIR/logs"

echo "Copying repository to scratch..."
rsync -a --quiet \
    --exclude 'venv' --exclude '.git' --exclude '__pycache__' \
    --exclude 'outputs' --exclude 'vector_stores' --exclude '*.log' \
    "$SUBMIT_DIR/" "$SCRATCH_DIR/"

cd "$SCRATCH_DIR"
source "$VENV_PATH/bin/activate"

pip install rank_bm25 --quiet --break-system-packages 2>/dev/null || true

export HF_HOME=/data/rech/kobeissa/hf
export HF_HUB_CACHE=/data/rech/kobeissa/hf/hub
export HF_DATASETS_CACHE=/data/rech/kobeissa/hf/datasets
export PYTORCH_ALLOC_CONF=expandable_segments:True

if [ -z "$HF_TOKEN" ]; then
    echo "[WARNING] HF_TOKEN not set — gated models (Qwen) may fail in Stage 2."
fi

# ── Stage 1: Training ───────────────────────────────────────────────────────
echo ""
echo ">>> STAGE 1: LoRA fine-tuning (cross-doc negs + scale=50)"
echo "    7 sentence columns: query, page_pos, 3×same-doc-neg, 2×cross-doc-neg"
echo "    All 115 FinQA training docs pre-loaded for cross-doc pool."
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
    echo "[ERROR] Training failed — aborting."; exit 1
fi

echo ""
echo ">>> Training complete. Syncing adapter back..."
rsync -a "${SCRATCH_DIR}/${MODEL_OUT}/" "${SUBMIT_DIR}/${MODEL_OUT}/"
echo ">>> Adapter persisted to ${SUBMIT_DIR}/${MODEL_OUT}"

# ── Stage 2: FinanceBench evaluation ────────────────────────────────────────
echo ""
echo ">>> STAGE 2: FinanceBench 150 evaluation"
echo "    Model: ${MODEL_OUT}  |  Index cache: ${EVAL_OUT}/index_cache/"
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
    --chunk-tokens      "${CHUNK_TOKENS_EVAL}" \
    --overlap-tokens    "${OVERLAP_TOKENS}" \
    --index-batch-size  "${INDEX_BATCH}" \
    --llm               "${LLM}"

if [ $? -ne 0 ]; then
    echo "[ERROR] Evaluation failed."; exit 1
fi

echo "Syncing eval results back..."
rsync -a "${SCRATCH_DIR}/${EVAL_OUT}/" "${SUBMIT_DIR}/${EVAL_OUT}/"

echo ""
echo "========================================================="
echo "Done!   End time : $(date)"
echo "Adapter          : ${SUBMIT_DIR}/${MODEL_OUT}"
echo "Results          : ${SUBMIT_DIR}/${EVAL_OUT}/"
echo "========================================================="
