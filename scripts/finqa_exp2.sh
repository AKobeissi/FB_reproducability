#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=exp2_bm25_negs
#SBATCH --output=logs/exp2_bm25_negs_%j.log
#SBATCH --error=logs/exp2_bm25_negs_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# =============================================================================
# Experiment 2 — BM25 hard negative mining only
#
# Ablation flag : --use-bm25-hard-negs
# Change vs job-6000 baseline:
#   Negs   : top-3 BM25-ranked non-gold pages per query (replaces random)
#            One BM25 index built per document and cached across queries.
#   Loss   : standard page-level MNR (unchanged)
#   LoRA   : unchanged (q/k/v only)
#   Batch  : unchanged (8 × 4 = effective 32)
#
# Why this matters: random negatives from a financial 10-K are topically
# similar to the gold page regardless of query (they're all financial tables).
# BM25 selects the pages that *lexically* resemble the query — exactly the
# failure cases at inference — so the model trains against harder confusions.
#
# Requires: rank_bm25  (pip install rank_bm25)
# =============================================================================

SUBMIT_DIR=$SLURM_SUBMIT_DIR
VENV_PATH="$SUBMIT_DIR/venv"
SCRATCH_DIR=/Tmp/$(whoami)/${SLURM_JOB_ID}

# ── Paths ──────────────────────────────────────────────────────────────────
JSONL="data/finqa_test_gold_pages.jsonl"
PDF_DIR_FINQA="Final-PDF"
PDF_DIR_FB="pdfs"
MODEL_OUT="models/exp2_bm25_negs"
EVAL_OUT="outputs/exp2_bm25_negs"
LLM="Qwen/Qwen2.5-7B-Instruct"

# ── Training hyperparameters ───────────────────────────────────────────────
EPOCHS=10
BATCH_SIZE=8        # same as baseline — 5 cols × 8 = 40 texts/step
GRAD_ACCUM=4
LR=2e-5
LORA_R=16
LORA_ALPHA=32
HARD_NEGS=3
CROSS_DOC_NEGS=0    # disabled — isolates BM25-negs effect only
LOSS_SCALE=20.0     # original scale — isolates BM25-negs effect only

# ── Inference hyperparameters (unchanged) ─────────────────────────────────
PAGE_K=30
RERANK_K=20
CHUNK_K=5
CHUNK_TOKENS=1024
OVERLAP_TOKENS=128
INDEX_BATCH=64

echo "========================================================="
echo "Experiment     : 2 — BM25 hard negative mining"
echo "Job ID         : ${SLURM_JOB_ID:-local}"
echo "Node           : ${SLURMD_NODENAME:-$(hostname)}"
echo "GPU            : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "GPU memory     : $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)"
echo "Start time     : $(date)"
echo "========================================================="
echo "Ablation flags : --use-bm25-hard-negs"
echo "Loss           : standard page-level MNR (baseline)"
echo "LoRA targets   : q/k/v only (baseline)"
echo "Negatives      : BM25-mined  (top-${HARD_NEGS} per query, cached per doc)"
echo "batch_size     : ${BATCH_SIZE} × grad_accum ${GRAD_ACCUM} = effective $(( BATCH_SIZE * GRAD_ACCUM ))"
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

# rank_bm25 is required for this experiment
pip install rank_bm25 --quiet --break-system-packages 2>/dev/null || \
    pip install rank_bm25 --quiet || \
    { echo "[ERROR] Could not install rank_bm25 — aborting."; exit 1; }

python -c "from rank_bm25 import BM25Okapi; print('rank_bm25 OK')" || \
    { echo "[ERROR] rank_bm25 import failed — aborting."; exit 1; }

export HF_HOME=/data/rech/kobeissa/hf
export HF_HUB_CACHE=/data/rech/kobeissa/hf/hub
export HF_DATASETS_CACHE=/data/rech/kobeissa/hf/datasets
export PYTORCH_ALLOC_CONF=expandable_segments:True

if [ -z "$HF_TOKEN" ]; then
    echo "[WARNING] HF_TOKEN not set — gated models (Qwen) may fail in Stage 2."
fi

# ── Stage 1: Training ───────────────────────────────────────────────────────
echo ""
echo ">>> STAGE 1: LoRA fine-tuning (BM25 hard negatives)"
echo "    5 sentence columns: query, page_pos, 3×bm25_page_neg"
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
    --loss-scale       "${LOSS_SCALE}" \
    --use-bm25-hard-negs

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
    --chunk-tokens      "${CHUNK_TOKENS}" \
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