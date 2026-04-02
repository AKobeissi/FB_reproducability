#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=exp4_all
#SBATCH --output=logs/exp4_all_%j.log
#SBATCH --error=logs/exp4_all_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# =============================================================================
# Experiment 4 — All three ablation flags combined
#
# Flags : --use-hierarchical-loss  --use-bm25-hard-negs  --extended-lora
#
# Summary of all changes vs job-6000 baseline:
#
#   Loss         : L = 0.7 * MNR(query, page) + 0.3 * MNR(query, chunk)
#   Page negs    : BM25-mined top-3 per query (instead of random)
#   Chunk gold   : BM25-best 512-tok chunk from the gold page
#   Chunk negs   : BM25-best chunk from each BM25-mined page negative
#   LoRA targets : q/k/v + attention.output.dense + intermediate.dense
#                  (~5.1M trainable params, 0.90%)
#
# Batch:
#   9 sentence columns × micro-batch 4 = 36 encoder calls per step
#   Page cols at 2048 tok, chunk cols at ≤512 tok → total activations
#   comparable to baseline 5 cols × 8 × 2048.
#   Effective batch = 4 × 8 = 32 (unchanged from baseline).
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
MODEL_OUT="models/exp4_all"
EVAL_OUT="outputs/exp4_all"
LLM="Qwen/Qwen2.5-7B-Instruct"

# ── Training hyperparameters ───────────────────────────────────────────────
EPOCHS=10
BATCH_SIZE=4        # 9 cols × 4 = 36 texts/step (hierarchical requires smaller batch)
GRAD_ACCUM=8        # effective batch = 4×8 = 32 (unchanged)
LR=2e-5
LORA_R=16
LORA_ALPHA=32
HARD_NEGS=3
CROSS_DOC_NEGS=0    # disabled — isolates the original 3-flag combo
LOSS_SCALE=20.0     # original scale — isolates the original 3-flag combo
HIER_ALPHA=0.7
CHUNK_TOKENS=512

# ── Inference hyperparameters (unchanged) ─────────────────────────────────
PAGE_K=30
RERANK_K=20
CHUNK_K=5
CHUNK_TOKENS_EVAL=1024
OVERLAP_TOKENS=128
INDEX_BATCH=64

echo "========================================================="
echo "Experiment     : 4 — All ablations combined"
echo "Job ID         : ${SLURM_JOB_ID:-local}"
echo "Node           : ${SLURMD_NODENAME:-$(hostname)}"
echo "GPU            : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "GPU memory     : $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)"
echo "Start time     : $(date)"
echo "========================================================="
echo "Ablation flags : --use-hierarchical-loss --use-bm25-hard-negs --extended-lora"
echo "Loss           : 0.7 * L_page + 0.3 * L_chunk"
echo "Negatives      : BM25-mined (page + chunk levels)"
echo "LoRA targets   : q/k/v + attention.output.dense + intermediate.dense"
echo "batch_size     : ${BATCH_SIZE} × grad_accum ${GRAD_ACCUM} = effective $(( BATCH_SIZE * GRAD_ACCUM ))"
echo "Dataset cols   : 9  (query, page_pos, 3×bm25_page_neg, chunk_pos, 3×chunk_neg)"
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

# rank_bm25 is required for --use-bm25-hard-negs
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
echo ">>> STAGE 1: LoRA fine-tuning (all ablations)"
echo "    9 cols: query, page_pos, 3×bm25_page_neg, chunk_pos, 3×bm25_chunk_neg"
echo "========================================================="

python src/training/train_finqa_page_scorer.py \
    --jsonl               "${JSONL}" \
    --pdf-dir             "${PDF_DIR_FINQA}" \
    --output              "${MODEL_OUT}" \
    --epochs              "${EPOCHS}" \
    --batch-size          "${BATCH_SIZE}" \
    --grad-accum          "${GRAD_ACCUM}" \
    --lr                  "${LR}" \
    --lora-r              "${LORA_R}" \
    --lora-alpha          "${LORA_ALPHA}" \
    --hard-negatives      "${HARD_NEGS}" \
    --cross-doc-negs      "${CROSS_DOC_NEGS}" \
    --loss-scale          "${LOSS_SCALE}" \
    --hierarchical-alpha  "${HIER_ALPHA}" \
    --chunk-tokens        "${CHUNK_TOKENS}" \
    --use-hierarchical-loss \
    --use-bm25-hard-negs \
    --extended-lora

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