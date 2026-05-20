#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=finqa_train_bienc
#SBATCH --output=logs/finqa_train_bienc_%j.log
#SBATCH --error=logs/finqa_train_bienc_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# =============================================================================
# FinQA-train → BGE-M3 bi-encoder (page scorer) fine-tuning + FB eval
# =============================================================================
#
# Trains the page-level bi-encoder on the FinQA TRAIN split (6,251 samples)
# instead of cross-validation on FinanceBench.  No leakage: the model is
# evaluated on FinanceBench, which has no overlap with FinQA questions.
#
# Key differences vs finqa_page_scorer.sh (which used the test split, 530 rows):
#   - JSONL: finqa_train_gold_pages.jsonl  (~6,251 pairs vs ~530)
#   - --use-bm25-hard-negs: harder same-doc negatives (BM25-mined)
#   - --cross-doc-negs 2: pages from other documents simulate global-index eval
#   - --loss-scale 50.0: sharper softmax for compressed BGE-M3 cosine scores
#
# Stage 0: generate finqa_train_gold_pages.jsonl if not present
# Stage 1: LoRA fine-tune BGE-M3 page scorer
# Stage 2: evaluate on FinanceBench 150
# =============================================================================

SUBMIT_DIR=$SLURM_SUBMIT_DIR
VENV_PATH="$SUBMIT_DIR/venv"
SCRATCH_DIR=/Tmp/$(whoami)/${SLURM_JOB_ID}

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
FINQA_TRAIN="finqa/train.json"
TRAIN_JSONL="data/finqa_train_gold_pages.jsonl"
REJECT_JSONL="data/finqa_train_gold_pages.rejects.jsonl"
PDF_DIR_FINQA="Final-PDF"
PDF_DIR_FB="pdfs"
MODEL_OUT="models/finqa_train_bi_encoder"
EVAL_OUT="outputs/finqa_train_bi_encoder"
LLM="Qwen/Qwen2.5-7B-Instruct"

# ─────────────────────────────────────────────────────────────────────────────
# Training hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
EPOCHS=10
BATCH_SIZE=8
GRAD_ACCUM=4           # effective batch = 32
LR=2e-5
LORA_R=16
LORA_ALPHA=32
HARD_NEGS=3            # same-doc BM25-mined hard negatives per positive
CROSS_DOC_NEGS=2       # cross-doc negatives — simulate global-index inference
LOSS_SCALE=50.0        # sharper than default 20.0 for BGE-M3 cosine range

# ─────────────────────────────────────────────────────────────────────────────
# Inference hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
PAGE_K=100
RERANK_K=20
CHUNK_K=5
CHUNK_TOKENS=1024
OVERLAP_TOKENS=128
INDEX_BATCH=64

# ─────────────────────────────────────────────────────────────────────────────
echo "========================================================="
echo "Job ID        : ${SLURM_JOB_ID:-local}"
echo "Node          : ${SLURMD_NODENAME:-$(hostname)}"
echo "GPU           : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "GPU memory    : $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)"
echo "Start time    : $(date)"
echo "========================================================="
echo "Training config:"
echo "  Training data  : ${TRAIN_JSONL}  (FinQA train split, ~6251 pairs)"
echo "  max_seq_length : 2048"
echo "  hard_negatives : ${HARD_NEGS} (BM25-mined same-doc)"
echo "  cross_doc_negs : ${CROSS_DOC_NEGS}"
echo "  loss_scale     : ${LOSS_SCALE}"
echo "  batch_size     : ${BATCH_SIZE} micro × ${GRAD_ACCUM} accum = $((BATCH_SIZE*GRAD_ACCUM)) effective"
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
export PYTORCH_ALLOC_CONF=expandable_segments:True

if [ -z "$HF_TOKEN" ]; then
    echo "[WARNING] HF_TOKEN not set — gated models (Qwen) may fail in Stage 2."
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage 0 — Generate finqa_train_gold_pages.jsonl if not present
# ─────────────────────────────────────────────────────────────────────────────
if [ ! -f "${TRAIN_JSONL}" ]; then
    echo ""
    echo ">>> STAGE 0: Preparing FinQA train gold-page JSONL"
    echo "    Input : ${FINQA_TRAIN}  (6,251 entries)"
    echo "    Output: ${TRAIN_JSONL}"
    echo "========================================================="

    python scripts/prepare_finqa_train_gold_pages.py \
        --finqa-train  "${FINQA_TRAIN}" \
        --pdf-dir      "${PDF_DIR_FINQA}" \
        --output       "${TRAIN_JSONL}" \
        --reject-log   "${REJECT_JSONL}" \
        --min-overlap  0.05 \
        --search-window 5

    if [ $? -ne 0 ]; then
        echo "[ERROR] JSONL preparation failed — aborting."
        exit 1
    fi

    echo ">>> Stage 0 complete: ${TRAIN_JSONL}"
    echo "    Syncing JSONL back to submit dir..."
    rsync -a "${SCRATCH_DIR}/data/" "${SUBMIT_DIR}/data/"
else
    echo ">>> Stage 0 skipped: ${TRAIN_JSONL} already exists"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — LoRA fine-tuning on FinQA train split
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo ">>> STAGE 1: LoRA fine-tuning on FinQA train split"
echo "    JSONL: ${TRAIN_JSONL}"
echo "    --use-bm25-hard-negs: same-doc negatives mined by BM25 overlap"
echo "    --cross-doc-negs ${CROSS_DOC_NEGS}: pages from other documents"
echo "========================================================="

python src/training/train_finqa_page_scorer.py \
    --jsonl            "${TRAIN_JSONL}" \
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
echo "    Scorer: BGE-M3 + LoRA (finqa-train)  |  Reranker: BGE-reranker-v2-m3"
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
