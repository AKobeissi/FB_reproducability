#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=exp7_within_doc
#SBATCH --output=logs/exp7_within_doc_%j.log
#SBATCH --error=logs/exp7_within_doc_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=04:00:00

# =============================================================================
# Experiment 7 — Within-doc hard negs + BGE-M3 query instruction
#
# ROOT CAUSE ANALYSIS of Exps 1–6 (full global search retained):
#
#   [WRONG TRAINING TASK]  Analysis of Exp 5 failures:
#     - 55/150 gold pages not in reranked top-20
#     - Of those 55: 53 had the right DOC in top-20 — wrong PAGE within it
#     - Only 2 were true document routing failures
#     → The problem is within-document page ranking, not document routing.
#     → DocRec@20 = 0.97 for vanilla BGE-M3 — doc routing is already solved.
#
#     Exps 1–6 all used cross-doc/temporal negatives, which train the model to
#     distinguish different company documents (trivially easy, already solved).
#     The result is that fine-tuning consistently regresses vs the baseline
#     because it is optimising for the wrong discrimination axis.
#
#     Fix: --cross-doc-negs 0  +  --use-bm25-hard-negs  +  --hard-negatives 5
#     Train to rank the gold page above hard same-document negatives — exactly
#     the task that is failing at eval time.
#
#   [QUERY-DOCUMENT ASYMMETRY]  BGE-M3 was trained with an asymmetric encoding
#     scheme: queries use a task instruction prefix; documents do not.  All
#     prior experiments encoded questions and pages with the same weights,
#     ignoring this asymmetry.
#
#     Without the prefix:
#       embed("What was Apple's revenue in FY2022?")
#       embed("Apple Inc. Annual Report ... revenue $394B ...")
#     are treated symmetrically — the model sees them as the same type of input.
#
#     With the BGE-M3 instruction:
#       embed("Represent this sentence for searching relevant passages: "
#             "What was Apple's revenue in FY2022?")
#     activates BGE-M3's asymmetric retrieval mode, separating the query
#     representation from the document representation in embedding space.
#
#     Fix: --query-prefix added to both training (sentence_0) and eval queries.
#
# Eval: full global search over all 84 documents / 12,013 pages (no oracle).
# Hierarchical pipeline: bi-encoder top-30 → cross-encoder reranker top-20
#   → chunk retrieval top-5 → Qwen generation.
#
# Negatives / row:  5 same-doc BM25  (no cross-doc, no temporal)
# Dataset columns:  7  (query + gold_page + 5 neg_pages)
# Encoder calls/step: 8 × 7 = 56  (same as exp5, ~30min training)
#
# FinanceBench data: NOT used in training.
# =============================================================================

SUBMIT_DIR=$SLURM_SUBMIT_DIR
VENV_PATH="$SUBMIT_DIR/venv"
SCRATCH_DIR=/Tmp/$(whoami)/${SLURM_JOB_ID}

# ── Paths ──────────────────────────────────────────────────────────────────
TRAIN_GOLD_JSONL="data/finqa_test_gold_pages.jsonl"   # 503 PDF-verified rows
PDF_DIR_FINQA="Final-PDF"                              # FinQA PDFs only
PDF_DIR_FB="pdfs"                                      # FinanceBench eval PDFs
MODEL_OUT="models/exp7_within_doc"
EVAL_OUT="outputs/exp7_within_doc"
LLM="Qwen/Qwen2.5-7B-Instruct"

# ── BGE-M3 query instruction ───────────────────────────────────────────────
# This prefix is applied to QUERIES only (not documents/pages).
# Must be identical at training time and eval time.
QUERY_PREFIX="Represent this sentence for searching relevant passages: "

# ── Training hyperparameters ───────────────────────────────────────────────
EPOCHS=10
BATCH_SIZE=8
GRAD_ACCUM=4            # effective batch = 32
LR=2e-5
LORA_R=16
LORA_ALPHA=32
HARD_NEGS=5             # same-doc BM25 hard negs (up from 3 — no cross-doc slots)
CROSS_DOC_NEGS=0        # OFF — cross-doc trains doc routing (already solved)
LOSS_SCALE=20.0         # lowered from 50; mean gap ~0.12 → softmax(20×0.12)≈0.91

# ── Inference hyperparameters ──────────────────────────────────────────────
PAGE_K=30
RERANK_K=20
CHUNK_K=5
CHUNK_TOKENS=1024
OVERLAP_TOKENS=128
INDEX_BATCH=64

echo "========================================================="
echo "Experiment     : 7 — Within-doc BM25 negs + BGE-M3 query prefix"
echo "Job ID         : ${SLURM_JOB_ID:-local}"
echo "Node           : ${SLURMD_NODENAME:-$(hostname)}"
echo "GPU            : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Start time     : $(date)"
echo "========================================================="
echo "Training data  : ${TRAIN_GOLD_JSONL}  (503 PDF-verified rows)"
echo "Negatives      : ${HARD_NEGS} same-doc BM25 (no cross-doc, no temporal)"
echo "Query prefix   : '${QUERY_PREFIX}'"
echo "LoRA           : r=${LORA_R} (q/k/v)"
echo "Loss scale     : ${LOSS_SCALE}"
echo "Eval mode      : global search (all 84 docs / 12,013 pages)"
echo "FB data        : NOT used in training"
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
    echo "[WARNING] HF_TOKEN not set — Qwen may fail in Stage 2."
fi

N_TRAIN=$(wc -l < "${TRAIN_GOLD_JSONL}")
echo "[INFO] Training on ${N_TRAIN} rows from ${TRAIN_GOLD_JSONL}"

# ── Stage 1: Training ───────────────────────────────────────────────────────
echo ""
echo ">>> STAGE 1: LoRA fine-tuning"
echo "    JSONL       : ${TRAIN_GOLD_JSONL}"
echo "    PDFs        : ${PDF_DIR_FINQA}"
echo "    Output      : ${MODEL_OUT}"
echo "    Task        : rank gold page above ${HARD_NEGS} same-doc BM25 hard negs"
echo "    Query prefix: '${QUERY_PREFIX}'"
echo "========================================================="

python src/training/train_finqa_page_scorer.py \
    --jsonl          "${TRAIN_GOLD_JSONL}" \
    --pdf-dir        "${PDF_DIR_FINQA}" \
    --output         "${MODEL_OUT}" \
    --epochs         "${EPOCHS}" \
    --batch-size     "${BATCH_SIZE}" \
    --grad-accum     "${GRAD_ACCUM}" \
    --lr             "${LR}" \
    --lora-r         "${LORA_R}" \
    --lora-alpha     "${LORA_ALPHA}" \
    --hard-negatives "${HARD_NEGS}" \
    --cross-doc-negs "${CROSS_DOC_NEGS}" \
    --loss-scale     "${LOSS_SCALE}" \
    --use-bm25-hard-negs \
    --query-prefix   "${QUERY_PREFIX}"

if [ $? -ne 0 ]; then
    echo "[ERROR] Training failed — aborting."; exit 1
fi

echo ""
echo ">>> Training complete. Syncing adapter back..."
rsync -a "${SCRATCH_DIR}/${MODEL_OUT}/" "${SUBMIT_DIR}/${MODEL_OUT}/"
echo ">>> Adapter persisted to ${SUBMIT_DIR}/${MODEL_OUT}"

# ── Stage 2: FinanceBench evaluation (global search) ────────────────────────
echo ""
echo ">>> STAGE 2: FinanceBench 150 evaluation  [global search, all 84 docs]"
echo "    Model       : ${MODEL_OUT}"
echo "    PDFs        : ${PDF_DIR_FB}"
echo "    Output      : ${EVAL_OUT}"
echo "    Query prefix: '${QUERY_PREFIX}'"
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
    --llm               "${LLM}" \
    --query-prefix      "${QUERY_PREFIX}"

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
