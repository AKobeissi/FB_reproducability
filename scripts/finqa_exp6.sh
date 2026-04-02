#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=exp6_temporal
#SBATCH --output=logs/exp6_temporal_%j.log
#SBATCH --error=logs/exp6_temporal_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00

# =============================================================================
# Experiment 6 — Temporal hard negatives + full FinQA training split
#
# Root causes addressed (analysis of Exps 1–5):
#
#   [NEGATIVE TYPE]  80 % of doc-level retrieval errors in Exp 5 were
#     same-company/different-year confusions (e.g. 3M_2022_10K vs 3M_2018_10K).
#     Exp 5 used 2 RANDOM cross-doc negatives from UNRELATED companies — these
#     are trivially easy and never expose the model to this failure pattern.
#     Fix: --temporal-negs 2  samples pages from the same company's OTHER
#     filings, BM25-mined against the query.
#
#   [DATA SIZE]  All prior experiments trained on 503 examples (FinQA test
#     split).  finqa/train.json contains 6,251 examples; of these ~1,700+ have
#     matching PDFs in Final-PDF/.  This 3–4× data increase provides better
#     generalisation across company styles and question types.
#     Fix: Stage 0 below converts finqa/train.json → JSONL.
#
#   [ARCHITECTURE]  Exp 3 (extended LoRA) reached the lowest training loss.
#     r=16 was conservative.  Extended LoRA at r=32 adds ~4× capacity over
#     the baseline q/k/v-only r=16 setup.
#     Fix: extended LoRA (q/k/v + attn-out + ffn-up), r=32.
#
#   [LOSS]  scale=50 in Exp 5 made softmax(50×0.12)≈0.9975 for the mean
#     score gap, killing gradients for moderate pairs.  scale=25 keeps the
#     gradient alive across the full difficulty range.
#     Hierarchical loss (page + chunk) gave the best NumericMatch in Exp 1.
#     Fix: --loss-scale 25  + hierarchical MNR.
#
#   [TRAINING STABILITY]  Exp 5 showed oscillating loss at epochs 7–10.
#     Linear LR schedule + late-epoch spikes from warmup end.
#     Fix: cosine schedule, 5 % warmup, 3 epochs + val split early stopping.
#
# Negatives / row:  3 same-doc BM25  +  2 temporal BM25  =  5 total
# Dataset columns:  2 + 5 + 1 + 5 = 13  (hierarchical)
# Encoder calls/step: 8 × 13 = 104  (fits L40S 46 GB with grad_checkpoint)
#
# FinanceBench data: NOT used anywhere in this script.
# =============================================================================

SUBMIT_DIR=$SLURM_SUBMIT_DIR
VENV_PATH="$SUBMIT_DIR/venv"
SCRATCH_DIR=/Tmp/$(whoami)/${SLURM_JOB_ID}

# ── Paths ──────────────────────────────────────────────────────────────────
FINQA_TRAIN_JSON="finqa/train.json"
TRAIN_GOLD_JSONL="data/finqa_train_gold_pages.jsonl"
TEST_GOLD_JSONL="data/finqa_test_gold_pages.jsonl"
PDF_DIR_FINQA="Final-PDF"       # ← FinQA PDFs only; FinanceBench pdfs/ NOT touched
PDF_DIR_FB="pdfs"               # ← eval only
MODEL_OUT="models/exp6_temporal"
EVAL_OUT="outputs/exp6_temporal"
LLM="Qwen/Qwen2.5-7B-Instruct"

# ── Training hyperparameters ───────────────────────────────────────────────
EPOCHS=3
BATCH_SIZE=8
GRAD_ACCUM=4          # effective batch = 32
LR=1e-5               # lower than exps 1-5 (2e-5) — more stable with extended LoRA
WARMUP_FRAC=0.05      # cosine schedule; shorter warmup = more training at low LR
LORA_R=32             # 2× rank vs baseline (r=16)
LORA_ALPHA=64
HARD_NEGS=3           # same-doc BM25
TEMPORAL_NEGS=2       # same-company different-year BM25  ← KEY FIX
CROSS_DOC_NEGS=0      # disabled — random cross-doc is too easy
LOSS_SCALE=25.0       # reduced from 50 — keeps gradients alive at gap≈0.12
HIER_ALPHA=0.85       # weight on page-level MNR term
CHUNK_TOKENS=512
VAL_FRAC=0.10

# ── Data prep hyperparameters ──────────────────────────────────────────────
MIN_OVERLAP=0.05      # token-overlap threshold for page sanity check
                      # Financial table pages score 0.05-0.15 even on correct page
SEARCH_WINDOW=5       # ±pages to search around html_page-1

# ── Inference hyperparameters (unchanged from exps 1–5) ────────────────────
PAGE_K=30
RERANK_K=20
CHUNK_K=5
CHUNK_TOKENS_EVAL=1024
OVERLAP_TOKENS=128
INDEX_BATCH=64

echo "========================================================="
echo "Experiment     : 6 — Temporal hard negs + full FinQA train"
echo "Job ID         : ${SLURM_JOB_ID:-local}"
echo "Node           : ${SLURMD_NODENAME:-$(hostname)}"
echo "GPU            : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "GPU memory     : $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)"
echo "Start time     : $(date)"
echo "========================================================="
echo "Negatives      : ${HARD_NEGS} same-doc BM25 + ${TEMPORAL_NEGS} temporal BM25 = $((HARD_NEGS + TEMPORAL_NEGS)) total"
echo "Loss           : Hierarchical MNR (alpha=${HIER_ALPHA}), scale=${LOSS_SCALE}"
echo "LoRA           : extended (q/k/v+attn-out+ffn-up), r=${LORA_R}"
echo "batch_size     : ${BATCH_SIZE} × grad_accum ${GRAD_ACCUM} = effective $(( BATCH_SIZE * GRAD_ACCUM ))"
echo "encoder calls  : $((BATCH_SIZE * (2 + (HARD_NEGS + TEMPORAL_NEGS) * 2 + 1))) texts/step (hierarchical)"
echo "FB data        : NOT used in training (only in eval)"
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

# ── Stage 0: Convert FinQA train.json → JSONL ──────────────────────────────
echo ""
echo ">>> STAGE 0: Prepare FinQA training data"
echo "    Input : ${FINQA_TRAIN_JSON}"
echo "    Output: ${TRAIN_GOLD_JSONL}"
echo "    PDFs  : ${PDF_DIR_FINQA}  (FinQA only — no FinanceBench data)"
echo "========================================================="

if [ -f "${TRAIN_GOLD_JSONL}" ]; then
    N_EXISTING=$(wc -l < "${TRAIN_GOLD_JSONL}")
    echo "[INFO] ${TRAIN_GOLD_JSONL} already exists (${N_EXISTING} rows) — skipping conversion."
    echo "       Delete the file to force re-generation."
else
    python scripts/prepare_finqa_train_gold_pages.py \
        --finqa-train   "${FINQA_TRAIN_JSON}" \
        --pdf-dir       "${PDF_DIR_FINQA}" \
        --output        "${TRAIN_GOLD_JSONL}" \
        --min-overlap   "${MIN_OVERLAP}" \
        --search-window "${SEARCH_WINDOW}"

    if [ $? -ne 0 ]; then
        echo "[ERROR] Stage 0 failed."; exit 1
    fi
fi

N_TRAIN=$(wc -l < "${TRAIN_GOLD_JSONL}")
N_TEST=$(wc -l < "${TEST_GOLD_JSONL}")
echo "[INFO] Training rows: ${N_TRAIN} (train, ~6k expected) + ${N_TEST} (test) = $(( N_TRAIN + N_TEST )) total (before dedup)"
echo "[INFO] Tier A (PDF-verified) and Tier B (JSON text) rows both included."
echo "[INFO] Tier B rows use pre_text+post_text as gold evidence; cross-doc random negs during training."

# ── Stage 1: Training ───────────────────────────────────────────────────────
echo ""
echo ">>> STAGE 1: LoRA fine-tuning (temporal negs + hierarchical loss)"
echo "    JSONL  : ${TRAIN_GOLD_JSONL} + ${TEST_GOLD_JSONL}"
echo "    PDFs   : ${PDF_DIR_FINQA}"
echo "    Output : ${MODEL_OUT}"
echo "========================================================="

python src/training/train_exp6_scorer.py \
    --jsonl          "${TRAIN_GOLD_JSONL}" "${TEST_GOLD_JSONL}" \
    --pdf-dir        "${PDF_DIR_FINQA}" \
    --output         "${MODEL_OUT}" \
    --epochs         "${EPOCHS}" \
    --batch-size     "${BATCH_SIZE}" \
    --grad-accum     "${GRAD_ACCUM}" \
    --lr             "${LR}" \
    --warmup-frac    "${WARMUP_FRAC}" \
    --lora-r         "${LORA_R}" \
    --lora-alpha     "${LORA_ALPHA}" \
    --hard-negatives "${HARD_NEGS}" \
    --temporal-negs  "${TEMPORAL_NEGS}" \
    --cross-doc-negs "${CROSS_DOC_NEGS}" \
    --loss-scale     "${LOSS_SCALE}" \
    --hier-alpha     "${HIER_ALPHA}" \
    --chunk-tokens   "${CHUNK_TOKENS}" \
    --val-frac       "${VAL_FRAC}"

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
echo "    Model  : ${MODEL_OUT}"
echo "    PDFs   : ${PDF_DIR_FB}  (FinanceBench evaluation PDFs)"
echo "    Output : ${EVAL_OUT}"
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
