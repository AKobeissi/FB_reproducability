#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=finqa_variants
#SBATCH --output=logs/finqa_variants_%j.log
#SBATCH --error=logs/finqa_variants_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=12:00:00

# Evaluates two ablation variants of the FinQA-trained page scorer on FB 150.
# Assumes models/finqa_page_scorer_lora and checkpoints/ft_cross_encoder already exist.
#
# Variant A — base_chunk:
#   FT bi-encoder (page retrieval) → no reranker → base BGE-M3 (chunk retrieval)
#
# Variant B — ft_reranker:
#   FT bi-encoder (page retrieval) → FT cross-encoder reranker → no chunk retrieval

SUBMIT_DIR=$SLURM_SUBMIT_DIR
VENV_PATH="$SUBMIT_DIR/venv"
SCRATCH_DIR=/Tmp/$(whoami)/${SLURM_JOB_ID}

MODEL_PATH="models/finqa_page_scorer_lora"
FT_RERANKER="checkpoints/ft_cross_encoder"
PDF_DIR="pdfs"
EVAL_OUT="outputs/finqa_page_scorer"
LLM="Qwen/Qwen2.5-7B-Instruct"

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

mkdir -p "$SCRATCH_DIR" "$SUBMIT_DIR/logs"

echo "Copying repository to scratch..."
rsync -a --quiet \
    --exclude 'venv' \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude 'outputs' \
    --exclude 'vector_stores' \
    --exclude '*.log' \
    "$SUBMIT_DIR/" "$SCRATCH_DIR/"

# Copy pre-built page index cache if it exists (saves ~30 min re-encoding)
if [ -d "$SUBMIT_DIR/${EVAL_OUT}/index_cache" ]; then
    echo "Copying existing page index cache..."
    mkdir -p "$SCRATCH_DIR/${EVAL_OUT}"
    rsync -a "$SUBMIT_DIR/${EVAL_OUT}/index_cache/" "$SCRATCH_DIR/${EVAL_OUT}/index_cache/"
fi

cd "$SCRATCH_DIR"

echo "Activating venv: $VENV_PATH"
source "$VENV_PATH/bin/activate"

export HF_HOME=/data/rech/kobeissa/hf
export HF_HUB_CACHE=/data/rech/kobeissa/hf/hub
export HF_DATASETS_CACHE=/data/rech/kobeissa/hf/datasets
export PYTORCH_ALLOC_CONF=expandable_segments:True

if [ -z "$HF_TOKEN" ]; then
    echo "[WARNING] HF_TOKEN not set — gated models (Qwen) may fail."
fi

# ─── Variant A: base_chunk ────────────────────────────────────────────────────
echo ""
echo ">>> VARIANT A: base_chunk"
echo "    FT bi-encoder → no reranker → base BGE-M3 chunk retrieval"
echo "========================================================="

python scripts/run_finqa_page_scorer_fb.py \
    --model-path        "${MODEL_PATH}" \
    --pdf-dir           "${PDF_DIR}" \
    --output-dir        "${EVAL_OUT}" \
    --variant           base_chunk \
    --page-k            "${PAGE_K}" \
    --rerank-k          "${RERANK_K}" \
    --chunk-k           "${CHUNK_K}" \
    --chunk-tokens      "${CHUNK_TOKENS}" \
    --overlap-tokens    "${OVERLAP_TOKENS}" \
    --index-batch-size  "${INDEX_BATCH}" \
    --llm               "${LLM}"

if [ $? -ne 0 ]; then
    echo "[ERROR] Variant A failed — aborting."
    exit 1
fi

echo ">>> Variant A done. Syncing results..."
rsync -a "${SCRATCH_DIR}/${EVAL_OUT}/base_chunk/" "${SUBMIT_DIR}/${EVAL_OUT}/base_chunk/"
# Sync updated cache back too (in case it was rebuilt)
rsync -a "${SCRATCH_DIR}/${EVAL_OUT}/index_cache/" "${SUBMIT_DIR}/${EVAL_OUT}/index_cache/"

# ─── Variant B: ft_reranker ───────────────────────────────────────────────────
echo ""
echo ">>> VARIANT B: ft_reranker"
echo "    FT bi-encoder → FT cross-encoder reranker → no chunk retrieval"
echo "========================================================="

python scripts/run_finqa_page_scorer_fb.py \
    --model-path        "${MODEL_PATH}" \
    --pdf-dir           "${PDF_DIR}" \
    --output-dir        "${EVAL_OUT}" \
    --variant           ft_reranker \
    --ft-reranker-path  "${FT_RERANKER}" \
    --page-k            "${PAGE_K}" \
    --rerank-k          "${RERANK_K}" \
    --index-batch-size  "${INDEX_BATCH}" \
    --llm               "${LLM}"

if [ $? -ne 0 ]; then
    echo "[ERROR] Variant B failed."
    exit 1
fi

echo ">>> Variant B done. Syncing results..."
rsync -a "${SCRATCH_DIR}/${EVAL_OUT}/ft_reranker/" "${SUBMIT_DIR}/${EVAL_OUT}/ft_reranker/"

echo ""
echo "========================================================="
echo "Done!  End time : $(date)"
echo "Results:"
echo "  Variant A (base_chunk)  : ${SUBMIT_DIR}/${EVAL_OUT}/base_chunk/"
echo "  Variant B (ft_reranker) : ${SUBMIT_DIR}/${EVAL_OUT}/ft_reranker/"
echo "========================================================="
