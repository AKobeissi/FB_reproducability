#!/bin/bash
#SBATCH --partition=rali
#SBATCH --job-name=hier_rag
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out

# ──────────────────────────────────────────────────────────────────────────────
# Hierarchical RAG: Learned Page Scorer + Chunk Re-Retrieval
#
# Runs both hierarchical algorithms on FinanceBench 150:
#   1. hier_ft_page_chunk        — fine-tuned page index → chunk re-retrieval
#   2. hier_bm25_ft_rerank_chunk — BM25 recall + neural rerank → chunk retrieval
#
# GPU memory plan (RTX 3090 / L40S both supported):
#   BGE-M3 inference: ~2 GB
#   No Qwen / cross-encoder loaded unless --generate is passed.
#
# To add answer generation (needs more VRAM / time):
#   sbatch run_hierarchical_rag.sh --generate
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(dirname "$(realpath "$0")")/..}"
cd "$PROJECT_ROOT" || exit 1

echo "=== Hierarchical RAG ==="
echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $(hostname)"
echo "GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Start  : $(date)"
echo ""

# ── Activate environment ──────────────────────────────────────────────────────
# Adjust the path below to match your conda/venv setup.
if [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
elif [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
elif command -v conda &>/dev/null; then
    # shellcheck disable=SC1090
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate base 2>/dev/null || true
fi

# ── Run the experiment ────────────────────────────────────────────────────────
python hierarchical_rag/hierarchical_rag.py \
    --data-path     data/financebench_open_source.jsonl \
    --doc-info-path data/financebench_document_information.jsonl \
    --pdf-dir       pdfs \
    --finqa-path    data/finqa_test_gold_pages.jsonl \
    --finqa-pdf-dir Final-PDF \
    --ft-model      models/fin_adapted_biencoder_bge_m3 \
    --results-dir   hierarchical_rag/results \
    --vs-dir        hierarchical_rag/vector_store \
    --M  20 \
    --N  50 \
    --k  5 \
    "$@"

echo ""
echo "=== Done: $(date) ==="
