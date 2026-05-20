#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=clip_visual_rag
#SBATCH --output=logs/colpali_byaldi_%j.log
#SBATCH --error=logs/colpali_byaldi_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:ls40:1
#SBATCH --mem=80G
#SBATCH --time=96:00:00

# =============================================================================
# CLIP ViT-L/14 + Qwen2-VL-7B  Visual RAG Baseline
# =============================================================================
#
# Retriever : CLIP ViT-L/14 embeds rendered PDF pages → FAISS cosine search
# Generator : Qwen2-VL-7B-Instruct answers from top-2 retrieved page images
#
# All packages already in venv (no pip installs needed):
#   transformers, torch, faiss-cpu, PyMuPDF, Pillow, qwen_vl_utils
#
# Memory budget (L40S 46 GB)
# --------------------------
#   CLIP ViT-L/14  fp16   ≈  1.2 GB  →  freed after indexing + retrieval
#   Qwen2-VL-7B    bf16   ≈ 15   GB  →  freed after generation
#   FAISS index    RAM    ≈  0.1 GB  (15k pages × 768d × 4B)
#
# Outputs  (baselines/results/)
# ------------------------------
#   predictions/colpali_byaldi_retrieval.json
#   predictions/colpali_byaldi_final.json
#   metrics/colpali_byaldi_metrics.json
#   metrics/all_variants_metrics.json     ← merged with text baselines
#   plots/colpali_byaldi/*.{pdf,png}
# =============================================================================

SUBMIT_DIR=$SLURM_SUBMIT_DIR
VENV_PATH="$SUBMIT_DIR/venv"
SCRATCH_DIR=/Tmp/$(whoami)/${SLURM_JOB_ID}

export HF_HOME=/data/rech/kobeissa/hf
export HF_HUB_CACHE=/data/rech/kobeissa/hf/hub
export HF_DATASETS_CACHE=/data/rech/kobeissa/hf/datasets
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

echo "========================================================="
echo "Job ID     : ${SLURM_JOB_ID:-local}"
echo "Node       : ${SLURMD_NODENAME:-$(hostname)}"
echo "GPU        : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
echo "Start time : $(date)"
echo "========================================================="

mkdir -p "$SCRATCH_DIR"
mkdir -p "$SUBMIT_DIR/logs"

# ---------------------------------------------------------------------------
# Copy repository to scratch
# ---------------------------------------------------------------------------
echo "Copying repository to scratch …"
rsync -a --quiet \
    --exclude 'venv' \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.log' \
    --exclude 'outputs' \
    --exclude 'vector_stores' \
    "$SUBMIT_DIR/" "$SCRATCH_DIR/"

mkdir -p "$SCRATCH_DIR/baselines/results/predictions"
mkdir -p "$SCRATCH_DIR/baselines/results/metrics"
mkdir -p "$SCRATCH_DIR/baselines/results/plots"

# Copy existing baseline metrics for comparison plots
for variant in dense_bge_m3 bge_reranker multi_hyde multi_hyde_reranker multi_hyde_ft_reranker; do
    src="$SUBMIT_DIR/baselines/results/metrics/${variant}_metrics.json"
    [ -f "$src" ] && cp "$src" "$SCRATCH_DIR/baselines/results/metrics/"
done
[ -f "$SUBMIT_DIR/baselines/results/metrics/all_variants_metrics.json" ] && \
    cp "$SUBMIT_DIR/baselines/results/metrics/all_variants_metrics.json" \
       "$SCRATCH_DIR/baselines/results/metrics/"

# Copy FinanceBench PDFs
echo "Copying PDFs to scratch …"
rsync -a --quiet "$SUBMIT_DIR/pdfs/" "$SCRATCH_DIR/pdfs/"

cd "$SCRATCH_DIR"

# ---------------------------------------------------------------------------
# Activate venv
# ---------------------------------------------------------------------------
echo "Activating venv: $VENV_PATH"
source "$VENV_PATH/bin/activate"

# ---------------------------------------------------------------------------
# Sanity check — no pip installs, everything must already be in the venv
# ---------------------------------------------------------------------------
python3 - <<'PYEOF'
import sys
checks = {
    "transformers": "transformers",
    "torch":        "torch",
    "faiss":        "faiss-cpu",
    "fitz":         "PyMuPDF",
    "PIL":          "Pillow",
    "numpy":        "numpy",
}
failed = []
for mod, pkg in checks.items():
    try:
        __import__(mod)
        print(f"[OK] {pkg}")
    except ImportError:
        print(f"[MISSING] {pkg} — not in venv")
        failed.append(pkg)
if failed:
    print(f"ERROR: Missing packages: {failed}")
    sys.exit(1)

# Qwen2-VL availability check
from transformers import Qwen2VLForConditionalGeneration  # noqa
print("[OK] Qwen2-VL in transformers")

# qwen_vl_utils is optional (PIL fallback exists in generator)
try:
    import qwen_vl_utils  # noqa
    print("[OK] qwen_vl_utils")
except ImportError:
    print("[INFO] qwen_vl_utils not found — PIL fallback will be used")

print("[OK] All required packages present.")
PYEOF

if [ $? -ne 0 ]; then
    echo "ERROR: Dependency check failed. Aborting."
    exit 1
fi

# ---------------------------------------------------------------------------
# NLTK data
# ---------------------------------------------------------------------------
python3 -c "
try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception:
    pass
"

export PYTHONPATH="${SCRATCH_DIR}:${SCRATCH_DIR}/baselines:${SCRATCH_DIR}/src:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# Run the Visual RAG pipeline
# ---------------------------------------------------------------------------
echo ""
echo ">>> Starting CLIP Visual RAG pipeline"
echo "========================================================="

python3 baselines/run_colpali_byaldi_baseline.py \
    --data-dir       data \
    --pdf-dir        pdfs \
    --output-dir     baselines/results \
    --index-root     .clip_index \
    --index-name     financebench_clip_vl \
    --colpali-model  openai/clip-vit-large-patch14 \
    --qwen-model     Qwen/Qwen2-VL-7B-Instruct \
    --top-k          20 \
    --top-k-images   2 \
    --max-new-tokens 512

PIPELINE_EXIT=$?
echo "Pipeline finished (exit code: ${PIPELINE_EXIT}) at $(date)"

# ---------------------------------------------------------------------------
# Sync outputs back
# ---------------------------------------------------------------------------
echo "Syncing outputs back …"

rsync -a "${SCRATCH_DIR}/baselines/results/predictions/" \
         "${SUBMIT_DIR}/baselines/results/predictions/"

rsync -a "${SCRATCH_DIR}/baselines/results/metrics/" \
         "${SUBMIT_DIR}/baselines/results/metrics/"

rsync -a "${SCRATCH_DIR}/baselines/results/plots/" \
         "${SUBMIT_DIR}/baselines/results/plots/"

# Sync the CLIP index back so future runs skip re-indexing
rsync -a "${SCRATCH_DIR}/.clip_index/" \
         "${SUBMIT_DIR}/.clip_index/" 2>/dev/null || true

rm -rf "${SCRATCH_DIR}"

echo "========================================================="
echo "Done!  End time: $(date)"
echo ""
echo "Outputs:"
echo "  baselines/results/predictions/colpali_byaldi_retrieval.json"
echo "  baselines/results/predictions/colpali_byaldi_final.json"
echo "  baselines/results/metrics/colpali_byaldi_metrics.json"
echo "  baselines/results/plots/colpali_byaldi/"
echo "========================================================="

exit ${PIPELINE_EXIT}
