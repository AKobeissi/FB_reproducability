#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=fusion_rag_ner
#SBATCH --output=logs/fusion_rag_%j.log
#SBATCH --error=logs/fusion_rag_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:ls40:1
#SBATCH --mem=60G
#SBATCH --time=48:00:00

# ─── Modality-Fusion Hierarchical RAG  (NER doc-filter, no oracle labels) ─────
#
# Runs two modality-fusion variants with NER-based document identification:
#   hier_marker_fusion — pdfplumber/marker table+text extraction + BM25/BGE-M3 RRF
#   hier_vlm_fusion    — Qwen2-VL-7B page extraction + BM25/BGE-M3 RRF
#
# Document filtering is performed via entity recognition (company name, year,
# doc type extracted from the query).  No oracle doc_name label is used.
# Ablation (oracle mode):  add --oracle-doc-filter
# Global search (no filter): add --no-ner-filter
#
# NER top-k:  --ner-top-k 3   (default: consider top-3 predicted documents)
#
# VRAM budget (L40S 48GB):
#   Phase 1: Qwen2-VL-7B in bfloat16 (~14GB) — freed before Phase 2
#   Phase 2: Fine-tuned BGE-M3 (~2GB)
#
# Metric output: DocRec@k / PageRec@k for k = 1, 3, 5, 10, 20
#
# Submit (NER filter, default):
#   sbatch scripts/run_modality_fusion_rag.sh
#
# Marker only (fast, no VLM):
#   sbatch scripts/run_modality_fusion_rag.sh --variants hier_marker_fusion
#
# Oracle ablation (cheating, for comparison):
#   sbatch scripts/run_modality_fusion_rag.sh --oracle-doc-filter
#
# Skip extraction (use cached content):
#   sbatch scripts/run_modality_fusion_rag.sh --skip-extract
#
# ──────────────────────────────────────────────────────────────────────────────

SUBMIT_DIR=$SLURM_SUBMIT_DIR
VENV_PATH="$SUBMIT_DIR/venv"
SCRATCH_DIR=/Tmp/$(whoami)/${SLURM_JOB_ID}

# HuggingFace cache
export HF_HOME=/data/rech/kobeissa/hf
export HF_HUB_CACHE=/data/rech/kobeissa/hf/hub
export HF_DATASETS_CACHE=/data/rech/kobeissa/hf/datasets
export PYTORCH_ALLOC_CONF=expandable_segments:True

EXTRA_ARGS="${@:-}"

echo "================================================="
echo "Job:         $SLURM_JOB_ID"
echo "Node:        $SLURMD_NODENAME"
echo "GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "VRAM:        $(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)"
echo "Start:       $(date)"
echo "Submit dir:  $SUBMIT_DIR"
echo "Scratch dir: $SCRATCH_DIR"
echo "Extra args:  ${EXTRA_ARGS:-none}"
echo "================================================="

mkdir -p "$SCRATCH_DIR"

# Save VLM cache back to persistent storage on SIGTERM (job time-limit kill)
_save_cache_on_cancel() {
  echo "SIGTERM received — saving VLM cache back to $SUBMIT_DIR before exit..."
  rsync -a --quiet \
    "$SCRATCH_DIR/modality_fusion_rag/cache/" \
    "$SUBMIT_DIR/modality_fusion_rag/cache/"
  echo "Cache saved."
}
trap '_save_cache_on_cancel' SIGTERM

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

echo "Activating venv: $VENV_PATH"
source "$VENV_PATH/bin/activate"

echo "Python: $(which python3)"
echo "Torch CUDA: $(python3 -c 'import torch; print(torch.cuda.is_available(), torch.version.cuda)')"
echo "GPU memory: $(python3 -c 'import torch; print(torch.cuda.get_device_properties(0).total_memory // 1e9, "GB")' 2>/dev/null)"

# ── Install optional dependencies if not already present ─────────────────────
echo "Checking optional dependencies..."
python3 -c "import marker" 2>/dev/null || {
  echo "Installing marker-pdf..."
  pip install marker-pdf --quiet
}
python3 -c "import pdfplumber" 2>/dev/null || {
  echo "Installing pdfplumber..."
  pip install pdfplumber --quiet
}
python3 -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null || {
  echo "Installing spaCy + model for NER doc filter..."
  pip install spacy --quiet
  python3 -m spacy download en_core_web_sm --quiet 2>/dev/null || true
  python3 -m spacy download en_core_web_lg --quiet 2>/dev/null || true
}

# ── Run the experiment ────────────────────────────────────────────────────────
# Default: NER-based doc filtering (no oracle labels).
# Add --oracle-doc-filter to compare with the oracle baseline.
python3 modality_fusion_rag.py \
  --data-path     data/financebench_open_source.jsonl \
  --doc-info-path data/financebench_document_information.jsonl \
  --pdf-dir       pdfs \
  --ft-model      models/fin_adapted_biencoder_bge_m3 \
  --results-dir   modality_fusion_rag/results \
  --vs-dir        modality_fusion_rag/vector_store \
  --cache-dir     modality_fusion_rag/cache \
  --k        5 \
  --k-text   100 \
  --k-table  100 \
  --rrf-k    60 \
  --ner-top-k 3 \
  $EXTRA_ARGS

STATUS=$?

# ── Copy results back ─────────────────────────────────────────────────────────
echo "Copying results back to $SUBMIT_DIR ..."
rsync -a --quiet \
  "$SCRATCH_DIR/modality_fusion_rag/" \
  "$SUBMIT_DIR/modality_fusion_rag/"

rm -rf "$SCRATCH_DIR"

echo "================================================="
echo "Finished: $(date)"
echo "Exit status: $STATUS"
echo ""
echo "Key results:"
echo "  modality_fusion_rag/results/metrics/all_variants_metrics.json"
echo "  modality_fusion_rag/results/metrics/fusion_table.csv"
echo "  modality_fusion_rag/results/metrics/by_question_type.csv"
echo "  modality_fusion_rag/results/plots/"
echo "================================================="

exit $STATUS
