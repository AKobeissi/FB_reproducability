#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=fb_meta_filter
#SBATCH --output=logs/meta_filter_%j.log
#SBATCH --error=logs/meta_filter_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:ls40:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00

# =============================================================================
# Metadata-Filtering Retrieval Experiment
#
# Tests how pre-filtering the corpus by metadata inferred from the query text
# (company name and/or year) affects retrieval.
#
# Five filter modes (each applied before retrieval):
#   company_only       — docs matching the extracted company name
#   year_exact         — docs from the exact year(s) in the query
#   year_window        — docs within [min_year-2, max_year+2]
#   company_year_exact — company AND exact year
#   company_year_window— company AND year ± 2 window
#
# Applied to two retrieval pipelines:
#   1. Base BGE-M3 dense
#   2. MultiHyDE + FT-ReRanker (best pipeline)
#
# All metadata extraction uses ONLY query text — no FinanceBench oracle fields.
#
# Outputs → outputs/metadata_filter/
#   summary.json / summary.csv
#   predictions/{variant}_retrieval.json  (20 files)
#   plots/recall_comparison.{pdf,png}
#   plots/recall_at_k.{pdf,png}
#   plots/filter_impact.{pdf,png}
# =============================================================================

SUBMIT_DIR=$SLURM_SUBMIT_DIR
VENV_PATH="$SUBMIT_DIR/venv"
SCRATCH_DIR=/Tmp/$(whoami)/${SLURM_JOB_ID}

export HF_HOME=/data/rech/kobeissa/hf
export HF_HUB_CACHE=/data/rech/kobeissa/hf/hub
export HF_DATASETS_CACHE=/data/rech/kobeissa/hf/datasets
export PYTORCH_ALLOC_CONF=expandable_segments:True

OUTPUT_DIR="outputs/metadata_filter"

echo "========================================================="
echo "Job ID     : ${SLURM_JOB_ID:-local}"
echo "Node       : ${SLURMD_NODENAME:-$(hostname)}"
echo "GPU        : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
echo "Output dir : ${OUTPUT_DIR}"
echo "Start time : $(date)"
echo "========================================================="

mkdir -p "$SCRATCH_DIR"
mkdir -p "$SUBMIT_DIR/logs"

echo "Copying repository to scratch…"
rsync -a --quiet \
    --exclude 'venv' \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.log' \
    --exclude 'outputs' \
    --exclude 'pdfs' \
    --exclude 'Final-PDF' \
    --exclude 'PDF-Opus*' \
    "$SUBMIT_DIR/" "$SCRATCH_DIR/"

# Copy vector store and bm25 cache (already built)
echo "Copying vector store…"
rsync -a --quiet "$SUBMIT_DIR/vector_stores/" "$SCRATCH_DIR/vector_stores/"

# Copy HyDE cache (pre-generated hypotheticals)
mkdir -p "$SCRATCH_DIR/baselines"
cp "$SUBMIT_DIR/baselines/hyde_cache.json" "$SCRATCH_DIR/baselines/hyde_cache.json"

# Copy FT reranker checkpoint (required — abort if missing)
if [ -d "$SUBMIT_DIR/checkpoints/ft_cross_encoder" ]; then
    echo "Copying FT reranker checkpoint…"
    mkdir -p "$SCRATCH_DIR/checkpoints"
    rsync -a --quiet \
        "$SUBMIT_DIR/checkpoints/ft_cross_encoder/" \
        "$SCRATCH_DIR/checkpoints/ft_cross_encoder/"
else
    echo "ERROR: FT reranker checkpoint not found at $SUBMIT_DIR/checkpoints/ft_cross_encoder"
    exit 1
fi

cd "$SCRATCH_DIR"

echo "Activating venv: $VENV_PATH"
source "$VENV_PATH/bin/activate"

python3 - <<'EOF'
try:
    import nltk
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
except Exception:
    pass
EOF

export PYTHONPATH="${SCRATCH_DIR}:${SCRATCH_DIR}/src:${SCRATCH_DIR}/baselines:${PYTHONPATH:-}"

echo ""
echo ">>> Running metadata-filter retrieval experiment"
echo "========================================================="

python3 baselines/run_metadata_filter_baselines.py \
    --data          "data/financebench_open_source.jsonl" \
    --doc-info      "data/financebench_document_information.jsonl" \
    --pdf-dir       "pdfs" \
    --vs-dir        "vector_stores/baselines" \
    --hyde-cache    "baselines/hyde_cache.json" \
    --ft-reranker   "checkpoints/ft_cross_encoder" \
    --output-dir    "${OUTPUT_DIR}" \
    --year-window   2

EXIT_CODE=$?
echo "Script finished (exit code: ${EXIT_CODE}) at $(date)"

# Sync results back
FINAL_OUT="${SUBMIT_DIR}/${OUTPUT_DIR}"
mkdir -p "$FINAL_OUT"

echo "Syncing outputs back to: $FINAL_OUT"
rsync -a "${SCRATCH_DIR}/${OUTPUT_DIR}/" "${FINAL_OUT}/"

rm -rf "$SCRATCH_DIR"

echo "========================================================="
echo "Done!  End time: $(date)"
echo ""
echo "Results:"
echo "  ${OUTPUT_DIR}/summary.json"
echo "  ${OUTPUT_DIR}/summary.csv"
echo "  ${OUTPUT_DIR}/plots/"
echo "========================================================="

exit ${EXIT_CODE}
