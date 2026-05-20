#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=fb_meta_filter_rr
#SBATCH --output=logs/meta_filter_rr_%j.log
#SBATCH --error=logs/meta_filter_rr_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:ls40:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00

# =============================================================================
# Pipeline: BGE-M3 dense → doc-name metadata filter → FT cross-encoder reranker
#
# Motivation: the metadata filter runs after dense retrieval and BEFORE the
# fine-tuned reranker.  It discards candidates from the wrong company/filing,
# giving the reranker a clean, company-scoped pool to work with.
#
# Requires (must exist before running this job):
#   baselines/results/predictions/dense_bge_m3_retrieval.json
#   checkpoints/ft_cross_encoder/          ← from scripts/run_ft_reranker.sh
#
# Stage 1 — Metadata filter + FT reranker  (run_metadata_filter_reranker.py)
#   Output: baselines/results/predictions/
#             dense_bge_m3_meta_filter_ft_reranker_retrieval.json
#           baselines/results/metrics/
#             dense_bge_m3_meta_filter_ft_reranker_metrics.json
#
# Stage 2 — LLM generation sweep  (llm_scaling_study.py)
#   Output: outputs/llm_scaling_meta_filter_rr/
#             summary.json / summary.csv
#             predictions/Qwen2.5-{3B,7B,14B}.json
#             oracle_predictions/
#             plots/
#
# Stage 3 — HTML report  (analyze_top_baseline.py)
#   Output: outputs/meta_filter_rr_analysis/report_Qwen2.5-{3B,7B,14B}.html
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
echo "Pipeline   : BGE-M3 → doc-name filter → FT ReRanker → Qwen2.5 sweep"
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
    --exclude 'vector_stores' \
    --exclude 'pdfs' \
    --exclude 'Final-PDF' \
    --exclude 'PDF-Opus*' \
    "$SUBMIT_DIR/" "$SCRATCH_DIR/"

# Copy required inputs
mkdir -p "$SCRATCH_DIR/baselines/results/predictions"
mkdir -p "$SCRATCH_DIR/baselines/results/metrics"

cp "$SUBMIT_DIR/baselines/results/predictions/dense_bge_m3_retrieval.json" \
   "$SCRATCH_DIR/baselines/results/predictions/"

# Copy existing baseline metrics for comparison printout (optional, no-op if missing)
for f in dense_bge_m3 bge_reranker dense_bge_m3_ft_reranker multi_hyde_ft_reranker; do
    src="$SUBMIT_DIR/baselines/results/metrics/${f}_metrics.json"
    [ -f "$src" ] && cp "$src" "$SCRATCH_DIR/baselines/results/metrics/"
done

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

export PYTHONPATH="${SCRATCH_DIR}:${SCRATCH_DIR}/src:${PYTHONPATH:-}"

# =============================================================================
# Stage 1 — Metadata filter + FT reranker
# =============================================================================
echo ""
echo ">>> Stage 1: doc-name metadata filter + FT cross-encoder reranker"
echo "========================================================="

RETRIEVAL_OUT="baselines/results/predictions/dense_bge_m3_meta_filter_ft_reranker_retrieval.json"
METRICS_OUT="baselines/results/metrics/dense_bge_m3_meta_filter_ft_reranker_metrics.json"

python3 baselines/run_metadata_filter_reranker.py \
    --input     "baselines/results/predictions/dense_bge_m3_retrieval.json" \
    --output    "$RETRIEVAL_OUT" \
    --metrics   "$METRICS_OUT" \
    --checkpoint "checkpoints/ft_cross_encoder" \
    --min-filter-k 5

STAGE1_EXIT=$?
echo "Stage 1 finished (exit code: ${STAGE1_EXIT}) at $(date)"

if [ $STAGE1_EXIT -ne 0 ]; then
    echo "ERROR: Stage 1 failed — aborting."
    exit $STAGE1_EXIT
fi

# Sync Stage 1 outputs back immediately (in case Stage 2 fails)
rsync -a "${SCRATCH_DIR}/baselines/results/predictions/" \
         "${SUBMIT_DIR}/baselines/results/predictions/"
rsync -a "${SCRATCH_DIR}/baselines/results/metrics/" \
         "${SUBMIT_DIR}/baselines/results/metrics/"

# =============================================================================
# Stage 2 — LLM scaling study (Qwen2.5-3B/7B/14B)
# =============================================================================
echo ""
echo ">>> Stage 2: LLM scaling study — Qwen2.5-3B/7B/14B"
echo "========================================================="

OUTPUT_DIR="outputs/llm_scaling_meta_filter_rr"

python3 baselines/llm_scaling_study.py \
    --retrieval-file "$RETRIEVAL_OUT" \
    --output-dir     "$OUTPUT_DIR"

STAGE2_EXIT=$?
echo "Stage 2 finished (exit code: ${STAGE2_EXIT}) at $(date)"

# Sync all outputs back
FINAL_OUT="${SUBMIT_DIR}/${OUTPUT_DIR}"
mkdir -p "$FINAL_OUT"
rsync -a "${SCRATCH_DIR}/${OUTPUT_DIR}/" "${FINAL_OUT}/"

rm -rf "$SCRATCH_DIR"

# =============================================================================
# Stage 3 — HTML reports (run locally, not on scratch)
# =============================================================================
if [ ${STAGE2_EXIT} -eq 0 ]; then
    echo ""
    echo ">>> Stage 3: Generating HTML reports"
    echo "========================================================="
    source "$VENV_PATH/activate" 2>/dev/null || true
    mkdir -p "${SUBMIT_DIR}/outputs/meta_filter_rr_analysis"
    cd "$SUBMIT_DIR"
    for MODEL in Qwen2.5-3B Qwen2.5-7B Qwen2.5-14B; do
        python3 analyze_top_baseline.py \
            --pred   "${OUTPUT_DIR}/predictions/${MODEL}.json" \
            --out    "outputs/meta_filter_rr_analysis/report_${MODEL}.html" \
            --model-label "${MODEL}-Instruct (4-bit) | Retrieval: BGE-M3 + MetaFilter + FT-ReRanker"
        echo "  → outputs/meta_filter_rr_analysis/report_${MODEL}.html"
    done
fi

FINAL_EXIT=$(( STAGE2_EXIT != 0 ? STAGE2_EXIT : 0 ))

echo "========================================================="
echo "Done!  End time: $(date)"
echo ""
echo "Results:"
echo "  baselines/results/predictions/dense_bge_m3_meta_filter_ft_reranker_retrieval.json"
echo "  baselines/results/metrics/dense_bge_m3_meta_filter_ft_reranker_metrics.json"
echo "  ${OUTPUT_DIR}/summary.json"
echo "  ${OUTPUT_DIR}/predictions/Qwen2.5-{3B,7B,14B}.json"
echo "  outputs/meta_filter_rr_analysis/report_Qwen2.5-{3B,7B,14B}.html"
echo "========================================================="

exit ${FINAL_EXIT}
