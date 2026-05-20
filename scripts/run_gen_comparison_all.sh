#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=fb_gen_compare_all
#SBATCH --output=logs/gen_compare_all_%j.log
#SBATCH --error=logs/gen_compare_all_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:ls40:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00

# =============================================================================
# Generative Comparison — Qwen2.5-7B-Instruct vs Llama-3.1-8B-Instruct
# Running for ALL 5 key retrieval configurations.
# =============================================================================

SUBMIT_DIR=$SLURM_SUBMIT_DIR
VENV_PATH="$SUBMIT_DIR/venv"
SCRATCH_DIR=/Tmp/$(whoami)/${SLURM_JOB_ID}

export HF_HOME=/data/rech/kobeissa/hf
export HF_HUB_CACHE=/data/rech/kobeissa/hf/hub
export HF_DATASETS_CACHE=/data/rech/kobeissa/hf/datasets
export PYTORCH_ALLOC_CONF=expandable_segments:True

declare -A EXPERIMENTS
EXPERIMENTS["dense_bge_m3"]="baselines/results/predictions/dense_bge_m3_retrieval.json"
EXPERIMENTS["multi_hyde_rr"]="baselines/results/predictions/multi_hyde_reranker_retrieval.json"
EXPERIMENTS["multi_hyde_ftrr"]="baselines/results/predictions/multi_hyde_ft_reranker_retrieval.json"
EXPERIMENTS["oracle_doc"]="baselines/results/predictions/oracle_doc_retrieval.json"
EXPERIMENTS["oracle_page"]="baselines/results/predictions/oracle_page_retrieval.json"

echo "========================================================="
echo "Job ID     : ${SLURM_JOB_ID:-local}"
echo "Node       : ${SLURMD_NODENAME:-$(hostname)}"
echo "GPU        : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
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
    --exclude 'vector_stores' \
    --exclude 'pdfs' \
    --exclude 'Final-PDF' \
    --exclude 'PDF-Opus*' \
    "$SUBMIT_DIR/" "$SCRATCH_DIR/"

# Copy all pre-computed retrieval files
echo "Copying retrieval predictions…"
mkdir -p "$SCRATCH_DIR/baselines/results/predictions"
for exp_name in "${!EXPERIMENTS[@]}"; do
    RETRIEVAL_FILE="${EXPERIMENTS[$exp_name]}"
    if [ -f "$SUBMIT_DIR/$RETRIEVAL_FILE" ]; then
        cp "$SUBMIT_DIR/$RETRIEVAL_FILE" "$SCRATCH_DIR/baselines/results/predictions/"
    else
        echo "WARNING: Retrieval file not found for $exp_name: $RETRIEVAL_FILE"
    fi
done

cd "$SCRATCH_DIR"

echo "Activating venv: $VENV_PATH"
source "$VENV_PATH/bin/activate"

export PYTHONPATH="${SCRATCH_DIR}:${SCRATCH_DIR}/src:${SCRATCH_DIR}/baselines:${PYTHONPATH:-}"

for exp_name in "dense_bge_m3" "multi_hyde_rr" "multi_hyde_ftrr" "oracle_doc" "oracle_page"; do
    RETRIEVAL_FILE="baselines/results/predictions/$(basename ${EXPERIMENTS[$exp_name]})"
    OUTPUT_DIR="outputs/gen_comparison_${exp_name}"
    
    if [ ! -f "$RETRIEVAL_FILE" ]; then
        echo ">>> Skipping $exp_name (file not found: $RETRIEVAL_FILE)"
        continue
    fi

    echo ""
    echo ">>> Running generative comparison for: $exp_name"
    echo ">>> Retrieval file: $RETRIEVAL_FILE"
    echo ">>> Output dir   : $OUTPUT_DIR"
    echo "========================================================="

    python3 baselines/run_gen_comparison.py \
        --retrieval-file "$RETRIEVAL_FILE" \
        --output-dir     "$OUTPUT_DIR"

    # Sync results back immediately after each experiment to be safe
    FINAL_OUT="${SUBMIT_DIR}/${OUTPUT_DIR}"
    mkdir -p "$FINAL_OUT"
    echo "Syncing outputs back to: $FINAL_OUT"
    rsync -a "${SCRATCH_DIR}/${OUTPUT_DIR}/" "${FINAL_OUT}/"
done

rm -rf "$SCRATCH_DIR"

echo "========================================================="
echo "Done!  End time: $(date)"
echo "========================================================="
