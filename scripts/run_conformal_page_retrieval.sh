#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=conformal_page_retrieval
#SBATCH --output=conformal_page_retrieval_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00

set -e

# --- 1. SETUP PATHS ---
# Assuming sbatch is launched from the repo root
SUBMIT_DIR=$SLURM_SUBMIT_DIR

# Path to the virtual environment
VENV_PATH="$SUBMIT_DIR/venv"

# Define Scratch Space
SCRATCH_DIR=/Tmp/$(whoami)/${SLURM_JOB_ID}
echo "Working in scratch: $SCRATCH_DIR"
mkdir -p $SCRATCH_DIR

# --- 2. COPY CODE TO SCRATCH ---
echo "Copying repository content from $SUBMIT_DIR to scratch..."
rsync -av \
	--exclude 'venv' \
	--exclude '.git' \
	--exclude '__pycache__' \
	--exclude 'outputs' \
	--exclude 'vector_stores' \
	--exclude 'results' \
	--exclude '*.log' \
	"$SUBMIT_DIR/" "$SCRATCH_DIR/"

# Move into the scratch directory
cd $SCRATCH_DIR

# --- 3. ACTIVATE ENVIRONMENT ---
echo "Activating venv from: $VENV_PATH"
source "$VENV_PATH/bin/activate"

# Ensure the scratch directory is in PYTHONPATH
export PYTHONPATH="$SCRATCH_DIR:$PYTHONPATH"

# Set environment variables
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# --- 4. CONFIGURATION ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

PDF_DIR="pdfs"
OUTPUT_DIR="results/conformal_page_retrieval_${TIMESTAMP}"

# Set this to the k-fold scorer directory that contains fold_assignments.json
# Example: results/kfold_page_scorer_bge_m3_p20/20260204_151121
LEARNED_KFOLD_DIR=""

# Optional: explicitly point to a fold_assignments.json location
FOLD_ASSIGNMENTS_DIR=""

EMBEDDING_MODEL="BAAI/bge-m3"
CHUNK_SIZE=1024
CHUNK_OVERLAP=128
MAX_PAGE_CHARS=8000
BATCH_SIZE=16
N_FOLDS=5
SEED=42

ALPHAS="0.05 0.10 0.20 0.30"
FIXED_PS="1 3 5 10 20"

CHUNK_TOP_K=5

# Generation settings
WITH_GENERATION=0
LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"
MAX_NEW_TOKENS=256
TEMPERATURE=0.2

if [[ -z "$FOLD_ASSIGNMENTS_DIR" && -n "$LEARNED_KFOLD_DIR" ]]; then
	FOLD_ASSIGNMENTS_DIR="$LEARNED_KFOLD_DIR"
fi

if [[ -z "$FOLD_ASSIGNMENTS_DIR" ]]; then
	echo "ERROR: Set FOLD_ASSIGNMENTS_DIR or LEARNED_KFOLD_DIR to a folder with fold_assignments.json"
	exit 1
fi

echo "=========================================="
echo "Starting Conformal Page Retrieval"
echo "Date: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Output Dir: $OUTPUT_DIR"
echo "=========================================="

CMD=(python scripts/run_conformal_page_retrieval.py
	--pdf-dir "$PDF_DIR"
	--output-dir "$OUTPUT_DIR"
	--fold-assignments-dir "$FOLD_ASSIGNMENTS_DIR"
	--embedding-model "$EMBEDDING_MODEL"
	--chunk-size $CHUNK_SIZE
	--chunk-overlap $CHUNK_OVERLAP
	--max-page-chars $MAX_PAGE_CHARS
	--batch-size $BATCH_SIZE
	--n-folds $N_FOLDS
	--seed $SEED
	--alphas $ALPHAS
	--fixed-ps $FIXED_PS
	--chunk-top-k $CHUNK_TOP_K
)

if [[ -n "$LEARNED_KFOLD_DIR" ]]; then
	CMD+=(--learned-kfold-dir "$LEARNED_KFOLD_DIR")
fi

if [[ $WITH_GENERATION -eq 1 ]]; then
	CMD+=(--with-generation --llm-model "$LLM_MODEL" --max-new-tokens $MAX_NEW_TOKENS --temperature $TEMPERATURE)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

# --- 5. SAVE RESULTS ---
FINAL_DEST="$SUBMIT_DIR/results"
echo "Copying results back to: $FINAL_DEST"
mkdir -p "$FINAL_DEST"
cp -r "$SCRATCH_DIR/$OUTPUT_DIR" "$FINAL_DEST/"

# Cleanup
rm -rf $SCRATCH_DIR

echo "Done!"
