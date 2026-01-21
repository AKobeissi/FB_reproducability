#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=hybrid_grid_search
#SBATCH --output=hybrid_grid_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00

# =================================================================
# HYBRID RAG GRID SEARCH (SLURM)
# =================================================================
# Models: Qwen 2.5 7B
# Embeddings: BGE-M3, MPNet
# Sparse: BM25, SPLADE
# Modes: Standard Hybrid (Gen), Hybrid Sweep (Params)
# Chunking: 1024 tokens / 128 overlap
# =================================================================

# --- 1. SETUP PATHS ---
SUBMIT_DIR=$SLURM_SUBMIT_DIR
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
    --exclude '*.log' \
    "$SUBMIT_DIR/" "$SCRATCH_DIR/"

cd $SCRATCH_DIR

# --- 3. ACTIVATE ENVIRONMENT ---
echo "Activating venv from: $VENV_PATH"
source "$VENV_PATH/bin/activate"
export PYTHONPATH="$SCRATCH_DIR:$PYTHONPATH"

# --- 4. DEFINE GRID ---
# Define the Embeddings to test
declare -a EMBEDDINGS=(
    "BAAI/bge-m3"
    "sentence-transformers/all-mpnet-base-v2"
)

# Define Sparse Models to test
declare -a SPARSE_MODELS=(
    "bm25"
    "splade"
)

# Output directory within scratch
SCRATCH_OUT="$SCRATCH_DIR/outputs"
mkdir -p "$SCRATCH_OUT"

# [OPTION A] Define Vector Store Path in Scratch
# This ensures databases are built in writable, fast local storage
VECTOR_STORE_DIR="$SCRATCH_DIR/vector_stores"
mkdir -p "$VECTOR_STORE_DIR"
echo "Vector Stores will be created in: $VECTOR_STORE_DIR"

# --- 5. RUN EXPERIMENTS ---

for emb in "${EMBEDDINGS[@]}"; do
    for sparse in "${SPARSE_MODELS[@]}"; do
        
        # Clean model name for folder/file naming (replace / with _)
        emb_safe=$(echo "$emb" | tr '/' '_')
        
        echo "----------------------------------------------------------------"
        echo "[RUNNING] Hybrid Grid | Emb: $emb | Sparse: $sparse"
        echo "----------------------------------------------------------------"

        # A. STANDARD HYBRID (Retrieval + Generation)
        #    Runs the pipeline with default alpha (usually 0.5 or 1.0 depending on code)
        echo "   > Mode: Standard Hybrid (Generative)"
        python -m src.core.rag_experiments qwen \
            --experiment hybrid \
            --embedding-model "$emb" \
            --sparse-model "$sparse" \
            --chunk-size 1024 \
            --chunk-overlap 128 \
            --chunking-unit tokens \
            --pdf-dir pdfs \
            --top-k 5 \
            --vector-store-dir "$VECTOR_STORE_DIR" \
            --output-dir "$SCRATCH_OUT/hybrid_${emb_safe}_${sparse}"

        # B. HYBRID SWEEP (Parameter Search)
        #    Runs retrieval only across many alpha/k values
        echo "   > Mode: Hybrid Sweep (Grid Search)"
        python -m src.core.rag_experiments qwen \
            --experiment hybrid_sweep \
            --embedding-model "$emb" \
            --sparse-model "$sparse" \
            --chunk-size 1024 \
            --chunk-overlap 128 \
            --chunking-unit tokens \
            --pdf-dir pdfs \
            --top-k 5 \
            --vector-store-dir "$VECTOR_STORE_DIR" \
            --output-dir "$SCRATCH_OUT/sweep_${emb_safe}_${sparse}"
            
    done
done

# --- 6. SAVE RESULTS ---
FINAL_DEST="$SUBMIT_DIR/outputs"
echo "Copying results back to: $FINAL_DEST"
mkdir -p "$FINAL_DEST"
# Copy recursively, preserving the subdirectories created above
cp -r "$SCRATCH_OUT/"* "$FINAL_DEST/"

# Cleanup
rm -rf $SCRATCH_DIR

echo "Job Complete!"