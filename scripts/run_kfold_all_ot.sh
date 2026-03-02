#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=kfold_all_ot
#SBATCH --output=kfold_all_ot_%j.log
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=96:00:00

SUBMIT_DIR=$SLURM_SUBMIT_DIR
VENV_PATH="$SUBMIT_DIR/venv"
SCRATCH_DIR=/Tmp/$(whoami)/${SLURM_JOB_ID}

mkdir -p "$SCRATCH_DIR"
rsync -av \
    --exclude 'venv' \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude 'outputs' \
    --exclude 'vector_stores' \
    --exclude 'results' \
    --exclude '*.log' \
    "$SUBMIT_DIR/" "$SCRATCH_DIR/"

cd "$SCRATCH_DIR"
source "$VENV_PATH/bin/activate"

export PYTHONPATH="$SCRATCH_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Keep all model/tokenizer caches in scratch (avoid persistent large files on submit dir/home)
export HF_HOME="$SCRATCH_DIR/.hf_cache"
export TRANSFORMERS_CACHE="$SCRATCH_DIR/.hf_cache/transformers"
export HUGGINGFACE_HUB_CACHE="$SCRATCH_DIR/.hf_cache/hub"
export SENTENCE_TRANSFORMERS_HOME="$SCRATCH_DIR/.hf_cache/sentence_transformers"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$SENTENCE_TRANSFORMERS_HOME"

SCRATCH_OUT="$SCRATCH_DIR/results"
mkdir -p "$SCRATCH_OUT"

COMMON_ARGS="--n-folds 5 --epochs 15 --batch-size 32 --lr 2e-5 --seed 42 --pdf-dir pdfs"
COMMON_OT_ARGS="--use-reranker --ot-model BAAI/bge-m3 --ot-query-sentences 8 --ot-doc-sentences 24 --ot-reg 0.05 --ot-iters 40"
CE_ARGS="--reranker-model BAAI/bge-reranker-v2-m3 --reranker-batch-size 16 --reranker-top-k 5"

# Unified (non-learned) OT experiment args
UNIFIED_COMMON_ARGS="-e unified --num-samples 150 --pdf-dir pdfs --embedding-model BAAI/bge-m3 --chunk-size 384 --chunk-overlap 64 --chunking-unit tokens --top-k 5 --k-cand 100 --eval-type retrieval --unified-retrieval dense --unified-rerank"

echo "=== OT EXP 1/4: global + OT ==="
python train_k_fold2.py \
  $COMMON_ARGS --page-k 20 --chunk-k 5 \
  $COMMON_OT_ARGS --reranker-type ot --reranker-candidates 100 \
  --output-dir "$SCRATCH_OUT/kfold_global_ot"

echo "=== OT EXP 2/4: global + OT->CE ==="
python train_k_fold2.py \
  $COMMON_ARGS --page-k 20 --chunk-k 5 \
  $COMMON_OT_ARGS $CE_ARGS --reranker-type ot_then_cross_encoder \
  --reranker-candidates 100 --ot-prune-k 20 \
  --output-dir "$SCRATCH_OUT/kfold_global_ot_then_ce"

echo "=== OT EXP 3/4: filtered + OT ==="
python train_k_fold2_filtered_chunks.py \
  $COMMON_ARGS --page-k 20 --chunk-k 5 \
  $COMMON_OT_ARGS --reranker-type ot --reranker-candidates 100 \
  --output-dir "$SCRATCH_OUT/kfold_filtered_ot"

echo "=== OT EXP 4/4: filtered + OT->CE ==="
python train_k_fold2_filtered_chunks.py \
  $COMMON_ARGS --page-k 20 --chunk-k 5 \
  $COMMON_OT_ARGS $CE_ARGS --reranker-type ot_then_cross_encoder \
  --reranker-candidates 100 --ot-prune-k 20 \
  --output-dir "$SCRATCH_OUT/kfold_filtered_ot_then_ce"

echo "=== OT EXP 5/6: unified (non-learned) + OT ==="
python -m src.core.rag_experiments qwen \
  $UNIFIED_COMMON_ARGS \
  --vector-store-dir "$SCRATCH_DIR/vector_stores_unified_ot" \
  --output-dir "$SCRATCH_OUT/unified_ot_outputs" \
  --unified-reranker-style ot \
  --unified-ot-model BAAI/bge-m3 \
  --unified-ot-query-sentences 8 \
  --unified-ot-doc-sentences 24 \
  --unified-ot-reg 0.05 \
  --unified-ot-iters 40

echo "=== OT EXP 6/6: unified (non-learned) + OT->CE ==="
python -m src.core.rag_experiments qwen \
  $UNIFIED_COMMON_ARGS \
  --vector-store-dir "$SCRATCH_DIR/vector_stores_unified_ot" \
  --output-dir "$SCRATCH_OUT/unified_ot_then_ce_outputs" \
  --unified-reranker-style ot_then_cross_encoder \
  --unified-ot-model BAAI/bge-m3 \
  --unified-ot-query-sentences 8 \
  --unified-ot-doc-sentences 24 \
  --unified-ot-reg 0.05 \
  --unified-ot-iters 40 \
  --unified-ot-prune-k 20

FINAL_DEST="$SUBMIT_DIR/results"
mkdir -p "$FINAL_DEST"
cp -r "$SCRATCH_OUT/"* "$FINAL_DEST/"
rm -rf "$SCRATCH_DIR"

echo "Done. OT experiment results copied to $FINAL_DEST"
