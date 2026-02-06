#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=page_baseline
#SBATCH --output=page_baseline_%j.log
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=03:00:00

# ========================================
# PAGE-THEN-CHUNK BASELINE (No Learned Scorer)
# ========================================
# 
# This runs the baseline page-then-chunk pipeline:
#   1. Retrieve top-5 pages using BGE-M3 embeddings
#   2. Chunk those 5 pages on-the-fly (1024 chars, 128 overlap)
#   3. Rank chunks using BGE-M3 embeddings
#   4. Use top-5 chunks for generation
#
# No learned page scorer - just vanilla BGE-M3 for everything
#

# Activate environment
source venv/bin/activate

# Configuration
LLM_MODEL="qwen"  # or "llama" for Llama 3.2 3B
EMBEDDING_MODEL="BAAI/bge-m3"
CHUNK_SIZE=1024
CHUNK_OVERLAP=128
TOP_K=5  # Final number of chunks used for generation

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/page_baseline_${TIMESTAMP}"

echo "=========================================="
echo "PAGE-THEN-CHUNK BASELINE EXPERIMENT"
echo "=========================================="
echo "LLM Model:       $LLM_MODEL"
echo "Embedding Model: $EMBEDDING_MODEL"
echo "Chunk Size:      $CHUNK_SIZE"
echo "Chunk Overlap:   $CHUNK_OVERLAP"
echo "Top K (chunks):  $TOP_K"
echo "Output Dir:      $OUTPUT_DIR"
echo ""
echo "Pipeline:"
echo "  1. Retrieve top-5 pages with $EMBEDDING_MODEL"
echo "  2. Chunk those pages (size=$CHUNK_SIZE)"  
echo "  3. Rank chunks with $EMBEDDING_MODEL"
echo "  4. Use top-$TOP_K chunks for generation"
echo "=========================================="
echo ""

# Run the experiment
python -m FB_reproducability.rag_experiments $LLM_MODEL \
  -e page_baseline \
  --embedding-model "$EMBEDDING_MODEL" \
  --chunk-size $CHUNK_SIZE \
  --chunk-overlap $CHUNK_OVERLAP \
  --top-k $TOP_K \
  --output-dir "$OUTPUT_DIR"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ Experiment complete!"
    echo "✓ Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Next steps:"
    echo "  1. Evaluate: python scripts/score_experiment.py --input-file \"$OUTPUT_DIR/page_baseline_*.json\""
    echo "  2. Compare with learned scorer results"
else
    echo ""
    echo "✗ Experiment failed with exit code $EXIT_CODE"
    echo "✗ Check the log file for details"
fi

exit $EXIT_CODE
