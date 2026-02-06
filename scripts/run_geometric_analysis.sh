#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=geometric_analysis
#SBATCH --output=geometric_analysis_%j.log
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -e  # Exit on error

echo "=============================================="
echo "GEOMETRIC ANALYSIS PIPELINE"
echo "=============================================="
echo ""

# Configuration
REPO_ROOT="/u/kobeissa/Documents/thesis/experiments/FB_reproducability"
cd "$REPO_ROOT"

# Experiments to analyze
declare -A EXPERIMENTS=(
    ["dense_bge_baseline"]="src/core/results/shared_vector/20260204/shared_vector_20260204_172553_scored.json"
    ["learned_page_scorer"]="results/kfold_page_scorer_bge_m3/20260204_151121/all_predictions.json"
)

# Embedding models
DENSE_BGE_MODEL="BAAI/bge-m3"
DATA_PATH="data/financebench_open_source.jsonl"

# Vector stores (if available)
DENSE_BGE_VECTOR_STORE="vector_stores/chroma/shared_vector_bge-m3"

echo "Step 1: Extract embeddings for Dense BGE baseline"
echo "=================================================="

DENSE_OUTPUT="analysis/geometric/dense_bge_baseline"

python scripts/extract_embeddings.py \
    --embedding-model "$DENSE_BGE_MODEL" \
    --data-path "$DATA_PATH" \
    --vector-store-path "$DENSE_BGE_VECTOR_STORE" \
    --output-dir "$DENSE_OUTPUT" \
    --use-sentence-transformers

echo ""
echo "Step 2: Run geometric analysis for Dense BGE baseline"
echo "======================================================"

if [ -f "$DENSE_OUTPUT/embeddings/query_embeddings.npy" ] && \
   [ -f "$DENSE_OUTPUT/embeddings/chunk_embeddings.npy" ]; then
    
    python scripts/geometric_analysis.py \
        --results-file "${EXPERIMENTS[dense_bge_baseline]}" \
        --embeddings-dir "$DENSE_OUTPUT/embeddings" \
        --output-dir "$DENSE_OUTPUT/analysis" \
        --method-name "Dense BGE-M3 Baseline" \
        --k-neighbors 50
    
    echo ""
    echo "✓ Dense BGE baseline analysis complete!"
    echo "  Results saved to: $DENSE_OUTPUT/analysis"
    echo ""
else
    echo "⚠ Warning: Chunk embeddings not found for Dense BGE baseline"
    echo "  Please ensure vectorstore exists or manually provide chunk embeddings"
    echo ""
fi

echo "Step 3: Extract embeddings for Learned Page Scorer"
echo "==================================================="

# For page scorer, we need to extract from the k-fold results
PAGE_SCORER_OUTPUT="analysis/geometric/learned_page_scorer"

# The page scorer uses BGE-M3 embeddings as well
python scripts/extract_embeddings.py \
    --embedding-model "$DENSE_BGE_MODEL" \
    --data-path "$DATA_PATH" \
    --vector-store-path "results/kfold_page_scorer_bge_m3/20260204_151121/vector_store" \
    --output-dir "$PAGE_SCORER_OUTPUT" \
    --use-sentence-transformers

echo ""
echo "Step 4: Run geometric analysis for Learned Page Scorer"
echo "======================================================="

if [ -f "$PAGE_SCORER_OUTPUT/embeddings/query_embeddings.npy" ] && \
   [ -f "$PAGE_SCORER_OUTPUT/embeddings/chunk_embeddings.npy" ]; then
    
    # Need to convert all_predictions.json format to have retrieval_metrics
    # For now, skip if format doesn't match
    echo "⚠ Note: all_predictions.json format may need conversion for analysis"
    echo "  Attempting analysis anyway..."
    
    python scripts/geometric_analysis.py \
        --results-file "${EXPERIMENTS[learned_page_scorer]}" \
        --embeddings-dir "$PAGE_SCORER_OUTPUT/embeddings" \
        --output-dir "$PAGE_SCORER_OUTPUT/analysis" \
        --method-name "Learned Page Scorer (BGE-M3)" \
        --k-neighbors 50 || echo "⚠ Analysis failed - format mismatch expected"
    
    echo ""
    echo "✓ Learned page scorer analysis attempted"
    echo "  Results saved to: $PAGE_SCORER_OUTPUT/analysis"
    echo ""
else
    echo "⚠ Warning: Chunk embeddings not found for Page Scorer"
    echo ""
fi

echo ""
echo "=============================================="
echo "GEOMETRIC ANALYSIS COMPLETE"
echo "=============================================="
echo ""
echo "Results locations:"
echo "  Dense BGE Baseline:     $DENSE_OUTPUT/analysis"
echo "  Learned Page Scorer:    $PAGE_SCORER_OUTPUT/analysis"
echo ""
echo "Generated files for each experiment:"
echo "  - query_curvatures.npy          # Curvature values"
echo "  - eigenvalue_spectra.npy        # Eigenvalue spectra"
echo "  - results_with_curvature.csv    # Results + curvature data"
echo "  - geometric_analysis_report.txt # Statistical summary"
echo "  - curvature_vs_*.pdf            # Performance correlation plots"
echo "  - curvature_by_question_type.pdf"
echo "  - eigenvalue_spectrum.pdf"
echo "  - curvature_heatmap_2d.pdf"
echo ""
echo "Next steps:"
echo "  1. Review plots in analysis directories"
echo "  2. Check geometric_analysis_report.txt for statistics"
echo "  3. Use results_with_curvature.csv for further analysis"
echo ""
