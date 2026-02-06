#!/bin/bash
#
# Aggregate evaluation results from FB_reproducability and backup directories
#

set -e

BASE_DIR="/u/kobeissa/Documents/thesis/experiments"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "========================================================================"
echo "Aggregating Evaluation Results"
echo "========================================================================"
echo ""

# Collect all result directories from both FB_reproducability and backup
RESULT_DIRS=()

for PROJECT_DIR in "FB_reproducability" "backup"; do
    FULL_PATH="$BASE_DIR/$PROJECT_DIR"
    
    if [ ! -d "$FULL_PATH" ]; then
        echo "Directory not found: $FULL_PATH, skipping..."
        continue
    fi
    
    echo "Collecting from $PROJECT_DIR..."
    
    # Add each subdirectory that exists
    for SUBDIR in "outputs" "results" "evaluation_results" "src/core/results"; do
        if [ -d "$FULL_PATH/$SUBDIR" ]; then
            RESULT_DIRS+=("$FULL_PATH/$SUBDIR")
            echo "  ✓ $SUBDIR"
        fi
    done
done

echo ""
echo "Found ${#RESULT_DIRS[@]} result directories"
echo ""

# Run aggregation
cd "$BASE_DIR/FB_reproducability"

python scripts/aggregate_evaluation_results.py \
    --results-dirs "${RESULT_DIRS[@]}" \
    --output-dir aggregated_results \
    --prefix "combined"

echo ""
echo "========================================================================"
echo "Done! Check aggregated_results/ directory for CSV files"
echo "========================================================================"
