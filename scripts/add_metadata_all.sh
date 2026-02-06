#!/bin/bash
#
# Add metadata to all result files in FB_reproducability and backup directories
#

set -e

BASE_DIR="/u/kobeissa/Documents/thesis/experiments"
DIRS=("FB_reproducability" "backup")

for PROJECT_DIR in "${DIRS[@]}"; do
    FULL_PATH="$BASE_DIR/$PROJECT_DIR"
    
    if [ ! -d "$FULL_PATH" ]; then
        echo "Directory not found: $FULL_PATH, skipping..."
        continue
    fi
    
    echo ""
    echo "========================================================================"
    echo "Processing $PROJECT_DIR"
    echo "========================================================================"
    echo ""
    
    cd "$FULL_PATH"
    
    # Process outputs directory
    if [ -d "outputs" ]; then
        echo "Processing outputs/ directory..."
        python scripts/add_metadata_to_results.py \
            --results-dir outputs \
            --pattern "*.json" \
            "$@"
        echo ""
    fi
    
    # Process results directory
    if [ -d "results" ]; then
        echo "Processing results/ directory..."
        python scripts/add_metadata_to_results.py \
            --results-dir results \
            --pattern "*.json" \
            "$@"
        echo ""
    fi
    
    # Process evaluation_results directory
    if [ -d "evaluation_results" ]; then
        echo "Processing evaluation_results/ directory..."
        python scripts/add_metadata_to_results.py \
            --results-dir evaluation_results \
            --pattern "*.json" \
            "$@"
        echo ""
    fi
    
    # Process src/core/results directory if it exists
    if [ -d "src/core/results" ]; then
        echo "Processing src/core/results/ directory..."
        python scripts/add_metadata_to_results.py \
            --results-dir src/core/results \
            --pattern "*.json" \
            "$@"
        echo ""
    fi
done

echo ""
echo "========================================================================"
echo "All directories processed!"
echo "========================================================================"
