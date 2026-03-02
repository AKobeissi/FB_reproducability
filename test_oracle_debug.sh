#!/bin/bash
# Test oracle document experiment with detailed logging

cd /u/kobeissa/Documents/thesis/experiments/FB_reproducability

echo "Testing Oracle Document Experiment..."
python -m src.core.rag_experiments \
    qwen \
    --experiment oracle_doc \
    --num-samples 5 \
    --chunk-size 1024 \
    --chunk-overlap 128 \
    --embedding-model bge-m3 \
    --chunking-unit tokens \
    --pdf-dir pdfs 2>&1 | tee oracle_debug.log

echo ""
echo "Checking results..."
LATEST_OUTPUT=$(find src/core/outputs/oracle_doc -name "*.json" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
echo "Latest output: $LATEST_OUTPUT"

if [ -f "$LATEST_OUTPUT" ]; then
    echo "Number of results:"
    python -c "import json; data=json.load(open('$LATEST_OUTPUT')); print(f\"  Total samples: {data.get('num_samples', 0)}\"); print(f\"  Results: {len(data.get('results', []))}\")"
    
    echo ""
    echo "First few lines of output:"
    head -30 "$LATEST_OUTPUT"
fi
