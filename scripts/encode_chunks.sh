#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=encode_chunks
#SBATCH --output=encode_chunks_%j.log
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# Activate environment
source venv/bin/activate

# Run encoding
python scripts/encode_chunks.py \
    --embedding-model "BAAI/bge-m3" \
    --data-path "data/financebench_open_source.jsonl" \
    --pdfs-dir "pdfs" \
    --chunk-size 1024 \
    --chunk-overlap 128 \
    --chunking-unit tokens \
    --output-dir "analysis/geometric/dense_bge_baseline" \
    --use-sentence-transformers

echo "Encoding complete!"
