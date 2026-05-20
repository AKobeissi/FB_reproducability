#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=finqa_gold_pages
#SBATCH --output=logs/finqa_gold_pages_%j.log
#SBATCH --error=logs/finqa_gold_pages_%j.log
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

mkdir -p logs

echo "========================================"
echo "Job ID     : ${SLURM_JOB_ID:-local}"
echo "Start time : $(date)"
echo "Working dir: $(pwd)"
echo "========================================"

python map_finqa_gold_pages.py \
    --lofin   data/finqa_test_pdf_only.jsonl \
    --finqa   finqa/test.json \
    --pdf-dir Final-PDF \
    --output  data/finqa_test_gold_pages.jsonl \
    --threshold 0.65 \
    --verbose

echo "========================================"
echo "End time : $(date)"
echo "========================================"