#!/usr/bin/env bash
# Stage 3: Train standard cross-encoder
#SBATCH --job-name=uq_edl_ce
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=results/runs/ce_%j.log
set -euo pipefail
cd "$(dirname "$0")/.."

python src/training/train_cross_encoder.py \
    --train_pairs data/processed/pairs/finqa_train_pairs.jsonl \
    --val_pairs data/processed/pairs/finqa_val_pairs.jsonl \
    --backbone BAAI/bge-reranker-v2-m3 \
    --max_length 512 \
    --batch_size 8 \
    --grad_accum 4 \
    --lr 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --epochs 3 \
    --fp16 \
    --seed 42 \
    --run_dir results/runs

echo "CE training done."
