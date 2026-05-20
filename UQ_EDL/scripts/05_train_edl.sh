#!/usr/bin/env bash
# Stage 4: Train evidential cross-encoder
#SBATCH --job-name=uq_edl_edl
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=results/runs/edl_%j.log
set -euo pipefail
cd "$(dirname "$0")/.."

python src/training/train_evidential.py \
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
    --lambda_kl 0.001 \
    --annealing_ratio 0.2 \
    --run_dir results/runs

echo "EDL training done."
