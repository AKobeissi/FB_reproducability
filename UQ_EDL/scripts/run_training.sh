#!/usr/bin/env bash
# Full training pipeline (both CE and EDL) — submit as SLURM jobs or run sequentially
# Usage: sbatch scripts/run_training.sh
#        or: bash scripts/run_training.sh
#SBATCH --job-name=uq_edl_train
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --output=results/runs/train_%j.log

set -euo pipefail
export HF_HOME=/data/rech/kobeissa/hf
cd "$(dirname "$0")/.."

source ../venv/bin/activate

echo "=== Training Standard Cross-Encoder ==="
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

echo "=== Training Evidential Cross-Encoder ==="
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

echo "Training complete."
