#!/bin/bash
#SBATCH --job-name=bootstrap_sig
#SBATCH --output=bootstrap_sig_%j.log
#SBATCH --error=bootstrap_sig_%j.log
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu

set -euo pipefail

REPO=/u/kobeissa/Documents/thesis/experiments/FB_reproducability

cd "$REPO"
source venv/bin/activate

echo "================================================"
echo "Bootstrap Significance Tests"
echo "Job: ${SLURM_JOB_ID:-local}  Start: $(date)"
echo "================================================"

python3 baselines/bootstrap_significance.py

echo ""
echo "Done: $(date)"
