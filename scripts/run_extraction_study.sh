#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=extraction_study
#SBATCH --output=logs/extraction_study_%j.log
#SBATCH --error=logs/extraction_study_%j.log
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

mkdir -p logs

echo "========================================"
echo "Job ID     : ${SLURM_JOB_ID:-local}"
echo "Start time : $(date)"
echo "Working dir: $(pwd)"
echo "========================================"

# Assuming the correct python is on the path or you have an environment to activate
# If you have a venv, you can uncomment the following:
# source venv/bin/activate

python3 baselines/extraction_study.py

echo "========================================"
echo "End time : $(date)"
echo "========================================"
