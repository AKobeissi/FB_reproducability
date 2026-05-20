#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=fb_regen_plots
#SBATCH --output=logs/regenerate_plots_%j.log
#SBATCH --error=logs/regenerate_plots_%j.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00

SUBMIT_DIR=$SLURM_SUBMIT_DIR
VENV_PATH="$SUBMIT_DIR/venv"

export HF_HOME=/data/rech/kobeissa/hf
export PYTHONPATH="${SUBMIT_DIR}:${SUBMIT_DIR}/src:${PYTHONPATH:-}"

echo "Activating venv: $VENV_PATH"
source "$VENV_PATH/bin/activate"

cd "$SUBMIT_DIR"

echo "Regenerating plots (including FT-ReRanker variants)…"
python3 baselines/regenerate_plots.py

echo "Done."
echo "Plots: baselines/results/plots/"
