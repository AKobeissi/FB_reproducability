#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=fb_pos_bias
#SBATCH --output=logs/positional_bias_%j.log
#SBATCH --error=logs/positional_bias_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=3:00:00

# =============================================================================
# Positional bias / lost-in-the-middle attention visualization
#
# Runs Qwen2.5-14B-Instruct on financebench_id_04735 (ADOBE_2015_10K)
# with gold evidence placed at the BEGINNING, MIDDLE, and END of a 20-chunk
# context to demonstrate the U-curve attention pattern.
#
# Memory budget (L40S 46 GB):
#   14B 4-bit model     ≈  8.5 GB
#   Prefill KV cache    ≈  4.0 GB  (3600 tok × 48 layers)
#   Decode attention    ≈  0.03 GB (3600 tok × 48 layers × 40 heads × 1 pos)
#   Total peak          ≈ 13 GB  → comfortably fits L40S
#
# Outputs → analysis/positional_bias/
#   attention_per_token_{beginning,middle,end}.png
#   attention_per_chunk.png
#   ucurve.png
#   results.json
#   summary.txt
# =============================================================================

SUBMIT_DIR=$SLURM_SUBMIT_DIR
VENV_PATH="$SUBMIT_DIR/venv"
SCRATCH_DIR=/Tmp/$(whoami)/${SLURM_JOB_ID}

export HF_HOME=/data/rech/kobeissa/hf
export HF_HUB_CACHE=/data/rech/kobeissa/hf/hub
export HF_DATASETS_CACHE=/data/rech/kobeissa/hf/datasets
export PYTORCH_ALLOC_CONF=expandable_segments:True
export MPLBACKEND=Agg

echo "========================================================="
echo "Job ID     : ${SLURM_JOB_ID:-local}"
echo "Node       : ${SLURMD_NODENAME:-$(hostname)}"
echo "GPU        : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
echo "Start time : $(date)"
echo "========================================================="

mkdir -p "$SCRATCH_DIR"
mkdir -p "$SUBMIT_DIR/logs"

echo "Copying repository to scratch…"
rsync -a --quiet \
    --exclude 'venv' \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.log' \
    --exclude 'vector_stores' \
    --exclude 'pdfs' \
    --exclude 'Final-PDF' \
    --exclude 'PDF-Opus*' \
    "$SUBMIT_DIR/" "$SCRATCH_DIR/"

cd "$SCRATCH_DIR"

echo "Activating venv: $VENV_PATH"
source "$VENV_PATH/bin/activate"

export PYTHONPATH="${SCRATCH_DIR}:${SCRATCH_DIR}/src:${PYTHONPATH:-}"

echo ""
echo ">>> Running positional bias attention visualization"
echo "    Sample: financebench_id_04735 (ADOBE_2015_10K)"
echo "    Model : Qwen/Qwen2.5-14B-Instruct (4-bit)"
echo "========================================================="

python3 analysis/positional_bias/run_positional_bias_attn.py

EXIT_CODE=$?
echo "Script finished (exit code: ${EXIT_CODE}) at $(date)"

# Sync results back
FINAL_OUT="${SUBMIT_DIR}/analysis/positional_bias"
mkdir -p "$FINAL_OUT"

echo "Syncing outputs back to: $FINAL_OUT"
rsync -a "${SCRATCH_DIR}/analysis/positional_bias/" "${FINAL_OUT}/"

rm -rf "$SCRATCH_DIR"

echo "========================================================="
echo "Done!  End time: $(date)"
echo ""
echo "Results:"
echo "  analysis/positional_bias/attention_per_token_beginning.png"
echo "  analysis/positional_bias/attention_per_token_middle.png"
echo "  analysis/positional_bias/attention_per_token_end.png"
echo "  analysis/positional_bias/attention_per_chunk.png"
echo "  analysis/positional_bias/ucurve.png"
echo "  analysis/positional_bias/results.json"
echo "  analysis/positional_bias/summary.txt"
echo "========================================================="

exit ${EXIT_CODE}
