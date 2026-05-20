#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=llm_judge_qwen14b
#SBATCH --output=logs/llm_judge_qwen14b_%j.log
#SBATCH --error=logs/llm_judge_qwen14b_%j.log
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00

# =============================================================================
# LLM-as-a-Judge Evaluation — Qwen2.5-14B answers on FinanceBench
#
# Uses GPT-4o to give a binary CORRECT / INCORRECT verdict for each of the
# 150 generated answers extracted from report_Qwen2.5-14B.html.
#
# No GPU needed — this is pure API calls to OpenAI.
#
# ── HOW TO ADD YOUR API KEY ──────────────────────────────────────────────────
#
#   Create a .env file in the project root with one line:
#
#     OPENAI_API_KEY=sk-...
#
#   .env is already in .gitignore so it will never be committed.
#
# ── SUBMIT ───────────────────────────────────────────────────────────────────
#
#   sbatch scripts/run_llm_judge_qwen14b.sh
#
# ── OUTPUT ───────────────────────────────────────────────────────────────────
#
#   baselines/results/llm_judge/qwen14b_judge_results.csv
#   baselines/results/llm_judge/qwen14b_judge_summary.txt
# =============================================================================

set -euo pipefail

PROJ="${SLURM_SUBMIT_DIR:-/u/kobeissa/Documents/thesis/experiments/FB_reproducability}"
ENV_FILE="$PROJ/.env"

# ── load API key from .env ────────────────────────────────────────────────────
if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: .env file not found at $ENV_FILE"
    echo ""
    echo "Create it with:"
    echo "  echo 'OPENAI_API_KEY=sk-...' > .env"
    exit 1
fi

# parse .env manually (handles KEY=value lines, ignores comments)
export $(grep -v '^\s*#' "$ENV_FILE" | xargs)

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY not found in $ENV_FILE"
    exit 1
fi

echo "API key loaded (${OPENAI_API_KEY:0:8}...)"

# ── environment ───────────────────────────────────────────────────────────────
cd "$PROJ"
mkdir -p logs baselines/results/llm_judge

echo "Activating venv: $PROJ/venv"
source "$PROJ/venv/bin/activate"

echo "Python: $(which python3)"

# install openai if not already present
python3 -c "import openai" 2>/dev/null || pip install openai --quiet

# ── run ───────────────────────────────────────────────────────────────────────
echo "Starting LLM judge evaluation at $(date)"

python baselines/llm_judge_qwen14b_eval.py --no-evidence

echo "Done at $(date)"
