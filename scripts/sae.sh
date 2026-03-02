#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=unified_sae_diag
#SBATCH --output=unified_sae_diag_%j.log
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00

set -euo pipefail

# ========================================================
# SAE Retrieval-Gap Diagnostics (Post-hoc on Unified Baseline)
# ========================================================
# This script:
#   1) copies repo to scratch
#   2) activates venv
#   3) locates an existing unified baseline results JSON
#   4) runs scripts/sae_diagnostic.py
#   5) copies outputs back
#
# Assumes you ALREADY ran your unified baseline and have outputs JSON somewhere
# (typically in repo outputs/).
#
# If your diagnostic script is named differently (e.g., unified_sae_gap_diagnostics.py),
# change SAE_SCRIPT below.
# ========================================================

# --- 1. SETUP PATHS ---
SUBMIT_DIR="${SLURM_SUBMIT_DIR}"
VENV_PATH="$SUBMIT_DIR/venv"

# Scratch
SCRATCH_DIR="/Tmp/$(whoami)/${SLURM_JOB_ID}"
echo "Working in scratch: $SCRATCH_DIR"
mkdir -p "$SCRATCH_DIR"

# --- 2. COPY CODE TO SCRATCH ---
echo "Copying repository content from $SUBMIT_DIR to scratch..."
rsync -av \
  --exclude 'venv' \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude 'vector_stores' \
  --exclude '*.log' \
  "$SUBMIT_DIR/" "$SCRATCH_DIR/"

cd "$SCRATCH_DIR"

# --- 3. ACTIVATE ENVIRONMENT ---
echo "Activating venv from: $VENV_PATH"
source "$VENV_PATH/bin/activate"

export PYTHONPATH="$SCRATCH_DIR${PYTHONPATH:+:$PYTHONPATH}"
# --- 4. CONFIG (EDIT THESE IF NEEDED) ---
# Diagnostic script path (inside repo copy)
SAE_SCRIPT="scripts/sae_diagnostic.py"

# Dataset + PDF corpus used by the unified baseline
DATASET_JSON="$SUBMIT_DIR/data/financebench_open_source_150.jsonl"
PDF_DIR="$SUBMIT_DIR/pdfs"

# Embedding model for diagnostics (BGE-M3 base)
EMBED_MODEL="BAAI/bge-m3"

# Where to search for unified baseline outputs JSON.
# Default: repo outputs copied into scratch.
UNIFIED_SEARCH_ROOT="$SCRATCH_DIR/outputs"

# If you want to force a specific unified JSON file, set this (absolute path in scratch after copy)
# Example:
# FORCED_UNIFIED_JSON="$SCRATCH_DIR/outputs/unified_hybrid_rerank/results.json"
FORCED_UNIFIED_JSON="${FORCED_UNIFIED_JSON:-}"

# SAE hyperparams
SPLIT_SEED=42
TRAIN_DOC_FRAC=0.8
SAE_HIDDEN_DIM=4096
SAE_TOPK=64
SAE_EPOCHS=12
SAE_BATCH_SIZE=256
MI_TOPK=128

# Output locations in scratch
SCRATCH_OUT="$SCRATCH_DIR/outputs"
SAE_OUT_DIR="$SCRATCH_OUT/unified_sae_gap_diag"
SAE_CACHE_DIR="$SCRATCH_OUT/cache_sae_gap_diag"
mkdir -p "$SAE_OUT_DIR" "$SAE_CACHE_DIR"

# --- 5. PRECHECKS ---
echo "==== PRECHECKS ===="
echo "PWD=$(pwd)"
echo "SUBMIT_DIR=$SUBMIT_DIR"
echo "SCRATCH_DIR=$SCRATCH_DIR"
echo "SCRATCH_OUT=$SCRATCH_OUT"
echo "UNIFIED_SEARCH_ROOT=$UNIFIED_SEARCH_ROOT"
echo "SAE_SCRIPT=$SAE_SCRIPT"
echo "DATASET_JSON=$DATASET_JSON"
echo "PDF_DIR=$PDF_DIR"
echo "==================="

if [ ! -f "$SAE_SCRIPT" ]; then
  echo "[ERROR] SAE diagnostic script not found: $SAE_SCRIPT"
  exit 1
fi

if [ ! -f "$DATASET_JSON" ]; then
  echo "[ERROR] Dataset JSON not found: $DATASET_JSON"
  exit 1
fi

if [ ! -d "$PDF_DIR" ]; then
  echo "[ERROR] PDF directory not found: $PDF_DIR"
  exit 1
fi

if [ ! -d "$UNIFIED_SEARCH_ROOT" ]; then
  echo "[ERROR] Unified search root not found: $UNIFIED_SEARCH_ROOT"
  exit 1
fi

echo "==== DEBUG: JSON files currently under outputs/ ===="
find "$UNIFIED_SEARCH_ROOT" -maxdepth 6 -type f -name "*.json" | sort | head -300 || true
echo "==================================================="

# --- 6. FIND UNIFIED RESULTS JSON ---
if [ -n "$FORCED_UNIFIED_JSON" ]; then
  UNIFIED_JSON="$FORCED_UNIFIED_JSON"
  if [ ! -f "$UNIFIED_JSON" ]; then
    echo "[ERROR] FORCED_UNIFIED_JSON does not exist: $UNIFIED_JSON"
    exit 1
  fi
else
  echo "Auto-discovering unified results JSON (looks for list with 'retrieved_chunks')..."
  export UNIFIED_SEARCH_ROOT
  UNIFIED_JSON=$(
    python - <<'PY'
import os, json, sys

root = os.environ["UNIFIED_SEARCH_ROOT"]
cands = []

for dp, _, fns in os.walk(root):
    for fn in fns:
        if not fn.endswith(".json"):
            continue
        p = os.path.join(dp, fn)
        try:
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, list) and obj and isinstance(obj[0], dict) and "retrieved_chunks" in obj[0]:
                cands.append((os.path.getsize(p), p))
        except Exception:
            continue

if not cands:
    sys.exit(1)

# Largest is usually the main unified results file
cands.sort(reverse=True)
print(cands[0][1])
PY
  ) || {
    echo "[ERROR] Could not auto-find a unified results JSON under $UNIFIED_SEARCH_ROOT"
    echo "Here are recent files for debugging:"
    find "$UNIFIED_SEARCH_ROOT" -maxdepth 6 -type f | sort | tail -200 || true
    exit 1
  }
fi

echo "Using unified results JSON:"
echo "  $UNIFIED_JSON"

# --- 7. RUN SAE DIAGNOSTIC ---
echo "========================================================"
echo "SAE Retrieval Gap Diagnostics (Post-hoc on Unified Baseline)"
echo "========================================================"

python "$SAE_SCRIPT" \
  --unified-results "$UNIFIED_JSON" \
  --dataset-json "$DATASET_JSON" \
  --pdf-dir "$PDF_DIR" \
  --embedding-model "$EMBED_MODEL" \
  --cache-dir "$SAE_CACHE_DIR" \
  --split-seed "$SPLIT_SEED" \
  --train-doc-frac "$TRAIN_DOC_FRAC" \
  --sae-hidden-dim "$SAE_HIDDEN_DIM" \
  --sae-topk "$SAE_TOPK" \
  --sae-epochs "$SAE_EPOCHS" \
  --sae-batch-size "$SAE_BATCH_SIZE" \
  --mi-topk "$MI_TOPK" \
  --out-prefix "$SAE_OUT_DIR/unified_bgem3_sae_gap_diag"

echo "==== DEBUG: SAE outputs produced ===="
find "$SAE_OUT_DIR" -maxdepth 3 -type f | sort || true
echo "====================================="

# --- 8. COPY RESULTS BACK ---
FINAL_DEST="$SUBMIT_DIR/outputs"
echo "Copying results back to: $FINAL_DEST"
mkdir -p "$FINAL_DEST"
cp -r "$SCRATCH_OUT/"* "$FINAL_DEST/" || true

# --- 9. CLEANUP ---
rm -rf "$SCRATCH_DIR"

echo "Done!"