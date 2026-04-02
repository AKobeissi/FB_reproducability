#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=pdf_opus
#SBATCH --output=pdf_opus_%j.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00

# =============================================================================
# pdf_opus_extraction.sh — SLURM job for EDGAR → PDF-Opus pipeline
#
# Downloads SEC filings and converts them to page-faithful PDFs where each
# logical page maps to exactly one PDF page.
#
# Strategy:
#   - Native PDFs on EDGAR → direct download
#   - HTM-only filings → page-break detection + CSS injection + Playwright
#   - Post-processing: blank page removal, validation
#
# Output: $SUBMIT_DIR/PDF-Opus/{doc_name}.pdf + manifest.jsonl
# =============================================================================

set -euo pipefail

SUBMIT_DIR=$SLURM_SUBMIT_DIR
VENV_PATH="$SUBMIT_DIR/venv"
SCRATCH_DIR=/Tmp/$(whoami)/${SLURM_JOB_ID}

echo "========================================================="
echo "Job:         $SLURM_JOB_ID"
echo "Node:        $SLURMD_NODENAME"
echo "Submit dir:  $SUBMIT_DIR"
echo "Scratch dir: $SCRATCH_DIR"
echo "Started:     $(date)"
echo "========================================================="

mkdir -p "$SCRATCH_DIR"

echo "Copying repository to scratch..."
rsync -a --quiet \
  --exclude 'venv' \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '*.log' \
  --exclude 'PDF-Opus' \
  --exclude 'pdfs' \
  --exclude 'pdfs-extended' \
  "$SUBMIT_DIR/" "$SCRATCH_DIR/"

cd "$SCRATCH_DIR"

echo "Activating venv: $VENV_PATH"
source "$VENV_PATH/bin/activate"

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
echo "Checking / installing dependencies..."

python -c "import requests" 2>/dev/null || pip install --quiet requests
python -c "import bs4"      2>/dev/null || pip install --quiet beautifulsoup4 lxml
python -c "import pypdf"    2>/dev/null || pip install --quiet pypdf
python -c "import playwright" 2>/dev/null || pip install --quiet playwright

# Install Chromium browser for Playwright (needed for HTM→PDF conversion)
python -m playwright install chromium 2>/dev/null || true

# Optional: redirect HF caches off home dir (avoid quota issues)
export HF_HOME="/data/rech/$(whoami)/hf"
export HF_DATASETS_CACHE="/data/rech/$(whoami)/hf_datasets"
export TRANSFORMERS_CACHE="/data/rech/$(whoami)/hf_transformers"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" 2>/dev/null || true

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# JSONL input files (relative to repo root)
JSONL_FILES=(
  "data/finqa_test.jsonl"
  "data/secqa_test.jsonl"
)

# Output directory name
OUT_REL="PDF-Opus"

# ---------------------------------------------------------------------------
# Validate inputs
# ---------------------------------------------------------------------------
JSONL_ARGS=()
for f in "${JSONL_FILES[@]}"; do
  if [[ ! -f "$SCRATCH_DIR/$f" ]]; then
    echo "[ERROR] Input file not found: $SCRATCH_DIR/$f"
    exit 1
  fi
  JSONL_ARGS+=("$SCRATCH_DIR/$f")
done

mkdir -p "$SCRATCH_DIR/$OUT_REL"

# ---------------------------------------------------------------------------
# Run the downloader
# ---------------------------------------------------------------------------
echo "========================================================"
echo "Running EDGAR PDF Opus Downloader"
echo "Inputs: ${JSONL_FILES[*]}"
echo "Output: $SCRATCH_DIR/$OUT_REL"
echo "Started: $(date)"
echo "========================================================"

python -u scripts/pdf_opus.py \
  --jsonl "${JSONL_ARGS[@]}" \
  --out_dir "$SCRATCH_DIR/$OUT_REL" \
  --failed-list "failed_downloads.txt"

EXIT_CODE=$?

echo "========================================================"
echo "Download finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "PDF count: $(find "$SCRATCH_DIR/$OUT_REL" -name '*.pdf' | wc -l)"
echo "Disk usage: $(du -sh "$SCRATCH_DIR/$OUT_REL" | cut -f1)"
echo "========================================================"

# ---------------------------------------------------------------------------
# Copy results back to submit dir
# ---------------------------------------------------------------------------
FINAL_DEST="$SUBMIT_DIR/$OUT_REL"
echo "Copying PDFs to: $FINAL_DEST"
mkdir -p "$FINAL_DEST"

# rsync with --ignore-existing so re-runs only copy new files
rsync -a --ignore-existing \
  "$SCRATCH_DIR/$OUT_REL/" \
  "$FINAL_DEST/"

RSYNC_EXIT=$?
if [[ $RSYNC_EXIT -ne 0 ]]; then
  echo "[ERROR] rsync failed (exit $RSYNC_EXIT)"
  echo "  Source still at: $SCRATCH_DIR/$OUT_REL"
  echo "  Scratch NOT cleaned so you can recover."
  exit 1
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
TOTAL=$(find "$FINAL_DEST" -name '*.pdf' | wc -l)

# Cleanup scratch
rm -rf "$SCRATCH_DIR"

echo "========================================================"
echo "All done!"
echo "PDFs saved to : $FINAL_DEST"
echo "Total PDFs    : $TOTAL"
echo "Manifest      : $FINAL_DEST/manifest.jsonl"
if [[ -f "$FINAL_DEST/failed_downloads.txt" ]]; then
  FAIL_COUNT=$(wc -l < "$FINAL_DEST/failed_downloads.txt")
  echo "Failed        : $FAIL_COUNT (see $FINAL_DEST/failed_downloads.txt)"
fi
echo "Finished      : $(date)"
echo "========================================================"