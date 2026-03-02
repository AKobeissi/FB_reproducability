#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=download_sec_pdfs
#SBATCH --output=download_sec_pdfs_%j.log
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
# Note: No GPU needed — this is a pure network I/O + PDF processing job.
# Adjust --time based on your dataset size:
#   ~100 docs  → 2-3h
#   ~300+ docs → 6h+

# ---------------------------------------------------------------------------
# 1. SETUP PATHS
# ---------------------------------------------------------------------------
SUBMIT_DIR=$SLURM_SUBMIT_DIR

VENV_PATH="$SUBMIT_DIR/venv"

SCRATCH_DIR=/Tmp/$(whoami)/${SLURM_JOB_ID}
echo "Working in scratch: $SCRATCH_DIR"
mkdir -p $SCRATCH_DIR

# ---------------------------------------------------------------------------
# 2. COPY CODE + DATA TO SCRATCH
# ---------------------------------------------------------------------------
echo "Copying repository content from $SUBMIT_DIR to scratch..."
rsync -av \
    --exclude 'venv' \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude 'outputs' \
    --exclude 'pdfs_extended' \
    --exclude '*.log' \
    "$SUBMIT_DIR/" "$SCRATCH_DIR/"

cd $SCRATCH_DIR

# ---------------------------------------------------------------------------
# 3. ACTIVATE ENVIRONMENT
# ---------------------------------------------------------------------------
echo "Activating venv from: $VENV_PATH"
source "$VENV_PATH/bin/activate"

export PYTHONPATH="$SCRATCH_DIR:$PYTHONPATH"

# Verify dependencies are installed
echo "Checking required packages..."
python -c "import requests, tqdm" || {
    echo "[ERROR] Missing required packages. Run inside venv:"
    echo "  pip install requests tqdm playwright"
    echo "  python -m playwright install chromium"
    exit 1
}

# ---------------------------------------------------------------------------
# 4. CONFIGURE DOWNLOAD
# ---------------------------------------------------------------------------
# Output directory in scratch (PDFs can be large — scratch has more space)
SCRATCH_PDF_DIR="$SCRATCH_DIR/pdfs_extended"
mkdir -p "$SCRATCH_PDF_DIR"

# SEC User-Agent — REQUIRED by SEC fair-access policy
# Format: "Full Name email@domain.com"
SEC_USER_AGENT="Amine Kobeissi amine.kobeissi@umontreal.ca"

# Input dataset files (relative to SCRATCH_DIR after rsync)
INPUT_FILES=(
    "data/finqa_test.jsonl"
    "data/secqa_test.jsonl"
)

# Check that input files exist
for f in "${INPUT_FILES[@]}"; do
    if [[ ! -f "$SCRATCH_DIR/$f" ]]; then
        echo "[ERROR] Input file not found: $SCRATCH_DIR/$f"
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# 5. RUN PDF DOWNLOAD
# ---------------------------------------------------------------------------
echo "========================================================"
echo "Downloading SEC PDFs for FinanceBench / SECQA datasets"
echo "========================================================"
echo "Inputs: ${INPUT_FILES[*]}"
echo "Output: $SCRATCH_PDF_DIR"
echo "Started: $(date)"
echo "========================================================"

python -u scripts/data_pdf_extract.py \
    --inputs "${INPUT_FILES[@]/#/$SCRATCH_DIR/}" \
    --out_dir "$SCRATCH_PDF_DIR" \
    --user_agent "$SEC_USER_AGENT" \
    --sleep_s 0.3 \
    --retries 4 \
    --workers 1 \
    --cache_dir "$SCRATCH_DIR/.sec_cache"

DOWNLOAD_EXIT=$?

echo "========================================================"
echo "Download finished at: $(date)"
echo "Exit code: $DOWNLOAD_EXIT"
echo "========================================================"

# Show what was produced
echo "PDF count : $(find "$SCRATCH_PDF_DIR" -name '*.pdf' | wc -l)"
echo "Disk usage: $(du -sh "$SCRATCH_PDF_DIR" | cut -f1)"

if [[ $DOWNLOAD_EXIT -ne 0 ]]; then
    echo "[WARN] Download script exited with code $DOWNLOAD_EXIT — some docs may have failed."
    echo "       Check the manifest for details: $SCRATCH_PDF_DIR/manifest.jsonl"
    echo "       You can re-run the same command; successful docs will be skipped."
fi

# ---------------------------------------------------------------------------
# 6. SAVE RESULTS BACK TO SUBMIT DIR
# ---------------------------------------------------------------------------
FINAL_DEST="$SUBMIT_DIR/pdfs_extended"

echo "Copying PDFs back to: $FINAL_DEST"
mkdir -p "$FINAL_DEST"

# Use rsync so re-runs only copy new/changed files
rsync -av --ignore-existing \
    "$SCRATCH_PDF_DIR/" \
    "$FINAL_DEST/"

RSYNC_EXIT=$?
if [[ $RSYNC_EXIT -ne 0 ]]; then
    echo "[ERROR] rsync failed (exit $RSYNC_EXIT) — PDFs may not have been saved!"
    echo "  Source still available at: $SCRATCH_PDF_DIR"
    echo "  Scratch will NOT be cleaned up so you can recover manually."
    exit 1
fi

# ---------------------------------------------------------------------------
# 7. CLEANUP
# ---------------------------------------------------------------------------
echo "Cleaning up scratch: $SCRATCH_DIR"
rm -rf $SCRATCH_DIR

echo "========================================================"
echo "All done!"
echo "PDFs saved to : $FINAL_DEST"
echo "Manifest      : $FINAL_DEST/manifest.jsonl"
echo "========================================================"