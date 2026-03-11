#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=chunking_sweep
#SBATCH --output=chunking_sweep_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:ls40:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00


SUBMIT_DIR=$SLURM_SUBMIT_DIR
VENV_PATH="$SUBMIT_DIR/venv"
SCRATCH_DIR=/Tmp/$(whoami)/${SLURM_JOB_ID}

# ── Configurable overrides ────────────────────────────────────────────────────
# Pass strategies as 1st arg:  sbatch run_chunking_sweep.sh "naive recursive semantic"
# Pass --standalone-only as 2nd arg to skip the full RAG pipeline
STRATEGIES="${1:-naive recursive semantic adaptive parent_child table_aware late contextual metadata}"
STANDALONE_FLAG="${2:-}"

echo "========================================================="
echo "Job:         $SLURM_JOB_ID"
echo "Node:        $SLURMD_NODENAME"
echo "Submit dir:  $SUBMIT_DIR"
echo "Scratch dir: $SCRATCH_DIR"
echo "Strategies:  $STRATEGIES"
echo "Standalone:  ${STANDALONE_FLAG:-no}"
echo "========================================================="

mkdir -p "$SCRATCH_DIR"

echo "Copying repository to scratch..."
rsync -a --quiet \
  --exclude 'venv' \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude 'outputs' \
  --exclude 'vector_stores' \
  --exclude '*.log' \
  "$SUBMIT_DIR/" "$SCRATCH_DIR/"

cd "$SCRATCH_DIR"
SCRATCH_OUT="${SCRATCH_DIR}/outputs"
SCRATCH_RES="${SCRATCH_DIR}/results"

echo "Activating venv: $VENV_PATH"
source "$VENV_PATH/bin/activate"

export HF_HOME=/data/rech/kobeissa/hf
export HF_HUB_CACHE=/data/rech/kobeissa/hf/hub
export HF_DATASETS_CACHE=/data/rech/kobeissa/hf/datasets

# ── Dependency check ──────────────────────────────────────────────────────────
python - <<'EOF'
import importlib, sys
missing = []
for pkg in ["sentence_transformers", "rouge_score", "nltk", "scipy", "fitz"]:
    try:
        importlib.import_module(pkg)
    except ImportError:
        missing.append(pkg)
if missing:
    print(f"[WARN] Missing packages: {missing}. Installing …")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                           "sentence-transformers", "rouge-score", "nltk", "scipy", "pymupdf"])
else:
    print("[OK] All dependencies present.")

# Ensure NLTK punkt tokenizer is available
import nltk
nltk.download("punkt_tab", quiet=True)
EOF

export PYTHONPATH="${SCRATCH_DIR}:${SCRATCH_DIR}/src:${SCRATCH_DIR}/src/chunking_study:${PYTHONPATH:-}"

# ── Paths ─────────────────────────────────────────────────────────────────────
PDF_DIR="./pdfs"
FB_JSON="${FB_JSON:-${SCRATCH_DIR}/data/financebench_open_source.jsonl}"
CHUNKING_SRC="${SCRATCH_DIR}/src/experiments"
OUTPUT_ROOT="${SCRATCH_OUT}/chunking_sweep"
VECTOR_STORE_DIR="${SCRATCH_DIR}/vector_stores"

# ── Phase 1: Run chunking experiments ─────────────────────────────────────────
echo ""
echo "[Phase 1] Running chunking experiments at $(date)"

python "${CHUNKING_SRC}/rag_chunking_experiments.py" \
  --pdf-dir           "${PDF_DIR}" \
  --output-root       "${OUTPUT_ROOT}" \
  --strategies        ${STRATEGIES} \
  --embedding-model   BAAI/bge-m3 \
  --llm-model         Qwen/Qwen2.5-7B-Instruct \
  --top-k             5 \
  --chunk-size        1024 \
  --chunk-overlap     128 \
  --semantic-threshold      0.5 \
  --semantic-min-sentences  2 \
  --semantic-max-sentences  40 \
  --adaptive-min      256 \
  --adaptive-max      2048 \
  --parent-size       2048 \
  --parent-overlap    256 \
  --child-size        512 \
  --child-overlap     64 \
  --late-chunk-size   512 \
  --late-chunk-overlap 64 \
  --late-model        jinaai/jina-embeddings-v3 \
  --late-max-tokens   8192 \
  --late-window-stride 512 \
  --context-budget    128 \
  --vector-store-dir  "${VECTOR_STORE_DIR}" \
  --use-faiss-chunking \
  --unified-retrieval dense \
  --eval-type         both \
  --eval-mode         static \
  ${STANDALONE_FLAG}

PHASE1_EXIT=$?
echo "[Phase 1] Finished (exit code: ${PHASE1_EXIT}) at $(date)"

# ── Phase 2: Chunk property analysis ─────────────────────────────────────────
echo ""
echo "[Phase 2] Running chunk property analysis at $(date)"

FB_FLAG=""
if [ -f "${FB_JSON}" ]; then
    FB_FLAG="--financebench-json ${FB_JSON}"
else
    echo "[WARN] FinanceBench JSON not found at ${FB_JSON} — running analysis without evidence overlap."
fi

python "${CHUNKING_SRC}/chunk_property_analysis.py" \
  --chunk-dir         "${OUTPUT_ROOT}/chunk_data" \
  --results-dir       "${OUTPUT_ROOT}" \
  ${FB_FLAG} \
  --output-dir        "${OUTPUT_ROOT}/analysis" \
  --embedding-model   BAAI/bge-m3

PHASE2_EXIT=$?
echo "[Phase 2] Finished (exit code: ${PHASE2_EXIT}) at $(date)"

# ── Copy outputs AND results back ─────────────────────────────────────────────
FINAL_OUTPUTS="${SUBMIT_DIR}/outputs"
FINAL_RESULTS="${SUBMIT_DIR}/results"
mkdir -p "${FINAL_OUTPUTS}" "${FINAL_RESULTS}"

echo "Copying outputs to: ${FINAL_OUTPUTS}"
if [ -d "${SCRATCH_OUT}" ] && [ "$(ls -A "${SCRATCH_OUT}" 2>/dev/null)" ]; then
    cp -r "${SCRATCH_OUT}/"* "${FINAL_OUTPUTS}/"
else
    echo "[WARN] outputs/ is empty — check experiment logs."
fi

echo "Copying scored results to: ${FINAL_RESULTS}"
if [ -d "${SCRATCH_RES}" ] && [ "$(ls -A "${SCRATCH_RES}" 2>/dev/null)" ]; then
    cp -r "${SCRATCH_RES}/"* "${FINAL_RESULTS}/"
else
    echo "[WARN] results/ is empty — scored JSON may be missing (evaluation may have failed)."
fi

# ── Cleanup ───────────────────────────────────────────────────────────────────
rm -rf "${SCRATCH_DIR}"

EXIT_CODE=$(( PHASE1_EXIT > PHASE2_EXIT ? PHASE1_EXIT : PHASE2_EXIT ))
echo "Done. Exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}