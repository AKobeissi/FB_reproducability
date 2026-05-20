#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=fb_ft_reranker
#SBATCH --output=logs/ft_reranker_%j.log
#SBATCH --error=logs/ft_reranker_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:ls40:1
#SBATCH --mem=64G
#SBATCH --time=36:00:00

# =============================================================================
# Fine-tune BAAI/bge-reranker-v2-m3 on FinQA, then evaluate on FinanceBench
# and the FinQA test set.
#
# Stage 1 — Training  (train_cross_encoder_reranker.py)
#   Input : finqa/train.json + finqa/dev.json
#   Output: checkpoints/ft_cross_encoder/   (final epoch checkpoint)
#
# Stage 2 — FinanceBench Evaluation  (run_ft_reranker_baselines.py)
#   Input : baselines/results/predictions/dense_bge_m3_retrieval.json
#           baselines/results/predictions/multi_hyde_retrieval.json
#   Output: baselines/results/predictions/dense_bge_m3_ft_reranker_retrieval.json
#           baselines/results/predictions/multi_hyde_ft_reranker_retrieval.json
#           baselines/results/metrics/*_ft_reranker_metrics.json
#           baselines/results/plots/ft_reranker_*.{pdf,png}
#
# Stage 3 — FinQA Test Evaluation  (run_finqa_ft_reranker.py)
#   Input : data/finqa_test_gold_pages.jsonl  (530 test questions)
#           Final-PDF/  (115 FinQA test PDFs)
#   Output: baselines/results/predictions/finqa_bge_m3_retrieval.json
#           baselines/results/predictions/finqa_ft_reranker_retrieval.json
#           baselines/results/metrics/finqa_{bge_m3,ft_reranker}_metrics.json
#
# Memory budget (L40S 46 GB):
#   bge-reranker-v2-m3 fp16  ≈  1.1 GB
#   training batch=32, seqlen=512 ≈ 18 GB activations
#   BGE-M3 indexing (115 docs)   ≈  2 GB
#   inference batch=64           ≈  3-4 GB
# =============================================================================

SUBMIT_DIR=$SLURM_SUBMIT_DIR
VENV_PATH="$SUBMIT_DIR/venv"
SCRATCH_DIR=/Tmp/$(whoami)/${SLURM_JOB_ID}

export HF_HOME=/data/rech/kobeissa/hf
export HF_HUB_CACHE=/data/rech/kobeissa/hf/hub
export HF_DATASETS_CACHE=/data/rech/kobeissa/hf/datasets
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

echo "========================================================="
echo "Job ID     : ${SLURM_JOB_ID:-local}"
echo "Node       : ${SLURMD_NODENAME:-$(hostname)}"
echo "GPU        : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
echo "Start time : $(date)"
echo "========================================================="

mkdir -p "$SCRATCH_DIR"
mkdir -p "$SUBMIT_DIR/logs"
mkdir -p "$SUBMIT_DIR/checkpoints"

# Copy repo to scratch (fast local I/O during training)
echo "Copying repository to scratch…"
rsync -a --quiet \
    --exclude 'venv' \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.log' \
    --exclude 'outputs' \
    --exclude 'vector_stores' \
    --exclude 'pdfs' \
    --exclude 'Final-PDF' \
    --exclude 'PDF-Opus*' \
    "$SUBMIT_DIR/" "$SCRATCH_DIR/"

# Copy required input files
mkdir -p "$SCRATCH_DIR/baselines/results/predictions"
mkdir -p "$SCRATCH_DIR/baselines/results/metrics"
mkdir -p "$SCRATCH_DIR/finqa"
mkdir -p "$SCRATCH_DIR/data"

cp "$SUBMIT_DIR/finqa/train.json" "$SCRATCH_DIR/finqa/"
cp "$SUBMIT_DIR/finqa/dev.json"   "$SCRATCH_DIR/finqa/"
cp "$SUBMIT_DIR/data/finqa_test_gold_pages.jsonl" "$SCRATCH_DIR/data/"

cp "$SUBMIT_DIR/baselines/results/predictions/dense_bge_m3_retrieval.json" \
   "$SCRATCH_DIR/baselines/results/predictions/"
cp "$SUBMIT_DIR/baselines/results/predictions/multi_hyde_retrieval.json" \
   "$SCRATCH_DIR/baselines/results/predictions/"

# Copy existing baseline metrics (needed for comparison plots)
for f in dense_bge_m3 bge_reranker multi_hyde; do
    src="$SUBMIT_DIR/baselines/results/metrics/${f}_metrics.json"
    if [ -f "$src" ]; then
        cp "$src" "$SCRATCH_DIR/baselines/results/metrics/"
    fi
done

# Copy FinQA test PDFs (115 unique docs from Final-PDF)
echo "Copying FinQA test PDFs to scratch…"
mkdir -p "$SCRATCH_DIR/Final-PDF"
SUBMIT_DIR_ESC="$SUBMIT_DIR" SCRATCH_DIR_ESC="$SCRATCH_DIR" \
python3 - <<'PYEOF'
import json, os, shutil
src_dir = os.environ["SUBMIT_DIR_ESC"] + "/Final-PDF"
dst_dir = os.environ["SCRATCH_DIR_ESC"] + "/Final-PDF"
jsonl   = os.environ["SCRATCH_DIR_ESC"] + "/data/finqa_test_gold_pages.jsonl"
with open(jsonl) as f:
    samples = [json.loads(l) for l in f]
docs = set()
for s in samples:
    for e in s.get("evidences_updated", []):
        docs.add(e.get("doc_name", ""))
copied = 0
for doc in sorted(docs):
    src = os.path.join(src_dir, doc + ".pdf")
    dst = os.path.join(dst_dir, doc + ".pdf")
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy2(src, dst)
        copied += 1
print(f"Copied {copied} FinQA test PDFs to scratch ({len(docs)} unique docs)")
PYEOF

cd "$SCRATCH_DIR"

echo "Activating venv: $VENV_PATH"
source "$VENV_PATH/bin/activate"

# Ensure NLTK data is available
python3 - <<'EOF'
try:
    import nltk
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
except Exception:
    pass
EOF

export PYTHONPATH="${SCRATCH_DIR}:${SCRATCH_DIR}/src:${PYTHONPATH:-}"

# =============================================================================
# Stage 1 — Training
# =============================================================================
echo ""
echo ">>> Stage 1: Training cross-encoder reranker on FinQA"
echo "========================================================="

python3 baselines/train_cross_encoder_reranker.py

TRAIN_EXIT=$?
echo "Training finished (exit code: ${TRAIN_EXIT}) at $(date)"

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "ERROR: Training failed — aborting."
    exit $TRAIN_EXIT
fi

# Sync checkpoint back immediately (in case Stage 2 crashes)
echo "Syncing checkpoint back…"
rsync -a "${SCRATCH_DIR}/checkpoints/ft_cross_encoder/" \
         "${SUBMIT_DIR}/checkpoints/ft_cross_encoder/"

# =============================================================================
# Stage 2 — Inference + Evaluation on FinanceBench
# =============================================================================
echo ""
echo ">>> Stage 2: Re-ranking and evaluating on FinanceBench"
echo "========================================================="

python3 baselines/run_ft_reranker_baselines.py

EVAL_EXIT=$?
echo "Evaluation finished (exit code: ${EVAL_EXIT}) at $(date)"

# =============================================================================
# Stage 3 — FinQA Test Evaluation
# =============================================================================
echo ""
echo ">>> Stage 3: Ingest FinQA test PDFs + evaluate FT reranker on FinQA"
echo "========================================================="

python3 baselines/run_finqa_ft_reranker.py

FINQA_EXIT=$?
echo "FinQA evaluation finished (exit code: ${FINQA_EXIT}) at $(date)"

# =============================================================================
# Sync all outputs back
# =============================================================================
echo "Syncing outputs back…"

rsync -a "${SCRATCH_DIR}/baselines/results/predictions/" \
         "${SUBMIT_DIR}/baselines/results/predictions/"

rsync -a "${SCRATCH_DIR}/baselines/results/metrics/" \
         "${SUBMIT_DIR}/baselines/results/metrics/"

rsync -a "${SCRATCH_DIR}/baselines/results/plots/" \
         "${SUBMIT_DIR}/baselines/results/plots/"

rsync -a "${SCRATCH_DIR}/checkpoints/ft_cross_encoder/" \
         "${SUBMIT_DIR}/checkpoints/ft_cross_encoder/"

# Sync FinQA vector store back (for future re-runs)
rsync -a "${SCRATCH_DIR}/vector_stores/finqa_test/" \
         "${SUBMIT_DIR}/vector_stores/finqa_test/" 2>/dev/null || true

rm -rf "$SCRATCH_DIR"

# Final exit code: fail if either eval stage failed
FINAL_EXIT=$(( EVAL_EXIT != 0 ? EVAL_EXIT : FINQA_EXIT ))

echo "========================================================="
echo "Done!  End time: $(date)"
echo ""
echo "Outputs:"
echo "  checkpoints/ft_cross_encoder/                    ← trained model"
echo "  baselines/results/predictions/*ft_reranker*      ← FinanceBench"
echo "  baselines/results/metrics/*ft_reranker*          ← FinanceBench metrics"
echo "  baselines/results/predictions/finqa_*            ← FinQA test"
echo "  baselines/results/metrics/finqa_*                ← FinQA test metrics"
echo "  baselines/results/plots/ft_reranker_*"
echo "========================================================="

exit ${FINAL_EXIT}
