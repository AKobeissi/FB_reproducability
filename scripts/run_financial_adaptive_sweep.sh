#!/bin/bash -l
#SBATCH --partition=rali
#SBATCH --job-name=finance_adaptive_sweep
#SBATCH --output=finance_adaptive_sweep_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00

set -euo pipefail

# =============================================================================
# FINANCE-ADAPTIVE PAGE RETRIEVAL SWEEPS
#
# Runs two parameter sweeps for scripts/finance_adaptive_page_retrieval.py:
#   1) Sweep A: balanced fusion + moderate neighborhood expansion
#   2) Sweep B: denser learned weighting + larger expansion window
# =============================================================================

# --- 1. RUN IN SUBMIT DIRECTORY (like run_page_baseline.sh) ---
cd "$SLURM_SUBMIT_DIR"

# --- 2. ACTIVATE ENVIRONMENT ---
echo "Activating venv from: venv/bin/activate"
source venv/bin/activate

export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# --- 3. CONFIGURATION ---
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="results/finance_adaptive_sweeps/${TS}"
MODEL_PATH="models/finetuned_page_scorer_v2/best_model"

mkdir -p "$OUT_DIR"

if [ ! -f "$MODEL_PATH/modules.json" ]; then
  echo "[ERROR] SentenceTransformer model not found at: $MODEL_PATH"
  echo "[ERROR] Expected file missing: $MODEL_PATH/modules.json"
  echo "[ERROR] Fix by training/saving model or changing MODEL_PATH in this script."
  exit 1
fi

# --- 4. RUN SWEEPS ---

echo "=========================================="
echo "Starting Finance-Adaptive Retrieval Sweeps"
echo "Date: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Model: $MODEL_PATH"
echo "Output: $OUT_DIR"
echo "=========================================="

echo ""
echo "==================== SWEEP A ===================="
echo "Balanced fusion + moderate neighborhood expansion"
echo "================================================="

python scripts/finance_adaptive_page_retrieval.py \
  --model-path "$MODEL_PATH" \
  --page-k 20 \
  --seed-ks "6,8,10" \
  --dense-weights "0.55,0.65,0.75" \
  --bm25-weights "0.20,0.30,0.40" \
  --section-weights "0.05,0.10,0.15" \
  --number-weights "0.00,0.05" \
  --base-windows "1,2" \
  --distance-penalty 0.08 \
  --output "$OUT_DIR/sweep_A.json" \
  | tee "$OUT_DIR/sweep_A.log"

if [ $? -eq 0 ]; then
  echo "✓ Sweep A completed successfully"
else
  echo "✗ Sweep A failed"
fi

echo ""
echo "==================== SWEEP B ===================="
echo "Denser learned weighting + larger expansion window"
echo "================================================="

python scripts/finance_adaptive_page_retrieval.py \
  --model-path "$MODEL_PATH" \
  --page-k 20 \
  --seed-ks "8,10,12" \
  --dense-weights "0.70,0.80,0.90" \
  --bm25-weights "0.10,0.20,0.30" \
  --section-weights "0.05,0.10" \
  --number-weights "0.05,0.10" \
  --base-windows "2,3" \
  --distance-penalty 0.10 \
  --output "$OUT_DIR/sweep_B.json" \
  --leakage-safe-mode both \
  --leakage-group-by both \
  --tune-frac 0.8 \
  --cv-folds 5 \
  --split-seed 42 \
  --leakage-output "$OUT_DIR/sweep_B_leakage_safe.json" \
  | tee "$OUT_DIR/sweep_B.log"

if [ $? -eq 0 ]; then
  echo "✓ Sweep B completed successfully"
else
  echo "✗ Sweep B failed"
fi

echo ""
echo "================= INFERENCE + EVAL ================="
echo "Using BEST config from Sweep B for full 150-question run"
echo "Outputs unified-style results + scored evaluation JSON"
echo "===================================================="

python scripts/finance_adaptive_page_retrieval.py \
  --model-path "$MODEL_PATH" \
  --page-k 20 \
  --seed-ks "8,10,12" \
  --dense-weights "0.70,0.80,0.90" \
  --bm25-weights "0.10,0.20,0.30" \
  --section-weights "0.05,0.10" \
  --number-weights "0.05,0.10" \
  --base-windows "2,3" \
  --distance-penalty 0.10 \
  --output "$OUT_DIR/sweep_B_recompute.json" \
  --run-inference \
  --best-from-leakage-safe "$OUT_DIR/sweep_B_leakage_safe.json" \
  --inference-output "$OUT_DIR/inference_unified_style.json" \
  --eval-mode static \
  --gen-model "meta-llama/Llama-3.2-3B-Instruct" \
  --gen-embedding-model "sentence-transformers/all-mpnet-base-v2" \
  --gen-output-dir "outputs/finance_adaptive_generation" \
  --gen-vector-store-dir "vector_stores" \
  | tee "$OUT_DIR/inference_eval.log"

if [ $? -eq 0 ]; then
  echo "✓ Inference + eval completed successfully"
else
  echo "✗ Inference + eval failed"
fi

echo ""
echo "=========================================="
echo "Sweep summaries"
echo "=========================================="

OUT_DIR_ENV="$OUT_DIR" python - <<'PY'
import json
import os
from pathlib import Path

base = Path(os.environ["OUT_DIR_ENV"])
for name in ["sweep_A.json", "sweep_B.json"]:
  p = base / name
  if not p.exists():
    continue
  data = json.loads(p.read_text())
  best = data.get("best") or {}
  metrics = best.get("metrics", {})
  print(f"{name}: page_recall@20={metrics.get('page_recall@k', 0):.4f} | page_hit@20={metrics.get('page_hit@k', 0):.4f}")

leak = base / "sweep_B_leakage_safe.json"
if leak.exists():
  data = json.loads(leak.read_text())
  groupings = data.get("groupings") or {}
  if groupings:
    for g in ["sample", "doc_name"]:
      grp = groupings.get(g) or {}
      hold = grp.get("holdout") or {}
      kf = grp.get("kfold") or {}
      if hold:
        tm = hold.get("test_metrics_for_best") or {}
        print(f"{g}.holdout_test: page_recall@20={tm.get('page_recall@k', 0):.4f} | page_hit@20={tm.get('page_hit@k', 0):.4f}")
      if kf:
        print(f"{g}.kfold_cv: recall_mean={kf.get('cv_mean_page_recall@k', 0):.4f}±{kf.get('cv_std_page_recall@k', 0):.4f} | hit_mean={kf.get('cv_mean_page_hit@k', 0):.4f}±{kf.get('cv_std_page_hit@k', 0):.4f}")
  else:
    hold = data.get("holdout") or {}
    kf = data.get("kfold") or {}
    if hold:
      tm = hold.get("test_metrics_for_best") or {}
      print(f"holdout_test: page_recall@20={tm.get('page_recall@k', 0):.4f} | page_hit@20={tm.get('page_hit@k', 0):.4f}")
    if kf:
      print(f"kfold_cv: recall_mean={kf.get('cv_mean_page_recall@k', 0):.4f}±{kf.get('cv_std_page_recall@k', 0):.4f} | hit_mean={kf.get('cv_mean_page_hit@k', 0):.4f}±{kf.get('cv_std_page_hit@k', 0):.4f}")

inf = base / "inference_unified_style_scored.json"
if inf.exists():
  data = json.loads(inf.read_text())
  agg = data.get("aggregate_stats", {})
  ret = (data.get("evaluation_summary") or {}).get("retrieval", {})
  gen = (data.get("evaluation_summary") or {}).get("generative", {})
  page_recall = (((ret.get("page_recall") or {}).get("20"))
                 if isinstance(ret.get("page_recall"), dict)
                 else None)
  print("inference_unified_style_scored.json:")
  print(f"  samples={agg.get('num_samples', 0)} avg_num_retrieved={agg.get('avg_num_retrieved', 0):.2f}")
  if page_recall is not None:
    print(f"  retrieval.page_recall@20={page_recall:.4f}")
  if gen:
    keys = [k for k in ["avg_bleu_4", "avg_rouge_l_f1", "avg_bertscore_f1"] if k in gen]
    if keys:
      print("  generative:", ", ".join(f"{k}={gen[k]:.4f}" for k in keys))

PY

echo "Done!"
