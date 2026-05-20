# Evidential Cross-Encoder Reranker for Financial RAG

**Knowing When Evidence Is Weak: Evidential Cross-Encoder Reranking for Financial RAG**

## Project Goal

Train and evaluate an uncertainty-aware cross-encoder reranker for financial question answering over SEC-style documents. The system outputs both a relevance score and an **uncertainty/vacuity** estimate using Evidential Deep Learning (EDL), enabling:

1. Better ranking via uncertainty-penalized scores
2. Detection of retrieval failures before they contaminate answer generation
3. Selective answering (abstain when evidence is weak)

## Quick Start

```bash
# All commands run from UQ_EDL/
source ../venv/bin/activate
export HF_HOME=/data/rech/kobeissa/hf

# 1. Data preparation (already done — outputs in data/processed/)
bash scripts/01_prepare_finqa.sh
bash scripts/02_prepare_financebench.sh
bash scripts/03b_build_pairs.sh

# 2. Training (GPU required)
bash scripts/run_training.sh
# or submit as SLURM: sbatch scripts/run_training.sh

# 3. Evaluation + paper outputs
bash scripts/run_eval_and_analysis.sh [ce_run_dir] [edl_run_dir]
```

## Data Status

| Dataset      | Pages   | Gold pages | Questions | Status |
|--------------|---------|------------|-----------|--------|
| FinQA train  | ~87,085 | ~68        | 355       | ✓ Done |
| FinQA val    | ~28,343 | 71         | 108       | ✓ Done |
| FinQA test   | ~9,690  | 45         | 67        | ✓ Done |
| FinanceBench | 11,921  | 163 (420 labeled) | 150 | ✓ Done |

Evidence alignments: **189/189 FinanceBench evidences aligned** (100% success rate).

## Directory Structure

```
UQ_EDL/
  configs/          YAML hyperparameter configs
  data/processed/   Extracted pages, splits, pairs
    pages/          finqa_pages.jsonl, financebench_pages.jsonl
    splits/         finqa_{train,val,test}.jsonl
    pairs/          *_pairs.jsonl, hard negatives
  src/
    data/           Page extraction, splits, pair construction, hard negatives
    retrieval/      BM25, BGE-M3 dense, RRF
    models/         CrossEncoder, EvidentialCrossEncoder, losses, calibration
    training/       Dataset, train_cross_encoder, train_evidential
    evaluation/     Ranking metrics, uncertainty metrics, selective eval
    analysis/       Tables, plots, error analysis, paper summaries
    utils/          IO, logging, seed, text utilities
  scripts/          Shell scripts for each pipeline stage
  results/          Run outputs, tables, plots, summaries, qualitative
  paper/            Draft paper sections
```

## Model Architecture

### Standard Cross-Encoder
```
[CLS] <merged question + metadata + page text> → XLM-RoBERTa → Linear(1024→2) → softmax
```

### Evidential Cross-Encoder
```
[CLS] <merged input> → XLM-RoBERTa → Linear(1024→2) → softplus → evidence
evidence + 1 = alpha (Dirichlet params)
p_relevant = alpha[1] / sum(alpha)
uncertainty = 2 / sum(alpha)     ← epistemic vacuity
score = p_relevant - beta * uncertainty
```

### EDL Loss (Sensoy et al. 2018)
```
L = MSE(y, p) + var(p) + annealing_coef * λ_KL * KL(Dir(α̃) || Dir(1,...,1))
```

KL is annealed over the first 20% of training steps to allow the model to learn before regularization kicks in.

## Key Experiments

| # | Experiment | Tag |
|---|------------|-----|
| 1 | CE on FinQA test | `ce_finqa_test` |
| 2 | EDL on FinQA test | `edl_finqa_test` |
| 3 | CE on FinanceBench | `ce_financebench` |
| 4 | EDL on FinanceBench (β=0) | `edl_financebench` |
| 5 | EDL on FinanceBench (β=0.25) | `edl_fb_beta025` |
| 6 | Beta ablation (0.0–1.0) | `edl_fb_beta*` |

## Backbone

`BAAI/bge-reranker-v2-m3` (XLM-RoBERTa, 568M params) — already a fine-tuned reranker, ideal for financial domain transfer.

## Hard Negatives (Finance-Aware)

| Type | Count |
|------|-------|
| same_doc_wrong_page | 551 |
| same_company_wrong_year | 596 |
| wrong_company_same_year | 710 |
| boilerplate_same_doc | 416 |
| **Total** | **2,273** |

## Output Files

After evaluation, find results in:
- `results/tables/` — CSV + LaTeX tables (main results, calibration, selective recall)
- `results/plots/` — reliability diagrams, risk-coverage curves, uncertainty distributions
- `results/qualitative/` — error analysis (30+ annotated failures)
- `results/paper_summaries/` — auto-generated paper paragraphs per experiment
