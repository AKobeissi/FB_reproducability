# RAG in Long-Document Financial Question Answering

Codebase for the thesis *Retrieval-Augmented Generation in Long-Document Financial Question Answering* (Amine Kobeissi, Université de Montréal, 2026).

This repository contains the full experimental pipeline used to evaluate retrieval strategies on the FinanceBench benchmark and a FinQA-503 benchmark extension, covering oracle-based failure decomposition, hierarchical page-then-chunk retrieval, visual page scoring, cross-encoder reranking, and LLM scaling.

---

## Research Overview

Financial QA is a particularly hard RAG setting: questions require precise numerical reasoning over long SEC filings (10-K, 10-Q, 8-K, earnings transcripts) spanning hundreds of pages. Evidence is often buried in dense tables, and structurally near-identical sections across companies and fiscal years make ranking difficult.

**Central finding:** The dominant bottleneck is *within-document page retrieval*, not document discovery. Systems frequently retrieve the correct filing while missing the evidence-bearing page.

This thesis studies retrieval as the central bottleneck through four contributions:

1. **Oracle-based evaluation framework** — Decomposes retrieval failure at document, page, and chunk levels. Oracle-document and oracle-page conditions establish empirical upper bounds and expose where performance gaps actually lie.
2. **Hierarchical page-then-chunk retrieval** — A two-stage pipeline where a fine-tuned BGE-M3 bi-encoder scores pages first (Stage 1), then standard chunk retrieval is applied only over the top-P pages (Stage 2). Also evaluates ColQwen2 as a visual alternative Stage-1 ranker.
3. **Fine-tuned cross-encoder reranker** — `bge-reranker-v2-m3` fine-tuned on FinQA financial page data and applied zero-shot to FinanceBench, isolating the effect of out-of-domain financial adaptation.
4. **Benchmark extension (FinQA-503)** — An alignment pipeline that maps 503 FinQA test questions to gold PDF pages from EDGAR, enabling a preliminary cross-dataset generalization test.

**Datasets:**
- [FinanceBench](https://huggingface.co/datasets/PatronusAI/financebench) — 150 expert-annotated QA pairs over 84 SEC filings (32 companies, FY2015–2024), with page-level evidence annotations
- **FinQA-503** — 503 FinQA test questions aligned to gold PDF pages in 115 annual 10-K filings from EDGAR

**Primary metric:** Page Recall@5 (`PageRec@5`) — fraction of queries for which the gold evidence page appears in the top-5 retrieved chunks.

---

## Repository Structure

```
.
├── src/                        # Core library code
│   ├── core/                   # RAG experiment orchestrator + mixins
│   ├── evaluation/             # Retrieval and generation evaluators
│   ├── experiments/            # Retrieval strategy implementations
│   ├── ingestion/              # PDF loading, chunking, page processing
│   ├── retrieval/              # BM25, vectorstore, OT reranker, late chunking
│   ├── training/               # Page scorer and bi-encoder training
│   └── utils/                  # Metadata helpers
│
├── baselines/                  # Experiment runner scripts and analysis utilities
│   ├── run_baselines.py        # Full retrieval baseline sweep
│   ├── run_benchmark_extension.py  # FinQA-503 evaluation
│   ├── run_ft_reranker_baselines.py  # Fine-tuned reranker evaluation
│   ├── llm_scaling_study.py    # LLM scaling (3B → 7B → 14B)
│   ├── chunking_study.py       # Chunking ablation
│   └── train_cross_encoder_reranker.py  # Cross-encoder fine-tuning on FinQA
│
├── scripts/                    # SLURM job scripts and standalone utilities
│   ├── run_baselines.sh        # Launch full baseline sweep
│   ├── run_benchmark_extension.sh  # Launch FinQA-503 evaluation
│   ├── run_chunking_sweep.sh   # Chunking ablation
│   ├── run_colpali_page.sh     # ColPali visual retrieval
│   ├── run_colpali_rerank.sh   # ColPali + reranking
│   ├── run_ft_reranker.sh      # Fine-tuned reranker pipeline
│   ├── run_hierarchical_rag.sh # Hierarchical page-then-chunk
│   ├── run_llm_scaling.sh      # LLM scaling study
│   └── ...                     # Other experiment entry points
│
├── analysis/                   # Post-hoc analysis scripts and outputs
│   ├── lexical_mismatch/       # Query–document term overlap analysis (FinanceBench)
│   ├── lexical_mismatch_benchmark_extension/  # Same, for FinQA-503
│   ├── query_difficulty/       # Difficulty regime analysis
│   └── page_stats/             # Page length distributions
│
├── data/                       # Dataset files
│   ├── financebench_open_source.jsonl
│   ├── finqa_test.jsonl
│   └── finqa_test_gold_pages.jsonl
│
├── docs/                       # Thesis document and supplementary references
│   ├── gabaraitmem.tex         # Thesis LaTeX source
│   ├── EVALUATION.md           # Evaluation pipeline reference
│   ├── LEARNED_SCORER_USAGE.md # Page scorer usage guide
│   └── LIGHTWEIGHT_CHECKPOINTS.md  # Checkpoint storage guide
│
├── runner.py                   # CLI for core RAG experiment modes
├── requirements.txt            # Python dependencies
└── .env.example                # Environment variable template
```

> **Runtime-generated directories** (gitignored): `outputs/`, `results/`, `vector_stores/`, `models/`, `checkpoints/`, `logs/`, `pdfs/`

---

## Setup

**Requirements:** Python 3.10+, NVIDIA GPU (≥24 GB VRAM recommended; experiments used RTX 3090).

```bash
# 1. Clone and create a virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Configure credentials
cp .env.example .env
# Fill in OPENAI_API_KEY (for HyDE generation and LLM-as-judge) and HF_TOKEN

# 4. HuggingFace login (for gated models)
huggingface-cli login
```

**PDF data:** Place FinanceBench PDF filings in `./pdfs/`. Filenames must loosely match the `doc_name` column in the dataset (the loader normalizes names). FinQA-503 PDFs (downloaded from EDGAR) go in `./pdfs-extended-v4/`.

**Vector store:** A ChromaDB collection is built once per (embedding model, chunk size, chunk overlap) configuration under `./vector_stores/` and reused across all runs sharing that configuration.

---

## Experiments

### Baseline Retrieval Sweep

All baselines operate over a shared BGE-M3 index (1024-token chunks, 128-token overlap, ChromaDB).

| Category | Method |
|---|---|
| Sparse | BM25, SPLADE |
| Dense | BGE-M3, all-mpnet-base-v2, FinLang/finance-investopedia |
| Hybrid | BM25 + BGE-M3 (RRF, 3 mixing ratios) |
| Query expansion | HyDE, Multi-HyDE (4 hypothetical passages, averaged embeddings) |
| Reranking | Zero-shot cross-encoder (`bge-reranker-v2-m3`), FT cross-encoder |
| Visual | ColPali, ColQwen2 (page-image late-interaction) |
| Oracle upper bounds | Oracle-document, Oracle-page |

```bash
# Run all baselines on FinanceBench
sbatch scripts/run_baselines.sh

# Or directly
python baselines/run_baselines.py --dataset financebench --output-dir outputs/baselines
```

### Oracle Analysis

Oracle-document and oracle-page conditions are built-in to all evaluations via ChromaDB metadata filtering. The oracle framework decomposes the retrieval performance gap into cross-document and within-document components.

### FinQA-503 Benchmark Extension

```bash
sbatch scripts/run_benchmark_extension.sh
# or: python baselines/run_benchmark_extension.py
```

### Hierarchical Page-then-Chunk Retrieval

Stage-1: fine-tuned BGE-M3 bi-encoder ranks all pages corpus-wide and selects top-P=20. Stage-2: standard BGE-M3 chunk retrieval over only those pages.

```bash
sbatch scripts/run_hierarchical_rag.sh
```

### Fine-Tuned Cross-Encoder Reranker

`bge-reranker-v2-m3` fine-tuned on FinQA page-level relevance data using binary cross-entropy loss, then evaluated zero-shot on FinanceBench.

```bash
# Train
python baselines/train_cross_encoder_reranker.py

# Evaluate
sbatch scripts/run_ft_reranker.sh
```

### Visual Page Retrieval (ColQwen2 / ColPali)

Retrieves PDF pages from page images using MaxSim over patch embeddings (ColBERT-style late interaction). No text extraction required.

```bash
sbatch scripts/run_colpali_page.sh     # ColPali
sbatch scripts/run_colpali_rerank.sh   # ColPali/ColQwen2 + reranking
```

### LLM Scaling Study

Evaluates generation quality with Qwen2.5 3B, 7B, and 14B under a fixed retrieval setup to separate retrieval from generation scaling effects.

```bash
sbatch scripts/run_llm_scaling.sh
```

### Chunking Ablation

Ablation over chunk sizes (256, 512, 1024 tokens) and overlap (0, 128), plus a parent-child (2048/512) variant.

```bash
sbatch scripts/run_chunking_sweep.sh
```

### Simple CLI (core RAG modes)

```bash
python runner.py [llama|qwen|both] [closed|single|random_single|shared|open] \
  --num-samples 50 \
  --pdf-dir ./pdfs \
  --output-dir ./outputs
```

---

## Evaluation

Experiments write JSON to `outputs/` (one file per run). Evaluation is a separate post-hoc step.

**Retrieval metrics** (PageRec@K, DocRec@K, MaxBLEU, MaxROUGE-L):
```bash
python src/evaluation/retrieval_evaluator.py --input outputs/<run>.json
```

**Generation metrics** (ROUGE-L, numeric match, BERTScore F1):
```bash
python src/evaluation/evaluate_outputs.py "outputs/*.json" \
  --output-dir outputs/scored
```

**LLM-as-judge** (GPT-4o, auxiliary comparison point only):
```bash
python src/evaluation/evaluate_outputs.py "outputs/*.json" \
  --judge-provider openai --judge-model gpt-4o \
  --output-dir outputs/scored
```

> Primary generation results in the thesis rely on **human evaluation** (binary correct/incorrect) and the deterministic metrics above. LLM-as-judge is used as a supplementary cross-check against published baselines.

**Aggregation**:
```bash
python scripts/aggregate_evaluation_results.py \
  --input-dir outputs/ --output-dir aggregated_results/
```

See `docs/EVALUATION.md` for full evaluation pipeline details.

---

## Analysis

| Script / Directory | Purpose |
|---|---|
| `analysis/lexical_mismatch/` | Query–document OOV rate, Jaccard, JSD by question/doc type |
| `analysis/query_difficulty/` | Difficulty regime analysis across retrieval signals |
| `analysis/page_stats/` | Page length distributions across the corpus |
| `scripts/geometric_analysis.py` | Intrinsic dimensionality and curvature of embedding space |
| `baselines/bootstrap_significance.py` | Pairwise bootstrap significance tests |

---

## Key Results (Summary)

- **Page retrieval is the dominant bottleneck.** Across all baselines, DocRec@5 is high (≥0.85) while PageRec@5 trails substantially, confirming the within-document gap.
- **Multi-HyDE + zero-shot reranker** is the strongest non-learned baseline on FinanceBench.
- **The FT cross-encoder reranker** improves PageRec@5 over the zero-shot reranker despite training only on FinQA data.
- **ColQwen2 visual retrieval** is competitive with text-based methods without requiring text extraction, demonstrating that visual layout carries retrieval signal in financial documents.
- **LLM scaling** from 3B to 7B improves answer quality substantially; gains from 7B to 14B are more modest and question-type dependent.
- **FinQA-503 generalizes differently.** BM25 is competitive on FinQA-503 (low lexical mismatch) but weak on FinanceBench (high mismatch), and no single method is uniformly best across both datasets.

---

## Citation

```bibtex
@mastersthesis{kobeissi2026rag,
  author  = {Amine Kobeissi},
  title   = {Retrieval-Augmented Generation in Long-Document Financial Question Answering},
  school  = {Université de Montréal},
  year    = {2026},
}
```

---

## License

MIT — see `LICENSE`.
