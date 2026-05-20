# RAG in Financial QA: A Systematic Study

Codebase for the thesis *Retrieval-Augmented Generation in Financial QA: A Systematic Study* (Amine Kobeissi, Université de Montréal, 2026).

This repository contains the full experimental pipeline used to evaluate 16+ retrieval strategies on the FinanceBench benchmark, with a benchmark extension to FinQA, and a series of studies on chunking strategies, modality fusion, visual retrieval, and LLM scaling for financial question answering.

---

## Research Overview

Financial QA is a particularly hard RAG setting: questions require precise numerical reasoning over long, multi-section SEC filings (10-K, 10-Q, earnings releases), where lexical mismatch between user queries and document text is high and evidence is often buried in tables or footnotes.

This work conducts a **systematic comparison** of retrieval strategies across two axes:

1. **Retrieval quality** — how well does the retriever surface the exact evidence page?
2. **Answer quality** — does better retrieval translate to better generated answers?

**Key benchmarks:**
- [FinanceBench](https://huggingface.co/datasets/PatronusAI/financebench) — 150 expert-annotated QA pairs over SEC filings
- **Benchmark extension** — 680 questions combining FinanceBench + FinQA, enabling cross-dataset generalization evaluation

**Primary metric:** Page Recall@K (`PageRec@K`) — the fraction of queries where the gold evidence page is retrieved in the top-K results.

---

## Repository Structure

```
.
├── src/                        # Core library code
│   ├── core/                   # RAG experiment orchestrator + mixins
│   ├── evaluation/             # Retrieval and generation evaluators
│   ├── experiments/            # Individual retrieval strategy implementations
│   ├── ingestion/              # PDF loading, chunking, page processing
│   ├── retrieval/              # BM25, vectorstore, OT reranker, late chunking
│   ├── training/               # Page scorer and bi-encoder training
│   └── utils/                  # Metadata helpers
│
├── baselines/                  # Baseline runner scripts and analysis utilities
│   ├── run_baselines.py        # Main baseline evaluation sweep
│   ├── run_benchmark_extension.py  # FinanceBench + FinQA combined evaluation
│   ├── run_ft_reranker_baselines.py  # Fine-tuned reranker evaluation
│   ├── llm_scaling_study.py    # LLM scaling (7B → 14B → 72B)
│   ├── chunking_study.py       # Chunking strategy comparison
│   └── train_cross_encoder_reranker.py  # Cross-encoder fine-tuning
│
├── scripts/                    # SLURM job scripts and standalone utilities
│   ├── run_baselines.sh        # Launch full baseline sweep
│   ├── run_benchmark_extension.sh  # Launch benchmark extension
│   ├── run_chunking_sweep.sh   # Launch chunking strategy study
│   ├── run_colpali_page.sh     # ColPali visual retrieval
│   ├── run_colpali_rerank.sh   # ColPali + cross-encoder reranking
│   ├── run_ft_reranker.sh      # Fine-tuned reranker pipeline
│   ├── run_hierarchical_rag.sh # Hierarchical RAG
│   ├── run_modality_fusion_rag.sh  # NER-based document filtering
│   ├── run_llm_scaling.sh      # LLM scaling study
│   └── ...                     # Other experiment entry points
│
├── analysis/                   # Post-hoc analysis scripts and outputs
│   ├── lexical_mismatch/       # Query–document term overlap analysis
│   ├── lexical_mismatch_benchmark_extension/
│   ├── query_difficulty/       # Difficulty regime analysis
│   └── page_stats/             # Page length and document statistics
│
├── data/                       # Dataset files (FinanceBench, FinQA)
│   ├── financebench_open_source.jsonl
│   ├── finqa_test.jsonl
│   └── finqa_test_gold_pages.jsonl
│
├── docs/                       # Thesis document and supplementary docs
│   ├── gabaraitmem.tex         # Thesis LaTeX source
│   ├── EVALUATION.md           # Evaluation pipeline reference
│   ├── LEARNED_SCORER_USAGE.md # Page scorer usage guide
│   └── LIGHTWEIGHT_CHECKPOINTS.md  # Checkpoint storage guide
│
├── runner.py                   # Friendly CLI for core RAG experiments
├── requirements.txt            # Python dependencies
└── .env.example                # Environment variable template
```

> **Runtime-generated directories** (gitignored): `outputs/`, `results/`, `vector_stores/`, `models/`, `checkpoints/`, `logs/`, `pdfs/`

---

## Setup

**Requirements:** Python 3.10+, CUDA-capable GPU (≥16 GB VRAM recommended for local inference).

```bash
# 1. Clone and create a virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Configure credentials
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY and/or HF_TOKEN

# 4. Authenticate with HuggingFace (for gated models like Llama)
huggingface-cli login
```

**PDF data:** Place FinanceBench PDF filings in `./pdfs/`. Filenames should loosely match the `doc_name` values in the dataset (the loader normalizes names automatically). FinQA PDFs go in `./pdfs-extended-v4/`.

---

## Experiments

### Baseline Sweep (16+ Retrieval Strategies)

The core comparison covers the following retrieval configurations:

| Category | Method |
|---|---|
| Lexical | BM25 |
| Dense | BGE-M3, all-mpnet-base-v2 |
| Hybrid | BM25 + BGE-M3 (RRF fusion) |
| Query expansion | HyDE, Multi-HyDE |
| Reranking | Cross-encoder (off-the-shelf), FT cross-encoder |
| Visual | ColPali, ColQwen2 |
| Oracle upper bounds | Oracle document, Oracle page |

```bash
# Run all baselines on FinanceBench
sbatch scripts/run_baselines.sh

# Or run directly
python baselines/run_baselines.py --dataset financebench --output-dir outputs/baselines
```

### Benchmark Extension (FinanceBench + FinQA, 680 questions)

```bash
sbatch scripts/run_benchmark_extension.sh
# or: python baselines/run_benchmark_extension.py
```

### Chunking Strategy Study

Compares 10 chunking strategies: fixed-size (256/512/1024 tokens), semantic, sentence-window, page-level, late chunking, and big2small retrieval.

```bash
sbatch scripts/run_chunking_sweep.sh
```

### Fine-Tuned Cross-Encoder Reranker

Trains a cross-encoder on FinanceBench page–query relevance pairs, then evaluates in the full retrieval pipeline.

```bash
# Train
python baselines/train_cross_encoder_reranker.py

# Evaluate
sbatch scripts/run_ft_reranker.sh
```

### Modality-Fusion RAG (NER Document Filtering)

Uses named entity recognition to filter the document index before retrieval, achieving 91.3% document recall @k=3 without oracle access.

```bash
sbatch scripts/run_modality_fusion_rag.sh
```

### Visual Retrieval (ColPali / ColQwen2)

Retrieves PDF pages directly from page images using multi-vector visual embeddings.

```bash
sbatch scripts/run_colpali_page.sh     # ColPali retrieval
sbatch scripts/run_colpali_rerank.sh   # ColPali + cross-encoder reranking
```

### LLM Scaling Study

Evaluates generation quality across model sizes (Qwen2.5 7B → 14B → 72B) on the top retrieval pipeline.

```bash
sbatch scripts/run_llm_scaling.sh
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

Experiments write lean JSON to `outputs/` (one file per run). Post-hoc evaluation is separate from the retrieval loop.

**Retrieval metrics** (PageRec@K, DocRec@K, MRR):
```bash
python src/evaluation/retrieval_evaluator.py --input outputs/<run>.json
```

**Generation metrics** (BLEU, ROUGE-L, BERTScore, LLM-as-judge):
```bash
python src/evaluation/evaluate_outputs.py "outputs/*.json" \
  --judge-provider openai \
  --judge-model gpt-4o-mini \
  --output-dir outputs/scored
```

**Aggregation** (multi-run comparison tables):
```bash
python scripts/aggregate_evaluation_results.py --input-dir outputs/ --output-dir aggregated_results/
```

See `docs/EVALUATION.md` for the full evaluation reference.

---

## Analysis

Post-experiment analysis scripts live under `analysis/` and `scripts/`:

- **Lexical mismatch** (`analysis/lexical_mismatch/`): query–document term overlap broken down by question type and document type.
- **Query difficulty** (`analysis/query_difficulty/`): regime analysis (easy/medium/hard) based on retrieval signals.
- **Page statistics** (`analysis/page_stats/`): document length distributions across the corpus.
- **Geometric analysis** (`scripts/geometric_analysis.py`): local intrinsic dimensionality and curvature of the embedding space.
- **Bootstrap significance** (`baselines/bootstrap_significance.py`): pairwise statistical significance tests across baselines.

---

## Key Findings (Summary)

- **HyDE and Multi-HyDE** consistently outperform vanilla dense retrieval on FinanceBench, closing ~30% of the gap to the oracle upper bound.
- **Fine-tuned cross-encoder reranking** on top of Multi-HyDE provides the best overall PageRec@5 on FinanceBench.
- **ColPali visual retrieval** is competitive without any text extraction, suggesting visual layout carries retrieval signal.
- **NER document filtering** substantially reduces the retrieval search space with minimal recall loss.
- **LLM scaling** improves answer quality significantly from 7B to 14B but shows diminishing returns to 72B, especially when retrieval quality is the bottleneck.
- **FinQA generalizes poorly** from FinanceBench-tuned systems, motivating the benchmark extension.

---

## Citation

```bibtex
@mastersthesis{kobeissi2026rag,
  author  = {Amine Kobeissi},
  title   = {Retrieval-Augmented Generation in Financial QA: A Systematic Study},
  school  = {Université de Montréal},
  year    = {2026},
}
```

---

## License

MIT — see `LICENSE`.
