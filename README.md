# FinanceBench RAG Experiments

FinanceBench RAG Experiments is a reproducible playground for the **PatronusAI/FinanceBench** benchmark. It bundles dataset loading, PDF chunking, vector-store construction, and multiple retrieval-augmented generation (RAG) baselines that mirror the official evaluation tracks (closed-book, single-vector, shared-vector, and gold-evidence/open-book). The project can run fully offline with local HuggingFace weights (Llama 3.2 3B Instruct or Qwen 2.5 7B Instruct) or call an OpenAI-compatible API endpoint when GPU resources are limited.

---

## Key Capabilities

- **Multi-mode experiments** – Closed-book, per-document vector stores, shared stores, and oracle/gold-evidence modes live under a single `RAGExperiment` orchestrator with mixins for chunking, prompting, vector stores, and result persistence (`rag_experiments.py`, `rag_experiment_mixins.py`).
- **Robust dependency handling** – `rag_dependencies.py` centralizes LangChain/Chroma/FAISS imports and provides safe fallbacks so scripts work even if some extras are missing.
- **FinanceBench-native data access** – `data_loader.py` downloads the HuggingFace dataset, logs stats, and exposes helper getters for batches, samples, and document filters.
- **PDF utilities + vector stores** – `pdf_utils.py` locates filings from a local `pdfs/` cache, while `vectorstore.py` builds persistent Chroma stores (preferred) or FAISS indices.
- **Post-run analytics** – Experiments emit lean JSON under `outputs/`. Use `evaluator.py` in notebooks or `posthoc_evaluator.py` on the CLI to compute BLEU/ROUGE/BERTScore, LLM-as-judge signals, RAGAS, and retrieval diagnostics.

---

## Repository Layout

| Path | Description |
| --- | --- |
| `runner.py` | Friendly CLI wrapper that maps `model`/`experiment` keywords to `RAGExperiment` constants. |
| `rag_experiments.py` | Core orchestration plus CLI, invoking mode-specific runners in `rag_*.py`. |
| `rag_closed_book.py`, `rag_single_vector.py`, `rag_shared_vector.py`, `rag_open_book.py` | Implement the individual experiment strategies. |
| `rag_experiment_mixins.py` | Chunking, prompting, vector-store, component-tracking, and result helpers mixed into `RAGExperiment`. |
| `vectorstore.py`, `pdf_utils.py` | Shared utilities for document ingestion and retrieval backends. |
| `data_loader.py` | FinanceBench dataset ingestion via HuggingFace `datasets`. |
| `evaluator.py`, `posthoc_evaluator.py` | Optional scoring utilities for saved experiment outputs. |
| `outputs/`, `vector_stores/` | Default locations for JSON results and persisted Chroma indices (created at runtime). |
| `requirements.txt` | Consolidated dependency list (un-pinned). |

---

## Environment Setup

1. **Python**: Use Python 3.10+ (matching current PyTorch/Transformers requirements).
2. **Virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. **HuggingFace access**: Authenticate once (`huggingface-cli login`) so `transformers` can pull the Llama/Qwen checkpoints. If you only plan to use `--use-api`, supply an OpenAI-compatible API key instead.
5. **GPU & quantization**: A GPU with ≥16 GB is recommended for local inference. The runner loads models in 8-bit by default via bitsandbytes; fall back to full precision with `--no-8bit` / `--no-8bit` if quantization is unavailable (expect higher VRAM usage).

> **Tip:** Large dependencies such as `faiss-cpu`, `bitsandbytes`, or GPU-enabled `torch` may require system packages (e.g., `cuda`, `g++`). Install the appropriate wheels for your platform.
---

## Data & PDF Preparation

1. **FinanceBench QA pairs**: The `FinanceBenchLoader` automatically downloads `PatronusAI/financebench` through HuggingFace `datasets`. No manual steps are needed unless you want to cache the dataset offline.
2. **Underlying filings**: Place FinanceBench PDF files inside `./pdfs` (relative to the repo root) or point `--pdf-dir`/`pdf_local_dir` to a directory that mirrors the dataset’s `doc_name` values. The loader matches filenames loosely (case/spacing agnostic) via `_normalize_name` in `pdf_utils.py`, so exact names are not required.
3. **Vector-store cache**: Chroma indices are persisted under `./vector_stores/chroma/<doc_or_shared>` so repeated runs skip re-chunking once a document has been ingested.

---

## Running Experiments

### Quick CLI (`runner.py`)

```bash
python runner.py [llama|qwen|both] [closed|single|shared|open] \
  --num-samples 50 \
  --pdf-dir /path/to/pdfs \
  --vector-store-dir ./vector_stores \
  --output-dir ./outputs \
  [--no-8bit] [--use-api --api-base-url https://router.huggingface.co --api-key-env HF_TOKEN] \
  [--llm-model gpt-4o-mini]
```

Key behavior:
- Omit positional args to run `both` models in `closed` mode.
- `--num-samples` limits the dataset slice without editing code.
- `--use-api` routes generation through an OpenAI-compatible endpoint (set your key in `--api-key-env`; default `OPENAI_API_KEY`).
- `--llm-model` overrides the built-in HF IDs when calling an API backend.
- Logs stream to stdout and `logs/<experiment>_<timestamp>.log`.

### Direct orchestrator (`rag_experiments.py`)

For finer control (chunk size, overlap, top-k, etc.), call the orchestrator CLI:

```bash
python rag_experiments.py llama --experiment single --chunk-size 768 \
  --chunk-overlap 80 --top-k 8 --num-samples 100 \
  --pdf-dir ./pdfs --vector-store-dir ./vector_stores --output-dir ./outputs
```

You can also import `RAGExperiment` from Python to embed the workflow in notebooks or other scripts.

---

## Experiment Types

| Mode | Description | Retrieval Backend |
| --- | --- | --- |
| `closed` | Pure generation baseline without context. | None |
| `single` | Builds a dedicated Chroma store per document and retrieves only from that filing. | Chroma (per-doc) |
| `shared` | Ingests all PDFs into a single shared Chroma store so cross-document evidence is possible. | Chroma (global) |
| `open` | Feeds the gold evidence segments directly to the generator (oracle upper bound). | Gold evidence only |

All retrieval modes rely on LangChain embeddings (`all-mpnet-base-v2` by default), `RecursiveCharacterTextSplitter`, and either LangChain retrieval chains or a manual fallback if advanced utilities are missing.

---

## Outputs & Persistence

- Each run saves `outputs/<experiment>_<timestamp>.json` containing `metadata` + a `results` array (`question`, `reference_answer`, `generated_answer`, `retrieved_chunks`, `gold_evidence_segments`, `final_prompt`, length stats, etc.).
- Chroma stores are persisted under `vector_stores/chroma/` and reused automatically. Delete the directory to force a rebuild.
- If a PDF is missing, the corresponding samples are marked with `skipped_reason` in the output so you can diagnose data coverage.

---

## Post-Hoc Evaluation

1. **Notebook/script workflow** – Import `Evaluator` and iterate over saved JSON to compute BLEU/ROUGE/BERTScore, RAGAS, and optional LLM-as-judge metrics:
   ```python
   from evaluator import Evaluator
   import json

   data = json.load(open("outputs/single_vector_20251118_174111.json"))
   samples = data["results"]

   ev = Evaluator(use_llm_judge=False)
   for sample in samples[:10]:
       metrics = ev.evaluate_generation(
           prediction=sample["generated_answer"],
           reference=sample["reference_answer"],
           question=sample["question"],
           contexts=[chunk["text"] for chunk in sample.get("retrieved_chunks", [])],
           gold_contexts=[seg["text"] for seg in sample.get("gold_evidence_segments", [])],
       )
   ```
2. **CLI summary** – Aggregate retrieval/generation diagnostics from any JSON file:
   ```bash
   python posthoc_evaluator.py --input outputs/single_vector_20251118_174111.json \
     --save-report outputs/single_vector_summary.json
   ```
   The script prints per-metric means/stds and highlights the top retrieval failure modes (e.g., `no_pdf_text`, `doc_match_page_miss`).

---

## Troubleshooting & Tips

- **Out-of-memory errors**: Enable 8-bit loading (default) or lower `--chunk-size`/`--top-k` to shrink prompt context. When 8-bit is unavailable, expect ~2× VRAM usage.
- **API runs**: Set `OPENAI_API_KEY` (or your chosen `--api-key-env`) and optionally override `--llm-model` to match the remote model name. Retrieval still happens locally; only generation moves to the API.
- **Missing PDFs**: Check the console warnings from `load_pdf_with_fallback`. Ensure filenames roughly match the `doc_name` column or pass a `--pdf-dir` with cleaner naming.
- **Vector store rebuilds**: Delete a subdirectory under `vector_stores/chroma/` to force re-chunking for a specific document, or pass a new `--vector-store-dir` to keep experiments isolated.
- **Logging**: Detailed logs (including chunk statistics and component summaries) are stored under `logs/` per run for reproducibility audits.

---

## Extending the Project

- **Add new generators**: Update `RAGExperiment` with additional model constants or pass a full HuggingFace model ID / API name via the CLI.
- **Swap embeddings or retrievers**: Override `embedding_model`, `chunk_size`, `chunk_overlap`, or implement a custom mixin method to introduce new preprocessing strategies.
- **Custom evaluation**: Consume the saved JSON in notebooks to train classifiers, compute bespoke KPIs, or feed `outputs/` into the provided evaluator modules.

Pull requests welcome! Open an issue if you run into missing dependencies or want to discuss extending the evaluation pipeline.
Footer
