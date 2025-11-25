## Evaluation Pipeline Overview

Experiments no longer compute BLEU/ROUGE/RAGAS metrics inline. Each `run_experiment` call now focuses solely on retrieval + generation so that we can run heavy evaluation passes separately (and repeatably) on the saved outputs.

Every row under `results` still contains:

- `gold_evidence_segments`: structured `{text, doc_name, page}` entries for the reference evidence.
- `retrieved_chunks`: the raw text + metadata for each chunk returned by the retriever (doc name, link, page number, etc.).
- `final_prompt`: the exact FinanceBench-style prompt that was fed to the generator (question + context block).
- Standard sample metadata: `doc_name`, `doc_link`, `question`, `reference_answer`, lengths, and experiment type.

There are **no** `generation_evaluation` or `retrieval_evaluation` blocks in the JSON anymore—those scores will be added later by the standalone evaluator.

## Running Experiments

Run experiments exactly as before, e.g.:

```bash
python rag_experiments.py llama --experiment single --num-samples 150
```

The resulting file in `outputs/` is slimmer (generation + metadata only) which keeps long runs fast and removes heavyweight LLM-as-judge dependencies from the main loop.

## Post-Hoc Evaluation

- `evaluator.py` exposes BLEU/ROUGE/BERTScore/RAGAS utilities that you can embed directly in notebooks or scripts.
- `evaluate_outputs.py` is the preferred CLI for batch-scoring the lean JSON artifacts written by `rag_experiments.py`.
- `posthoc_evaluator.py` supports the older schema with inline metrics; keep using it only if you rely on legacy files.

## Standalone Evaluation CLI (BERTScore + Judge + RAGAS)

`evaluate_outputs.py` loads any JSON in `outputs/`, computes BLEU/ROUGE/BERTScore, spins up an LLM-as-judge (HuggingFace weights or an OpenAI API model), optionally runs RAGAS holistic metrics, and writes an updated file (optionally in-place).

Example (score everything under `outputs/` and save annotated copies under `outputs/scored/`):

```bash
python evaluate_outputs.py "outputs/*.json" \
  --judge-model meta-llama/Meta-Llama-3-8B-Instruct \
  --judge-provider huggingface \
  --device-map auto \
  --retrieval-top-k 5 \
  --output-dir outputs/scored \
  --ragas-llm-provider auto
```

Or use GPT-4o-mini (OpenAI API) for both the judge and RAGAS LLM:

```bash
python evaluate_outputs.py "outputs/*.json" \
  --judge-provider openai \
  --judge-model gpt-4o-mini \
  --openai-api-key-env OPENAI_API_KEY \
  --ragas-llm-provider openai
```

Key behavior:

- Uses BERTScore by default (requires `bert-score` weights, downloaded automatically).
- Selects either a local HuggingFace `text-generation` pipeline or an OpenAI-compatible chat model for the judge (`--judge-provider {huggingface,openai}`).
- Adds full per-sample metrics under `generation_evaluation` (BLEU/ROUGE/BERTScore, judge verdict, optional RAGAS) plus an aggregated `evaluation_summary`, and fills `retrieval_evaluation` when `--retrieval-top-k` is provided.
- RAGAS is enabled whenever the dependency is installed. Disable with `--skip-ragas`, override embeddings via `--ragas-embedding-model`, or point to a specific LangChain LLM via `--ragas-llm-provider/--ragas-llm-model`.
- Supports `--overwrite` if you want to annotate files in place, or `--suffix _scored` (default) to keep originals untouched.
- Prints a one-line dashboard per file (BLEU-4, ROUGE-L, BERTScore F1, judge accuracy, etc.) so you can monitor progress in long batches.
