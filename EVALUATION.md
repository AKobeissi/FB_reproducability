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

## Post-Hoc Evaluation (work in progress)

- `evaluator.py` still provides BLEU/ROUGE/BERTScore/RAGAS utilities, but they now need to be invoked separately (e.g., via a future `evaluate_outputs.py` script).
- `posthoc_evaluator.py` currently expects the legacy schema with precomputed metrics; it will be updated alongside the new standalone evaluator.

Until the dedicated evaluation CLI lands, you can import `Evaluator` in a notebook or script, iterate over the saved JSON, and score the fields you care about without rerunning the expensive retrieval/generation stages.

## Standalone Evaluation CLI (BERTScore + HF judge)

`evaluate_outputs.py` is a thin wrapper around `Evaluator` that loads any JSON in `outputs/`, computes BLEU/ROUGE/BERTScore as well as HuggingFace-based LLM-judge scores, and writes an updated file (optionally in-place).

Example (scores every JSON file and stores annotated copies under `outputs/scored/`):

```bash
python evaluate_outputs.py "outputs/*.json" \
  --judge-model meta-llama/Meta-Llama-3-8B-Instruct \
  --device-map auto \
  --retrieval-top-k 5 \
  --output-dir outputs/scored
```

Key behavior:

- Uses BERTScore by default (requires `bert-score` weights, downloaded automatically).
- Spins up a local HuggingFace `text-generation` pipeline for the judge—no OpenAI key is needed. Pass any chat instruct model you have locally (Qwen, Llama, etc.) via `--judge-model`.
- Adds full per-sample metrics under `generation_evaluation` (and optional `retrieval_evaluation`) plus an aggregated `evaluation_summary` block at the top of the JSON.
- Supports `--overwrite` if you want to annotate files in place, or `--suffix _scored` (default) to keep originals untouched.
- Prints a one-line dashboard per file (BLEU-4, ROUGE-L, BERTScore F1, judge accuracy) so you can monitor progress in long batches.
