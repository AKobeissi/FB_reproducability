## Post-Run Evaluation

The `post_run_evaluator.py` script lets you audit one or more completed RAG experiment
outputs (`outputs/*.json`) without re-running the models. It enriches each record with
FinanceBench metadata and reports both retrieval and generation gaps.

### Prerequisites

- Install the project requirements (`pip install -r requirements.txt`).
- Ensure Hugging Face credentials are configured so `FinanceBenchLoader` can access
  the `PatronusAI/financebench` dataset.
- Place the JSON output files you want to inspect under `outputs/` (or pass explicit
  paths/globs).

### Basic usage

```bash
python3 post_run_evaluator.py \
  --outputs "outputs/single_vector_*.json" \
  --split train \
  --topk 1 3 5 \
  --chunk-match-threshold 0.45 \
  --low-rank-threshold 3 \
  --failure-threshold 0.5 \
  --use-bertscore
```

Key flags:

- `--outputs`: file paths or glob patterns; multiple values allowed.
- `--topk`: retrieval cutoffs for Hit/Recall (default: 1 3 5).
- `--chunk-match-threshold`: token-recall required for a retrieved chunk to count as
  containing the gold snippet (default 0.4).
- `--low-rank-threshold`: ranks greater than this are labeled
  `right_chunk_low_rank`.
- `--failure-threshold`: F1 threshold that separates “failed” vs “passed” answers.
- `--use-bertscore` / `--use-llm-judge`: opt-in semantic metrics; expect longer
  runtimes and extra model downloads.
- `--save-summary` / `--save-details`: override the default output locations.

Running the script produces two files under `outputs/post_eval/` (timestamps omitted
below for clarity):

| File | Description |
| --- | --- |
| `post_eval_summary_*.json` | Aggregated metrics across all inputs, including retrieval hit/recall/mrr, generation scores (EM/F1/BLEU/ROUGE/BERT), LLM-judge accuracy, and failure breakdowns. |
| `post_eval_details_*.json` | Per-sample diagnostics with inferred gold doc/pages, per-level retrieval metrics, generation metrics, and assigned failure reasons. |

You will also see a pretty-printed summary in the console for quick inspection.

### Interpreting the summary

- **Retrieval metrics** are grouped by `doc`, `page`, and `chunk`. Each group contains
  average `hit@k`, `recall@k`, and MRR. Use these to see how often the correct
  document/page/chunk surfaced in the top ranks.
- **Classification breakdown** highlights why retrieval failed:
  - `wrong_document`
  - `right_doc_wrong_page`
  - `right_page_wrong_chunk`
  - `right_chunk_low_rank`
  - `no_retrieval`
  - `retrieval_on_point`
- **Failure analysis** ties generation misses (F1 below the configured threshold or
  LLM judge marking answers incorrect) back to the retrieval classification so you can
  quantify questions like: “What % of failed answers stem from wrong documents vs
  wrong pages vs low-ranked chunks?”

To dive deeper into individual questions, open the details JSON and look at:

- `gold.doc_names` / `gold.pages`: the canonical ground-truth metadata sourced from
  FinanceBench.
- `retrieval.doc/page/chunk.matches`: which retrieved chunks counted as hits and at
  which ranks.
- `failure.retrieval_reason`: the category used in the aggregated breakdown.

This post-run pass is designed to be idempotent: you can re-run it whenever new
experiment outputs land, point it at subsets (e.g., only `open_book` runs), or
tighten/loosen thresholds to study different failure definitions.
