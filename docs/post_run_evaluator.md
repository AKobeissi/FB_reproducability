## Post-Run Evaluation

## Post-Run Evaluation

The `post_run_evaluator.py` script turns raw experiment outputs (which now only contain
questions, reference answers, gold evidence, generated answers, and retrieved chunks) into
fully scored evaluations. It enriches each record with FinanceBench metadata and reports
both retrieval and generation gaps.

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

Running the script produces two files under `outputs/evaluations/` (timestamps omitted
below for clarity):

| File | Description |
| --- | --- |
| `evaluation_summary_*.json` | Aggregated metrics across all inputs, including retrieval hit/recall/mrr, generation scores (EM/F1/BLEU/ROUGE/BERT), LLM-judge accuracy, and failure breakdowns. |
| `evaluation_details_*.json` | Per-sample diagnostics that retain the generated/reference answers, structured gold evidence (with doc + page info), retrieved chunks (with metadata), plus all computed metrics and failure tags. |

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
- `gold_evidence`: structured entries (text, doc name, page number, source) that the raw
  experiment emitted.
- `retrieved_chunks`: every retrieved chunk with its metadata (doc, page, score, rank).
- `retrieval.doc/page/chunk.matches`: which retrieved chunks counted as hits and at
  which ranks.
- `failure.retrieval_reason`: the category used in the aggregated breakdown.

This post-run pass is designed to be idempotent: you can re-run it whenever new experiment
outputs land, point it at subsets (e.g., only `open_book` runs), or tighten/loosen
thresholds to study different failure definitions. The raw experiment JSONs remain untouched;
all scoring artifacts live under `outputs/evaluations/`.
