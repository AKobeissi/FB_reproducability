## Evaluation Pipeline Overview

The evaluation stack now captures end‑to‑end quality signals for every FinanceBench sample, combining lexical, semantic, retrieval, and judge-based metrics. Each run emits enriched JSON rows under `results` with the following additions:

- `gold_evidence_segments`: structured list of `{text, doc_name, page}` entries for downstream analysis.
- `final_prompt`: exact FinanceBench-style prompt (question + context) that fed the generator or API.
- `retrieval_evaluation`: extended diagnostics including `doc/page/chunk_hit_at_k`, `*_recall_at_k`, `failure_reason`, and legacy overlap statistics.
- `generation_evaluation.ragas`: per-sample RAGAS scores when a LangChain-compatible LLM is available.

## Metrics

| Category | Details |
| --- | --- |
| Statistical | BLEU‑1…4 (SacreBLEU fallback), ROUGE‑1/2/L, optional BERTScore. |
| RAGAS | Faithfulness, Answer Relevancy, Context Precision, Context Recall via `ragas>=0.3.9` and LangChain `HuggingFacePipeline`. |
| Retrieval | Doc/Page/Chunk Hit@K, Recall@K, max/mean token overlap, failure mode histogram (`no_doc_match`, `doc_match_page_miss`, `page_match_chunk_miss`, `chunk_match`). |
| Judge | LlamaIndex CorrectnessEvaluator system/user prompt rendered for the local HuggingFace Mistral‑7B pipeline. First line = 1‑5 score, second line = rationale, with `correct` defined as score ≥ 4. |

## Dependencies & Configuration

- `requirements.txt` now includes `ragas>=0.3.9` and `llama-index-core>=0.14.0`. Install extras before running `runner.py`.
- `RAGExperiment` wires the evaluator with your embedding model/device and supplies the LangChain HF pipeline for both retrieval chains and RAGAS scoring.
- The judge model defaults to `mistralai/Mistral-7B-Instruct-v0.3`. Set `use_llm_judge=True` when instantiating `RAGExperiment` to enable it; otherwise the pipeline is skipped.

## Output Consumption Tips

- Use `results[*].final_prompt` to reproduce user/context payloads that produced each answer (both local weights and API mode).
- `gold_evidence_segments` carries the `evidence_page_num` metadata so you can attribute misses to document or pagination drift.
- Retrieval miss attribution is available via `retrieval_evaluation.failure_reason`. Combine with `doc_hit_rank`/`page_hit_rank` to see how deep the relevant content appeared.
- RAGAS metrics appear under `generation_evaluation.ragas`. Aggregated summaries are logged (mean/std) whenever the metrics are active.

## Post-Hoc CLI

You can aggregate metrics from any saved JSON (no rerun required):

```bash
python posthoc_evaluator.py --input outputs/single_vector_20251118_174111.json \
    --save-report reports/single_vector_summary.json
```

The CLI reads the per-sample `generation_evaluation` / `retrieval_evaluation` blocks already stored in the results file and prints mean/std/range summaries, length stats, and retrieval failure mode histograms. No additional configuration is needed, and the optional `--save-report` flag writes the aggregated view as JSON for dashboards.

## Extending / Tuning

- To disable RAGAS (e.g., API-only runs), instantiate `Evaluator(use_ragas=False, ...)`.
- To override the embedding model for RAGAS comparisons, pass `ragas_embedding_model` when constructing `Evaluator` or call `configure_ragas`.
- The LLM judge uses the LlamaIndex prompt verbatim. If you swap in a different model, ensure it can follow `[INST]` formatting or adjust `_create_judge_prompt`.
