# instructions.md

# Project: Finance-Aware Evidential Cross-Encoder Reranker for Selective Financial RAG

## 0. Project Goal

Build, train, and evaluate an uncertainty-aware cross-encoder reranker for financial question answering over long SEC-style documents.

The core idea is to replace a standard cross-encoder relevance head with an Evidential Deep Learning (EDL) head that outputs both:

1. a relevance estimate for each candidate page/chunk, and
2. an uncertainty/evidence estimate indicating how reliable the relevance judgment is.

The project should be designed as a research-grade implementation that can support a paper. Code must be modular, reproducible, and produce saved tables, plots, logs, and paper-ready result summaries.

---

## 1. Problem Formulation

### 1.1 Motivation

Financial RAG systems often fail not because the generator is incapable, but because the retrieved evidence is incomplete, ambiguous, or subtly wrong.

Common financial retrieval errors include:

- retrieving the correct company but wrong fiscal year,
- retrieving the correct metric name but wrong table,
- retrieving a relevant-looking page that does not contain the answer,
- retrieving a page with stale or duplicate values,
- retrieving boilerplate text instead of answer-bearing evidence,
- ranking a semantically similar but numerically unsupported page above the gold evidence page.

A standard reranker outputs a scalar relevance score. This is not enough for financial QA because two candidates can have similar relevance scores but very different uncertainty. The system should know when it has strong evidence and when it is guessing.

### 1.2 Main Research Question

Can an evidential cross-encoder reranker improve financial RAG reliability by estimating both passage relevance and uncertainty?

More specifically:

1. Does an evidential reranker improve ranking quality over a standard cross-encoder?
2. Does the uncertainty estimate predict retrieval failure?
3. Can uncertainty-aware ranking reduce confidently wrong evidence selection?
4. Can passage-level uncertainty be aggregated into question-level answerability?
5. Does training on FinQA-style financial evidence transfer to FinanceBench?

### 1.3 Proposed Contribution

The contribution should not be framed as simply "EDL for reranking", because generic evidential reranking already exists.

The stronger contribution is:

> A finance-aware evidential cross-encoder reranker for selective financial RAG, trained with financial hard negatives and evaluated not only on retrieval quality but also on calibration, uncertainty, failure detection, and selective answering.

The novelty should come from:

- applying evidential reranking to financial QA,
- using finance-specific hard negatives,
- evaluating uncertainty as a retrieval-failure signal,
- testing cross-dataset transfer from FinQA-derived supervision to FinanceBench,
- connecting passage uncertainty to question-level answerability or abstention.

---

## 2. Dataset Plan

Use two main datasets:

1. FinQA-derived page/chunk supervision for training and validation.
2. FinanceBench for out-of-domain evaluation.

The user already has:

- FinQA test data,
- PDFs corresponding to many FinQA examples,
- page-level alignment between FinQA evidence and PDFs for a subset.

Even though FinQA "test" is originally a benchmark test split, it can be used as training data for this project if the external evaluation is on FinanceBench and this choice is stated transparently. Do not report results on FinQA as if it were a held-out official test benchmark if it was used for training.

### 2.1 Recommended Split

Use FinQA-aligned examples as development data:

- Train: 70 percent of aligned FinQA questions.
- Validation: 15 percent.
- Internal test: 15 percent.

Split by company/ticker or document when possible, not randomly by question, to reduce leakage.

Use FinanceBench as external test:

- Do not train on FinanceBench.
- Use FinanceBench only for final evaluation and error analysis.
- If hyperparameter tuning on FinanceBench is necessary, create a small dev split and a final untouched split, but the preferred setup is no tuning on FinanceBench.

### 2.2 Unit of Retrieval

Primary unit: page-level reranking.

Each candidate should be a page from a PDF, optionally with metadata.

For each example:

```json
{
  "qid": "unique_question_id",
  "question": "What was the company's revenue in 2019?",
  "doc_id": "company_form_year",
  "candidate_id": "doc_id_page_42",
  "candidate_text": "text extracted from page 42",
  "page_number": 42,
  "metadata": {
    "company": "...",
    "ticker": "...",
    "form_type": "10-K",
    "fiscal_year": "2019",
    "filing_date": "...",
    "section": "..."
  },
  "label": 1
}
```

Use chunk-level reranking as an optional extension after page-level results are working.

### 2.3 Positive Labels

A candidate page is positive if it corresponds to a gold evidence page.

For FinQA:

- Use the existing evidence-to-PDF alignment.
- If gold evidence is mapped to page `p`, then page `p` is positive.
- If evidence spans multiple pages, all mapped pages are positive.
- If the mapping confidence is low, either exclude the example or label it as weak supervision.

For FinanceBench:

- Use annotated evidence pages if available.
- If only evidence text is available, align evidence text to PDF pages using fuzzy matching or lexical overlap.
- Store alignment scores and exclude uncertain alignments below a threshold.

### 2.4 Negative Labels

Use several types of negatives.

#### Easy negatives

Pages from unrelated filings or unrelated companies.

#### Retriever negatives

Top-k pages retrieved by BM25/BGE-M3 that are not gold evidence pages.

#### Finance-aware hard negatives

These are crucial. Generate negatives that are very close to the positive but wrong.

Examples:

- same company, wrong fiscal year,
- same company, wrong quarter,
- same filing, wrong page,
- same metric name, wrong table,
- same page section, wrong value,
- similar table from a different year,
- page containing the correct entity but not the correct metric,
- page containing the correct metric but not the answer value,
- boilerplate page with strong lexical overlap.

The paper should emphasize that these negatives simulate realistic financial RAG errors.

### 2.5 Candidate Generation

For each question, generate top-N candidates before reranking.

Recommended first-stage retrievers:

1. BM25
2. BGE-M3 dense retrieval
3. BM25 + BGE-M3 reciprocal rank fusion

For training:

- retrieve top 50 or top 100 candidates per question,
- include positives even if first-stage retrieval misses them,
- sample 4 to 8 negatives per positive.

For evaluation:

- rerank top 20, top 50, and top 100 candidates,
- report whether gold evidence was present in the first-stage candidate set,
- separate candidate-generation failure from reranking failure.

---

## 3. Model Design

### 3.1 Baseline Cross-Encoder

Input:

```text
[CLS] question [SEP] candidate_page_text [SEP]
```

Output:

```text
scalar relevance score
```

or two logits:

```text
irrelevant logit, relevant logit
```

Recommended backbones:

- `BAAI/bge-reranker-v2-m3` if easy to fine-tune,
- `microsoft/deberta-v3-base`,
- `cross-encoder/ms-marco-MiniLM-L-6-v2` for a lightweight baseline.

Start with a manageable model first. Make the whole pipeline work before using a larger reranker.

### 3.2 Evidential Cross-Encoder

Use the same backbone, but replace the normal classification head with an evidential head.

For binary relevance:

- class 0: irrelevant
- class 1: relevant

Let the model output raw logits:

```python
logits = W h + b
```

Convert logits to non-negative evidence:

```python
evidence = softplus(logits)
```

Dirichlet parameters:

```python
alpha = evidence + 1.0
```

Total evidence:

```python
S = alpha.sum(dim=-1)
```

Predictive probability:

```python
prob = alpha / S.unsqueeze(-1)
p_relevant = prob[:, 1]
```

Uncertainty/vacuity:

```python
uncertainty = K / S
```

For binary classification, `K = 2`.

### 3.3 Ranking Scores

Evaluate multiple ranking functions.

#### Mean relevance

```python
score = p_relevant
```

#### Uncertainty-penalized relevance

```python
score = p_relevant - lambda_uncertainty * uncertainty
```

Tune `lambda_uncertainty` on validation.

#### Evidence-weighted relevance

```python
score = p_relevant * (1.0 - uncertainty)
```

#### Lower confidence bound style score

```python
score = p_relevant - beta * uncertainty
```

Start with:

```python
beta in [0.0, 0.1, 0.25, 0.5, 1.0]
```

---

## 4. Loss Functions

Implement two model variants.

### 4.1 Standard Cross-Entropy Baseline

For standard cross-encoder:

```python
loss = cross_entropy(logits, labels)
```

### 4.2 Evidential Classification Loss

Implement the EDL classification loss.

Use one-hot labels `y`.

Expected class probabilities:

```python
p = alpha / alpha.sum(dim=-1, keepdim=True)
```

Mean squared error-style EDL loss:

```python
err = (y - p) ** 2
var = alpha * (S - alpha) / (S * S * (S + 1))
loss_data = (err + var).sum(dim=-1).mean()
```

Add KL regularization toward a uniform Dirichlet prior.

The KL term should be annealed:

```python
annealing_coef = min(1.0, global_step / annealing_steps)
loss = loss_data + annealing_coef * lambda_kl * loss_kl
```

Start with:

```yaml
lambda_kl: 0.001
annealing_steps: 10% to 20% of total training steps
```

Also test:

```yaml
lambda_kl: [0.0001, 0.001, 0.01]
```

Important: if the model collapses to high uncertainty for everything, reduce `lambda_kl`.

### 4.3 Optional Utility Regression Head

Only implement this after the classification version works.

Continuous utility target examples:

- whether adding this page lets the generator answer correctly,
- ROUGE/BLEU overlap with gold evidence,
- numeric match improvement,
- answer correctness improvement with vs. without the page.

This can use a Normal-Inverse-Gamma evidential regression head, but it is optional and should be treated as a second-stage extension.

---

## 5. Question-Level Uncertainty

Passage-level uncertainty is useful, but the paper needs question-level decisions.

For each question, after reranking top-k candidates, compute:

```python
top1_uncertainty
mean_top5_uncertainty
max_top5_uncertainty
mean_top5_relevance
top1_relevance
entropy_top5_scores
evidence_mass_top5
score_gap_top1_top2
```

Define retrieval failure labels:

```python
failure_at_1 = gold page not in top 1
failure_at_3 = gold page not in top 3
failure_at_5 = gold page not in top 5
failure_at_10 = gold page not in top 10
```

Then evaluate whether uncertainty predicts these failures.

Optional: train a lightweight logistic regression calibrator:

```python
features = [
    top1_uncertainty,
    mean_top5_uncertainty,
    top1_relevance,
    mean_top5_relevance,
    score_gap_top1_top2,
    num_unique_pages,
    num_unique_sections
]
target = failure_at_5
```

This becomes the question-level answerability/failure predictor.

---

## 6. Evaluation

### 6.1 Retrieval Metrics

Report:

- Recall@1
- Recall@3
- Recall@5
- Recall@10
- MRR@10
- nDCG@5
- nDCG@10
- MAP@10

Use page-level evidence recall as the main metric.

For each model, report results on:

1. FinQA internal validation/test
2. FinanceBench external test

Separate:

- first-stage retrieval performance,
- reranking performance conditioned on gold evidence being in candidates,
- end-to-end retrieval performance.

### 6.2 Calibration and Uncertainty Metrics

Report:

- Expected Calibration Error
- Brier score
- Negative log-likelihood if applicable
- AUROC for detecting retrieval failure
- AUPRC for detecting retrieval failure
- risk-coverage curves
- AURC
- selective recall/accuracy at fixed coverage levels

Coverage levels:

```text
100%, 90%, 80%, 70%, 60%, 50%
```

Example question:

> If we only answer the 70% of questions with lowest uncertainty, how much does PageRecall@5 or answer accuracy improve?

### 6.3 Downstream QA Metrics

If generation is included:

- numeric exact match,
- answer F1 where applicable,
- citation/evidence support,
- answer correctness,
- abstention accuracy,
- hallucination rate if labels are available.

But first prioritize retrieval and uncertainty. Generation is optional for v1.

### 6.4 Error Analysis

Create manual error categories:

- wrong fiscal year,
- wrong quarter,
- wrong company,
- wrong metric,
- correct page not in candidate set,
- correct page in candidate set but reranked too low,
- correct page retrieved but uncertainty high,
- irrelevant page with overconfident score,
- table extraction failure,
- annotation/evidence ambiguity.

Save at least 30 qualitative examples from FinanceBench.

---

## 7. Baselines

Implement these in order.

### 7.1 Retrieval Baselines

1. BM25
2. BGE-M3 dense retrieval
3. BM25 + BGE-M3 RRF

### 7.2 Reranker Baselines

1. Zero-shot BGE reranker
2. Fine-tuned standard cross-encoder
3. Temperature-scaled cross-encoder
4. MC-dropout cross-encoder, if feasible
5. Evidential cross-encoder

Optional but useful:

6. Small ensemble of 3 cross-encoders
7. Generic EDRR reproduction if code is easy to run

### 7.3 Ablations

Run these ablations:

1. Standard CE vs. Evidential CE with same backbone.
2. Generic negatives vs. finance-aware hard negatives.
3. Ranking by probability vs. probability minus uncertainty.
4. With metadata tokens vs. without metadata tokens.
5. Train on FinQA only vs. train on FinQA plus synthetic hard negatives.
6. FinanceBench performance with no FinanceBench tuning.

---

## 8. Expected Directory Structure

Create the following structure:

```text
evidential-finance-reranker/
  README.md
  instructions.md
  requirements.txt
  pyproject.toml

  configs/
    base.yaml
    train_ce.yaml
    train_edl.yaml
    eval_financebench.yaml

  data/
    raw/
      finqa/
      financebench/
      pdfs/
    processed/
      pages/
      chunks/
      pairs/
      candidates/
      splits/
    external/

  src/
    __init__.py

    data/
      build_finqa_pages.py
      build_financebench_pages.py
      align_evidence_to_pages.py
      make_splits.py
      make_pairs.py
      generate_hard_negatives.py

    retrieval/
      bm25.py
      dense_bge.py
      rrf.py
      build_index.py
      retrieve_candidates.py

    models/
      cross_encoder.py
      evidential_cross_encoder.py
      losses.py
      calibration.py

    training/
      train_cross_encoder.py
      train_evidential.py
      callbacks.py

    evaluation/
      evaluate_ranking.py
      evaluate_uncertainty.py
      evaluate_selective.py
      evaluate_generation.py
      metrics.py

    analysis/
      error_analysis.py
      make_tables.py
      make_plots.py
      qualitative_cases.py

    utils/
      io.py
      logging.py
      seed.py
      text.py

  scripts/
    01_prepare_finqa.sh
    02_prepare_financebench.sh
    03_retrieve_candidates.sh
    04_train_ce.sh
    05_train_edl.sh
    06_eval_all.sh
    07_make_paper_tables.sh

  results/
    runs/
    tables/
    plots/
    qualitative/
    paper_summaries/

  paper/
    outline.md
    related_work.md
    method.md
    experiments.md
    results.md
    limitations.md
```

---

## 9. Implementation Details

### 9.1 Reproducibility

Every run must save:

```text
config.yaml
metrics.json
predictions.jsonl
ranking_results.jsonl
uncertainty_results.jsonl
model_checkpoint/
train.log
```

Use deterministic seeds where possible:

```python
seed = 42
```

Log:

- model name,
- data split,
- number of questions,
- number of candidate pairs,
- positive/negative ratio,
- max sequence length,
- learning rate,
- batch size,
- random seed,
- GPU type if available.

### 9.2 Candidate Pair Format

Use JSONL.

Each line:

```json
{
  "qid": "finqa_001",
  "question": "...",
  "candidate_id": "doc123_page_4",
  "doc_id": "doc123",
  "page_number": 4,
  "candidate_text": "...",
  "metadata_text": "Company: Apple | Form: 10-K | Fiscal year: 2019 | Page: 4",
  "label": 1,
  "negative_type": null,
  "source_dataset": "finqa"
}
```

For negatives:

```json
"negative_type": "same_company_wrong_year"
```

### 9.3 Model Input Text

Concatenate metadata with text:

```text
Question: {question}

Candidate metadata:
Company: {company}
Ticker: {ticker}
Form: {form_type}
Fiscal year: {fiscal_year}
Filing date: {filing_date}
Page: {page_number}
Section: {section}

Candidate page:
{candidate_text}
```

This is important because financial relevance often depends on metadata.

### 9.4 Training Hyperparameters

Start with:

```yaml
model_name: microsoft/deberta-v3-base
max_length: 512
learning_rate: 2e-5
weight_decay: 0.01
batch_size: 8
gradient_accumulation_steps: 4
epochs: 3
warmup_ratio: 0.1
fp16: true
seed: 42
negative_ratio: 6
```

For EDL:

```yaml
lambda_kl: 0.001
annealing_ratio: 0.2
ranking_score: p_relevant_minus_beta_uncertainty
beta: 0.25
```

---

## 10. Metrics Implementation

Implement metrics from scratch or using sklearn where appropriate.

### 10.1 Ranking Metrics

Inputs:

```python
qid_to_ranked_candidates
qid_to_gold_candidate_ids
```

Compute:

- Recall@k
- MRR@k
- nDCG@k
- MAP@k

### 10.2 Calibration

For binary relevance probabilities:

- ECE with 10 and 15 bins,
- Brier score,
- reliability diagrams.

### 10.3 Failure Detection

For each question:

```python
y_failure = 1 if no gold page in top_k else 0
uncertainty_score = question_level_uncertainty
```

Compute:

- AUROC
- AUPRC
- precision/recall at thresholds
- risk-coverage.

### 10.4 Selective Evaluation

Sort questions by uncertainty ascending.

For each coverage level:

```python
keep lowest uncertainty questions
compute Recall@k on kept subset
compute failure rate on kept subset
```

Save:

```text
results/tables/selective_recall.csv
results/plots/risk_coverage_curve.png
```

---

## 11. Paper-Oriented Output

The code must automatically generate paper-ready outputs.

### 11.1 Tables

Generate CSV and LaTeX tables.

Required tables:

1. Dataset statistics.
2. Main FinanceBench reranking results.
3. Calibration and uncertainty results.
4. Selective retrieval risk-coverage results.
5. Ablation study.
6. Error analysis categories.

Save to:

```text
results/tables/
paper/tables/
```

### 11.2 Plots

Generate:

1. reliability diagram,
2. risk-coverage curve,
3. uncertainty distribution for success vs. failure,
4. score vs. uncertainty scatterplot,
5. bar chart of error categories.

Save to:

```text
results/plots/
paper/figures/
```

### 11.3 Written Summaries

For each experiment, automatically write a short markdown paragraph with:

- what was tested,
- key result,
- interpretation,
- possible paper sentence.

Save to:

```text
results/paper_summaries/
```

Example:

```markdown
## Experiment: Evidential CE vs Standard CE on FinanceBench

The evidential cross-encoder achieved Recall@5 of X compared with Y for the standard cross-encoder. More importantly, its uncertainty score achieved AUROC Z for predicting missing gold evidence at top-5. This suggests that evidential uncertainty is useful not only for ranking but also for identifying questions where the retrieved context is unreliable.
```

---

## 12. Paper Framing

### 12.1 Tentative Title

Recommended:

```text
Knowing When Evidence Is Weak: Evidential Cross-Encoder Reranking for Financial RAG
```

Alternative:

```text
Beyond Relevance Scores: Uncertainty-Aware Reranking for Financial Question Answering
```

### 12.2 Abstract Draft

This project studies uncertainty-aware evidence selection for financial retrieval-augmented question answering. Financial documents are long, repetitive, and numerically dense, making retrieval errors difficult to detect from relevance scores alone. We propose a finance-aware evidential cross-encoder reranker that outputs both relevance estimates and uncertainty through a Dirichlet evidence head. The model is trained on page-level financial evidence supervision with finance-specific hard negatives and evaluated on FinanceBench as an external test bed. Beyond ranking metrics, we evaluate calibration, retrieval-failure detection, and selective answering. The goal is to determine whether evidential uncertainty can identify when retrieved financial evidence is weak, stale, or insufficient before it contaminates answer generation.

### 12.3 Main Claims to Test

Do not assume these are true. Test them.

1. EDL improves calibration over a standard cross-encoder.
2. EDL uncertainty predicts retrieval failure.
3. Finance-aware hard negatives improve uncertainty quality.
4. Uncertainty-aware ranking improves selective retrieval.
5. FinQA-derived page supervision transfers to FinanceBench.

### 12.4 Paper Sections

Use this structure:

```text
1. Introduction
2. Related Work
   2.1 Financial Question Answering
   2.2 Reranking for RAG
   2.3 Uncertainty Quantification in Retrieval and RAG
   2.4 Evidential Deep Learning
3. Problem Setup
4. Method
   4.1 Candidate Generation
   4.2 Evidential Cross-Encoder
   4.3 Finance-Aware Hard Negatives
   4.4 Question-Level Uncertainty Aggregation
5. Experiments
   5.1 Datasets
   5.2 Baselines
   5.3 Metrics
   5.4 Implementation Details
6. Results
   6.1 Reranking Performance
   6.2 Calibration
   6.3 Failure Detection
   6.4 Selective Retrieval
   6.5 Ablations
7. Error Analysis
8. Limitations
9. Conclusion
```

### 12.5 Important Limitations to Mention

- FinQA-derived training labels may be noisy due to evidence-page alignment.
- FinanceBench public subset is small.
- EDL does not guarantee perfect calibration.
- Passage-level uncertainty cannot catch all generator reasoning errors.
- If using FinQA test data as training data, the paper must clearly state that FinQA is used only as a source of supervision and not as an official test benchmark.
- FinanceBench evaluation should remain external and untouched.

---

## 13. Step-by-Step Tasks for Codex or Claude Code

### Step 1: Create the repository skeleton

Create all directories listed in Section 8.

Add:

```text
README.md
requirements.txt
pyproject.toml
configs/base.yaml
```

### Step 2: Implement data ingestion

Implement scripts to load:

- FinQA JSON,
- FinanceBench JSON/CSV,
- extracted PDF page text,
- evidence annotations or page alignments.

Outputs:

```text
data/processed/pages/*.jsonl
data/processed/splits/*.json
```

### Step 3: Implement evidence-page alignment checks

If page alignments already exist, validate them.

If not, implement fuzzy matching:

- normalize text,
- compute token overlap,
- compute fuzzy score,
- map evidence to best page,
- save alignment confidence.

Exclude examples below threshold.

### Step 4: Implement candidate generation

Implement:

- BM25 retrieval,
- BGE-M3 dense retrieval,
- RRF fusion.

For each question, save top-N candidates:

```text
data/processed/candidates/{dataset}_{retriever}_top100.jsonl
```

### Step 5: Implement pair construction

Build training pairs from candidates.

Include:

- positives,
- easy negatives,
- retriever negatives,
- finance-aware hard negatives.

Save:

```text
data/processed/pairs/finqa_train_pairs.jsonl
data/processed/pairs/finqa_val_pairs.jsonl
data/processed/pairs/finqa_test_pairs.jsonl
data/processed/pairs/financebench_eval_pairs.jsonl
```

### Step 6: Implement standard cross-encoder

Create:

```text
src/models/cross_encoder.py
src/training/train_cross_encoder.py
```

Train and evaluate.

### Step 7: Implement evidential cross-encoder

Create:

```text
src/models/evidential_cross_encoder.py
src/models/losses.py
src/training/train_evidential.py
```

Implement:

- evidence head,
- Dirichlet alpha,
- p_relevant,
- uncertainty,
- EDL loss,
- KL annealing.

### Step 8: Implement evaluation scripts

Create:

```text
src/evaluation/evaluate_ranking.py
src/evaluation/evaluate_uncertainty.py
src/evaluation/evaluate_selective.py
```

Each script must save JSON, CSV, and human-readable markdown summaries.

### Step 9: Implement plots and tables

Create:

```text
src/analysis/make_tables.py
src/analysis/make_plots.py
```

Outputs:

```text
results/tables/*.csv
results/tables/*.tex
results/plots/*.png
```

### Step 10: Implement error analysis

Create:

```text
src/analysis/error_analysis.py
```

For each failed FinanceBench question, save:

- question,
- gold evidence page,
- top retrieved pages,
- predicted relevance,
- uncertainty,
- suspected error category,
- model explanation if available.

Manual labels can be added later.

### Step 11: Run all experiments

Minimum experiments:

1. BM25 only.
2. BGE-M3 only.
3. BM25 + BGE-M3 RRF.
4. Standard cross-encoder reranker.
5. Temperature-scaled cross-encoder.
6. Evidential cross-encoder with probability ranking.
7. Evidential cross-encoder with uncertainty-penalized ranking.
8. Evidential cross-encoder with finance-aware hard negatives.
9. Ablation without metadata tokens.
10. Selective retrieval using question-level uncertainty.

### Step 12: Produce paper package

Create:

```text
paper/outline.md
paper/related_work.md
paper/method.md
paper/experiments.md
paper/results.md
paper/limitations.md
```

Each should contain concise draft text using actual experiment outputs when available.

---

## 14. Definition of Done

The project is complete when:

1. FinQA-derived training data is built.
2. FinanceBench external evaluation data is built.
3. At least three first-stage retrieval baselines are run.
4. A standard cross-encoder reranker is trained.
5. An evidential cross-encoder reranker is trained.
6. Ranking metrics are reported.
7. Calibration metrics are reported.
8. Failure-detection AUROC/AUPRC are reported.
9. Risk-coverage curves are generated.
10. At least one ablation on hard negatives is completed.
11. At least one ablation on uncertainty-aware scoring is completed.
12. Paper-ready tables and plots are saved.
13. A qualitative error analysis file is created.
14. Draft paper sections are generated.

---

## 15. Most Important Experimental Comparison

The most important comparison is:

```text
Fine-tuned standard cross-encoder
vs.
Fine-tuned evidential cross-encoder
```

with the same:

- backbone,
- training data,
- candidates,
- max length,
- optimizer,
- random seed,
- evaluation set.

This isolates the effect of the evidential objective.

The second most important comparison is:

```text
Evidential CE with generic negatives
vs.
Evidential CE with finance-aware hard negatives
```

This tests whether the finance-specific part matters.

The third most important comparison is:

```text
Ranking by p_relevant
vs.
Ranking by p_relevant - beta * uncertainty
```

This tests whether uncertainty helps ranking or is only useful for failure detection.

---

## 16. Recommended First Milestone

Do not start with the full project. First implement a minimal version:

1. Use FinQA-aligned pages.
2. Build train/val pairs.
3. Train standard CE.
4. Train EDL CE.
5. Evaluate Recall@5 and ECE on validation.
6. Run on FinanceBench.
7. Generate one risk-coverage curve.

Once this works, add hard negatives, metadata, and downstream QA.

