"""
Configuration for the Domain-Adapted Financial RAG Experiment.

Design goals:
- Fine-tune a bi-encoder on FinQA financial question-to-page pairs
- Hard negatives from within-document pages to improve page-level discrimination
- Doc-filtered retrieval: for each FinanceBench question, search ONLY within the
  target document's pages (doc_name is known per question). This eliminates cross-
  document noise and turns PageRec into a pure within-doc page ranking problem.
- Hierarchical retrieval: build a chunk-level index; retrieve chunks, then aggregate
  to the parent page level for consistent @page evaluation.
- Multi-HyDE with Qwen for query augmentation at inference time
- Cross-encoder reranking for final precision
- Evaluate on FinanceBench (no leakage: training uses FinQA data only)
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# Root of the FB_reproducability project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class TrainingConfig:
    """Bi-encoder fine-tuning settings."""

    # Base model to fine-tune.
    # bge-m3 (~570M) matches the baseline model so the fine-tuned model starts
    # from the same strong representations. Using a weaker base (bge-base-en-v1.5)
    # caused the fine-tuned model to be worse than the baseline at DocRec.
    base_model: str = "BAAI/bge-m3"

    # Where to save the fine-tuned model weights.
    output_model_path: str = os.path.join(BASE_DIR, "models", "fin_adapted_biencoder_bge_m3")

    # Training hyperparameters.
    # Lower LR (5e-6 vs 2e-5) and fewer epochs (5 vs 10) prevent catastrophic
    # forgetting when fine-tuning a large model on a small dataset (~450 examples).
    num_epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 5e-6
    warmup_ratio: float = 0.1
    max_seq_length: int = 512
    max_page_chars: int = 2000

    # Layer freezing to prevent catastrophic forgetting.
    # We freeze all transformer layers except the last `trainable_layers` layers.
    # This keeps 90%+ of parameters fixed, letting only the top layers adapt to
    # the financial domain. Set to 0 to train all layers (not recommended for
    # small datasets).
    trainable_layers: int = 3      # last 3 transformer layers + pooling are trainable

    # Hard negative strategy:
    # Intra-document hard negatives (pages within the same filing) force the model
    # to learn to distinguish the cash-flow page from the income-statement page.
    num_hard_negatives_per_example: int = 5
    hard_neg_window: int = 10      # pages within ±10 of the gold page

    # Validation split
    eval_split: float = 0.1


@dataclass
class HyDEConfig:
    """Hypothetical Document Embedding (HyDE) settings."""

    enabled: bool = True

    qwen_model_name: str = "Qwen/Qwen2.5-7B-Instruct"

    # Number of hypothetical passages per query (Multi-HyDE)
    num_hypotheticals: int = 3

    max_new_tokens: int = 150
    temperature: float = 0.7
    do_sample: bool = True
    load_in_4bit: bool = True

    prompt_template: str = (
        "You are a financial analyst reviewing SEC 10-K and 10-Q filings. "
        "Given the following question, write a short passage (3-4 sentences) "
        "that would directly answer the question, as if excerpted from a "
        "company's annual report. Include specific financial figures and "
        "terminology as they would appear in the filing.\n\n"
        "Question: {question}\n\n"
        "Passage from annual report:"
    )


@dataclass
class RetrievalConfig:
    """Retrieval pipeline settings."""

    # Pages retrieved before reranking
    candidate_pages: int = 20

    # Final pages returned
    final_k: int = 5

    # Chunking for generation context (not for hierarchical index)
    chunk_size: int = 800
    chunk_overlap: int = 100

    # Cross-encoder reranker
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    enable_reranking: bool = True

    # RRF
    rrf_k: int = 60

    # -----------------------------------------------------------------------
    # Doc-filtered retrieval
    # -----------------------------------------------------------------------
    # When True, every ChromaDB query is filtered to only the pages of the
    # target document using `where={"doc_name": doc_name}`.  This is ORACLE
    # mode (doc_name comes from the FinanceBench ground truth) and gives
    # DocRec@k=1.0 by construction.  Disabled by default for honest evaluation.
    # Enable explicitly only for ablation comparison purposes.
    doc_filter_retrieval: bool = False

    # -----------------------------------------------------------------------
    # Hierarchical (chunk-level) retrieval
    # -----------------------------------------------------------------------
    # When True, we build and search a chunk-level index instead of the page-
    # level index.  Chunks (~400 chars) capture the specific table or paragraph
    # that answers a question, avoiding dilution from long pages.  After
    # retrieval, chunks are aggregated back to their parent pages.
    hierarchical_retrieval: bool = True

    # Chunk size for the hierarchical chunk-level index
    hier_chunk_size: int = 400
    hier_chunk_overlap: int = 50

    # Number of chunks to retrieve before aggregating to pages
    hier_n_chunks: int = 40


@dataclass
class DataConfig:
    """Input data paths."""

    # ---- Training data (FinQA only — never FinanceBench) ----
    finqa_gold_pages_path: str = os.path.join(BASE_DIR, "data", "finqa_test_gold_pages.jsonl")
    finqa_pdf_dir: str = os.path.join(BASE_DIR, "Final-PDF")

    # ---- Evaluation data (FinanceBench 150 questions) ----
    financebench_data_path: str = os.path.join(BASE_DIR, "data", "financebench_open_source.jsonl")
    financebench_pdf_dir: str = os.path.join(BASE_DIR, "pdfs")

    # ---- ChromaDB: page-level indexes ----
    # v3: embed_text now prefixed with PDF-filename-derived company/year/doc_type
    # so global (non-filtered) retrieval has document-level context.
    chroma_persist_dir: str = os.path.join(BASE_DIR, "vector_stores", "domain_adapted_chroma_v3")
    collection_name: str = "fb_ft_bge_m3_pages_v3"           # fine-tuned model pages
    baseline_collection_name: str = "fb_baseline_bge_m3_pages_v3"  # baseline BGE-M3 pages

    # ---- ChromaDB: chunk-level indexes (for hierarchical retrieval) ----
    chunk_collection_name: str = "fb_ft_bge_m3_chunks_v3"
    baseline_chunk_collection_name: str = "fb_baseline_bge_m3_chunks_v3"


@dataclass
class ExperimentConfig:
    """Top-level configuration."""

    training: TrainingConfig = field(default_factory=TrainingConfig)
    hyde: HyDEConfig = field(default_factory=HyDEConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    data: DataConfig = field(default_factory=DataConfig)

    output_dir: str = os.path.join(BASE_DIR, "domain_adapted_retrieval", "results")

    k_eval_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10, 20])
    main_k: int = 5
    seed: int = 42

    # ---- Published baselines for comparison plots ----
    prior_results: Dict = field(default_factory=lambda: {
        "Dense BGE-M3": {
            "doc_recall@5": 0.88, "page_recall@5": 0.34,
            "context_bleu@5": 0.26, "context_rougeL@5": 0.35,
        },
        "BM25": {
            "doc_recall@5": 0.32, "page_recall@5": 0.07,
            "context_bleu@5": 0.04, "context_rougeL@5": 0.12,
        },
        "SPLADE": {
            "doc_recall@5": 0.50, "page_recall@5": 0.16,
            "context_bleu@5": 0.18, "context_rougeL@5": 0.29,
        },
        "Hybrid (BM25+BGE-M3)": {
            "doc_recall@5": 0.61, "page_recall@5": 0.23,
            "context_bleu@5": 0.13, "context_rougeL@5": 0.27,
        },
        "HyDE": {
            "doc_recall@5": 0.86, "page_recall@5": 0.40,
            "context_bleu@5": 0.25, "context_rougeL@5": 0.37,
        },
        "Multi-HyDE": {
            "doc_recall@5": 0.85, "page_recall@5": 0.42,
            "context_bleu@5": 0.27, "context_rougeL@5": 0.39,
        },
        "BGE-M3 + ReRanker": {
            "doc_recall@5": 0.87, "page_recall@5": 0.41,
            "context_bleu@5": 0.19, "context_rougeL@5": 0.34,
        },
        "BGE-M3 + Multi-HyDE + ReRanker": {
            "doc_recall@5": 0.93, "page_recall@5": 0.46,
            "context_bleu@5": 0.28, "context_rougeL@5": 0.40,
        },
        "Oracle Document": {
            "doc_recall@5": 1.00, "page_recall@5": 0.60,
            "context_bleu@5": 0.25, "context_rougeL@5": 0.42,
        },
    })
