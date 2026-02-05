"""
Proper K-Fold Cross-Validation for Page Scorer with Retraining

This script implements TRUE k-fold cross-validation:
1. Split documents into k folds
2. For each fold: train on k-1 folds, predict on held-out fold
3. Combine predictions from all folds (unbiased estimates for ALL samples)
4. Compare against baselines on the full dataset

Output:
- Training curves for each fold
- Per-fold metrics
- Aggregated metrics with mean +/- std
- Full predictions file for comparison with other methods
- Comprehensive evaluation (doc/page/chunk recall, BLEU, ROUGE, etc.)
"""

import json
import logging
import os
import random
import shutil
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

# LangChain text splitter for chunking
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Training imports
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.vectorstores import Chroma

# Local imports
from src.ingestion.data_loader import FinanceBenchLoader
from src.ingestion.page_processor import extract_pages_from_pdf
from src.evaluation.retrieval_evaluator import RetrievalEvaluator
from src.evaluation.evaluator import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# HyDE imports (after logger is defined)
try:
    from src.ingestion.hyde import generate_hypothetical_documents
except ImportError:
    generate_hypothetical_documents = None
    logger.warning("HyDE module not available. HyDE features will be disabled.")

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class KFoldConfig:
    """Configuration for k-fold cross-validation."""
    # K-Fold settings
    n_folds: int = 5
    random_seed: int = 42

    # Model settings
    base_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    max_seq_length: int = 512

    # Training settings
    epochs: int = 15
    batch_size: int = 16
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    eval_every_n_epochs: int = 3
    patience: int = 3

    # Page scorer settings
    max_page_chars: int = 2000
    page_k: int = 5  # Top P pages to retrieve
    chunk_k: int = 5  # Top K chunks from those pages

    # Data settings
    pdf_dir: str = "pdfs"

    # Output settings
    output_dir: str = "results/kfold_page_scorer"
    save_models: bool = True  # Whether to save each fold's model

    # Inference settings
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    chunk_size: int = 1024  # in tokens
    chunk_overlap: int = 128  # in tokens

    # LLM generation settings
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    max_new_tokens: int = 256
    temperature: float = 0.2
    
    # HyDE settings (for inference query reformulation)
    use_hyde: bool = False  # Enable HyDE query reformulation at inference
    hyde_model: str = "Qwen/Qwen2.5-7B-Instruct"  # Model for HyDE generation
    hyde_num_generations: int = 1  # 1 for HyDE, >1 for Multi-HyDE
    hyde_aggregate: str = "mean"  # How to aggregate Multi-HyDE: 'mean' or 'max'


# =============================================================================
# Page Record and Training Data Classes
# =============================================================================

@dataclass
class PageRecord:
    """Represents a single page."""
    doc_name: str
    page_num: int
    page_text: str
    page_id: str = field(init=False)

    def __post_init__(self):
        self.page_id = f"{self.doc_name}_p{self.page_num}"
        self.page_text = self._normalize_text(self.page_text)

    @staticmethod
    def _normalize_text(text: str, max_chars: int = 2000) -> str:
        if not text:
            return " "
        import re
        text = text.replace("\x00", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()
        return text[:max_chars] if text else " "


@dataclass
class TrainingMetrics:
    """Stores training metrics for a fold."""
    fold: int
    epoch_losses: List[float] = field(default_factory=list)
    eval_metrics: List[Dict[str, float]] = field(default_factory=list)
    best_epoch: int = 0
    best_page_recall: float = 0.0
    training_time_seconds: float = 0.0


# =============================================================================
# Data Loading and Preprocessing
# =============================================================================

def load_financebench_data() -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Load FinanceBench dataset."""
    logger.info("Loading FinanceBench data...")
    loader = FinanceBenchLoader()
    df = loader.load_data()
    data = df.to_dict('records')
    logger.info(f"Loaded {len(data)} samples from {len(df['doc_name'].unique())} documents")
    return df, data


def build_page_records(
    pdf_dir: Path,
    doc_names: Set[str],
    max_chars: int = 2000
) -> Dict[str, List[PageRecord]]:
    """Build PageRecord objects for specified documents."""
    pages_by_doc = {}

    for doc_name in tqdm(doc_names, desc="Loading PDFs"):
        pdf_path = pdf_dir / f"{doc_name}.pdf"
        if not pdf_path.exists():
            logger.warning(f"PDF not found: {pdf_path}")
            continue

        try:
            raw_pages = extract_pages_from_pdf(pdf_path, doc_name)
            page_records = [
                PageRecord(
                    doc_name=doc_name,
                    page_num=p['page'],
                    page_text=p['text']
                )
                for p in raw_pages
            ]
            pages_by_doc[doc_name] = page_records
        except Exception as e:
            logger.warning(f"Failed to load {pdf_path}: {e}")

    total_pages = sum(len(pages) for pages in pages_by_doc.values())
    logger.info(f"Loaded {total_pages} pages from {len(pages_by_doc)} documents")
    return pages_by_doc


def create_document_folds(
    all_docs: List[str],
    n_folds: int,
    seed: int
) -> List[Set[str]]:
    """Split documents into k folds."""
    random.seed(seed)
    docs_shuffled = all_docs.copy()
    random.shuffle(docs_shuffled)

    # Split into n_folds roughly equal parts
    fold_size = len(docs_shuffled) // n_folds
    folds = []

    for i in range(n_folds):
        start_idx = i * fold_size
        if i == n_folds - 1:
            # Last fold gets remaining documents
            end_idx = len(docs_shuffled)
        else:
            end_idx = start_idx + fold_size
        folds.append(set(docs_shuffled[start_idx:end_idx]))

    return folds


# =============================================================================
# Training Functions
# =============================================================================

def create_training_examples(
    df: pd.DataFrame,
    train_docs: Set[str],
    pages_by_doc: Dict[str, List[PageRecord]],
    examples_per_question: int = 4
) -> List[InputExample]:
    """Create training examples using MultipleNegativesRankingLoss format."""
    train_examples = []
    skipped = 0

    for _, row in df.iterrows():
        doc_name = row['doc_name']

        if doc_name not in train_docs or doc_name not in pages_by_doc:
            continue

        question = row['question']
        evidence_list = row['evidence']

        # Extract gold page numbers
        gold_pages = set()
        for ev in evidence_list:
            p = ev.get('evidence_page_num') or ev.get('page_ix') or ev.get('page')
            if p is not None:
                gold_pages.add(int(p))

        if not gold_pages:
            skipped += 1
            continue

        # Validate gold pages exist
        doc_pages = pages_by_doc[doc_name]
        valid_gold_pages = {p for p in gold_pages if p < len(doc_pages)}

        if not valid_gold_pages:
            skipped += 1
            continue

        # Create positive pairs for MNRL
        for _ in range(examples_per_question):
            pos_page_num = random.choice(list(valid_gold_pages))
            pos_page = doc_pages[pos_page_num]
            train_examples.append(InputExample(texts=[question, pos_page.page_text]))

    logger.info(f"Created {len(train_examples)} training pairs (skipped {skipped})")
    return train_examples


def evaluate_page_retrieval(
    model: SentenceTransformer,
    df: pd.DataFrame,
    eval_docs: Set[str],
    pages_by_doc: Dict[str, List[PageRecord]],
    top_k: int = 10
) -> Dict[str, float]:
    """Evaluate page retrieval on specified documents."""
    page_hits = 0
    page_recalls = []
    total_questions = 0

    for _, row in df.iterrows():
        doc_name = row['doc_name']

        if doc_name not in eval_docs or doc_name not in pages_by_doc:
            continue

        question = row['question']
        evidence_list = row['evidence']

        # Extract gold pages
        gold_pages = set()
        for ev in evidence_list:
            p = ev.get('evidence_page_num') or ev.get('page_ix') or ev.get('page')
            if p is not None:
                gold_pages.add(int(p))

        if not gold_pages:
            continue

        doc_pages = pages_by_doc[doc_name]
        valid_gold_pages = {p for p in gold_pages if p < len(doc_pages)}

        if not valid_gold_pages:
            continue

        total_questions += 1

        # Retrieve pages
        query_embedding = model.encode(question, convert_to_tensor=True)
        page_texts = [p.page_text for p in doc_pages]
        page_embeddings = model.encode(page_texts, convert_to_tensor=True)

        scores = torch.nn.functional.cosine_similarity(
            query_embedding.unsqueeze(0), page_embeddings
        )
        top_indices = torch.argsort(scores, descending=True)[:top_k].cpu().tolist()

        # Check hits
        retrieved_pages = set(top_indices)
        hits = retrieved_pages & valid_gold_pages

        if hits:
            page_hits += 1

        recall = len(hits) / len(valid_gold_pages) if valid_gold_pages else 0.0
        page_recalls.append(recall)

    if total_questions == 0:
        return {"page_hit@k": 0.0, "page_recall@k": 0.0, "n_questions": 0}

    return {
        "page_hit@k": page_hits / total_questions,
        "page_recall@k": np.mean(page_recalls) if page_recalls else 0.0,
        "n_questions": total_questions
    }


def train_fold_model(
    fold_idx: int,
    train_docs: Set[str],
    val_docs: Set[str],
    df: pd.DataFrame,
    pages_by_doc: Dict[str, List[PageRecord]],
    config: KFoldConfig,
    output_dir: Path
) -> Tuple[SentenceTransformer, TrainingMetrics]:
    """Train a model for a single fold."""
    import time
    start_time = time.time()

    logger.info(f"\n{'='*80}")
    logger.info(f"TRAINING FOLD {fold_idx + 1}/{config.n_folds}")
    logger.info(f"Train docs: {len(train_docs)}, Val docs: {len(val_docs)}")
    logger.info(f"{'='*80}")

    # Set seeds for reproducibility
    fold_seed = config.random_seed + fold_idx
    random.seed(fold_seed)
    np.random.seed(fold_seed)
    torch.manual_seed(fold_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(fold_seed)

    # Create training examples
    train_examples = create_training_examples(
        df, train_docs, pages_by_doc,
        examples_per_question=4
    )

    if not train_examples:
        raise ValueError(f"No training examples created for fold {fold_idx}")

    # Initialize model
    model = SentenceTransformer(config.base_model_name)
    model.max_seq_length = config.max_seq_length

    # Training setup
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=config.batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Metrics tracking
    metrics = TrainingMetrics(fold=fold_idx)
    best_model_state = None

    # Calculate warmup steps
    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)

    logger.info(f"Training with {len(train_examples)} examples")
    logger.info(f"Steps per epoch: {steps_per_epoch}, Total steps: {total_steps}")
    logger.info(f"Warmup steps: {warmup_steps}")

    # Baseline evaluation
    baseline_metrics = evaluate_page_retrieval(model, df, val_docs, pages_by_doc, top_k=config.page_k)
    logger.info(f"Baseline: {baseline_metrics}")
    metrics.eval_metrics.append({"epoch": 0, **baseline_metrics})

    epochs_without_improvement = 0

    for epoch in range(config.epochs):
        logger.info(f"\n--- Epoch {epoch + 1}/{config.epochs} ---")

        # Train one epoch
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            warmup_steps=warmup_steps if epoch == 0 else 0,
            show_progress_bar=True,
            output_path=None,
            use_amp=True,
            optimizer_params={"lr": config.learning_rate}
        )

        # Evaluate periodically
        if (epoch + 1) % config.eval_every_n_epochs == 0 or epoch == config.epochs - 1:
            eval_metrics = evaluate_page_retrieval(model, df, val_docs, pages_by_doc, top_k=config.page_k)
            eval_metrics["epoch"] = epoch + 1
            metrics.eval_metrics.append(eval_metrics)

            logger.info(f"Epoch {epoch + 1}: {eval_metrics}")

            # Check for improvement
            if eval_metrics["page_recall@k"] > metrics.best_page_recall:
                metrics.best_page_recall = eval_metrics["page_recall@k"]
                metrics.best_epoch = epoch + 1
                best_model_state = model.state_dict().copy()
                epochs_without_improvement = 0
                logger.info(f"New best model (page_recall@k: {metrics.best_page_recall:.4f})")
            else:
                epochs_without_improvement += 1

                if epochs_without_improvement >= config.patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Save model if requested
    if config.save_models:
        model_path = output_dir / f"fold_{fold_idx}" / "model"
        model.save(str(model_path))
        logger.info(f"Saved model to {model_path}")

    metrics.training_time_seconds = time.time() - start_time
    logger.info(f"Fold {fold_idx + 1} training complete in {metrics.training_time_seconds:.1f}s")
    logger.info(f"Best epoch: {metrics.best_epoch}, Best page_recall@k: {metrics.best_page_recall:.4f}")

    return model, metrics


# =============================================================================
# Inference Functions
# =============================================================================

class _SentenceTransformerEmbeddings:
    """LangChain-compatible embeddings wrapper for SentenceTransformer."""

    def __init__(self, model: SentenceTransformer):
        self._model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [v.tolist() for v in self._model.encode(texts, convert_to_tensor=False)]

    def embed_query(self, text: str) -> List[float]:
        return self._model.encode(text, convert_to_tensor=False).tolist()

def run_page_then_chunk_inference(
    model: SentenceTransformer,
    samples: List[Dict[str, Any]],
    pages_by_doc: Dict[str, List[PageRecord]],
    chunk_db: Any,  # Chroma vectorstore
    config: KFoldConfig,
    chunk_embeddings_model: SentenceTransformer
) -> List[Dict[str, Any]]:
    """Run page-then-chunk inference on samples."""
    results = []

    # Use token-based chunking instead of character-based
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        length_function=lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    )

    # Build global page index across all documents
    all_pages: List[PageRecord] = []
    for _doc, pages in pages_by_doc.items():
        all_pages.extend(pages)

    all_page_texts = [p.page_text for p in all_pages]
    all_page_meta = [(p.doc_name, p.page_num) for p in all_pages]

    # Precompute page embeddings once
    all_page_embeddings = model.encode(all_page_texts, convert_to_tensor=True)

    # Build ONE global chunk index with ALL chunks from ALL documents
    logger.info("Building global chunk index (one-time setup)...")
    all_chunk_texts: List[str] = []
    all_chunk_metas: List[Dict[str, Any]] = []
    
    for doc_name, pages in pages_by_doc.items():
        for page in pages:
            page_chunks = text_splitter.split_text(page.page_text)
            for chunk_text in page_chunks:
                cleaned = " ".join(chunk_text.split())
                if not cleaned:
                    continue
                all_chunk_texts.append(cleaned)
                all_chunk_metas.append({
                    "doc_name": doc_name,
                    "page": page.page_num
                })
    
    # Create global chunk vectorstore ONCE
    embeddings = _SentenceTransformerEmbeddings(chunk_embeddings_model)
    global_chunk_db = Chroma.from_texts(
        texts=all_chunk_texts,
        embedding=embeddings,
        metadatas=all_chunk_metas,
        collection_name="global_chunks"
    )
    logger.info(f"Global chunk index built: {len(all_chunk_texts)} chunks from {len(pages_by_doc)} documents")

    # Pre-generate HyDE reformulations if enabled
    hyde_queries = {}
    if config.use_hyde and generate_hypothetical_documents:
        logger.info(f"Pre-generating HyDE reformulations (k={config.hyde_num_generations})...")
        for sample in tqdm(samples, desc="HyDE generation"):
            question = sample['question']
            try:
                hyp_docs = generate_hypothetical_documents(
                    query=question,
                    model_name=config.hyde_model,
                    num_generations=config.hyde_num_generations
                )
                hyde_queries[question] = hyp_docs
            except Exception as e:
                logger.warning(f"HyDE generation failed for query: {e}")
                hyde_queries[question] = [question]  # Fallback to original
        logger.info(f"Generated HyDE reformulations for {len(hyde_queries)} queries")
    elif config.use_hyde:
        logger.warning("HyDE enabled but generation module not available. Using original queries.")

    # Lazy-load Qwen for generation
    llm_pipeline = None
    try:
        gen_tokenizer = AutoTokenizer.from_pretrained(config.llm_model)
        llm = AutoModelForCausalLM.from_pretrained(
            config.llm_model,
            device_map="auto",
            torch_dtype=torch.float16
        )
        llm_pipeline = pipeline(
            "text-generation",
            model=llm,
            tokenizer=gen_tokenizer
        )
        logger.info(f"Loaded LLM: {config.llm_model}")
    except Exception as e:
        logger.warning(f"Failed to load LLM '{config.llm_model}': {e}")

    for sample in tqdm(samples, desc="Inference"):
        question = sample['question']
        doc_name = sample['doc_name']

        result = {
            **sample,
            "retrieved_chunks": [],
            "retrieved_pages": [],
            "model_answer": "Generation skipped",
            "hyde_used": config.use_hyde
        }

        if not all_pages:
            results.append(result)
            continue

        # Step 1: Apply HyDE if enabled (use pre-generated reformulations)
        if config.use_hyde and question in hyde_queries:
            hyp_docs = hyde_queries[question]
            
            if config.hyde_num_generations == 1:
                # Single HyDE: use the one reformulated query
                query_for_retrieval = hyp_docs[0]
            else:
                # Multi-HyDE: encode all hypotheticals and aggregate
                hyp_embeddings = model.encode(hyp_docs, convert_to_tensor=True)
                if config.hyde_aggregate == "mean":
                    query_embedding = torch.mean(hyp_embeddings, dim=0)
                else:  # max
                    query_embedding = torch.max(hyp_embeddings, dim=0)[0]
                query_for_retrieval = None  # Already have embedding
        else:
            query_for_retrieval = question
            query_embedding = None

        # Step 2: Retrieve top-P pages across ALL documents
        if query_embedding is None:
            query_embedding = model.encode(query_for_retrieval, convert_to_tensor=True)
            
        scores = torch.nn.functional.cosine_similarity(
            query_embedding.unsqueeze(0), all_page_embeddings
        )
        top_page_indices = torch.argsort(scores, descending=True)[:config.page_k].cpu().tolist()

        retrieved_pages_meta = [all_page_meta[idx] for idx in top_page_indices]
        result["retrieved_pages"] = [f"{doc}_p{page}" for doc, page in retrieved_pages_meta]

        # Step 3: Filter global chunk index to only include chunks from top-P pages
        # Build metadata filter: chunks must match one of the retrieved (doc_name, page) pairs
        retrieved_chunks = []
        
        # Chroma metadata filtering: retrieve from chunks that match any of the top pages
        # We'll do this by querying with an OR filter on (doc_name, page) combinations
        # Since Chroma's filtering can be complex, we'll retrieve more chunks and filter in-memory
        
        # Get candidate chunks (retrieve more than k, then filter by page metadata)
        candidate_k = config.chunk_k * 10  # Retrieve extra to ensure we have enough after filtering
        
        try:
            if hasattr(global_chunk_db, "similarity_search_with_relevance_scores"):
                hits = global_chunk_db.similarity_search_with_relevance_scores(question, k=candidate_k)
                candidates = [(doc.page_content, doc.metadata, score) for doc, score in hits]
            elif hasattr(global_chunk_db, "similarity_search_with_score"):
                hits = global_chunk_db.similarity_search_with_score(question, k=candidate_k)
                candidates = [(doc.page_content, doc.metadata, score) for doc, score in hits]
            else:
                hits = global_chunk_db.similarity_search(question, k=candidate_k)
                candidates = [(doc.page_content, doc.metadata, 0.0) for doc in hits]
            
            # Filter to only chunks from retrieved pages
            retrieved_page_set = set(retrieved_pages_meta)
            for text, metadata, score in candidates:
                chunk_doc = metadata.get("doc_name")
                chunk_page = metadata.get("page")
                if (chunk_doc, chunk_page) in retrieved_page_set:
                    retrieved_chunks.append({
                        "text": text,
                        "metadata": metadata,
                        "score": float(score)
                    })
                    if len(retrieved_chunks) >= config.chunk_k:
                        break
        
        except Exception as e:
            logger.warning(f"Chunk retrieval failed for question: {e}")
        
        result["retrieved_chunks"] = retrieved_chunks

        # Add gold evidence for evaluation
        evidence_list = sample.get('evidence', [])
        if not isinstance(evidence_list, list):
            try:
                evidence_list = list(evidence_list)
            except Exception:
                evidence_list = []
        gold_segments = []
        for ev in evidence_list:
            gold_segments.append({
                "doc_name": ev.get('doc_name', doc_name),
                "page": ev.get('evidence_page_num') or ev.get('page_ix') or ev.get('page'),
                "text": ev.get('evidence_text', '')
            })
        result["gold_evidence_segments"] = gold_segments
        result["gold_evidence"] = "\n\n".join([s.get("text", "") for s in gold_segments])
        result["reference_answer"] = sample.get('answer', '')

        # Step 3: Generate answer using Qwen 2.5 7B Instruct
        if llm_pipeline and result["retrieved_chunks"]:
            context = "\n\n".join([c["text"] for c in result["retrieved_chunks"]])
            prompt = (
                "You are a helpful financial analyst. Use the context to answer the question.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}\nAnswer:"
            )
            try:
                outputs = llm_pipeline(
                    prompt,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    do_sample=True
                )
                if outputs and isinstance(outputs, list):
                    generated = outputs[0].get("generated_text", "")
                    result["model_answer"] = generated.replace(prompt, "").strip()
            except Exception as e:
                logger.warning(f"Generation failed: {e}")

        results.append(result)

    return results


# =============================================================================
# Evaluation Functions
# =============================================================================

def compute_full_metrics(
    results: List[Dict[str, Any]],
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, Any]:
    """Compute comprehensive evaluation metrics."""
    retrieval_evaluator = RetrievalEvaluator()
    retrieval_metrics = retrieval_evaluator.compute_metrics(results, k_values=k_values)

    # Add generation metrics if answers are present
    predictions = [r.get("model_answer", "") for r in results]
    references = [r.get("reference_answer", "") for r in results]

    gen_metrics = {}
    if any(predictions) and any(references):
        try:
            evaluator = Evaluator(use_bertscore=False, use_llm_judge=False, use_ragas=False)
            for pred, ref in zip(predictions, references):
                if pred and ref and pred != "Generation skipped":
                    bleu = evaluator.compute_bleu(pred, ref)
                    rouge = evaluator.compute_rouge(pred, ref)
                    for k, v in bleu.items():
                        gen_metrics.setdefault(k, []).append(v)
                    for k, v in rouge.items():
                        gen_metrics.setdefault(k, []).append(v)

            # Average generation metrics
            gen_metrics = {k: np.mean(v) for k, v in gen_metrics.items() if v}
        except Exception as e:
            logger.warning(f"Error computing generation metrics: {e}")

    return {**retrieval_metrics, **gen_metrics}


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_training_curves(
    all_fold_metrics: List[TrainingMetrics],
    output_dir: Path
):
    """Plot training curves for all folds."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Page Recall@k over epochs
    ax = axes[0]
    for metrics in all_fold_metrics:
        epochs = [m["epoch"] for m in metrics.eval_metrics]
        recalls = [m["page_recall@k"] for m in metrics.eval_metrics]
        ax.plot(epochs, recalls, 'o-', label=f'Fold {metrics.fold + 1}', linewidth=2, markersize=6)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Page Recall@k', fontsize=12)
    ax.set_title('Training Progress: Page Recall@k', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 2: Best performance per fold
    ax = axes[1]
    folds = [m.fold + 1 for m in all_fold_metrics]
    best_recalls = [m.best_page_recall for m in all_fold_metrics]
    colors = plt.cm.viridis(np.linspace(0.3, 0.7, len(folds)))

    bars = ax.bar(folds, best_recalls, color=colors, edgecolor='black', linewidth=1.2)
    ax.axhline(np.mean(best_recalls), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(best_recalls):.4f}')

    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('Best Page Recall@k', fontsize=12)
    ax.set_title('Best Performance per Fold', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars, best_recalls):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved training curves to {output_dir / 'training_curves.png'}")


def plot_final_results(
    aggregated_metrics: Dict[str, Any],
    output_dir: Path
):
    """Plot final aggregated results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Retrieval metrics
    ax = axes[0]
    retrieval_keys = ['doc_hit@5', 'page_hit@5', 'chunk_hit@5', 'doc_recall@5', 'page_recall@5', 'chunk_recall@5']
    retrieval_keys = [k for k in retrieval_keys if k in aggregated_metrics]

    if retrieval_keys:
        means = [aggregated_metrics[k] for k in retrieval_keys]
        x = np.arange(len(retrieval_keys))

        bars = ax.bar(x, means, color='steelblue', edgecolor='black', linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels([k.replace('@5', '\n@5') for k in retrieval_keys], fontsize=10)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Retrieval Metrics @5', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)

        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # Plot 2: Generation metrics (if available)
    ax = axes[1]
    gen_keys = ['bleu_4', 'rouge_l_f1', 'rouge_1_f1']
    gen_keys = [k for k in gen_keys if k in aggregated_metrics]

    if gen_keys:
        means = [aggregated_metrics[k] for k in gen_keys]
        x = np.arange(len(gen_keys))

        bars = ax.bar(x, means, color='coral', edgecolor='black', linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels([k.replace('_', ' ').upper() for k in gen_keys], fontsize=10)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Generation Metrics', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)

        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    else:
        ax.text(0.5, 0.5, 'No generation metrics\n(generation skipped)',
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.set_title('Generation Metrics', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / "final_results.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved final results plot to {output_dir / 'final_results.png'}")


# =============================================================================
# Main K-Fold CV Function
# =============================================================================

def run_kfold_cross_validation(config: KFoldConfig) -> Dict[str, Any]:
    """Run complete k-fold cross-validation with retraining."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*80)
    logger.info(f"K-FOLD CROSS-VALIDATION WITH RETRAINING")
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*80)

    # Save config
    with open(output_dir / "config.json", 'w') as f:
        json.dump(asdict(config), f, indent=2)

    # Load data
    df, data = load_financebench_data()
    all_docs = list(df['doc_name'].unique())

    # Create document folds
    folds = create_document_folds(all_docs, config.n_folds, config.random_seed)

    logger.info(f"\nDocument distribution across {config.n_folds} folds:")
    for i, fold in enumerate(folds):
        logger.info(f"  Fold {i+1}: {len(fold)} documents")

    # Save fold assignments
    fold_assignments = {f"fold_{i}": list(fold) for i, fold in enumerate(folds)}
    with open(output_dir / "fold_assignments.json", 'w') as f:
        json.dump(fold_assignments, f, indent=2)

    # Load all pages
    pdf_dir = Path(config.pdf_dir)
    pages_by_doc = build_page_records(pdf_dir, set(all_docs), config.max_page_chars)

    # Storage for results
    all_fold_metrics: List[TrainingMetrics] = []
    all_predictions: List[Dict[str, Any]] = []
    fold_eval_results: List[Dict[str, Any]] = []

    # Run k-fold CV
    for fold_idx in range(config.n_folds):
        fold_output_dir = output_dir / f"fold_{fold_idx}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)

        # Define train/test docs for this fold
        test_docs = folds[fold_idx]
        train_docs = set()
        for i, fold in enumerate(folds):
            if i != fold_idx:
                train_docs.update(fold)

        # Get samples for this fold
        test_samples = [s for s in data if s['doc_name'] in test_docs]

        logger.info(f"\nFold {fold_idx + 1}: Train={len(train_docs)} docs, Test={len(test_docs)} docs, Test samples={len(test_samples)}")

        # Train model for this fold
        model, training_metrics = train_fold_model(
            fold_idx=fold_idx,
            train_docs=train_docs,
            val_docs=test_docs,  # Use test as validation during training
            df=df,
            pages_by_doc=pages_by_doc,
            config=config,
            output_dir=fold_output_dir
        )

        all_fold_metrics.append(training_metrics)

        # Save training metrics for this fold
        with open(fold_output_dir / "training_metrics.json", 'w') as f:
            json.dump({
                "fold": fold_idx,
                "best_epoch": training_metrics.best_epoch,
                "best_page_recall": training_metrics.best_page_recall,
                "training_time_seconds": training_metrics.training_time_seconds,
                "eval_metrics": training_metrics.eval_metrics
            }, f, indent=2)

        # Run inference on held-out fold
        logger.info(f"\nRunning inference on fold {fold_idx + 1} test set ({len(test_samples)} samples)...")

        # For simplicity, we'll use page-level retrieval results
        # In a full implementation, you'd also retrieve chunks
        fold_predictions = run_page_then_chunk_inference(
            model=model,
            samples=test_samples,
            pages_by_doc=pages_by_doc,
            chunk_db=None,  # Would need actual chunk DB
            config=config,
            chunk_embeddings_model=model  # Use same model for chunks
        )

        # Compute metrics for this fold
        fold_metrics = compute_full_metrics(fold_predictions)
        fold_eval_results.append({
            "fold": fold_idx,
            "n_samples": len(test_samples),
            "metrics": fold_metrics
        })

        logger.info(f"Fold {fold_idx + 1} metrics: {fold_metrics}")

        # Save fold predictions
        with open(fold_output_dir / "predictions.json", 'w') as f:
            json.dump(fold_predictions, f, indent=2, default=str)

        # Add to all predictions (for combined evaluation)
        all_predictions.extend(fold_predictions)

        # Clean up GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ==========================================================================
    # Aggregate results across all folds
    # ==========================================================================

    logger.info("\n" + "="*80)
    logger.info("AGGREGATING RESULTS ACROSS ALL FOLDS")
    logger.info("="*80)

    # Compute metrics on ALL predictions (combined from all folds)
    all_metrics = compute_full_metrics(all_predictions)

    # Compute per-fold statistics
    metric_keys = set()
    for fold_result in fold_eval_results:
        metric_keys.update(fold_result["metrics"].keys())

    aggregated_stats = {}
    for key in metric_keys:
        values = [f["metrics"].get(key, 0) for f in fold_eval_results]
        aggregated_stats[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "values": values
        }

    # Training statistics
    training_stats = {
        "best_page_recalls": [m.best_page_recall for m in all_fold_metrics],
        "best_epochs": [m.best_epoch for m in all_fold_metrics],
        "training_times": [m.training_time_seconds for m in all_fold_metrics],
        "mean_best_page_recall": float(np.mean([m.best_page_recall for m in all_fold_metrics])),
        "std_best_page_recall": float(np.std([m.best_page_recall for m in all_fold_metrics])),
        "total_training_time": sum(m.training_time_seconds for m in all_fold_metrics)
    }

    # ==========================================================================
    # Save all results
    # ==========================================================================

    # 1. All predictions (for comparison with other methods)
    with open(output_dir / "all_predictions.json", 'w') as f:
        json.dump(all_predictions, f, indent=2, default=str)
    logger.info(f"Saved {len(all_predictions)} predictions to all_predictions.json")

    # 2. Aggregated metrics
    final_results = {
        "config": asdict(config),
        "n_folds": config.n_folds,
        "n_total_samples": len(all_predictions),
        "timestamp": timestamp,
        "overall_metrics": all_metrics,
        "per_fold_metrics": fold_eval_results,
        "aggregated_statistics": aggregated_stats,
        "training_statistics": training_stats
    }

    with open(output_dir / "final_results.json", 'w') as f:
        json.dump(final_results, f, indent=2, default=str)

    # 3. Summary table (easy to read)
    summary_lines = [
        "="*80,
        f"K-FOLD CROSS-VALIDATION RESULTS ({config.n_folds} folds)",
        "="*80,
        f"Total samples: {len(all_predictions)}",
        f"Total training time: {training_stats['total_training_time']:.1f}s",
        "",
        "RETRIEVAL METRICS (mean +/- std across folds):",
        "-"*40,
    ]

    for key in ['doc_hit@5', 'page_hit@5', 'chunk_hit@5', 'doc_recall@5', 'page_recall@5', 'chunk_recall@5', 'mrr']:
        if key in aggregated_stats:
            stats = aggregated_stats[key]
            summary_lines.append(f"  {key}: {stats['mean']:.4f} +/- {stats['std']:.4f}")

    summary_lines.extend([
        "",
        "TRAINING STATISTICS:",
        "-"*40,
        f"  Best page_recall@k: {training_stats['mean_best_page_recall']:.4f} +/- {training_stats['std_best_page_recall']:.4f}",
        f"  Best epochs: {training_stats['best_epochs']}",
        "",
        "="*80
    ])

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    with open(output_dir / "summary.txt", 'w') as f:
        f.write(summary_text)

    # 4. Generate plots
    plot_training_curves(all_fold_metrics, output_dir)
    plot_final_results(all_metrics, output_dir)

    logger.info(f"\nAll results saved to: {output_dir}")
    logger.info("Files generated:")
    logger.info("  - config.json: Configuration used")
    logger.info("  - fold_assignments.json: Document assignments to folds")
    logger.info("  - all_predictions.json: All predictions (for comparison)")
    logger.info("  - final_results.json: Complete metrics and statistics")
    logger.info("  - summary.txt: Human-readable summary")
    logger.info("  - training_curves.png: Training progress visualization")
    logger.info("  - final_results.png: Final metrics visualization")
    logger.info("  - fold_*/: Per-fold models, predictions, and metrics")

    return final_results


# =============================================================================
# Train Final Production Model
# =============================================================================

def train_final_production_model(config: KFoldConfig, output_dir: Optional[str] = None):
    """Train final model on ALL data for production use."""

    if output_dir is None:
        output_dir = Path(config.output_dir) / "production_model"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*80)
    logger.info("TRAINING FINAL PRODUCTION MODEL (ALL DATA)")
    logger.info("="*80)

    # Load data
    df, data = load_financebench_data()
    all_docs = set(df['doc_name'].unique())

    # Load pages
    pdf_dir = Path(config.pdf_dir)
    pages_by_doc = build_page_records(pdf_dir, all_docs, config.max_page_chars)

    # Train on ALL documents
    model, metrics = train_fold_model(
        fold_idx=999,  # Special marker for production
        train_docs=all_docs,
        val_docs=all_docs,  # Evaluate on all (just for monitoring)
        df=df,
        pages_by_doc=pages_by_doc,
        config=config,
        output_dir=output_dir
    )

    # Save final model
    model.save(str(output_dir / "final_model"))

    # Save training info
    with open(output_dir / "training_info.json", 'w') as f:
        json.dump({
            "config": asdict(config),
            "n_documents": len(all_docs),
            "n_pages": sum(len(pages) for pages in pages_by_doc.values()),
            "best_epoch": metrics.best_epoch,
            "best_page_recall": metrics.best_page_recall,
            "training_time_seconds": metrics.training_time_seconds
        }, f, indent=2)

    logger.info(f"\nProduction model saved to: {output_dir / 'final_model'}")
    return model


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="K-Fold CV for Page Scorer")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of folds")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs per fold")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--page-k", type=int, default=5, help="Top P pages to retrieve")
    parser.add_argument("--chunk-k", type=int, default=5, help="Top K chunks to retrieve")
    parser.add_argument("--base-model", type=str, default="sentence-transformers/all-mpnet-base-v2", help="Base embedding model")
    parser.add_argument("--output-dir", type=str, default="results/kfold_page_scorer", help="Output directory")
    parser.add_argument("--pdf-dir", type=str, default="pdfs", help="PDF directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--production", action="store_true", help="Also train production model on all data")
    
    # HyDE arguments
    parser.add_argument("--use-hyde", action="store_true", help="Enable HyDE query reformulation at inference")
    parser.add_argument("--hyde-model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model for HyDE generation")
    parser.add_argument("--hyde-num-generations", type=int, default=1, help="Number of HyDE generations (1=HyDE, >1=Multi-HyDE)")
    parser.add_argument("--hyde-aggregate", type=str, default="mean", choices=["mean", "max"], help="Multi-HyDE aggregation method")

    args = parser.parse_args()

    config = KFoldConfig(
        n_folds=args.n_folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        page_k=args.page_k,
        chunk_k=args.chunk_k,
        base_model_name=args.base_model,
        output_dir=args.output_dir,
        pdf_dir=args.pdf_dir,
        random_seed=args.seed,
        use_hyde=args.use_hyde,
        hyde_model=args.hyde_model,
        hyde_num_generations=args.hyde_num_generations,
        hyde_aggregate=args.hyde_aggregate
    )

    # Run k-fold CV
    results = run_kfold_cross_validation(config)

    # Optionally train production model
    if args.production:
        logger.info("\n" + "="*80)
        logger.info("TRAINING PRODUCTION MODEL")
        logger.info("="*80)
        train_final_production_model(config)