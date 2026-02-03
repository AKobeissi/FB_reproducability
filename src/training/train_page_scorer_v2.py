"""
src/training/train_page_scorer_v2.py

Improved Page-Level Bi-Encoder Training with Hard Negative Mining.

Key improvements:
1. BM25-based hard negative mining within documents (70-90%)
2. Cross-document hard negatives (10-30%) 
3. Proper evaluation metrics during training
4. Page-level dev set evaluation
5. Query augmentation with company name + fiscal year
6. Deterministic reproducibility
"""
import logging
import os
import json
import random
import numpy as np
import torch
from pathlib import Path
from typing import List, Set, Dict, Any, Tuple
import re
from collections import defaultdict
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from rank_bm25 import BM25Okapi

from src.ingestion.data_loader import FinanceBenchLoader
from src.ingestion.page_processor import extract_pages_from_pdf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _normalize_page_text(raw_txt: str, max_chars: int = 2000) -> str:
    """Normalize page text for embedding (matches inference)."""
    if not raw_txt:
        return " "
    txt = raw_txt.replace("\x00", " ")
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    txt = txt.strip()
    return (txt[:max_chars] if txt else " ")


def _tokenize_for_bm25(text: str) -> List[str]:
    """Simple tokenization for BM25."""
    return text.lower().split()


def _extract_company_name(doc_name: str) -> str:
    """Extract company name from doc_name like '3M_2018_10K'."""
    # Remove year and form type
    name = re.sub(r'_\d{4}.*', '', doc_name)
    name = re.sub(r'_10[KQ].*', '', name)
    return name.strip('_')


def _extract_fiscal_year(question: str, doc_name: str) -> str:
    """Extract fiscal year from question or doc_name."""
    # Try question first
    year_match = re.search(r'(FY\s?)?20\d{2}|2023Q[1-4]', question, re.IGNORECASE)
    if year_match:
        return year_match.group(0)
    
    # Fallback to doc_name
    year_match = re.search(r'20\d{2}', doc_name)
    if year_match:
        return year_match.group(0)
    
    return ""


def _augment_query(question: str, doc_name: str) -> str:
    """Augment query with company name and fiscal year."""
    company = _extract_company_name(doc_name)
    fiscal_year = _extract_fiscal_year(question, doc_name)
    
    # Don't duplicate if already present
    augmented = question
    if company and company.lower() not in question.lower():
        augmented = f"{company} {augmented}"
    if fiscal_year and fiscal_year not in question:
        augmented = f"{augmented} {fiscal_year}"
    
    return augmented


class PageRecord:
    """Canonical page record."""
    def __init__(self, doc_name: str, page_num: int, page_text: str):
        self.doc_name = doc_name
        self.page_num = page_num  # 0-indexed
        self.page_id = f"{doc_name}_p{page_num}"
        self.page_text = _normalize_page_text(page_text)
        self.tokens = _tokenize_for_bm25(self.page_text)


class HardNegativeSampler:
    """BM25-based hard negative sampler for in-document negatives."""
    
    def __init__(self, pages_by_doc: Dict[str, List[PageRecord]]):
        self.pages_by_doc = pages_by_doc
        self.bm25_by_doc: Dict[str, BM25Okapi] = {}
        
        # Build BM25 index per document
        for doc_name, pages in pages_by_doc.items():
            if pages:
                corpus = [p.tokens for p in pages]
                self.bm25_by_doc[doc_name] = BM25Okapi(corpus)
    
    def sample_in_doc_hard_negatives(
        self, 
        question: str, 
        doc_name: str, 
        gold_page_nums: Set[int],
        n_samples: int = 3
    ) -> List[PageRecord]:
        """Sample hard negatives from same document using BM25."""
        if doc_name not in self.bm25_by_doc:
            return []
        
        pages = self.pages_by_doc[doc_name]
        bm25 = self.bm25_by_doc[doc_name]
        
        # Get BM25 scores
        query_tokens = _tokenize_for_bm25(question)
        scores = bm25.get_scores(query_tokens)
        
        # Rank pages by BM25 score, excluding gold pages
        candidates = [
            (i, score) for i, score in enumerate(scores)
            if i not in gold_page_nums and i < len(pages)
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Sample from top-ranked negatives (hard negatives)
        # Mix of very hard (top ranks) and medium hard
        hard_pool_size = min(20, len(candidates))
        hard_pool = candidates[:hard_pool_size]
        
        if not hard_pool:
            return []
        
        # Sample with bias toward harder negatives
        sampled_indices = []
        for _ in range(min(n_samples, len(hard_pool))):
            # Weighted sampling: higher rank = higher probability
            weights = [1.0 / (rank + 1) for rank, _ in enumerate(hard_pool)]
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            
            idx = np.random.choice(len(hard_pool), p=weights)
            page_idx, _ = hard_pool[idx]
            if page_idx not in sampled_indices:
                sampled_indices.append(page_idx)
        
        return [pages[i] for i in sampled_indices]
    
    def sample_cross_doc_negatives(
        self,
        question: str,
        source_doc: str,
        n_samples: int = 1
    ) -> List[PageRecord]:
        """Sample hard negatives from other documents."""
        other_docs = [d for d in self.pages_by_doc.keys() if d != source_doc]
        if not other_docs:
            return []
        
        # Sample random other documents
        sampled_docs = random.sample(other_docs, min(n_samples, len(other_docs)))
        
        negatives = []
        for doc_name in sampled_docs:
            pages = self.pages_by_doc[doc_name]
            if pages:
                # Sample top BM25 page from this document
                bm25 = self.bm25_by_doc.get(doc_name)
                if bm25:
                    query_tokens = _tokenize_for_bm25(question)
                    scores = bm25.get_scores(query_tokens)
                    # Take top page from this document
                    best_idx = int(np.argmax(scores))
                    if best_idx < len(pages):
                        negatives.append(pages[best_idx])
                else:
                    negatives.append(random.choice(pages))
        
        return negatives


def build_page_records(
    pdf_dir: Path,
    doc_names: Set[str]
) -> Dict[str, List[PageRecord]]:
    """Build PageRecord objects for all documents."""
    pages_by_doc = {}
    
    first_doc_checked = False
    
    for doc_name in doc_names:
        pdf_path = pdf_dir / f"{doc_name}.pdf"
        if not pdf_path.exists():
            logger.warning(f"PDF not found: {pdf_path}")
            continue
        
        raw_pages = extract_pages_from_pdf(pdf_path, doc_name)
        
        # DEBUG: Check first document's page structure
        if not first_doc_checked and raw_pages:
            logger.info(f"\n=== Debug: First PDF page structure ===")
            logger.info(f"Doc: {doc_name}")
            logger.info(f"Total pages: {len(raw_pages)}")
            logger.info(f"First page keys: {list(raw_pages[0].keys())}")
            logger.info(f"First page 'page' field: {raw_pages[0].get('page')}")
            logger.info(f"Page text length: {len(raw_pages[0].get('text', ''))}")
            logger.info(f"Last page 'page' field: {raw_pages[-1].get('page')}")
            
            # Check if page numbers are sequential 0-indexed
            page_nums = [p.get('page') for p in raw_pages]
            logger.info(f"Page numbers: first 5 = {page_nums[:5]}, last 5 = {page_nums[-5:]}")
            
            # Check normalized text
            normalized = _normalize_page_text(raw_pages[0].get('text', ''))
            logger.info(f"Normalized text length: {len(normalized)}")
            logger.info(f"Normalized text preview: {normalized[:200]}")
            
            first_doc_checked = True
        
        page_records = [
            PageRecord(doc_name, page_dict['page'], page_dict['text'])
            for page_dict in raw_pages
        ]
        pages_by_doc[doc_name] = page_records
        logger.info(f"Loaded {len(page_records)} pages from {doc_name}")
    
    return pages_by_doc


def create_training_examples_simple(
    df,
    train_docs: Set[str],
    pages_by_doc: Dict[str, List[PageRecord]],
    examples_per_question: int = 4
) -> List[InputExample]:
    """Create training examples WITHOUT hard negative mining - simpler approach."""
    
    train_examples = []
    skipped = 0
    
    for idx, row in df.iterrows():
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
            logger.warning(f"Gold pages {gold_pages} out of range for {doc_name} ({len(doc_pages)} pages)")
            skipped += 1
            continue
        
        # NO query augmentation - keep it simple
        # Create multiple positive pairs (for MNRL)
        for _ in range(examples_per_question):
            pos_page_num = random.choice(list(valid_gold_pages))
            pos_page = doc_pages[pos_page_num]
            
            # Simple pair: (query, positive_page)
            train_examples.append(InputExample(texts=[question, pos_page.page_text]))
    
    logger.info(f"Created {len(train_examples)} training pairs from {len(df[df['doc_name'].isin(train_docs)])} questions")
    if skipped > 0:
        logger.info(f"Skipped {skipped} questions (missing pages or invalid indices)")
    
    return train_examples


def create_training_examples_with_hard_negatives(
    df,
    train_docs: Set[str],
    pages_by_doc: Dict[str, List[PageRecord]],
    sampler: HardNegativeSampler,
    examples_per_question: int = 4,
    in_doc_negative_ratio: float = 0.75
) -> List[InputExample]:
    """Create training examples WITH hard negative mining using TripletLoss."""
    
    train_examples = []
    skipped = 0
    
    n_in_doc = int(examples_per_question * in_doc_negative_ratio)
    n_cross_doc = examples_per_question - n_in_doc
    
    for idx, row in df.iterrows():
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
            logger.warning(f"Gold pages {gold_pages} out of range for {doc_name} ({len(doc_pages)} pages)")
            skipped += 1
            continue
        
        # Sample hard negatives
        in_doc_negs = sampler.sample_in_doc_hard_negatives(
            question, doc_name, valid_gold_pages, n_samples=n_in_doc
        )
        cross_doc_negs = sampler.sample_cross_doc_negatives(
            question, doc_name, n_samples=n_cross_doc
        )
        
        all_negatives = in_doc_negs + cross_doc_negs
        
        if not all_negatives:
            skipped += 1
            continue
        
        # Create triplets: (query, positive, negative)
        for neg_page in all_negatives:
            pos_page_num = random.choice(list(valid_gold_pages))
            pos_page = doc_pages[pos_page_num]
            
            # Triplet: (query, positive_page, negative_page)
            train_examples.append(InputExample(
                texts=[question, pos_page.page_text, neg_page.page_text]
            ))
    
    logger.info(f"Created {len(train_examples)} triplets with hard negatives from {len(df[df['doc_name'].isin(train_docs)])} questions")
    logger.info(f"  In-doc negatives: ~{n_in_doc}/{examples_per_question}, Cross-doc: ~{n_cross_doc}/{examples_per_question}")
    if skipped > 0:
        logger.info(f"Skipped {skipped} questions (missing pages or negatives)")
    
    return train_examples


def evaluate_on_dev(
    model: SentenceTransformer,
    df,
    dev_docs: Set[str],
    pages_by_doc: Dict[str, List[PageRecord]],
    top_k: int = 10,
    debug: bool = False
) -> Dict[str, float]:
    """Evaluate page retrieval on dev set."""
    
    page_hits = 0
    page_recalls = []
    total_questions = 0
    
    debug_examples = []
    
    for idx, row in df.iterrows():
        doc_name = row['doc_name']
        
        if doc_name not in dev_docs or doc_name not in pages_by_doc:
            continue
        
        question = row['question']
        evidence_list = row['evidence']
        
        # Extract gold pages
        gold_pages = set()
        for ev in evidence_list:
            # FinanceBench uses 'evidence_page_num' (0-indexed)
            p = ev.get('evidence_page_num') or ev.get('page_ix') or ev.get('page')
            if p is not None:
                gold_pages.add(int(p))
        
        if not gold_pages:
            continue
        
        # Validate gold pages exist in extracted pages
        doc_pages = pages_by_doc[doc_name]
        max_page_idx = len(doc_pages) - 1
        valid_gold_pages = {p for p in gold_pages if p <= max_page_idx}
        
        if not valid_gold_pages:
            if debug and total_questions < 3:
                logger.warning(f"Question {idx}: Gold pages {gold_pages} out of range for {doc_name} (max={max_page_idx})")
            continue
        
        total_questions += 1
        
        # Retrieve pages
        # DISABLED: Query augmentation inflates metrics and doesn't match inference
        # augmented_question = _augment_query(question, doc_name)
        # query_embedding = model.encode(augmented_question, convert_to_tensor=True)
        
        # Use raw question to match real pipeline
        query_embedding = model.encode(question, convert_to_tensor=True)
        page_texts = [p.page_text for p in doc_pages]
        page_embeddings = model.encode(page_texts, convert_to_tensor=True)
        
        # Compute similarities
        scores = torch.nn.functional.cosine_similarity(
            query_embedding.unsqueeze(0), page_embeddings
        )
        top_indices = torch.argsort(scores, descending=True)[:top_k].cpu().tolist()
        
        # Check hits and recall
        retrieved_pages = set(top_indices)
        hits = retrieved_pages & valid_gold_pages
        
        if hits:
            page_hits += 1
        
        recall = len(hits) / len(valid_gold_pages) if valid_gold_pages else 0.0
        page_recalls.append(recall)
        
        # Debug first few examples
        if debug and total_questions <= 3:
            debug_examples.append({
                "question": question[:80],
                # "augmented_question": augmented_question[:100],
                "doc_name": doc_name,
                "total_pages": len(doc_pages),
                "gold_pages": sorted(valid_gold_pages),
                "retrieved_pages": top_indices[:5],
                "hit": bool(hits),
                "top_scores": scores[top_indices[:5]].cpu().tolist(),
                "gold_page_scores": [scores[p].item() if p < len(scores) else -1.0 for p in sorted(valid_gold_pages)[:3]]
            })
    
    if debug and debug_examples:
        logger.info("\n=== Debug Examples ===")
        for ex in debug_examples:
            logger.info(f"Q: {ex['question']}...")
            # logger.info(f"  Augmented: {ex['augmented_question']}...")
            logger.info(f"  Doc: {ex['doc_name']} ({ex['total_pages']} pages)")
            logger.info(f"  Gold pages: {ex['gold_pages']}")
            logger.info(f"  Gold page scores: {[f'{s:.4f}' for s in ex['gold_page_scores']]}")
            logger.info(f"  Retrieved (top 5): {ex['retrieved_pages']}")
            logger.info(f"  Retrieved scores: {[f'{s:.4f}' for s in ex['top_scores']]}")
            logger.info(f"  Hit: {ex['hit']}\n")
    
    if total_questions == 0:
        logger.warning("No valid questions in dev set!")
        return {"page_hit@k": 0.0, "page_recall@k": 0.0, "n_questions": 0}
    
    return {
        "page_hit@k": page_hits / total_questions,
        "page_recall@k": np.mean(page_recalls) if page_recalls else 0.0,
        "n_questions": total_questions
    }


def train_page_scorer_v2(
    base_model_name: str = "sentence-transformers/all-mpnet-base-v2",
    output_path: str = "models/finetuned_page_scorer_v2",
    pdf_dir: str = "pdfs",
    epochs: int = 20,  # Reduced from 30
    batch_size: int = 16,
    max_seq_length: int = 512,
    train_split_ratio: float = 0.85,
    examples_per_question: int = 12,  # Increased from 8 for more training signal
    in_doc_negative_ratio: float = 0.75,  # 75% in-doc, 25% cross-doc
    eval_every_n_epochs: int = 5,
    patience: int = 3,  # Early stopping: stop if no improvement for 3 evaluations
    use_hard_negatives: bool = False  # NEW: Enable hard negative mining
):
    """Train page scorer with optional hard negative mining and proper evaluation."""
    
    # Set seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.cuda.empty_cache()
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    loader = FinanceBenchLoader()
    df = loader.load_data()
    
    # DIAGNOSTIC: Check evidence structure
    logger.info("\n=== Checking Evidence Data Structure ===")
    sample_row = df.iloc[0]
    logger.info(f"Sample row keys: {list(sample_row.keys())}")
    logger.info(f"Evidence type: {type(sample_row['evidence'])}")
    if isinstance(sample_row['evidence'], list) and len(sample_row['evidence']) > 0:
        logger.info(f"First evidence item: {sample_row['evidence'][0]}")
        logger.info(f"Evidence item keys: {list(sample_row['evidence'][0].keys())}")
    logger.info(f"Doc name: {sample_row['doc_name']}")
    
    # Split by document
    all_docs = list(df['doc_name'].unique())
    random.shuffle(all_docs)
    
    split_idx = int(len(all_docs) * train_split_ratio)
    train_docs = set(all_docs[:split_idx])
    dev_docs = set(all_docs[split_idx:])
    
    logger.info(f"Total docs: {len(all_docs)}")
    logger.info(f"Train docs: {len(train_docs)}")
    logger.info(f"Dev docs: {len(dev_docs)}")
    
    # Save splits
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "splits.json"), "w") as f:
        json.dump({
            "train_docs": list(train_docs),
            "dev_docs": list(dev_docs)
        }, f, indent=2)
    
    # Build page records
    pdf_path = Path(pdf_dir)
    logger.info("Building page records...")
    pages_by_doc = build_page_records(pdf_path, set(all_docs))
    
    logger.info(f"Total pages loaded: {sum(len(pages) for pages in pages_by_doc.values())}")
    
    # Create training examples
    if use_hard_negatives:
        logger.info("Creating training examples with HARD NEGATIVES (BM25-based)...")
        # Initialize hard negative sampler
        sampler = HardNegativeSampler(pages_by_doc)
        train_examples = create_training_examples_with_hard_negatives(
            df, train_docs, pages_by_doc, sampler,
            examples_per_question=examples_per_question,
            in_doc_negative_ratio=in_doc_negative_ratio
        )
    else:
        logger.info("Creating training examples with WEAK NEGATIVES (in-batch for MNRL)...")
        train_examples = create_training_examples_simple(
            df, train_docs, pages_by_doc,
            examples_per_question=4  # Fewer examples, higher quality
        )
    
    if not train_examples:
        logger.error("No training examples created!")
        return
    
    # Initialize model
    model = SentenceTransformer(base_model_name)
    model.max_seq_length = max_seq_length
    
    # Training setup
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    
    # Choose loss function based on negative type
    if use_hard_negatives:
        # TripletLoss for explicit hard negatives
        train_loss = losses.TripletLoss(model)
        loss_name = "TripletLoss with hard negatives"
    else:
        # MultipleNegativesRankingLoss with in-batch negatives
        train_loss = losses.MultipleNegativesRankingLoss(model)
        loss_name = "MNRL with in-batch weak negatives"
    
    logger.info(f"Training with {len(train_examples)} examples using {loss_name}")
    logger.info(f"Batch size: {batch_size}, Epochs: {epochs}")
    
    # Baseline evaluation
    logger.info("\n=== Baseline Evaluation (before training) ===")
    
    # SANITY CHECK: Verify embeddings are distinct
    logger.info("\n=== Sanity Check: Embedding Diversity ===")
    if dev_docs and pages_by_doc:
        first_dev_doc = list(dev_docs)[0]
        if first_dev_doc in pages_by_doc:
            test_pages = pages_by_doc[first_dev_doc][:5]
            if test_pages:
                test_texts = [p.page_text for p in test_pages]
                test_embeddings = model.encode(test_texts, convert_to_tensor=True)
                
                # Check if embeddings are all identical (broken) or diverse (working)
                similarities = []
                for i in range(len(test_embeddings)):
                    for j in range(i+1, len(test_embeddings)):
                        sim = torch.nn.functional.cosine_similarity(
                            test_embeddings[i].unsqueeze(0),
                            test_embeddings[j].unsqueeze(0)
                        ).item()
                        similarities.append(sim)
                
                if similarities:
                    avg_sim = np.mean(similarities)
                    min_sim = min(similarities)
                    max_sim = max(similarities)
                    logger.info(f"Page embedding similarities (same doc): avg={avg_sim:.4f}, min={min_sim:.4f}, max={max_sim:.4f}")
                    
                    if avg_sim > 0.99:
                        logger.error("WARNING: All page embeddings are nearly identical! Text normalization may be broken.")
    
    baseline_metrics = evaluate_on_dev(model, df, dev_docs, pages_by_doc, top_k=10, debug=True)
    logger.info(f"Baseline: {baseline_metrics}")
    
    # DIAGNOSTIC: Check a few training examples too
    logger.info("\n=== Checking Training Data ===")
    train_check_count = 0
    for idx, row in df.iterrows():
        if train_check_count >= 3:
            break
        doc_name = row['doc_name']
        if doc_name not in train_docs or doc_name not in pages_by_doc:
            continue
        
        evidence_list = row['evidence']
        gold_pages = set()
        for ev in evidence_list:
            p = ev.get('evidence_page_num') or ev.get('page_ix') or ev.get('page')
            if p is not None:
                gold_pages.add(int(p))
        
        if not gold_pages:
            continue
        
        doc_pages = pages_by_doc[doc_name]
        max_page_idx = len(doc_pages) - 1
        valid_gold = {p for p in gold_pages if p <= max_page_idx}
        
        logger.info(f"Train example {train_check_count + 1}:")
        logger.info(f"  Doc: {doc_name}, Total pages: {len(doc_pages)}")
        logger.info(f"  Gold pages: {sorted(gold_pages)}, Valid: {sorted(valid_gold)}")
        logger.info(f"  Question: {row['question'][:80]}...")
        
        # Check if gold page has content
        if valid_gold:
            first_gold = min(valid_gold)
            gold_page_text = doc_pages[first_gold].page_text
            logger.info(f"  Gold page {first_gold} text length: {len(gold_page_text)}")
            logger.info(f"  Gold page {first_gold} preview: {gold_page_text[:150]}...")
        
        train_check_count += 1
    
    best_page_recall = 0.0
    epochs_without_improvement = 0
    
    # Train with periodic evaluation
    steps_per_epoch = len(train_dataloader)
    warmup = int(steps_per_epoch * 0.1 * epochs)
    
    # Lower learning rate to prevent overfitting
    optimizer_params = {"lr": 2e-5}  # Default is 2e-5 for sentence-transformers, making it explicit
    
    for epoch in range(epochs):
        logger.info(f"\n=== Epoch {epoch + 1}/{epochs} ===")
        
        # Train one epoch
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            warmup_steps=warmup if epoch == 0 else 0,
            show_progress_bar=True,
            output_path=None,  # Don't save every epoch
            use_amp=True,
            optimizer_params=optimizer_params
        )
        
        # Evaluate periodically
        if (epoch + 1) % eval_every_n_epochs == 0 or epoch == epochs - 1:
            logger.info(f"\n=== Evaluating after epoch {epoch + 1} ===")
            metrics = evaluate_on_dev(model, df, dev_docs, pages_by_doc, top_k=10)
            logger.info(f"Metrics: {metrics}")
            
            # Save best model
            if metrics["page_recall@k"] > best_page_recall:
                best_page_recall = metrics["page_recall@k"]
                best_path = os.path.join(output_path, "best_model")
                model.save(best_path)
                logger.info(f"New best model saved (page_recall@10: {best_page_recall:.4f})")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                logger.info(f"No improvement ({epochs_without_improvement}/{patience})")
                
                # Early stopping
                if epochs_without_improvement >= patience:
                    logger.info(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    logger.info(f"Best page_recall@10: {best_page_recall:.4f}")
                    break
    
    # Save final model
    final_path = os.path.join(output_path, "final_model")
    model.save(final_path)
    logger.info(f"\nTraining complete! Final model saved to {final_path}")
    logger.info(f"Best page_recall@10: {best_page_recall:.4f}")


if __name__ == "__main__":
    import sys
    
    # Check command line argument for hard negatives
    use_hard_negatives = "--hard-negatives" in sys.argv
    
    if use_hard_negatives:
        logger.info("\n" + "="*80)
        logger.info("TRAINING WITH HARD NEGATIVES (BM25-based mining)")
        logger.info("="*80 + "\n")
        output_path = "models/finetuned_page_scorer_v2_hard"
    else:
        logger.info("\n" + "="*80)
        logger.info("TRAINING WITH WEAK NEGATIVES (in-batch MNRL)")
        logger.info("="*80 + "\n")
        output_path = "models/finetuned_page_scorer_v2_weak"
    
    train_page_scorer_v2(
        output_path=output_path,
        use_hard_negatives=use_hard_negatives
    )
