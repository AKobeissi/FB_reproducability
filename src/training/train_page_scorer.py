"""
src/training/train_page_scorer.py
Training script for the Page-Level Bi-Encoder.
Updated to use TripletLoss and Document-level splits.
"""
import logging
import os
import random
import numpy as np
from pathlib import Path
from typing import List, Set
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from src.ingestion.data_loader import FinanceBenchLoader
from src.ingestion.page_processor import extract_pages_from_pdf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_page_scorer(
    base_model_name: str = "sentence-transformers/all-mpnet-base-v2",
    output_path: str = "models/finetuned_page_scorer",
    pdf_dir: str = "pdfs",
    epochs: int = 3,
    batch_size: int = 16,
    max_seq_length: int = 512,
    train_split_ratio: float = 0.8
):
    # 1. Load Data
    loader = FinanceBenchLoader()
    df = loader.load_data()
    
    # 2. Split by Document (Prevent Leakage)
    all_docs = df['doc_name'].unique()
    np.random.seed(42)
    random.seed(42)
    
    train_docs = set(np.random.choice(all_docs, size=int(len(all_docs) * train_split_ratio), replace=False))
    test_docs = set(all_docs) - train_docs
    
    logger.info(f"Total Documents: {len(all_docs)}")
    logger.info(f"Training on {len(train_docs)} docs | Holding out {len(test_docs)} docs")

    # 3. Prepare Training Examples
    train_examples = []
    pdf_cache = {}
    pdf_path_obj = Path(pdf_dir)
    
    skipped_count = 0
    
    for idx, row in df.iterrows():
        doc_name = row['doc_name']
        
        # SKIP if in test set
        if doc_name not in train_docs:
            continue
            
        question = row['question']
        evidence_list = row['evidence']
        
        # Get Gold Page indices
        gold_pages_indices = set()
        for ev in evidence_list:
            if 'page_ix' in ev:
                gold_pages_indices.add(ev['page_ix'])
        
        if not gold_pages_indices:
            skipped_count += 1
            continue
            
        # Load PDF Pages (Cache them)
        if doc_name not in pdf_cache:
            f_path = pdf_path_obj / f"{doc_name}.pdf"
            if not f_path.exists():
                logger.warning(f"PDF not found: {doc_name}")
                continue
            pdf_cache[doc_name] = extract_pages_from_pdf(f_path, doc_name)
            
        doc_pages = pdf_cache.get(doc_name, [])
        if not doc_pages:
            continue
            
        # Create Triplets: (Anchor, Positive, Negative)
        # We pick ONE positive and ONE negative per question to avoid false negatives in batch
        
        # 1. Pick a random positive page
        pos_idx = random.choice(list(gold_pages_indices))
        if pos_idx >= len(doc_pages): 
            continue
        pos_text = doc_pages[pos_idx]['digest']
        
        # 2. Pick a random NEGATIVE page from the SAME document (Hard Negative)
        # Identify valid negative indices (pages that are not gold)
        all_indices = set(range(len(doc_pages)))
        neg_indices = list(all_indices - gold_pages_indices)
        
        if not neg_indices:
            # If all pages are gold (unlikely), skip
            continue
            
        neg_idx = random.choice(neg_indices)
        neg_text = doc_pages[neg_idx]['digest']
        
        train_examples.append(InputExample(texts=[question, pos_text, neg_text]))

    logger.info(f"Created {len(train_examples)} training triplets.")
    if skipped_count > 0:
        logger.info(f"Skipped {skipped_count} questions due to missing page info.")

    # 4. Initialize Model
    model = SentenceTransformer(base_model_name)
    model.max_seq_length = max_seq_length

    # 5. Training Setup
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    
    # FIX: Use TripletLoss
    train_loss = losses.TripletLoss(model)

    # 6. Train
    logger.info(f"Starting training for {epochs} epochs...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=int(len(train_dataloader) * 0.1),
        show_progress_bar=True,
        output_path=output_path
    )
    
    logger.info(f"Model saved to {output_path}")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train_page_scorer()