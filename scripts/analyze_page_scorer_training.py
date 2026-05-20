"""
Analyze Page Scorer Training Performance

This script evaluates the trained page scorer on both training and dev sets
to detect overfitting and visualize performance.
"""

import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict

from sentence_transformers import SentenceTransformer
from src.ingestion.data_loader import FinanceBenchLoader
from src.ingestion.page_processor import extract_pages_from_pdf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_retrieval(
    model: SentenceTransformer,
    questions: List[Dict],
    pages_by_doc: Dict,
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, float]:
    """Evaluate page retrieval metrics."""
    
    metrics = {f"page_hit@{k}": [] for k in k_values}
    metrics.update({f"page_recall@{k}": [] for k in k_values})
    metrics["mrr"] = []
    
    for q_idx, row in enumerate(questions):
        doc_name = row['doc_name']
        question = row['question']
        
        if doc_name not in pages_by_doc:
            continue
        
        # Get gold pages
        evidence_list = row.get('evidence', [])
        if isinstance(evidence_list, dict):
            evidence_list = [evidence_list]
        
        gold_pages = set()
        for ev in evidence_list:
            p = ev.get('page_ix') or ev.get('evidence_page_num') or ev.get('page')
            if p is not None:
                gold_pages.add(int(p))
        
        if not gold_pages:
            continue
        
        # Score all pages
        doc_pages = pages_by_doc[doc_name]
        page_texts = [p.page_text for p in doc_pages]
        
        # Encode
        query_emb = model.encode([question], convert_to_numpy=True)[0]
        page_embs = model.encode(page_texts, convert_to_numpy=True)
        
        # Compute similarities
        similarities = np.dot(page_embs, query_emb)
        ranked_indices = np.argsort(similarities)[::-1]
        
        # Compute metrics
        for k in k_values:
            top_k_pages = set(ranked_indices[:k].tolist())
            hit = 1 if (top_k_pages & gold_pages) else 0
            recall = len(top_k_pages & gold_pages) / len(gold_pages)
            
            metrics[f"page_hit@{k}"].append(hit)
            metrics[f"page_recall@{k}"].append(recall)
        
        # MRR
        for rank_idx, page_idx in enumerate(ranked_indices):
            if page_idx in gold_pages:
                metrics["mrr"].append(1.0 / (rank_idx + 1))
                break
        else:
            metrics["mrr"].append(0.0)
    
    # Average
    return {k: np.mean(v) if v else 0.0 for k, v in metrics.items()}


def load_page_records(pdf_dir: Path, doc_names: Set[str]) -> Dict:
    """Load page records for documents."""
    from src.training.train_page_scorer_v2 import PageRecord
    
    pages_by_doc = {}
    
    for doc_name in doc_names:
        pdf_path = pdf_dir / f"{doc_name}.pdf"
        if not pdf_path.exists():
            continue
        
        try:
            page_dicts = extract_pages_from_pdf(pdf_path, doc_name)
            pages_by_doc[doc_name] = [
                PageRecord(p['doc_name'], p['page'], p['text'])
                for p in page_dicts
            ]
        except Exception as e:
            logger.warning(f"Failed to load {doc_name}: {e}")
    
    return pages_by_doc


def main():
    """Analyze trained page scorer for overfitting."""
    
    logger.info("="*80)
    logger.info("PAGE SCORER TRAINING ANALYSIS")
    logger.info("="*80)
    
    # Paths
    model_dir = Path("models/finetuned_page_scorer_v2")
    best_model_path = model_dir / "best_model"
    final_model_path = model_dir / "final_model"
    splits_path = model_dir / "splits.json"
    pdf_dir = Path("pdfs")
    
    # Load splits
    logger.info("\nLoading train/dev splits...")
    with open(splits_path) as f:
        splits = json.load(f)
    
    train_docs = set(splits["train_docs"])
    dev_docs = set(splits["dev_docs"])
    
    logger.info(f"Train docs: {len(train_docs)}")
    logger.info(f"Dev docs: {len(dev_docs)}")
    
    # Load data
    logger.info("\nLoading FinanceBench data...")
    loader = FinanceBenchLoader()
    df = loader.load_data()
    
    # Split questions
    train_questions = [row for _, row in df.iterrows() if row['doc_name'] in train_docs]
    dev_questions = [row for _, row in df.iterrows() if row['doc_name'] in dev_docs]
    
    logger.info(f"Train questions: {len(train_questions)}")
    logger.info(f"Dev questions: {len(dev_questions)}")
    
    # Load pages
    logger.info("\nLoading page records...")
    all_docs = train_docs | dev_docs
    pages_by_doc = load_page_records(pdf_dir, all_docs)
    logger.info(f"Loaded pages for {len(pages_by_doc)} documents")
    
    # Load models
    logger.info("\nLoading models...")
    base_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    best_model = SentenceTransformer(str(best_model_path))
    final_model = SentenceTransformer(str(final_model_path))
    
    logger.info("✓ Loaded: Base model (untrained)")
    logger.info("✓ Loaded: Best model (best validation)")
    logger.info("✓ Loaded: Final model (last epoch)")
    
    # Evaluate on train set
    logger.info("\n" + "="*80)
    logger.info("EVALUATING ON TRAIN SET")
    logger.info("="*80)
    
    logger.info("\nBase model (untrained):")
    base_train = evaluate_retrieval(base_model, train_questions, pages_by_doc)
    for k, v in base_train.items():
        logger.info(f"  {k}: {v:.4f}")
    
    logger.info("\nBest model:")
    best_train = evaluate_retrieval(best_model, train_questions, pages_by_doc)
    for k, v in best_train.items():
        logger.info(f"  {k}: {v:.4f}")
    
    logger.info("\nFinal model:")
    final_train = evaluate_retrieval(final_model, train_questions, pages_by_doc)
    for k, v in final_train.items():
        logger.info(f"  {k}: {v:.4f}")
    
    # Evaluate on dev set
    logger.info("\n" + "="*80)
    logger.info("EVALUATING ON DEV SET")
    logger.info("="*80)
    
    logger.info("\nBase model (untrained):")
    base_dev = evaluate_retrieval(base_model, dev_questions, pages_by_doc)
    for k, v in base_dev.items():
        logger.info(f"  {k}: {v:.4f}")
    
    logger.info("\nBest model:")
    best_dev = evaluate_retrieval(best_model, dev_questions, pages_by_doc)
    for k, v in best_dev.items():
        logger.info(f"  {k}: {v:.4f}")
    
    logger.info("\nFinal model:")
    final_dev = evaluate_retrieval(final_model, dev_questions, pages_by_doc)
    for k, v in final_dev.items():
        logger.info(f"  {k}: {v:.4f}")
    
    # Overfitting analysis
    logger.info("\n" + "="*80)
    logger.info("OVERFITTING ANALYSIS")
    logger.info("="*80)
    
    # Compare train vs dev for best model
    train_dev_gap = {}
    for metric in ["page_hit@5", "page_recall@5", "mrr"]:
        gap = best_train[metric] - best_dev[metric]
        train_dev_gap[metric] = gap
        logger.info(f"{metric}:")
        logger.info(f"  Train: {best_train[metric]:.4f}")
        logger.info(f"  Dev:   {best_dev[metric]:.4f}")
        logger.info(f"  Gap:   {gap:.4f} {'⚠️ HIGH' if gap > 0.15 else '✓ OK'}")
    
    # Improvement over baseline
    logger.info("\n" + "="*80)
    logger.info("IMPROVEMENT OVER BASELINE")
    logger.info("="*80)
    
    for metric in ["page_hit@5", "page_recall@5", "mrr"]:
        train_improvement = best_train[metric] - base_train[metric]
        dev_improvement = best_dev[metric] - base_dev[metric]
        
        logger.info(f"{metric}:")
        logger.info(f"  Train improvement: {train_improvement:+.4f}")
        logger.info(f"  Dev improvement:   {dev_improvement:+.4f}")
    
    # Create visualization
    logger.info("\n" + "="*80)
    logger.info("GENERATING PLOTS")
    logger.info("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Page Scorer Training Analysis", fontsize=16, fontweight='bold')
    
    # Plot 1: Train vs Dev performance
    ax = axes[0, 0]
    metrics_to_plot = ["page_hit@1", "page_hit@3", "page_hit@5", "page_hit@10"]
    train_vals = [best_train[m] for m in metrics_to_plot]
    dev_vals = [best_dev[m] for m in metrics_to_plot]
    
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    
    ax.bar(x - width/2, train_vals, width, label='Train', alpha=0.8)
    ax.bar(x + width/2, dev_vals, width, label='Dev', alpha=0.8)
    ax.set_xlabel('Metric')
    ax.set_ylabel('Score')
    ax.set_title('Train vs Dev Performance (Best Model)')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('page_', '') for m in metrics_to_plot], rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 2: Improvement over baseline
    ax = axes[0, 1]
    metrics_to_plot = ["page_hit@5", "page_recall@5", "mrr"]
    baseline_vals = [base_dev[m] for m in metrics_to_plot]
    trained_vals = [best_dev[m] for m in metrics_to_plot]
    
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    
    ax.bar(x - width/2, baseline_vals, width, label='Baseline', alpha=0.8)
    ax.bar(x + width/2, trained_vals, width, label='Trained', alpha=0.8)
    ax.set_xlabel('Metric')
    ax.set_ylabel('Score')
    ax.set_title('Dev Set: Baseline vs Trained')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot, rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 3: Recall@K curves
    ax = axes[1, 0]
    k_vals = [1, 3, 5, 10]
    base_recalls = [base_dev[f"page_recall@{k}"] for k in k_vals]
    best_recalls = [best_dev[f"page_recall@{k}"] for k in k_vals]
    
    ax.plot(k_vals, base_recalls, 'o-', label='Baseline', linewidth=2, markersize=8)
    ax.plot(k_vals, best_recalls, 's-', label='Trained', linewidth=2, markersize=8)
    ax.set_xlabel('K (number of pages)')
    ax.set_ylabel('Recall')
    ax.set_title('Page Recall@K on Dev Set')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(k_vals)
    
    # Plot 4: Overfitting indicators
    ax = axes[1, 1]
    metrics = ["page_hit@5", "page_recall@5", "mrr"]
    gaps = [train_dev_gap[m] for m in metrics]
    
    colors = ['red' if abs(g) > 0.15 else 'green' for g in gaps]
    ax.barh(metrics, gaps, color=colors, alpha=0.7)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.axvline(x=0.15, color='orange', linestyle='--', linewidth=0.8, label='Warning threshold')
    ax.set_xlabel('Train - Dev Gap')
    ax.set_title('Overfitting Indicator (Best Model)')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = model_dir / "training_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Plot saved: {output_path}")
    
    # Save metrics to JSON
    results = {
        "splits": {
            "train_docs": len(train_docs),
            "dev_docs": len(dev_docs),
            "train_questions": len(train_questions),
            "dev_questions": len(dev_questions)
        },
        "baseline": {
            "train": base_train,
            "dev": base_dev
        },
        "best_model": {
            "train": best_train,
            "dev": best_dev
        },
        "final_model": {
            "train": final_train,
            "dev": final_dev
        },
        "overfitting_analysis": {
            "train_dev_gap": train_dev_gap,
            "overfitting_detected": any(abs(g) > 0.15 for g in train_dev_gap.values())
        }
    }
    
    output_json = model_dir / "training_analysis.json"
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"✓ Metrics saved: {output_json}")
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    
    if any(abs(g) > 0.15 for g in train_dev_gap.values()):
        logger.warning("⚠️  Some train-dev gaps > 0.15 - possible overfitting")
    else:
        logger.info("✓ Train-dev gaps < 0.15 - no significant overfitting")
    
    logger.info(f"\nBest model dev performance:")
    logger.info(f"  Page Hit@5:    {best_dev['page_hit@5']:.4f}")
    logger.info(f"  Page Recall@5: {best_dev['page_recall@5']:.4f}")
    logger.info(f"  MRR:           {best_dev['mrr']:.4f}")
    
    logger.info(f"\nImprovement over baseline:")
    logger.info(f"  Page Hit@5:    {best_dev['page_hit@5'] - base_dev['page_hit@5']:+.4f}")
    logger.info(f"  Page Recall@5: {best_dev['page_recall@5'] - base_dev['page_recall@5']:+.4f}")
    logger.info(f"  MRR:           {best_dev['mrr'] - base_dev['mrr']:+.4f}")


if __name__ == "__main__":
    main()
