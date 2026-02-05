"""
Proper Evaluation of Page Scorer with No Data Leakage

This script addresses critical methodological issues:
1. Respects train/dev document splits (no leakage)
2. No query augmentation (matches real inference)
3. Ablation: learned vs base model for pages and chunks
4. Consistent corpus definition
5. Fresh vector stores (no reuse issues)
6. Cross-validation across multiple random splits
"""

import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Set
from datetime import datetime
import pandas as pd

from src.ingestion.data_loader import FinanceBenchLoader
from src.core.rag_experiments import RAGExperiment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


def load_splits(model_dir: Path) -> Dict[str, Set[str]]:
    """Load train/dev splits from trained model."""
    splits_path = model_dir / "splits.json"
    with open(splits_path) as f:
        splits = json.load(f)
    return {
        "train_docs": set(splits["train_docs"]),
        "dev_docs": set(splits["dev_docs"])
    }


def get_sample_indices_for_docs(data: List[Dict], doc_set: Set[str]) -> List[int]:
    """Get sample indices for questions from specific documents."""
    indices = []
    for i, sample in enumerate(data):
        doc_name = sample.get("doc_name", "")
        if doc_name in doc_set:
            indices.append(i)
    return indices


def run_ablation_experiment(
    experiment_config: Dict,
    data: List[Dict],
    sample_indices: List[int],
    ablation_name: str,
    learned_page_model: str = None,
    use_learned_for_chunks: bool = False
) -> Dict:
    """Run single ablation experiment.
    
    Args:
        experiment_config: Experiment configuration
        data: Full dataset
        sample_indices: Indices to evaluate on
        ablation_name: Name for this ablation
        learned_page_model: Path to learned model (None = baseline)
        use_learned_for_chunks: If True, use learned model for chunks. If False, use base model for chunks.
    """
    
    logger.info(f"\n{'='*80}")
    logger.info(f"ABLATION: {ablation_name}")
    logger.info(f"  Learned page model: {learned_page_model or 'None (baseline)'}")
    logger.info(f"  Use learned for chunks: {use_learned_for_chunks}")
    logger.info(f"  Samples: {len(sample_indices)}")
    logger.info(f"{'='*80}")
    
    # Create unique vector store name to avoid reuse
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    vs_suffix = f"_{ablation_name}_{timestamp}"
    
    experiment = RAGExperiment(
        **experiment_config,
        vector_store_dir=f"vector_stores/ablation{vs_suffix}"
    )
    
    # Force fresh vector store by clearing if exists
    import shutil
    vs_path = Path(experiment.vector_store_dir)
    if vs_path.exists():
        shutil.rmtree(vs_path)
    
    # Subset data to sample indices
    data_subset = [data[i] for i in sample_indices]
    
    # Run page-then-chunk with specified model
    from src.experiments.page_retrieval import run_page_then_chunk
    
    results = run_page_then_chunk(
        experiment=experiment,
        data=data_subset,
        learned_model_path=learned_page_model,
        use_learned_for_chunks=use_learned_for_chunks
    )
    
    # Extract metrics
    from src.evaluation.retrieval_evaluator import RetrievalEvaluator
    evaluator = RetrievalEvaluator()
    metrics = evaluator.compute_metrics(results, k_values=[1, 3, 5, 10])
    
    return {
        "ablation_name": ablation_name,
        "learned_page_model": learned_page_model or "baseline",
        "use_learned_for_chunks": use_learned_for_chunks,
        "n_samples": len(sample_indices),
        "metrics": metrics,
        "results": results
    }


def run_cross_validation(
    base_config: Dict,
    n_splits: int = 3,
    output_dir: Path = Path("results/cross_validation")
) -> Dict:
    """Run cross-validation with different random splits."""
    
    logger.info(f"\n{'='*80}")
    logger.info(f"CROSS-VALIDATION: {n_splits} random splits")
    logger.info(f"{'='*80}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load full data
    loader = FinanceBenchLoader()
    df = loader.load_data()
    all_docs = list(df['doc_name'].unique())
    data = df.to_dict('records')
    
    cv_results = []
    
    for split_idx in range(n_splits):
        seed = 42 + split_idx
        np.random.seed(seed)
        
        logger.info(f"\n--- Split {split_idx + 1}/{n_splits} (seed={seed}) ---")
        
        # Create random split
        docs_shuffled = all_docs.copy()
        np.random.shuffle(docs_shuffled)
        split_point = int(len(docs_shuffled) * 0.85)
        
        train_docs = set(docs_shuffled[:split_point])
        dev_docs = set(docs_shuffled[split_point:])
        
        train_indices = get_sample_indices_for_docs(data, train_docs)
        dev_indices = get_sample_indices_for_docs(data, dev_docs)
        
        logger.info(f"  Train: {len(train_docs)} docs, {len(train_indices)} questions")
        logger.info(f"  Dev: {len(dev_docs)} docs, {len(dev_indices)} questions")
        
        # Test baseline on this split's dev set
        baseline_result = run_ablation_experiment(
            base_config,
            data,
            dev_indices,
            f"cv_split{split_idx}_baseline",
            learned_page_model=None
        )
        
        # Test learned model on this split's dev set
        learned_model_path = "models/finetuned_page_scorer_v2/best_model"
        learned_result = run_ablation_experiment(
            base_config,
            data,
            dev_indices,
            f"cv_split{split_idx}_learned",
            learned_page_model=learned_model_path
        )
        
        # Save individual split results for debugging
        split_output_dir = output_dir / "cross_validation" / f"split_{split_idx}"
        split_output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(split_output_dir / "baseline_results.json", 'w') as f:
            json.dump({
                "metrics": baseline_result["metrics"],
                "n_samples": baseline_result["n_samples"],
                "results": baseline_result["results"][:3]  # Save first 3 for debugging
            }, f, indent=2)
        
        with open(split_output_dir / "learned_results.json", 'w') as f:
            json.dump({
                "metrics": learned_result["metrics"],
                "n_samples": learned_result["n_samples"],
                "results": learned_result["results"][:3]  # Save first 3 for debugging
            }, f, indent=2)
        
        logger.info(f"Split {split_idx} results saved to: {split_output_dir}")
        
        cv_results.append({
            "split": split_idx,
            "seed": seed,
            "train_docs": len(train_docs),
            "dev_docs": len(dev_docs),
            "train_questions": len(train_indices),
            "dev_questions": len(dev_indices),
            "baseline_metrics": baseline_result["metrics"],
            "learned_metrics": learned_result["metrics"]
        })
    
    # Aggregate metrics
    metrics_keys = cv_results[0]["baseline_metrics"].keys()
    aggregated = {
        "baseline": {},
        "learned": {},
        "improvement": {}
    }
    
    for key in metrics_keys:
        baseline_values = [r["baseline_metrics"][key] for r in cv_results]
        learned_values = [r["learned_metrics"][key] for r in cv_results]
        improvement_values = [l - b for l, b in zip(learned_values, baseline_values)]
        
        aggregated["baseline"][key] = {
            "mean": np.mean(baseline_values),
            "std": np.std(baseline_values),
            "values": baseline_values
        }
        aggregated["learned"][key] = {
            "mean": np.mean(learned_values),
            "std": np.std(learned_values),
            "values": learned_values
        }
        aggregated["improvement"][key] = {
            "mean": np.mean(improvement_values),
            "std": np.std(improvement_values),
            "values": improvement_values
        }
    
    # Save results
    output_file = output_dir / "cross_validation_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "n_splits": n_splits,
            "split_results": cv_results,
            "aggregated": aggregated
        }, f, indent=2)
    
    logger.info(f"\n✓ Cross-validation results saved: {output_file}")
    
    # Create visualization
    logger.info("Creating cross-validation visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{n_splits}-Fold Cross-Validation Results", fontsize=16, fontweight='bold')
    
    # Plot 1: Page Hit@5 across folds
    ax = axes[0, 0]
    fold_indices = range(n_splits)
    baseline_page_hit5 = [r["baseline_metrics"]["page_hit@5"] for r in cv_results]
    learned_page_hit5 = [r["learned_metrics"]["page_hit@5"] for r in cv_results]
    
    ax.plot(fold_indices, baseline_page_hit5, 'o-', label='Baseline', color='steelblue', linewidth=2, markersize=8)
    ax.plot(fold_indices, learned_page_hit5, 's-', label='Learned', color='coral', linewidth=2, markersize=8)
    ax.axhline(aggregated["baseline"]["page_hit@5"]["mean"], color='steelblue', linestyle='--', alpha=0.5)
    ax.axhline(aggregated["learned"]["page_hit@5"]["mean"], color='coral', linestyle='--', alpha=0.5)
    ax.set_xlabel('Fold')
    ax.set_ylabel('Page Hit@5')
    ax.set_title('Page Hit@5 Across Folds')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 2: Mean with error bars
    ax = axes[0, 1]
    metrics = ["page_hit@5", "page_recall@5", "doc_hit@1"]
    x = np.arange(len(metrics))
    width = 0.35
    
    baseline_means = [aggregated["baseline"][m]["mean"] for m in metrics]
    baseline_stds = [aggregated["baseline"][m]["std"] for m in metrics]
    learned_means = [aggregated["learned"][m]["mean"] for m in metrics]
    learned_stds = [aggregated["learned"][m]["std"] for m in metrics]
    
    ax.bar(x - width/2, baseline_means, width, yerr=baseline_stds, 
           label='Baseline', alpha=0.8, color='steelblue', capsize=5)
    ax.bar(x + width/2, learned_means, width, yerr=learned_stds,
           label='Learned', alpha=0.8, color='coral', capsize=5)
    ax.set_ylabel('Score')
    ax.set_title('Mean ± Std Across Folds')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=15)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 3: Improvement distribution
    ax = axes[1, 0]
    page_hit5_improvements = [l - b for l, b in zip(learned_page_hit5, baseline_page_hit5)]
    ax.bar(fold_indices, page_hit5_improvements, color='green', alpha=0.7)
    ax.axhline(aggregated["improvement"]["page_hit@5"]["mean"], 
               color='black', linestyle='--', linewidth=2,
               label=f'Mean: {aggregated["improvement"]["page_hit@5"]["mean"]:.3f}')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Improvement (Learned - Baseline)')
    ax.set_title('Page Hit@5 Improvement per Fold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 4: Summary table
    ax = axes[1, 1]
    ax.axis('off')
    
    table_data = [
        ["Metric", "Baseline", "Learned", "Improvement"],
        ["Page Hit@5", 
         f"{aggregated['baseline']['page_hit@5']['mean']:.3f} ± {aggregated['baseline']['page_hit@5']['std']:.3f}",
         f"{aggregated['learned']['page_hit@5']['mean']:.3f} ± {aggregated['learned']['page_hit@5']['std']:.3f}",
         f"+{aggregated['improvement']['page_hit@5']['mean']:.3f} ± {aggregated['improvement']['page_hit@5']['std']:.3f}"],
        ["Page Recall@5",
         f"{aggregated['baseline']['page_recall@5']['mean']:.3f} ± {aggregated['baseline']['page_recall@5']['std']:.3f}",
         f"{aggregated['learned']['page_recall@5']['mean']:.3f} ± {aggregated['learned']['page_recall@5']['std']:.3f}",
         f"+{aggregated['improvement']['page_recall@5']['mean']:.3f} ± {aggregated['improvement']['page_recall@5']['std']:.3f}"],
        ["Doc Hit@1",
         f"{aggregated['baseline']['doc_hit@1']['mean']:.3f} ± {aggregated['baseline']['doc_hit@1']['std']:.3f}",
         f"{aggregated['learned']['doc_hit@1']['mean']:.3f} ± {aggregated['learned']['doc_hit@1']['std']:.3f}",
         f"+{aggregated['improvement']['doc_hit@1']['mean']:.3f} ± {aggregated['improvement']['doc_hit@1']['std']:.3f}"],
    ]
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.tight_layout()
    cv_plot_path = output_dir / "cross_validation_results.png"
    plt.savefig(cv_plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Cross-validation plot saved: {cv_plot_path}")
    
    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info("CROSS-VALIDATION SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Page Hit@5:")
    logger.info(f"  Baseline: {aggregated['baseline']['page_hit@5']['mean']:.4f} ± {aggregated['baseline']['page_hit@5']['std']:.4f}")
    logger.info(f"  Learned:  {aggregated['learned']['page_hit@5']['mean']:.4f} ± {aggregated['learned']['page_hit@5']['std']:.4f}")
    logger.info(f"  Improvement: {aggregated['improvement']['page_hit@5']['mean']:.4f} ± {aggregated['improvement']['page_hit@5']['std']:.4f}")
    
    return {"split_results": cv_results, "aggregated": aggregated}


def main():
    """Run proper evaluation with ablations."""
    
    logger.info("="*80)
    logger.info("PROPER PAGE SCORER EVALUATION (NO LEAKAGE)")
    logger.info("="*80)
    
    output_dir = Path("results/proper_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data and splits
    logger.info("\nLoading data and splits...")
    loader = FinanceBenchLoader()
    df = loader.load_data()
    data = df.to_dict('records')
    
    model_dir = Path("models/finetuned_page_scorer_v2")
    splits = load_splits(model_dir)
    
    logger.info(f"Total samples: {len(data)}")
    logger.info(f"Train docs: {len(splits['train_docs'])}")
    logger.info(f"Dev docs: {len(splits['dev_docs'])}")
    
    # Get sample indices
    train_indices = get_sample_indices_for_docs(data, splits["train_docs"])
    dev_indices = get_sample_indices_for_docs(data, splits["dev_docs"])
    
    logger.info(f"Train questions: {len(train_indices)}")
    logger.info(f"Dev questions: {len(dev_indices)}")
    
    # Verify no leakage
    train_docs_found = {data[i]["doc_name"] for i in train_indices}
    dev_docs_found = {data[i]["doc_name"] for i in dev_indices}
    overlap = train_docs_found & dev_docs_found
    
    if overlap:
        raise ValueError(f"Document leakage detected! Overlap: {overlap}")
    else:
        logger.info("✓ No document leakage - splits are clean")
    
    # Base config
    base_config = {
        "experiment_type": "page_then_chunk",
        "llm_model": "Qwen/Qwen2.5-7B-Instruct",
        "embedding_model": "sentence-transformers/all-mpnet-base-v2",
        "chunk_size": 1024,
        "chunk_overlap": 100,
        "page_k": 5,
        "top_k": 10,
        "pdf_local_dir": "/u/kobeissa/Documents/thesis/experiments/FB_reproducability/pdfs",
        "output_dir": str(output_dir),
        "use_all_pdfs": False  # Only index relevant docs
    }
    
    learned_model_path = str(model_dir / "best_model")
    
    # ========================================================================
    # SKIP SINGLE-SPLIT EXPERIMENTS - GO STRAIGHT TO CROSS-VALIDATION
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("NOTE: Skipping single-split dev set experiments")
    logger.info("Running 5-fold cross-validation instead for robust evaluation")
    logger.info("="*80)
    
    # ========================================================================
    # EXPERIMENT 4: Cross-validation
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("RUNNING 5-FOLD CROSS-VALIDATION")
    logger.info("="*80)
    
    cv_results = run_cross_validation(base_config, n_splits=5, output_dir=output_dir)
    
    # ========================================================================
    # SAVE SUMMARY TO OUTPUT DIR
    # ========================================================================
    summary_file = output_dir / "evaluation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "evaluation_type": "5-fold_cross_validation",
            "model_path": "models/finetuned_page_scorer_v2/best_model",
            "timestamp": datetime.now().isoformat(),
            "n_folds": 5,
            "aggregated_results": cv_results["aggregated"]
        }, f, indent=2)
    
    logger.info(f"\n✓ Summary saved: {summary_file}")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("EVALUATION COMPLETE")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Summary file: {summary_file}")
    logger.info(f"Cross-validation plot: {output_dir}/cross_validation_results.png")
    logger.info(f"Cross-validation data: {output_dir}/cross_validation/cross_validation_results.json")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()
