"""
example_use_learned_page_scorer.py

Complete example of using trained page scorer in page-then-chunk retrieval pipeline.

Pipeline Overview:
1. Load trained page scorer model
2. Score all pages with learned embeddings
3. Retrieve top M pages
4. Chunk those M pages
5. Retrieve top K chunks from those pages
"""

import logging
from pathlib import Path
from typing import List, Dict, Any
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ingestion.data_loader import FinanceBenchLoader
from src.experiments.page_retrieval import run_page_then_chunk
from src.core.rag_experiments import RAGExperiment

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Example: Using trained page scorer for page-then-chunk retrieval.
    """
    
    # ========================================================================
    # STEP 1: Specify your trained model path
    # ========================================================================
    # This should point to the directory where you saved your trained model
    # After running train_page_scorer_v2.py, the model is saved to:
    TRAINED_MODEL_PATH = "models/finetuned_page_scorer_v2"
    
    # Verify the model exists
    model_path = Path(TRAINED_MODEL_PATH)
    if not model_path.exists():
        logger.error(f"❌ Trained model not found at: {TRAINED_MODEL_PATH}")
        logger.info("Train the model first using: python src/training/train_page_scorer_v2.py")
        return
    
    logger.info(f"✓ Found trained page scorer at: {TRAINED_MODEL_PATH}")
    
    # ========================================================================
    # STEP 2: Load your dataset
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("Loading FinanceBench dataset...")
    logger.info("="*80)
    
    loader = FinanceBenchLoader()
    data = loader.load_data()
    
    # Convert to list of dicts for the pipeline
    dataset = data.to_dict('records')
    
    logger.info(f"Loaded {len(dataset)} questions")
    
    # ========================================================================
    # STEP 3: Configure your experiment
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("Configuring RAG experiment...")
    logger.info("="*80)
    
    # Create experiment instance with your desired configuration
    experiment = RAGExperiment(
        # Model configuration (for answer generation)
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        
        # Chunk embedding model (used for stage 2: chunk scoring)
        # This is separate from the page scorer
        embedding_model_name="sentence-transformers/all-mpnet-base-v2",
        
        # Chunking configuration (for stage 3: chunking retrieved pages)
        chunk_size=1024,  # characters
        chunk_overlap=100,
        chunking_unit="chars",
        
        # Retrieval configuration
        page_k=5,   # ← Top M pages to retrieve in stage 1
        top_k=3,    # ← Top K chunks to use for generation (from those M pages)
        
        # Paths
        pdf_dir="pdfs",
        output_dir="outputs/learned_page_scorer",
        vector_store_dir="vector_stores",
        
        # Other settings
        use_all_pdfs=False,  # Only index PDFs relevant to the dataset
    )
    
    logger.info(f"✓ Experiment configured:")
    logger.info(f"  - Page retrieval (stage 1): Top {experiment.page_k} pages")
    logger.info(f"  - Chunk retrieval (stage 2): Top {experiment.top_k} chunks")
    logger.info(f"  - Chunk size: {experiment.chunk_size} {experiment.chunking_unit}")
    
    # ========================================================================
    # STEP 4: Run page-then-chunk retrieval with learned page scorer
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("Running Page-Then-Chunk Retrieval Pipeline with Learned Scorer")
    logger.info("="*80)
    logger.info("\nPipeline stages:")
    logger.info(f"  1. Score pages using trained model: {TRAINED_MODEL_PATH}")
    logger.info(f"  2. Retrieve top {experiment.page_k} pages")
    logger.info(f"  3. Chunk those {experiment.page_k} pages on-the-fly")
    logger.info(f"  4. Score chunks using base embeddings (NOT learned model)")
    logger.info(f"  5. Use top {experiment.top_k} chunks for answer generation")
    logger.info("")
    
    # Run the pipeline with your trained model for pages, base model for chunks
    results = run_page_then_chunk(
        experiment=experiment,
        data=dataset,
        learned_model_path=TRAINED_MODEL_PATH,  # ← Trained model for pages
        use_learned_for_chunks=False  # ← Use base model for chunks (RECOMMENDED)
    )
    
    logger.info("\nℹ️  NOTE: To use learned model for BOTH pages and chunks:")
    logger.info("   Set use_learned_for_chunks=True in run_page_then_chunk()")
    
    # ========================================================================
    # STEP 5: Save and display results
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("Pipeline Complete!")
    logger.info("="*80)
    
    # Save results
    import json
    from datetime import datetime
    
    output_path = Path(experiment.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_path / f"learned_page_scorer_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"✓ Results saved to: {output_file}")
    
    # Display sample results
    logger.info("\n" + "="*80)
    logger.info("Sample Results (first 3 questions):")
    logger.info("="*80)
    
    for i, result in enumerate(results[:3]):
        logger.info(f"\n--- Question {i+1} ---")
        logger.info(f"Question: {result['question']}")
        logger.info(f"Retrieved {result['num_retrieved']} chunks from pages: "
                   f"{result.get('metadata_page_retrieval', {}).get('retrieved_pages', [])}")
        logger.info(f"Generated Answer: {result['generated_answer'][:200]}...")
        logger.info(f"Reference Answer: {result['reference_answer'][:200]}...")
    
    # ========================================================================
    # STEP 6: Evaluate results (optional)
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("To evaluate these results, run:")
    logger.info("="*80)
    logger.info(f"python src/evaluation/evaluate_outputs.py --input {output_file}")
    
    return results


def run_comparison():
    """
    Compare learned page scorer vs baseline (no learned scorer).
    """
    logger.info("\n" + "="*80)
    logger.info("COMPARISON: Learned vs Baseline Page Scorer")
    logger.info("="*80)
    
    # Load data once
    loader = FinanceBenchLoader()
    data = loader.load_data()
    dataset = data.to_dict('records')[:10]  # Test on 10 samples
    
    # Shared config
    config = {
        "model_name": "meta-llama/Llama-3.2-3B-Instruct",
        "embedding_model_name": "sentence-transformers/all-mpnet-base-v2",
        "chunk_size": 1024,
        "chunk_overlap": 100,
        "page_k": 5,
        "top_k": 3,
        "pdf_dir": "pdfs",
        "vector_store_dir": "vector_stores",
    }
    
    # Run 1: Baseline (no learned scorer)
    logger.info("\n--- Running BASELINE (no learned scorer) ---")
    exp_baseline = RAGExperiment(**config, output_dir="outputs/baseline_comparison")
    results_baseline = run_page_then_chunk(
        experiment=exp_baseline,
        data=dataset,
        learned_model_path=None  # No learned model
    )
    
    # Run 2: Learned scorer
    logger.info("\n--- Running LEARNED PAGE SCORER ---")
    exp_learned = RAGExperiment(**config, output_dir="outputs/learned_comparison")
    results_learned = run_page_then_chunk(
        experiment=exp_learned,
        data=dataset,
        learned_model_path="models/finetuned_page_scorer_v2"
    )
    
    # Compare
    logger.info("\n" + "="*80)
    logger.info("Comparison Summary:")
    logger.info("="*80)
    
    for i in range(min(3, len(dataset))):
        logger.info(f"\n--- Question {i+1}: {dataset[i]['question'][:80]}... ---")
        
        baseline_pages = results_baseline[i].get('metadata_page_retrieval', {}).get('retrieved_pages', [])
        learned_pages = results_learned[i].get('metadata_page_retrieval', {}).get('retrieved_pages', [])
        
        logger.info(f"Baseline retrieved: {baseline_pages}")
        logger.info(f"Learned retrieved:  {learned_pages}")
        
        overlap = set(baseline_pages) & set(learned_pages)
        logger.info(f"Overlap: {len(overlap)}/{len(baseline_pages)} pages")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Use trained page scorer in retrieval pipeline")
    parser.add_argument(
        "--compare", 
        action="store_true",
        help="Run comparison between learned and baseline scorer"
    )
    
    args = parser.parse_args()
    
    if args.compare:
        run_comparison()
    else:
        main()
