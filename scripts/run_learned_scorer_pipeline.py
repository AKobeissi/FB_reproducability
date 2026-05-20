#!/usr/bin/env python3
"""
run_learned_scorer_pipeline.py

Simple script to run page-then-chunk retrieval with your trained page scorer.
Usage:
    python run_learned_scorer_pipeline.py                    # Run on full dataset
    python run_learned_scorer_pipeline.py --samples 10       # Test on 10 samples
    python run_learned_scorer_pipeline.py --page-k 10        # Retrieve 10 pages
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ingestion.data_loader import FinanceBenchLoader
from src.experiments.page_retrieval import run_page_then_chunk
from src.core.rag_experiments import RAGExperiment

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(args):
    """Run the pipeline with your trained page scorer."""
    
    # ========================================================================
    # Configuration
    # ========================================================================
    TRAINED_MODEL_PATH = args.model_path
    PAGE_K = args.page_k  # Top M pages to retrieve
    CHUNK_K = args.top_k   # Top K chunks to use
    CHUNK_SIZE = args.chunk_size
    CHUNK_OVERLAP = args.chunk_overlap
    NUM_SAMPLES = args.samples  # None = all samples
    
    # ========================================================================
    # Validate model exists
    # ========================================================================
    model_path = Path(TRAINED_MODEL_PATH)
    if not model_path.exists():
        logger.error(f"❌ Model not found: {TRAINED_MODEL_PATH}")
        logger.info("\nTrain the model first:")
        logger.info("  python src/training/train_page_scorer_v2.py")
        return 1
    
    logger.info("="*80)
    logger.info("PAGE-THEN-CHUNK RETRIEVAL WITH LEARNED PAGE SCORER")
    logger.info("="*80)
    logger.info(f"Model: {TRAINED_MODEL_PATH}")
    logger.info(f"Pipeline: Retrieve {PAGE_K} pages → Chunk → Use top {CHUNK_K} chunks")
    logger.info("")
    
    # ========================================================================
    # Load data
    # ========================================================================
    logger.info("Loading dataset...")
    loader = FinanceBenchLoader()
    data = loader.load_data()
    dataset = data.to_dict('records')
    
    if NUM_SAMPLES:
        dataset = dataset[:NUM_SAMPLES]
        logger.info(f"Testing on first {NUM_SAMPLES} samples")
    else:
        logger.info(f"Running on all {len(dataset)} samples")
    
    # ========================================================================
    # Configure experiment
    # ========================================================================
    logger.info("\nConfiguring experiment...")
    
    experiment = RAGExperiment(
        # Generation model
        model_name=args.model_name,
        
        # Embedding model (for chunks, NOT pages)
        embedding_model_name=args.embedding_model,
        
        # Retrieval params
        page_k=PAGE_K,
        top_k=CHUNK_K,
        
        # Chunking params
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        chunking_unit="chars",
        
        # Paths
        pdf_dir=args.pdf_dir,
        output_dir=args.output_dir,
        vector_store_dir=args.vector_store_dir,
        
        use_all_pdfs=False,
    )
    
    logger.info(f"  ✓ Page retrieval (stage 1): Top {PAGE_K} pages")
    logger.info(f"  ✓ Chunking (stage 3): {CHUNK_SIZE} chars, {CHUNK_OVERLAP} overlap")
    logger.info(f"  ✓ Chunk retrieval (stage 4): Top {CHUNK_K} chunks")
    logger.info(f"  ✓ Generation model: {args.model_name}")
    
    # ========================================================================
    # Run pipeline
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("STARTING PIPELINE")
    logger.info("="*80)
    
    try:
        results = run_page_then_chunk(
            experiment=experiment,
            data=dataset,
            learned_model_path=TRAINED_MODEL_PATH
        )
        
        logger.info("\n" + "="*80)
        logger.info("✓ PIPELINE COMPLETE")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # ========================================================================
    # Save results
    # ========================================================================
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_path / f"learned_scorer_p{PAGE_K}_k{CHUNK_K}_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✓ Results saved: {output_file}")
    
    # ========================================================================
    # Display summary
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    
    total_retrieved = sum(r['num_retrieved'] for r in results)
    avg_retrieved = total_retrieved / len(results) if results else 0
    
    total_context_length = sum(r.get('context_length', 0) for r in results)
    avg_context_length = total_context_length / len(results) if results else 0
    
    logger.info(f"Samples processed: {len(results)}")
    logger.info(f"Avg chunks retrieved: {avg_retrieved:.1f}")
    logger.info(f"Avg context length: {avg_context_length:.0f} chars")
    
    # Show first few examples
    logger.info("\n" + "-"*80)
    logger.info("SAMPLE RESULTS (first 3)")
    logger.info("-"*80)
    
    for i, result in enumerate(results[:3]):
        logger.info(f"\n[{i+1}] Question: {result['question']}")
        
        # Show which pages were retrieved
        page_meta = result.get('metadata_page_retrieval', {})
        retrieved_pages = page_meta.get('retrieved_pages', [])
        logger.info(f"    Retrieved pages: {retrieved_pages}")
        
        # Show chunks
        logger.info(f"    Chunks used: {result['num_retrieved']}")
        
        # Show answer preview
        answer = result.get('generated_answer', '')
        logger.info(f"    Answer: {answer[:150]}...")
    
    # ========================================================================
    # Next steps
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("NEXT STEPS")
    logger.info("="*80)
    logger.info("\n1. Evaluate retrieval quality:")
    logger.info(f"   python src/evaluation/evaluate_outputs.py --input {output_file}")
    logger.info("\n2. Evaluate answer quality:")
    logger.info(f"   python src/evaluation/generative_evaluator.py --input {output_file}")
    logger.info("\n3. Compare with baseline:")
    logger.info(f"   python {__file__} --no-learned-model")
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run page-then-chunk retrieval with learned page scorer"
    )
    
    # Model paths
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/finetuned_page_scorer_v2",
        help="Path to trained page scorer model"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="LLM for answer generation"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="Embedding model for chunk scoring (NOT used for pages)"
    )
    
    # Retrieval parameters
    parser.add_argument(
        "--page-k",
        type=int,
        default=5,
        help="Number of pages to retrieve (M)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of chunks to use for generation (K)"
    )
    
    # Chunking parameters
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
        help="Chunk size in characters"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Chunk overlap in characters"
    )
    
    # Data parameters
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of samples to process (default: all)"
    )
    
    # Paths
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default="pdfs",
        help="Directory containing PDFs"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/learned_page_scorer",
        help="Output directory for results"
    )
    parser.add_argument(
        "--vector-store-dir",
        type=str,
        default="vector_stores",
        help="Directory for vector stores"
    )
    
    args = parser.parse_args()
    
    sys.exit(main(args))
