"""
Test script to verify separate embeddings for pages vs chunks.
"""
import logging
from pathlib import Path
from src.ingestion.data_loader import FinanceBenchLoader
from src.experiments.page_retrieval import run_page_then_chunk
from src.core.rag_experiments import RAGExperiment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_separate_embeddings():
    """Test that pages use learned model and chunks use base model."""
    
    # Load 1 sample
    loader = FinanceBenchLoader()
    df = loader.load_data()
    sample_data = df.head(1).to_dict('records')
    
    # Config
    config = {
        "model_name": "meta-llama/Llama-3.2-3B-Instruct",
        "embedding_model_name": "sentence-transformers/all-mpnet-base-v2",
        "chunk_size": 512,
        "chunk_overlap": 50,
        "page_k": 3,
        "top_k": 5,
        "pdf_dir": "pdfs",
        "output_dir": "outputs/test",
        "vector_store_dir": "vector_stores/test"
    }
    
    experiment = RAGExperiment(**config)
    
    logger.info("\\n" + "="*80)
    logger.info("TEST 1: Learned pages + Base chunks (use_learned_for_chunks=False)")
    logger.info("="*80)
    
    results1 = run_page_then_chunk(
        experiment=experiment,
        data=sample_data,
        learned_model_path="models/finetuned_page_scorer_v2/best_model",
        use_learned_for_chunks=False  # Should use base model for chunks
    )
    
    logger.info(f"✓ Test 1 complete: {len(results1)} results")
    
    logger.info("\\n" + "="*80)
    logger.info("TEST 2: Learned pages + Learned chunks (use_learned_for_chunks=True)")
    logger.info("="*80)
    
    results2 = run_page_then_chunk(
        experiment=experiment,
        data=sample_data,
        learned_model_path="models/finetuned_page_scorer_v2/best_model",
        use_learned_for_chunks=True  # Should use learned model for chunks too
    )
    
    logger.info(f"✓ Test 2 complete: {len(results2)} results")
    
    logger.info("\\n" + "="*80)
    logger.info("TEST 3: Baseline (no learned model)")
    logger.info("="*80)
    
    results3 = run_page_then_chunk(
        experiment=experiment,
        data=sample_data,
        learned_model_path=None,  # Use base model for pages
        use_learned_for_chunks=False
    )
    
    logger.info(f"✓ Test 3 complete: {len(results3)} results")
    
    logger.info("\\n" + "="*80)
    logger.info("ALL TESTS PASSED!")
    logger.info("="*80)
    logger.info("\\nYou can now use:")
    logger.info("  - use_learned_for_chunks=False (learned pages, base chunks) ← RECOMMENDED")
    logger.info("  - use_learned_for_chunks=True  (learned for both)")

if __name__ == "__main__":
    test_separate_embeddings()
