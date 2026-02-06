"""
Extract and save embeddings from existing experiments for geometric analysis.
This script loads your retrieval experiments and saves the embeddings.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import json
import logging
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_financebench_data(data_path: str = "data/financebench_open_source.jsonl") -> List[Dict]:
    """Load FinanceBench dataset."""
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def extract_embeddings_for_experiment(
    embedding_model_name: str,
    data_path: str,
    vector_store_path: str,
    output_dir: str,
    use_sentence_transformers: bool = True
):
    """
    Extract embeddings from an experiment.
    
    Args:
        embedding_model_name: Name of embedding model (e.g., 'BAAI/bge-m3')
        data_path: Path to FinanceBench data
        vector_store_path: Path to Chroma vectorstore (if available)
        output_dir: Where to save embeddings
        use_sentence_transformers: Use sentence-transformers library
    """
    logger.info("="*80)
    logger.info(f"EXTRACTING EMBEDDINGS FOR: {embedding_model_name}")
    logger.info("="*80)
    
    # Load data
    logger.info(f"Loading data from {data_path}")
    data = load_financebench_data(data_path)
    queries = [sample['question'] for sample in data]
    logger.info(f"Loaded {len(queries)} queries")
    
    # Load embedding model
    logger.info(f"Loading embedding model: {embedding_model_name}")
    
    if use_sentence_transformers:
        model = SentenceTransformer(embedding_model_name, trust_remote_code=True)
        logger.info("Using sentence-transformers")
        
        # Encode queries
        logger.info("Encoding queries...")
        query_embeddings = model.encode(queries, show_progress_bar=True, convert_to_numpy=True)
        
    else:
        # Use LangChain embeddings
        model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cuda', 'trust_remote_code': True},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("Using LangChain HuggingFaceEmbeddings")
        
        # Encode queries
        logger.info("Encoding queries...")
        query_embeddings = []
        for query in tqdm(queries, desc="Encoding queries"):
            emb = model.embed_query(query)
            query_embeddings.append(emb)
        query_embeddings = np.array(query_embeddings)
    
    logger.info(f"Query embeddings shape: {query_embeddings.shape}")
    
    # Try to load chunk embeddings from vectorstore
    chunk_embeddings = None
    
    if vector_store_path and Path(vector_store_path).exists():
        logger.info(f"Loading chunk embeddings from vectorstore: {vector_store_path}")
        
        try:
            import chromadb
            client = chromadb.PersistentClient(path=vector_store_path)
            collections = client.list_collections()
            
            if len(collections) > 0:
                collection = collections[0]
                logger.info(f"Loading from collection: {collection.name}")
                
                # Get all embeddings
                results = collection.get(include=['embeddings'])
                if results['embeddings'] is not None and len(results['embeddings']) > 0:
                    chunk_embeddings = np.array(results['embeddings'])
                    logger.info(f"Loaded {len(chunk_embeddings)} chunk embeddings from vectorstore")
                    logger.info(f"Chunk embeddings shape: {chunk_embeddings.shape}")
                else:
                    logger.warning(f"Collection '{collection.name}' exists but contains no embeddings")
        except Exception as e:
            logger.warning(f"Could not load from vectorstore: {e}")
    
    # If no vectorstore, encode all PDFs (expensive but necessary)
    if chunk_embeddings is None:
        logger.warning("No vectorstore found. You need to provide chunk embeddings separately.")
        logger.warning("Options:")
        logger.warning("  1. Run your retrieval experiment first to generate vectorstore")
        logger.warning("  2. Manually encode all chunks from your corpus")
        logger.warning("\nFor now, saving only query embeddings...")
    
    # Save embeddings
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    emb_dir = output_path / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(emb_dir / "query_embeddings.npy", query_embeddings)
    logger.info(f"Saved query embeddings to {emb_dir}/query_embeddings.npy")
    
    if chunk_embeddings is not None:
        np.save(emb_dir / "chunk_embeddings.npy", chunk_embeddings)
        logger.info(f"Saved chunk embeddings to {emb_dir}/chunk_embeddings.npy")
    
    logger.info("="*80)
    logger.info("EXTRACTION COMPLETE")
    logger.info("="*80)
    
    return query_embeddings, chunk_embeddings


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract embeddings for geometric analysis")
    parser.add_argument('--embedding-model', type=str, default='BAAI/bge-m3',
                       help='Embedding model name')
    parser.add_argument('--data-path', type=str, 
                       default='data/financebench_open_source.jsonl',
                       help='Path to FinanceBench data')
    parser.add_argument('--vector-store-path', type=str, default=None,
                       help='Path to Chroma vectorstore (optional)')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for embeddings')
    parser.add_argument('--use-sentence-transformers', action='store_true',
                       help='Use sentence-transformers library (faster)')
    
    args = parser.parse_args()
    
    extract_embeddings_for_experiment(
        embedding_model_name=args.embedding_model,
        data_path=args.data_path,
        vector_store_path=args.vector_store_path,
        output_dir=args.output_dir,
        use_sentence_transformers=args.use_sentence_transformers
    )


if __name__ == "__main__":
    main()
