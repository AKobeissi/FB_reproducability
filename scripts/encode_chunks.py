#!/usr/bin/env python3
"""
Encode all document chunks from PDFs with BGE-M3 embeddings.
This creates a complete corpus of chunk embeddings for geometric analysis.
"""

import argparse
import json
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_documents(data_path: str, pdfs_dir: str):
    """Load all PDFs referenced in the data file."""
    logger.info(f"Loading data from {data_path}")
    
    # Load questions to get PDF list
    with open(data_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    # Get unique PDF names
    pdf_names = set()
    for item in data:
        pdf_names.add(item['doc_name'])
    
    logger.info(f"Found {len(pdf_names)} unique PDFs")
    
    # Load all PDFs
    all_docs = []
    pdf_path = Path(pdfs_dir)
    
    for pdf_name in tqdm(pdf_names, desc="Loading PDFs"):
        pdf_file = pdf_path / f"{pdf_name}.pdf"
        if not pdf_file.exists():
            logger.warning(f"PDF not found: {pdf_file}")
            continue
        
        try:
            loader = PyMuPDFLoader(str(pdf_file))
            docs = loader.load()
            all_docs.extend(docs)
        except Exception as e:
            logger.error(f"Error loading {pdf_file}: {e}")
    
    logger.info(f"Loaded {len(all_docs)} pages from PDFs")
    return all_docs


def chunk_documents(docs, chunk_size: int = 1024, chunk_overlap: int = 128, chunking_unit: str = "chars"):
    """Split documents into chunks using RecursiveCharacterTextSplitter."""
    logger.info(f"Chunking documents (unit={chunking_unit}, size={chunk_size}, overlap={chunk_overlap})")
    
    # For tokens, we need a tokenizer-based length function
    if chunking_unit == "tokens":
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
            length_function = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
            logger.info("Using tokenizer-based length function for tokens")
        except Exception as e:
            logger.warning(f"Could not load tokenizer: {e}. Falling back to character-based.")
            length_function = len
    else:
        length_function = len
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_function,
        is_separator_regex=False,
    )
    
    chunks = text_splitter.split_documents(docs)
    logger.info(f"Created {len(chunks)} chunks")
    
    return chunks


def encode_chunks(chunks, model_name: str, use_sentence_transformers: bool = True):
    """Encode all chunks with the specified embedding model."""
    logger.info(f"Encoding {len(chunks)} chunks with {model_name}")
    
    # Extract text from chunks
    texts = [chunk.page_content for chunk in chunks]
    
    if use_sentence_transformers:
        logger.info("Using sentence-transformers")
        from sentence_transformers import SentenceTransformer
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        if device == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        
        model = SentenceTransformer(model_name, device=device)
        
        # Use larger batch size for GPU
        batch_size = 128 if device == "cuda" else 32
        logger.info(f"Batch size: {batch_size}")
        
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
    else:
        logger.info("Using LangChain HuggingFaceEmbeddings")
        from langchain.embeddings import HuggingFaceEmbeddings
        
        model = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={'normalize_embeddings': True}
        )
        
        embeddings = []
        for text in tqdm(texts, desc="Encoding chunks"):
            emb = model.embed_query(text)
            embeddings.append(emb)
        embeddings = np.array(embeddings)
    
    logger.info(f"Chunk embeddings shape: {embeddings.shape}")
    return embeddings, texts


def main():
    parser = argparse.ArgumentParser(description="Encode document chunks for geometric analysis")
    parser.add_argument("--embedding-model", type=str, default="BAAI/bge-m3",
                        help="HuggingFace model name for embeddings")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to financebench_open_source.jsonl")
    parser.add_argument("--pdfs-dir", type=str, default="pdfs",
                        help="Directory containing PDF files")
    parser.add_argument("--chunk-size", type=int, default=1024,
                        help="Chunk size for text splitting")
    parser.add_argument("--chunk-overlap", type=int, default=128,
                        help="Chunk overlap for text splitting")
    parser.add_argument("--chunking-unit", type=str, default="chars", choices=["chars", "tokens"],
                        help="Unit for chunking: chars or tokens")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for embeddings")
    parser.add_argument("--use-sentence-transformers", action="store_true",
                        help="Use sentence-transformers instead of LangChain")
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info(f"ENCODING CHUNKS FOR: {args.embedding_model}")
    logger.info("=" * 80)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    embeddings_dir = output_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and chunk documents, args.chunking_unit
    docs = load_documents(args.data_path, args.pdfs_dir)
    chunks = chunk_documents(docs, args.chunk_size, args.chunk_overlap)
    
    # Encode chunks
    chunk_embeddings, chunk_texts = encode_chunks(
        chunks,
        args.embedding_model,
        args.use_sentence_transformers
    )
    
    # Save embeddings
    chunk_emb_path = embeddings_dir / "chunk_embeddings.npy"
    np.save(chunk_emb_path, chunk_embeddings)
    logger.info(f"Saved chunk embeddings to {chunk_emb_path}")
    
    # Save chunk texts for reference
    chunk_texts_path = embeddings_dir / "chunk_texts.json"
    with open(chunk_texts_path, 'w') as f:
        json.dump(chunk_texts, f, indent=2)
    logger.info(f"Saved chunk texts to {chunk_texts_path}")
    
    # Also encode queries for convenience
    logger.info("Encoding queries...")
    with open(args.data_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    queries = [item['question'] for item in data]
    logger.info(f"Loaded {len(queries)} queries")
    
    if args.use_sentence_transformers:
        from sentence_transformers import SentenceTransformer
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(args.embedding_model, device=device)
        batch_size = 128 if device == "cuda" else 32
        
        query_embeddings = model.encode(
            queries,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
    else:
        from langchain.embeddings import HuggingFaceEmbeddings
        model = HuggingFaceEmbeddings(
            model_name=args.embedding_model,
            encode_kwargs={'normalize_embeddings': True}
        )
        query_embeddings = []
        for query in tqdm(queries, desc="Encoding queries"):
            emb = model.embed_query(query)
            query_embeddings.append(emb)
        query_embeddings = np.array(query_embeddings)
    
    logger.info(f"Query embeddings shape: {query_embeddings.shape}")
    
    # Save query embeddings
    query_emb_path = embeddings_dir / "query_embeddings.npy"
    np.save(query_emb_path, query_embeddings)
    logger.info(f"Saved query embeddings to {query_emb_path}")
    
    logger.info("=" * 80)
    logger.info("ENCODING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Chunk embeddings: {chunk_embeddings.shape}")
    logger.info(f"Query embeddings: {query_embeddings.shape}")
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
