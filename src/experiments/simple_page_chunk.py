"""
Simple Page-then-Chunk Retrieval
1. Extract all pages from PDFs (0-indexed)
2. Embed pages with embedding model
3. Retrieve top M pages (e.g., 20)
4. Chunk those M pages
5. Retrieve top K chunks (e.g., 5)
6. Generate answer with those chunks
"""

import os
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import re

try:
    import fitz  # PyMuPDF
    if not hasattr(fitz, "open"):
        raise ImportError("fitz missing open")
except Exception:  # pragma: no cover
    import pymupdf as fitz
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

logger = logging.getLogger(__name__)


def _normalize_doc_name(doc_name: str) -> str:
    """Normalize document name for consistent comparison."""
    if not doc_name:
        return ""
    name = str(doc_name).lower().strip()
    if name.endswith(".pdf"):
        name = name[:-4]
    return name


def _collect_unique_docs(data: List[Dict[str, Any]]) -> set:
    """Extract unique document names from dataset."""
    doc_names = set()
    for sample in data:
        doc_name = sample.get('doc_name') or sample.get('document')
        if doc_name:
            doc_names.add(_normalize_doc_name(doc_name))
    return doc_names


def extract_all_pages_from_pdfs(pdf_dir: Path, target_docs: Optional[set] = None) -> List[Dict[str, Any]]:
    """
    Extract all pages from PDFs in directory.
    
    Args:
        pdf_dir: Directory containing PDFs
        target_docs: Optional set of normalized doc names to index (None = index all)
    
    Returns:
        List of dicts with: {text, doc_name, page (0-indexed), pdf_path}
    """
    logger.info(f"Extracting pages from PDFs in: {pdf_dir}")
    
    all_pages = []
    all_pdf_files = list(pdf_dir.glob("**/*.pdf"))
    
    # Filter to only target documents if specified
    if target_docs:
        pdf_files = []
        for pdf_path in all_pdf_files:
            doc_name = _normalize_doc_name(pdf_path.stem)
            if doc_name in target_docs:
                pdf_files.append(pdf_path)
        logger.info(f"Filtered to {len(pdf_files)} PDFs matching dataset (out of {len(all_pdf_files)} total)")
    else:
        pdf_files = all_pdf_files
        logger.info(f"Found {len(pdf_files)} PDF files (indexing all)")
    
    for pdf_path in pdf_files:
        doc_name = pdf_path.stem
        try:
            doc = fitz.open(str(pdf_path))
            num_pages = len(doc)
            for page_idx in range(num_pages):
                page = doc[page_idx]
                text = page.get_text("text")
                
                all_pages.append({
                    "text": text,
                    "doc_name": doc_name,
                    "page": page_idx,  # 0-indexed
                    "pdf_path": str(pdf_path)
                })
            doc.close()
            logger.debug(f"Extracted {num_pages} pages from {doc_name}")
        except Exception as e:
            logger.error(f"Error extracting pages from {pdf_path}: {e}")
            continue
    
    logger.info(f"Total pages extracted: {len(all_pages)}")
    return all_pages


def embed_pages(pages: List[Dict[str, Any]], model_name: str = "BAAI/bge-m3") -> np.ndarray:
    """
    Embed all page texts using the specified model.
    Returns: numpy array of shape (num_pages, embedding_dim)
    """
    logger.info(f"Embedding {len(pages)} pages with {model_name}")
    
    model = SentenceTransformer(model_name)
    
    # Extract texts
    texts = [p["text"] for p in pages]
    
    # Embed in batches
    batch_size = 128
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        embeddings = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_embeddings.append(embeddings)
        if (i // batch_size + 1) % 10 == 0:
            logger.info(f"  Embedded batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
    
    embeddings = np.vstack(all_embeddings)
    logger.info(f"✓ Embeddings shape: {embeddings.shape}")
    
    return embeddings


def retrieve_top_pages(query: str, pages: List[Dict[str, Any]], 
                       page_embeddings: np.ndarray, model: SentenceTransformer,
                       top_m: int = 20) -> List[Dict[str, Any]]:
    """
    Retrieve top M pages by similarity to query.
    Returns: List of page dicts sorted by relevance
    """
    # Embed query
    query_embedding = model.encode([query], convert_to_numpy=True)[0]
    
    # Compute similarities
    similarities = np.dot(page_embeddings, query_embedding)
    
    # Get top M indices
    top_indices = np.argsort(similarities)[::-1][:top_m]
    
    # Return top pages with scores
    top_pages = []
    for idx in top_indices:
        page = pages[idx].copy()
        page["similarity_score"] = float(similarities[idx])
        top_pages.append(page)
    
    return top_pages


def chunk_pages(pages: List[Dict[str, Any]], chunk_size: int = 1024, 
                chunk_overlap: int = 128, chunking_unit: str = "chars") -> List[Dict[str, Any]]:
    """
    Chunk the given pages.
    Returns: List of chunk dicts with {text, doc_name, page, chunk_idx}
    """
    # Adjust sizes if using tokens
    if chunking_unit == "tokens":
        chunk_size = chunk_size * 4
        chunk_overlap = chunk_overlap * 4
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    
    all_chunks = []
    for page in pages:
        text = page["text"]
        if not text or not text.strip():
            continue
        
        chunks = text_splitter.split_text(text)
        for chunk_idx, chunk_text in enumerate(chunks):
            cleaned = re.sub(r"[ \t]+", " ", chunk_text).strip()
            if not cleaned:
                continue
            
            all_chunks.append({
                "text": cleaned,
                "doc_name": page["doc_name"],
                "page": page["page"],
                "chunk_idx": chunk_idx,
                "page_similarity": page.get("similarity_score", 0.0)
            })
    
    return all_chunks


def retrieve_top_chunks(query: str, chunks: List[Dict[str, Any]], 
                        model: SentenceTransformer, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve top K chunks by similarity to query.
    Returns: List of chunk dicts sorted by relevance
    """
    if not chunks:
        return []
    
    # Embed all chunks
    chunk_texts = [c["text"] for c in chunks]
    chunk_embeddings = model.encode(chunk_texts, convert_to_numpy=True, show_progress_bar=False)
    
    # Embed query
    query_embedding = model.encode([query], convert_to_numpy=True)[0]
    
    # Compute similarities
    similarities = np.dot(chunk_embeddings, query_embedding)
    
    # Get top K indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Return top chunks with scores
    top_chunks = []
    for idx in top_indices:
        chunk = chunks[idx].copy()
        chunk["chunk_similarity_score"] = float(similarities[idx])
        top_chunks.append(chunk)
    
    return top_chunks


def run_simple_page_chunk(experiment, data: List[Dict[str, Any]], 
                          top_m_pages: int = 20, top_k_chunks: int = 5) -> List[Dict[str, Any]]:
    """
    Run simple page-then-chunk retrieval.
    
    Args:
        experiment: RAGExperiment instance
        data: List of question samples
        top_m_pages: Number of pages to retrieve
        top_k_chunks: Number of chunks to retrieve
    
    Returns:
        List of result dicts ready for evaluation
    """
    logger.info("\n" + "=" * 80)
    logger.info("SIMPLE PAGE-THEN-CHUNK RETRIEVAL")
    logger.info(f"Strategy: Retrieve {top_m_pages} Pages → Chunk → Retrieve {top_k_chunks} Chunks")
    logger.info("=" * 80)
    
    # Collect unique documents from dataset
    target_docs = _collect_unique_docs(data)
    logger.info(f"\nDataset references {len(target_docs)} unique documents")
    if len(target_docs) <= 10:
        logger.info(f"  Documents: {sorted(target_docs)}")
    else:
        logger.info(f"  Sample documents: {list(sorted(target_docs))[:5]}...")
    
    # Get PDF directory
    pdf_dir = Path(getattr(experiment, "pdf_local_dir", None) or 
                   getattr(experiment, "pdf_dir", None) or "pdfs")
    
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")
    
    # Step 1: Extract all pages from PDFs (filtered to dataset docs)
    logger.info("\n[Step 1/6] Extracting pages from PDFs...")
    all_pages = extract_all_pages_from_pdfs(pdf_dir, target_docs=target_docs)
    
    if not all_pages:
        raise ValueError("No pages extracted from PDFs!")
    
    # Step 2: Embed all pages
    logger.info("\n[Step 2/6] Embedding pages...")
    embedding_model_name = experiment.embedding_model
    model = SentenceTransformer(embedding_model_name)
    page_embeddings = embed_pages(all_pages, embedding_model_name)
    
    # Get chunking parameters
    chunk_size = getattr(experiment, "chunk_size", 1024)
    chunk_overlap = getattr(experiment, "chunk_overlap", 128)
    chunking_unit = getattr(experiment, "chunking_unit", "chars")
    
    # Initialize results
    results = []
    
    logger.info(f"\n[Step 3/6] Processing {len(data)} questions...")
    
    for i, sample in enumerate(data):
        t0 = time.time()
        query = sample.get("question", "") or ""
        
        # Step 3: Retrieve top M pages for this query
        top_pages = retrieve_top_pages(query, all_pages, page_embeddings, model, top_m=top_m_pages)
        
        logger.info(f"\nSample {i}: Retrieved {len(top_pages)} pages")
        if top_pages:
            logger.debug(f"  Top page: {top_pages[0]['doc_name']}_p{top_pages[0]['page']} (score: {top_pages[0]['similarity_score']:.4f})")
        
        # Step 4: Chunk the top M pages
        chunks = chunk_pages(top_pages, chunk_size, chunk_overlap, chunking_unit)
        logger.info(f"Sample {i}: Generated {len(chunks)} chunks from {len(top_pages)} pages")
        
        # Step 5: Retrieve top K chunks from those chunks
        top_chunks = retrieve_top_chunks(query, chunks, model, top_k=top_k_chunks)
        logger.info(f"Sample {i}: Retrieved {len(top_chunks)} chunks")
        
        # Step 6: Prepare result
        result = {
            "sample_id": i,
            "doc_name": sample.get("doc_name", ""),
            "doc_link": sample.get("doc_link", ""),
            "doc_type": sample.get("doc_type", ""),
            "question": query,
            "reference_answer": sample.get("answer", ""),
            "question_type": sample.get("question_type", ""),
            "question_reasoning": sample.get("question_reasoning", ""),
            
            # Retrieved pages
            "retrieved_pages": [
                {
                    "doc_name": p["doc_name"],
                    "page": p["page"],
                    "score": p["similarity_score"]
                }
                for p in top_pages
            ],
            
            # Retrieved chunks for generation
            "retrieved_chunks": [
                {
                    "text": c["text"],
                    "metadata": {
                        "doc_name": c["doc_name"],
                        "page": c["page"],
                        "chunk_idx": c["chunk_idx"]
                    }
                }
                for c in top_chunks
            ],
            
            # Context for generation
            "context": "\n\n".join([c["text"] for c in top_chunks]),
            
            # Gold evidence for evaluation
            "gold_evidence": sample.get("evidence", []),
            
            # Timing
            "retrieval_time": time.time() - t0,
        }
        
        results.append(result)
        
        if (i + 1) % 10 == 0:
            logger.info(f"Completed {i + 1}/{len(data)} samples")
    
    logger.info("\n" + "=" * 80)
    logger.info(f"✓ Retrieval complete for {len(results)} samples")
    logger.info("=" * 80)
    
    # Step 7: Generate answers
    logger.info("\n[Step 4/6] Generating answers...")
    for i, result in enumerate(results):
        question = result["question"]
        context = result["context"]
        
        t0 = time.time()
        answer, prompt = experiment._generate_answer(question, context, return_prompt=True)
        gen_time = time.time() - t0
        
        result["generated_answer"] = answer
        result["prompt"] = prompt
        result["generation_time"] = gen_time
        result["generation_length"] = len(answer) if answer else 0
        
        if (i + 1) % 10 == 0:
            logger.info(f"Generated {i + 1}/{len(results)} answers")
    
    logger.info("\n" + "=" * 80)
    logger.info(f"✓ Generation complete")
    logger.info("=" * 80)
    
    return results
