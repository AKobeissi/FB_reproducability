"""
BM25 Retrieval Runner - Improved Implementation.

1. Robust Caching: Uses corpus fingerprint (files+size+mtime) to detect stale data.
2. Safe Serialization: Pickles only the Document chunks, not the Retriever object.
3. Finance Tokenization: Custom preprocessing for financial terms (numbers, currency).
4. BM25 Scores: access underlying rank_bm25 to return scores.
5. Context Safety: Hard limits on generation context size.
6. Stable IDs: Generates metadata IDs for debugging.
"""
import os
import logging
import pickle
import hashlib
import re
import numpy as np
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from pathlib import Path

# Imports
from .rag_dependencies import BM25Retriever
try:
    from langchain_community.document_loaders import PyMuPDFLoader
except ImportError:
    try:
        from langchain.document_loaders import PyMuPDFLoader
    except ImportError:
        PyMuPDFLoader = None

logger = logging.getLogger(__name__)

# --- 1. Helper: Corpus Fingerprint (Fix for Stale Index) ---
def _compute_corpus_fingerprint(pdf_dir: Path, extension: str = "*.pdf") -> str:
    """
    Generate a stable hash based on file content metadata (Name + Size + MTime).
    This ensures that if a PDF is changed/added/removed, the cache is invalidated.
    """
    files = sorted(list(pdf_dir.glob(extension)))
    if not files:
        return "empty_corpus"
    
    parts = []
    for p in files:
        try:
            st = p.stat()
            # signature: filename:size:mtime
            parts.append(f"{p.name}:{st.st_size}:{int(st.st_mtime)}")
        except OSError:
            pass # Skip files that vanished during run
            
    # MD5 is sufficient for cache invalidation
    return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()[:12]

# --- 2. Helper: Finance Tokenization (Fix for precision) ---
def finance_preprocess_func(text: str) -> List[str]:
    """
    Custom tokenizer for Finance.
    - Lowercases text.
    - Preserves numbers, currency symbols ($, €, £), and percentages.
    - Normalizes "1,000" -> "1000" is optional, but often keeping commas is safer for distinct string matching 
      unless we do specific number parsing. Here we keep it simple but preserve typical finance tokens.
    """
    if not text:
        return []
    text = text.lower()
    # Regex logic:
    # 1. Matches currency symbols followed by numbers: [$€£]\d+
    # 2. Matches percentages: \d+%
    # 3. Matches words/numbers (alphanumeric including dots/commas inside numbers): \w+(?:[.,]\w+)*
    # This is a simplified tokenizer.
    tokens = re.findall(r"\b\w+(?:[.,]\w+)*\b|[$€£%]", text)
    return tokens

# --- 3. Helper: Cache Paths ---
def _get_chunk_cache_path(experiment, fingerprint: str) -> str:
    """
    Cache path includes Chunk Size, Overlap, AND Corpus Fingerprint.
    """
    filename = (
        f"chunks_sz{experiment.chunk_size}_"
        f"ol{experiment.chunk_overlap}_"
        f"fp{fingerprint}.pkl"
    )
    return os.path.join(experiment.vector_store_dir, "bm25_cache", filename)

def run_bm25(experiment, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Run the BM25 experiment with enhanced features.
    """
    # 1. Validation
    if BM25Retriever is None:
        raise ImportError("BM25Retriever not available. Please install rank_bm25: 'pip install rank_bm25'")
    
    if PyMuPDFLoader is None:
        raise RuntimeError("PyMuPDFLoader is not available. Install pymupdf.")

    # Fix Path Typing (Review Point 4)
    pdf_dir = Path(experiment.pdf_local_dir)
    if not pdf_dir.exists():
        logger.error(f"PDF Directory not found: {pdf_dir}")
        return []

    # 2. Cache Management
    fingerprint = _compute_corpus_fingerprint(pdf_dir)
    cache_path = _get_chunk_cache_path(experiment, fingerprint)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    chunks = None
    
    # Try Loading Cache
    if os.path.exists(cache_path):
        logger.info(f"Loading cached chunks from: {cache_path}")
        try:
            with open(cache_path, 'rb') as f:
                chunks = pickle.load(f)
            logger.info(f"Loaded {len(chunks)} chunks from cache.")
        except Exception as e:
            logger.error(f"Failed to load cache: {e}. Will rebuild.")
            chunks = None

    # 3. Build Chunks (if not cached)
    if chunks is None:
        logger.info("Building Corpus Chunks (this happens in-memory)...")
        
        pdf_files = sorted(list(pdf_dir.glob("*.pdf")))
        if not pdf_files:
            logger.error(f"No PDF files found in {pdf_dir}")
            return []

        all_docs = []
        for pdf_path in tqdm(pdf_files, desc="Loading PDFs"):
            try:
                loader = PyMuPDFLoader(str(pdf_path))
                docs = loader.load()
                # Clean metadata
                for d in docs:
                    d.metadata["doc_name"] = pdf_path.name
                    d.metadata["source"] = str(pdf_path.name)
                    # Ensure page is present
                    if "page" not in d.metadata:
                        d.metadata["page"] = -1
                all_docs.extend(docs)
            except Exception as e:
                logger.error(f"Failed to load {pdf_path}: {e}")
        
        if not all_docs:
            return []

        # Chunking
        logger.info(f"Chunking {len(all_docs)} documents (Size={experiment.chunk_size})...")
        chunks = experiment.text_splitter.split_documents(all_docs)
        
        # Add Stable IDs (Review Point 7)
        for idx, chunk in enumerate(chunks):
            doc_name = chunk.metadata.get("doc_name", "unknown")
            page = chunk.metadata.get("page", "x")
            # Create a simple stable-ish ID. 
            # Note: For perfect stability across runs if files are reordered, 
            # we'd need to hash the content, but this is good for debugging within a run.
            chunk.metadata["chunk_id"] = f"{doc_name}:{page}:{idx}"

        logger.info(f"Generated {len(chunks)} chunks.")

        # Save to Cache (Safe Pickle of List[Document])
        logger.info(f"Saving chunks to {cache_path}...")
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(chunks, f)
        except Exception as e:
            logger.warning(f"Could not save chunk cache: {e}")

    # 4. Initialize Retriever (Rebuild Index)
    # We rebuild the BM25 index in memory every time. It's fast (~seconds) and avoids pickling lambda functions.
    logger.info("Initializing BM25Retriever with Finance Preprocessing...")
    
    # NOTE: BM25Retriever.from_documents automatically tokenizes 'chunks' using preprocess_func
    retriever = BM25Retriever.from_documents(
        chunks, 
        preprocess_func=finance_preprocess_func
    )
    retriever.k = experiment.top_k

    # 5. Run Inference
    results = experiment._create_skipped_results(
        data, "various", "various", "local_pdfs", experiment.experiment_type, start_id=0
    )
    
    logger.info(f"Processing {len(data)} questions with BM25...")
    
    # Context Limit Constant (Review Point 5)
    MAX_CONTEXT_CHARS = 15000 
    
    for i, item in enumerate(tqdm(data, desc="Inference")):
        question = item.get("question")
        if not question:
            continue
        
        # A. Retrieval with Scores (Review Point 3)
        # BM25Retriever standard invoke() doesn't return scores.
        # We access the internal 'vectorizer' (rank_bm25 object) to get scores.
        
        # 1. Preprocess query
        processed_query = retriever.preprocess_func(question)
        
        # 2. Get all scores from rank_bm25
        doc_scores = retriever.vectorizer.get_scores(processed_query)
        
        # 3. Sort and get top_k indices
        top_n_indices = np.argsort(doc_scores)[::-1][:experiment.top_k]
        
        # 4. Fetch docs and scores
        retrieved_chunks_formatted = []
        retrieved_texts = []
        
        for idx in top_n_indices:
            doc = retriever.docs[idx]
            score = float(doc_scores[idx])
            
            chunk_obj = {
                "text": doc.page_content,
                "metadata": doc.metadata,
                "score": score, # Exposing the real BM25 score
                "chunk_id": doc.metadata.get("chunk_id")
            }
            retrieved_chunks_formatted.append(chunk_obj)
            retrieved_texts.append(doc.page_content)
        
        # B. Context Construction with Cap (Review Point 5)
        raw_context = "\n\n".join(retrieved_texts)
        if len(raw_context) > MAX_CONTEXT_CHARS:
            context_text = raw_context[:MAX_CONTEXT_CHARS] + "...(truncated)"
        else:
            context_text = raw_context

        # C. Generation
        # (Skip generation if we are just evaluating retrieval, but here we include it)
        answer, prompt = experiment._generate_answer(
            question, 
            context_text, 
            mode=experiment.experiment_type, 
            return_prompt=True
        )

        # D. Gold Evidence Prep
        gold_segments, gold_text = experiment._prepare_gold_evidence(item.get("evidence"))
        results[i]['gold_evidence'] = gold_text
        results[i]['gold_evidence_segments'] = gold_segments

        # E. Store Results
        results[i]['retrieved_chunks'] = retrieved_chunks_formatted
        results[i]['num_retrieved'] = len(retrieved_chunks_formatted)
        results[i]['context_length'] = len(context_text)
        results[i]['generated_answer'] = answer
        results[i]['final_prompt'] = prompt
        
        # Fix: Ensure notification happens
        experiment.notify_sample_complete(1)
        
    return results