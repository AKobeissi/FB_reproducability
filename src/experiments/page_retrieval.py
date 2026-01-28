"""
src/experiments/page_retrieval.py
Two-stage retrieval: Page Retrieval (Top-P) -> Chunk Scoring (Top-K).
Corrected for signature mismatches, metadata consistency, and caching.
"""
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from src.retrieval.vectorstore import build_chroma_store, populate_chroma_store
from src.ingestion.page_processor import extract_pages_from_pdf
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

def _get_chunk_map(experiment) -> Dict[str, List[Dict[str, Any]]]:
    """
    Builds an in-memory map of (doc_name, page_num) -> [Chunks].
    Handles pagination to ensure all chunks are fetched.
    """
    logger.info("Building in-memory chunk map for re-ranking...")
    
    # 1. Get the chunk vectorstore (lazy load to check existence)
    # Returns 3 values: retriever, vectordb, is_empty
    _, chunk_db, is_empty = build_chroma_store(experiment, "all", lazy_load=True)
    
    if is_empty:
        logger.warning("Chunk store is empty! Cannot map chunks.")
        return {}

    chunk_map = {}
    
    # 2. Robust Fetching with Pagination
    # Chroma's .get() can have limits; fetch in batches or use large limit
    try:
        # First get all IDs to know count
        all_ids = chunk_db.get()['ids']
        total_count = len(all_ids)
        batch_size = 5000
        
        logger.info(f"Fetching {total_count} chunks in batches of {batch_size}...")
        
        for i in range(0, total_count, batch_size):
            batch_ids = all_ids[i : i + batch_size]
            data = chunk_db.get(ids=batch_ids, include=['embeddings', 'metadatas', 'documents'])
            
            ids = data['ids']
            embeddings = data['embeddings']
            metadatas = data['metadatas']
            documents = data['documents']
            
            for j, doc_id in enumerate(ids):
                meta = metadatas[j] or {}
                doc_name = meta.get('doc_name')
                
                # Robust Page Number Extraction
                # Try common keys: 'page', 'page_num', 'page_number', 'page_ix'
                page_raw = meta.get('page')
                if page_raw is None:
                    page_raw = meta.get('page_num', meta.get('page_number', meta.get('page_ix')))
                
                try:
                    page = int(page_raw) if page_raw is not None else -1
                except:
                    page = -1
                    
                key = (doc_name, page)
                
                chunk_obj = {
                    "text": documents[j],
                    "metadata": meta,
                    "embedding": embeddings[j],
                    "id": doc_id
                }
                
                if key not in chunk_map:
                    chunk_map[key] = []
                chunk_map[key].append(chunk_obj)
                
    except Exception as e:
        logger.error(f"Failed to fetch data from Chroma: {e}")
        return {}
        
    logger.info(f"Mapped chunks for {len(chunk_map)} pages.")
    return chunk_map


def run_page_then_chunk(
    experiment, 
    data: List[Dict[str, Any]], 
    learned_model_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Execute Page-First Retrieval Experiment.
    """
    
    # --- 1. Initialize/Load Page Store ---
    
    # Setup Embeddings for Pages
    if learned_model_path:
        logger.info(f"Loading Learned Page Scorer from: {learned_model_path}")
        from langchain.embeddings.base import Embeddings
        class STEmbeddings(Embeddings):
            def __init__(self, path):
                self.model = SentenceTransformer(path)
            def embed_documents(self, texts):
                return self.model.encode(texts, convert_to_numpy=True).tolist()
            def embed_query(self, text):
                return self.model.encode([text], convert_to_numpy=True)[0].tolist()
        
        page_embeddings = STEmbeddings(learned_model_path)
        collection_suffix = "learned"
    else:
        page_embeddings = experiment.embeddings
        collection_suffix = "baseline"

    # Check if DB exists first (lazy_load=True)
    db_name = f"pages_{collection_suffix}"
    # FIX: Correct unpacking (3 values)
    _, page_vectordb, is_empty = build_chroma_store(
        experiment, 
        db_name,
        embeddings=page_embeddings,
        lazy_load=True
    )

    # Only extract and populate if it's actually empty/new
    if is_empty:
        logger.info(f"Page store '{db_name}' is empty. Extracting pages...")
        all_pages = []
        if experiment.use_all_pdfs:
            pdf_files = list(experiment.pdf_local_dir.glob("*.pdf"))
            logger.info(f"Processing {len(pdf_files)} PDFs...")
            for pdf in pdf_files:
                doc_name = pdf.stem 
                # extract_pages_from_pdf returns list of dicts with 'digest'
                pages = extract_pages_from_pdf(pdf, doc_name)
                all_pages.extend(pages)
        
        page_docs = []
        for p in all_pages:
            # FIX: Do not store full raw text in metadata to avoid bloat
            # We index the 'digest'
            d = Document(
                page_content=p['digest'], 
                metadata={
                    "doc_name": p['doc_name'],
                    "page": p['page'], # int
                    "source_path": p.get('source', '')
                    # Removed "source_text"
                }
            )
            page_docs.append(d)
        
        # Populate using the existing helper
        populate_chroma_store(experiment, page_vectordb, page_docs, db_name)
        logger.info(f"Populated page store with {len(page_docs)} pages.")
    else:
        logger.info(f"Using existing page store '{db_name}'.")

    # --- 2. Prepare Chunk Map ---
    # Used for the second stage scoring
    chunk_map = _get_chunk_map(experiment)

    # --- 3. Run Inference ---
    results = []
    page_k = 5
    chunk_k = experiment.top_k
    
    logger.info(f"Running Inference: Top-{page_k} Pages -> Top-{chunk_k} Chunks")
    
    for sample in data:
        query = sample['question']
        
        # Step A: Retrieve Pages
        retrieved_pages = page_vectordb.similarity_search(query, k=page_k)
        
        selected_pages_meta = []
        candidate_chunks = []
        
        for p in retrieved_pages:
            d_name = p.metadata.get('doc_name')
            p_num = p.metadata.get('page')
            selected_pages_meta.append((d_name, p_num))
            
            key = (d_name, p_num)
            if key in chunk_map:
                candidate_chunks.extend(chunk_map[key])
        
        # Step B: Score Chunks
        final_chunks = []
        
        if candidate_chunks:
            # FIX: Robust Cosine Similarity (Normalize vectors)
            q_vec = np.array(experiment.embeddings.embed_query(query), dtype=np.float32)
            q_norm = np.linalg.norm(q_vec)
            if q_norm > 1e-12:
                q_vec /= q_norm
            
            scores = []
            for c in candidate_chunks:
                c_vec = np.array(c['embedding'], dtype=np.float32)
                c_norm = np.linalg.norm(c_vec)
                if c_norm > 1e-12:
                    c_vec /= c_norm
                
                # Dot product of normalized vectors = Cosine Similarity
                score = float(np.dot(q_vec, c_vec))
                scores.append((score, c))
            
            # Sort descending
            scores.sort(key=lambda x: x[0], reverse=True)
            top_chunks = scores[:chunk_k]
            
            for score, c in top_chunks:
                final_chunks.append({
                    "text": c['text'],
                    "metadata": c['metadata'],
                    "score": score
                })
                
        # Generate Answer
        model_answer = "Generation skipped"
        if experiment.llm_model:
            context = "\n\n".join([c['text'] for c in final_chunks])
            prompt = experiment.format_prompt(query, context)
            model_answer = experiment.generate_response(prompt)

        results.append({
            **sample,
            "retrieved_chunks": final_chunks,
            "retrieved_pages": [f"{p[0]}_p{p[1]}" for p in selected_pages_meta],
            "model_answer": model_answer
        })
        
        if len(results) % 10 == 0:
            logger.info(f"Processed {len(results)} samples")

    return results