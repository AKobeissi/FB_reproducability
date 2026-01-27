"""
Experiment: Metadata Extraction & Propagation + Metadata-Aware Reranking.
Part A: Page-aware chunking with metadata.
Part B: Reranking using metadata injection.
"""
import logging
import os
import torch
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm
from sentence_transformers import CrossEncoder

from src.ingestion.pdf_utils import load_pdf_with_fallback
from src.ingestion.chunking import chunk_docs_page_aware
from src.retrieval.vectorstore import build_chroma_store, populate_chroma_store
from src.utils.metadata_helper import MetadataProcessor, RerankerCache

logger = logging.getLogger(__name__)

class MetadataAwareReranker:
    def __init__(self, model_id: str, cache_dir: str):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Reranker {model_id} on {self.device}...")
        self.model = CrossEncoder(
            model_id,
            device=self.device,
            trust_remote_code=True,
            max_length=512 # Ensure consistent length
        )
        self.cache = RerankerCache(cache_dir)
        self.processor = MetadataProcessor()

    def score_batch(self, query: str, chunks: List[Any], use_meta: bool, meta_fields: List[str]) -> List[float]:
        """
        B1. Batch scoring with Caching.
        """
        scores = [None] * len(chunks)
        indices_to_compute = []
        pairs_to_compute = []
        
        # 1. Check Cache
        for i, chunk in enumerate(chunks):
            chunk_id = chunk.metadata.get("chunk_id", "unknown")
            cached_score = self.cache.get(query, chunk_id, self.model_id, use_meta)
            
            if cached_score is not None:
                scores[i] = cached_score
            else:
                indices_to_compute.append(i)
                # B2. Format Input
                if use_meta:
                    text_a, text_b = self.processor.format_reranker_input(query, chunk, meta_fields)
                else:
                    text_a, text_b = query, chunk.page_content
                pairs_to_compute.append([text_a, text_b])
        
        # 2. Compute Missing
        if pairs_to_compute:
            # Predict returns numpy array
            computed_scores = self.model.predict(pairs_to_compute, batch_size=16, show_progress_bar=False)
            
            # 3. Update Cache & Result List
            for local_idx, score in enumerate(computed_scores):
                original_idx = indices_to_compute[local_idx]
                chunk = chunks[original_idx]
                chunk_id = chunk.metadata.get("chunk_id", "unknown")
                
                self.cache.set(query, chunk_id, self.model_id, use_meta, float(score))
                scores[original_idx] = float(score)
                
        return scores

def run_metadata_reranking_experiment(experiment, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    logger.info("="*80)
    logger.info("RUNNING METADATA RERANKING EXPERIMENT")
    logger.info("="*80)

    # --- Configuration ---
    use_meta_in_rerank = True
    active_meta_fields = ["doc_id", "page", "year", "filing_type", "section"]
    reranker_model = "BAAI/bge-reranker-v2-m3"
    
    # Setup Cache Dir
    cache_dir = os.path.join(experiment.output_dir, "rerank_cache")
    
    # Initialize Reranker
    reranker = MetadataAwareReranker(reranker_model, cache_dir)
    
    # --- Part A: Ingestion with Page-Aware Chunking ---
    
    # 1. Identify Docs
    unique_docs = {}
    for sample in data:
        unique_docs.setdefault(sample.get('doc_name'), sample.get('doc_link'))
    
    # 2. Build Store (Using 'metadata_exp' collection to avoid polluting others)
    _, vectordb, _ = build_chroma_store(experiment, "metadata_exp", lazy_load=True)
    
    # 3. Ingest Loop
    logger.info("Ingesting Documents with Page-Aware Chunking...")
    # Note: In a real run, you might want to check what's already in the DB.
    # For this snippet, we assume we push distinct chunks.
    
    for doc_name, doc_link in tqdm(unique_docs.items(), desc="Ingesting"):
        if not doc_name: continue
        
        pdf_pages, _ = load_pdf_with_fallback(doc_name, doc_link, experiment.pdf_local_dir)
        
        if pdf_pages:
            # A3. Page-Aware Chunking
            chunks = chunk_docs_page_aware(pdf_pages, experiment.text_splitter, doc_name_override=doc_name)
            
            # Enrich with optional metadata (A2)
            for c in chunks:
                MetadataProcessor.enrich_metadata(c)
                
            # Verify A3 Acceptance Criteria (Sample check)
            if len(chunks) > 0:
                if "page" not in chunks[0].metadata:
                    logger.error(f"CRITICAL: Chunk missing page metadata! {doc_name}")
            
            # Populate Vector Store
            populate_chroma_store(experiment, vectordb, chunks, "metadata_exp")

    # --- Part B: Execution ---
    results = experiment._create_skipped_results(data, "metadata_rerank", "metadata_rerank", "pdf", "metadata_rerank", start_id=0)

    for i, sample in enumerate(tqdm(data, desc="Running Queries")):
        question = sample.get("question")
        if not question: continue
        
        results[i]["doc_name"] = sample.get("doc_name")
        
        # 1. Retrieve Candidates (Dense)
        # Fetch more to allow reranker to work
        candidates_k = experiment.top_k * 4
        
        try:
            docs = vectordb.similarity_search(question, k=candidates_k)
        except Exception as e:
            logger.warning(f"Retrieval failed for {i}: {e}")
            docs = []
            
        # 2. Metadata-Aware Reranking
        if docs:
            # Score
            scores = reranker.score_batch(
                question, 
                docs, 
                use_meta=use_meta_in_rerank, 
                meta_fields=active_meta_fields
            )
            
            # Assign scores and Sort
            scored_docs = []
            for d, s in zip(docs, scores):
                d.metadata['rerank_score'] = s
                scored_docs.append(d)
            
            scored_docs.sort(key=lambda x: x.metadata['rerank_score'], reverse=True)
            
            # Cut to top_k
            final_docs = scored_docs[:experiment.top_k]
        else:
            final_docs = []
            
        # 3. Generation
        context = "\n\n".join([d.page_content for d in final_docs])
        ans, prompt = experiment._generate_answer(question, context, return_prompt=True)
        
        # 4. Format Output
        formatted_chunks = []
        for rank, d in enumerate(final_docs):
            formatted_chunks.append({
                "rank": rank + 1,
                "text": d.page_content,
                "score": d.metadata.get('rerank_score'),
                "metadata": d.metadata # Includes year, filing, etc.
            })
            
        results[i]["retrieved_chunks"] = formatted_chunks
        results[i]["generated_answer"] = ans
        results[i]["final_prompt"] = prompt
        
        gold_segs, gold_str = experiment._prepare_gold_evidence(sample.get('evidence', ''))
        results[i]["gold_evidence"] = gold_str

    return results