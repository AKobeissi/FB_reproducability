"""
src/experiments/page_retrieval.py
Page-First Retrieval with On-the-Fly Chunking (0-indexed pages).

Pipeline:
1) Index Pages: Extract pages from PDFs and embed PAGE TEXT (not digests).
   Store only small metadata in Chroma: {doc_name, page, pdf_path}.  (No raw_text in metadata.)
2) Retrieve Pages: Retrieve Top-P pages for each query.
3) Chunk Live: Re-load the original page text from disk for those Top-P pages, then chunk on-the-fly.
4) Score Chunks: Embed and rank candidate chunks against the query.
5) Generate: ALWAYS generate an answer using the experiment's LLM based on the retrieved chunks.
6) Output: Compatible with existing retrieval_evaluator.py + generative_evaluator.py AND core aggregate stats:
   - retrieved_chunks: List[{text, metadata, score}] where metadata includes {doc_name, page}
   - generated_answer: str
   - generation_length: int
   - model_answer: str (legacy)
"""

import logging
import time
import re
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set

import numpy as np
import fitz  # PyMuPDF

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.retrieval.vectorstore import build_chroma_store, populate_chroma_store
from src.ingestion.page_processor import extract_pages_from_pdf
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def _local_format_prompt(query: str, context_str: str) -> str:
    return (
        "You are a careful financial analyst. Answer the question using ONLY the context.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )


def _normalize_page_text_for_embedding(raw_txt: str, max_chars: int = 2000) -> str:
    """Normalize page text for embedding.
    
    Args:
        raw_txt: Raw PDF page text
        max_chars: Maximum characters to use
    
    Returns:
        Normalized text suitable for embedding
    """
    if not raw_txt:
        return " "
    
    txt = raw_txt.replace("\x00", " ")
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    txt = txt.strip()
    
    return (txt[:max_chars] if txt else " ")


def _canonical_doc_key(doc_name: str) -> str:
    """Normalize document name (lowercase, strip .pdf)."""
    if not doc_name:
        return ""
    key = str(doc_name).lower().strip()
    if key.endswith(".pdf"):
        key = key[:-4]
    return key


def _find_pdf_file(doc_key: str, pdf_dir: Path) -> Optional[Path]:
    """Find PDF file with flexible matching (handles case and underscore variations).
    
    Args:
        doc_key: Document key (e.g., '3m_2022_10k' or '3M_2022_10K')
        pdf_dir: Directory containing PDFs
        
    Returns:
        Path to PDF file if found, None otherwise
    """
    # Normalize the search key
    search_key = doc_key.lower().replace("_", "").replace("-", "")
    
    # Try direct match first
    direct_path = pdf_dir / f"{doc_key}.pdf"
    if direct_path.exists():
        return direct_path
    
    # Try uppercase version
    upper_path = pdf_dir / f"{doc_key.upper()}.pdf"
    if upper_path.exists():
        return upper_path
    
    # Try with underscores preserved but uppercase
    upper_underscore = pdf_dir / f"{doc_key.replace('_', '_').upper()}.pdf"
    if upper_underscore.exists():
        return upper_underscore
    
    # Fuzzy search: normalize all PDFs and compare
    for pdf_file in pdf_dir.glob("*.pdf"):
        pdf_normalized = pdf_file.stem.lower().replace("_", "").replace("-", "")
        if pdf_normalized == search_key:
            return pdf_file
    
    # Try recursive search as last resort
    pattern = f"**/{doc_key}.pdf"
    matches = list(pdf_dir.glob(pattern))
    if matches:
        return matches[0]
    
    # Case-insensitive recursive search
    for pdf_file in pdf_dir.glob("**/*.pdf"):
        if pdf_file.stem.lower() == doc_key.lower():
            return pdf_file
    
    return None


def _collect_unique_doc_keys(data: List[Dict[str, Any]]) -> Set[str]:
    """Extract unique document names from dataset."""
    doc_keys = set()
    for sample in data:
        doc = sample.get("doc_name") or sample.get("document")
        if doc:
            doc_keys.add(_canonical_doc_key(doc))
    return doc_keys


def _load_page_text(pdf_path: str, page_idx: int) -> str:
    """Load raw text for a specific page from PDF using PyMuPDFLoader for consistency.
    
    Args:
        pdf_path: Path to PDF file
        page_idx: 0-indexed page number (matching PyMuPDFLoader convention)
    """
    try:
        # Try to use PyMuPDFLoader for consistency
        try:
            from langchain_community.document_loaders import PyMuPDFLoader
        except:
            try:
                from langchain.document_loaders import PyMuPDFLoader
            except:
                PyMuPDFLoader = None
        
        if PyMuPDFLoader:
            # Use PyMuPDFLoader which ensures 0-indexed pages
            loader = PyMuPDFLoader(pdf_path)
            docs = loader.load()
            if page_idx < 0 or page_idx >= len(docs):
                logger.warning(f"Page {page_idx} out of range in {pdf_path} (has {len(docs)} pages)")
                return ""
            # PyMuPDFLoader docs are ordered by page (0-indexed)
            return docs[page_idx].page_content
        else:
            # Fallback to fitz
            doc = fitz.open(pdf_path)
            if page_idx < 0 or page_idx >= len(doc):
                logger.warning(f"Page {page_idx} out of range in {pdf_path}")
                return ""
            page = doc[page_idx]
            text = page.get_text("text")
            doc.close()
            return text
    except Exception as e:
        logger.error(f"Error loading page {page_idx} from {pdf_path}: {e}")
        return ""


def run_page_then_chunk(
    experiment,
    data: List[Dict[str, Any]],
    learned_model_path: Optional[str] = None,
    use_learned_for_chunks: bool = False
) -> List[Dict[str, Any]]:
    """
    Execute Page-First Retrieval with ON-THE-FLY chunking and forced generation.
    Pages are 0-indexed throughout (matches your dataset assumption).
    
    Args:
        experiment: RAGExperiment instance
        data: List of question samples
        learned_model_path: Path to learned page scorer (None = use baseline)
        use_learned_for_chunks: If True, use learned model for chunks too. If False, use base model for chunks.
    """

    logger.info("\n" + "=" * 80)
    logger.info("RUNNING PAGE-FIRST RETRIEVAL WITH ON-THE-FLY CHUNKING")
    logger.info("Strategy: Retrieve Pages -> Chunk Retrieved Pages -> Rank Chunks -> Generate")
    logger.info("=" * 80)

    # =========================================================================
    # PART 1: Initialize (or build) the Page Vector Store
    # =========================================================================

    # Determine page embedding model
    if learned_model_path:
        logger.info(f"Loading Learned Page Scorer from: {learned_model_path}")
        
        # Verify the model exists
        if not os.path.exists(learned_model_path):
            raise FileNotFoundError(f"Learned model not found at: {learned_model_path}")

        from langchain.embeddings.base import Embeddings

        class STEmbeddings(Embeddings):
            def __init__(self, path: str):
                self.model = SentenceTransformer(path)
                logger.info(f"✓ Loaded SentenceTransformer from {path}")
                logger.info(f"  Model max_seq_length: {self.model.max_seq_length}")

            def embed_documents(self, texts: List[str]):
                return self.model.encode(texts, convert_to_numpy=True).tolist()

            def embed_query(self, text: str):
                return self.model.encode([text], convert_to_numpy=True)[0].tolist()

        page_embeddings = STEmbeddings(learned_model_path)
        collection_suffix = "learned"
        logger.info(f"Using learned page scorer with collection suffix: {collection_suffix}")
    else:
        page_embeddings = experiment.embeddings
        collection_suffix = "baseline"
        logger.info(f"Using baseline embeddings: {experiment.embedding_model}")
    
    # Determine chunk embedding model (separate from page embeddings)
    if use_learned_for_chunks and learned_model_path:
        chunk_embeddings = page_embeddings  # Reuse learned model
        logger.info(f"Using learned model for BOTH pages and chunks")
    else:
        chunk_embeddings = experiment.embeddings  # Always use base model for chunks
        logger.info(f"Using base model ({experiment.embedding_model}) for chunk scoring")

    # Force a fresh store name to avoid reusing older broken indexes
    db_name = f"pages_v12_text_{collection_suffix}"

    # Connect to Chroma
    try:
        _, page_vectordb, is_empty_flag = build_chroma_store(
            experiment, db_name, embeddings=page_embeddings, lazy_load=True
        )
    except ValueError:
        _, page_vectordb = build_chroma_store(
            experiment, db_name, embeddings=page_embeddings, lazy_load=True
        )
        is_empty_flag = False

    try:
        actual_count = page_vectordb._collection.count()
    except Exception:
        actual_count = 0

    should_populate = is_empty_flag or actual_count == 0

    # ------------------------------------------------
    # Build list of PDFs to index
    # ------------------------------------------------
    pdf_local_dir = getattr(experiment, "pdf_local_dir", None) or getattr(experiment, "pdf_dir", None) or "pdfs"
    pdf_dir = Path(pdf_local_dir)

    if should_populate:
        logger.info(f"Populating page store '{db_name}' (currently {actual_count} pages).")

        if not pdf_dir.exists():
            logger.error(f"PDF directory not found: {pdf_dir}")
        else:
            unique_doc_keys = _collect_unique_doc_keys(data)
            use_all_pdfs = bool(getattr(experiment, "use_all_pdfs", False))

            target_pdfs: List[Path] = []

            if use_all_pdfs or not unique_doc_keys:
                # Fall back: index all PDFs if requested OR if the dataset doesn't expose doc_name at top-level
                target_pdfs = list(pdf_dir.glob("**/*.pdf"))
                logger.info(f"Indexing ALL PDFs: found {len(target_pdfs)} under {pdf_dir}")
            else:
                logger.info(f"Indexing {len(unique_doc_keys)} relevant PDFs under {pdf_dir}")
                for doc_key in sorted(unique_doc_keys):
                    pdf_path = _find_pdf_file(doc_key, pdf_dir)
                    if pdf_path:
                        target_pdfs.append(pdf_path)
                        logger.debug(f"Found PDF: {doc_key} -> {pdf_path.name}")
                    else:
                        logger.warning(f"PDF missing for doc_key: {doc_key}")

            if not target_pdfs:
                logger.error(f"CRITICAL: No PDFs found in {pdf_dir}. Cannot build page index.")
            else:
                all_pages: List[Dict[str, Any]] = []
                logger.info(f"Extracting pages from {len(target_pdfs)} PDFs.")

                for pdf in target_pdfs:
                    doc_key = pdf.stem  # canonical doc id
                    pages = extract_pages_from_pdf(pdf, doc_key)
                    all_pages.extend(pages)

                page_docs: List[Document] = []
                for p in all_pages:
                    raw_txt = p.get("text") or ""
                    emb_txt = _normalize_page_text_for_embedding(raw_txt)
                    page_idx = int(p.get("page", 0))  # 0-indexed by extractor

                    doc_key = _canonical_doc_key(p.get("doc_name") or "")
                    pdf_path = p.get("source") or str(pdf_dir / f"{doc_key}.pdf")
                    pdf_path = str(Path(pdf_path))  # ensure serializable

                    page_docs.append(
                        Document(
                            page_content=emb_txt,
                            metadata={
                                "doc_name": doc_key,   # ✅ must match evaluator normalization
                                "page": page_idx,      # ✅ 0-indexed
                                "pdf_path": pdf_path,  # for re-loading text at retrieval-time
                            },
                        )
                    )

                if page_docs:
                    populate_chroma_store(experiment, page_vectordb, page_docs, db_name)
                    logger.info(f"✓ Populated page store with {len(page_docs)} pages.")
                    
                    # Verify the store was populated
                    try:
                        new_count = page_vectordb._collection.count()
                        logger.info(f"✓ Verified: Store now contains {new_count} pages.")
                        if new_count == 0:
                            logger.error("CRITICAL: Store population appeared to succeed but count is 0!")
                    except Exception as e:
                        logger.warning(f"Could not verify store count: {e}")
                else:
                    logger.warning("Extraction resulted in 0 pages.")
    else:
        logger.info(f"Using existing page store '{db_name}' ({actual_count} pages).")

    # =========================================================================
    # PART 2: Initialize on-the-fly chunker
    # =========================================================================
    chunk_size_adjusted = experiment.chunk_size
    chunk_overlap_adjusted = experiment.chunk_overlap
    if getattr(experiment, "chunking_unit", "chars") == "tokens":
        chunk_size_adjusted = int(experiment.chunk_size) * 4
        chunk_overlap_adjusted = int(experiment.chunk_overlap) * 4

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_adjusted,
        chunk_overlap=chunk_overlap_adjusted,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # =========================================================================
    # PART 3: Ensure LLM is loaded for generation
    # =========================================================================
    logger.info("Ensuring LLM is loaded for answer generation...")
    experiment.ensure_langchain_llm()
    logger.info(f"✓ LLM ready: {experiment.llm_model_name}")

    # =========================================================================
    # PART 4: Inference loop
    # =========================================================================
    page_k = int(getattr(experiment, "page_k", 5))
    chunk_k = int(getattr(experiment, "top_k", 5))

    logger.info(f"Starting Inference: Retrieve {page_k} Pages -> Chunk -> Rank Top {chunk_k}")
    
    # Create results skeleton matching unified pipeline format
    results = experiment._create_skipped_results(
        data, 
        "page_then_chunk", 
        "page_then_chunk", 
        "pdf", 
        "page_retrieval", 
        start_id=0
    )

    for i, sample in enumerate(data):
        t0 = time.time()
        query = sample.get("question", "") or ""

        # Copy over sample fields
        results[i]["sample_id"] = i
        results[i]["doc_name"] = sample.get("doc_name", "")
        results[i]["doc_link"] = sample.get("doc_link", "")
        results[i]["question"] = query
        results[i]["reference_answer"] = sample.get("answer", "")
        results[i]["question_type"] = sample.get("question_type", "")
        results[i]["question_reasoning"] = sample.get("question_reasoning", "")
        
        # Parse gold evidence from the sample (handle pandas/numpy array-like objects)
        gold_evidence_segments = []
        evidence_data = sample.get("evidence")
        
        # Safe check for array-like objects (avoid "truth value of array is ambiguous" error)
        has_evidence = False
        if evidence_data is not None:
            try:
                # For arrays/Series, check length
                if hasattr(evidence_data, '__len__'):
                    has_evidence = len(evidence_data) > 0
                else:
                    has_evidence = bool(evidence_data)
            except (ValueError, TypeError):
                # Fallback: try to convert to list
                try:
                    evidence_data = list(evidence_data)
                    has_evidence = len(evidence_data) > 0
                except:
                    has_evidence = False
        
        if has_evidence:
            # Convert to list if array-like
            if not isinstance(evidence_data, list):
                try:
                    evidence_data = list(evidence_data)
                except:
                    evidence_data = [evidence_data]
            
            # FinanceBench evidence is a list of dicts with structure:
            # [{"doc_name": str, "evidence_text": str, "evidence_page_num": int}]
            # NOTE: Ground truth pages are 0-indexed (matching PyMuPDFLoader)
            for ev in evidence_data:
                if isinstance(ev, dict):
                    # Get page from correct field name: evidence_page_num
                    page_val = ev.get("evidence_page_num") or ev.get("page")
                    if page_val is not None:
                        # Ensure it's 0-indexed (PyMuPDFLoader standard)
                        page_str = str(page_val).strip()
                    else:
                        page_str = ""
                    
                    gold_evidence_segments.append({
                        "doc_name": _canonical_doc_key(ev.get("doc_name", "")),
                        "text": ev.get("evidence_text", ""),
                        "page": page_str
                    })
        
        results[i]["gold_evidence_segments"] = gold_evidence_segments
        
        # Debug first few samples
        if i < 3:
            logger.info(f"Sample {i} gold evidence: {gold_evidence_segments}")

        # --- Step A: Retrieve Pages ---
        retrieved_pages = []
        try:
            # Try with scores first
            if hasattr(page_vectordb, "similarity_search_with_score"):
                retrieved_pages = [d for d, _ in page_vectordb.similarity_search_with_score(query, k=page_k)]
                logger.debug(f"Sample {i}: Retrieved {len(retrieved_pages)} pages with similarity_search_with_score")
            elif hasattr(page_vectordb, "similarity_search_with_relevance_scores"):
                retrieved_pages = [d for d, _ in page_vectordb.similarity_search_with_relevance_scores(query, k=page_k)]
                logger.debug(f"Sample {i}: Retrieved {len(retrieved_pages)} pages with similarity_search_with_relevance_scores")
            else:
                retrieved_pages = page_vectordb.similarity_search(query, k=page_k)
                logger.debug(f"Sample {i}: Retrieved {len(retrieved_pages)} pages with similarity_search")
        except Exception as e:
            logger.warning(f"Page retrieval failed for sample {i}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            retrieved_pages = []

        # --- Step B: Load page text and chunk ---
        generated_chunks: List[Dict[str, Any]] = []
        retrieved_pages_debug: List[str] = []

        for pdoc in retrieved_pages:
            doc_key = _canonical_doc_key(pdoc.metadata.get("doc_name") or "")
            page_idx = int(pdoc.metadata.get("page", 0))
            pdf_path = pdoc.metadata.get("pdf_path", "")

            retrieved_pages_debug.append(f"{doc_key}_p{page_idx}")

            raw_text = _load_page_text(pdf_path, page_idx)
            if not raw_text or not raw_text.strip():
                logger.debug(f"Sample {i}: Empty page text for {doc_key} page {page_idx} at {pdf_path}")
                continue

            chunks_from_page = text_splitter.split_text(raw_text)
            logger.debug(f"Sample {i}: Generated {len(chunks_from_page)} chunks from {doc_key} page {page_idx}")
            
            for chunk_text in chunks_from_page:
                cleaned = re.sub(r"[ \t]+", " ", chunk_text).strip()
                if not cleaned:
                    continue
                generated_chunks.append(
                    {
                        "text": cleaned,
                        "metadata": {
                            "doc_name": doc_key,
                            "page": page_idx,
                        },
                    }
                )
        
        logger.info(f"Sample {i}: Generated {len(generated_chunks)} total chunks from {len(retrieved_pages)} pages")
        
        # Debug chunk metadata for first sample
        if i < 3 and generated_chunks:
            logger.info(f"Sample {i} chunk metadata (first 3):")
            for idx, chunk in enumerate(generated_chunks[:3]):
                meta = chunk.get("metadata", {})
                logger.info(f"  Chunk {idx}: doc={meta.get('doc_name')}, page={meta.get('page')} (type={type(meta.get('page'))})")

        # --- Step C: Embed and Score Chunks ---
        final_chunks_list: List[Dict[str, Any]] = []
        if generated_chunks:
            logger.debug(f"Sample {i}: Embedding and scoring {len(generated_chunks)} chunks")
            # Use chunk_embeddings (base model unless use_learned_for_chunks=True)
            q_vec = np.array(chunk_embeddings.embed_query(query), dtype=np.float32)
            q_norm = float(np.linalg.norm(q_vec)) or 1.0

            chunk_texts = [c["text"] for c in generated_chunks]
            chunk_vecs = chunk_embeddings.embed_documents(chunk_texts)

            scored: List[Tuple[float, Dict[str, Any]]] = []
            for idx, c_vec in enumerate(chunk_vecs):
                c_vec = np.array(c_vec, dtype=np.float32)
                c_norm = float(np.linalg.norm(c_vec)) or 1.0
                score = float(np.dot(q_vec, c_vec) / (q_norm * c_norm))
                scored.append((score, generated_chunks[idx]))

            scored.sort(key=lambda x: x[0], reverse=True)

            for score, c in scored[:chunk_k]:
                final_chunks_list.append(
                    {
                        "text": c["text"],
                        "metadata": c["metadata"],
                        "score": float(score),
                    }
                )
            
            logger.info(f"Sample {i}: Selected top {len(final_chunks_list)} chunks (scores: {[f'{c['score']:.3f}' for c in final_chunks_list[:3]]}...)")
        else:
            logger.warning(f"Sample {i}: No chunks generated from {len(retrieved_pages)} pages")

        results[i]["retrieved_chunks"] = final_chunks_list

        # --- Step D: Generate Answer ---
        if final_chunks_list:
            context_str = "\n\n".join([c["text"] for c in final_chunks_list])
            prompt = _local_format_prompt(query, context_str)
            
            try:
                # Use the LangChain LLM wrapper or pipeline
                if experiment.use_api and experiment.api_client:
                    # API generation
                    response = experiment.api_client.chat.completions.create(
                        model=experiment.llm_model_name,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=experiment.max_new_tokens,
                        temperature=0.7
                    )
                    generated_answer = response.choices[0].message.content
                elif experiment.llm_pipeline:
                    # HuggingFace pipeline
                    output = experiment.llm_pipeline(prompt)
                    generated_answer = output[0]["generated_text"]
                elif experiment.langchain_llm:
                    # LangChain wrapper
                    generated_answer = experiment.langchain_llm.invoke(prompt)
                else:
                    raise RuntimeError("No LLM available for generation")
                    
                results[i]["generated_answer"] = generated_answer
                results[i]["generation_length"] = len(generated_answer)
                results[i]["model_answer"] = generated_answer  # Legacy field
            except Exception as e:
                logger.error(f"Generation failed for sample {i}: {e}")
                results[i]["generated_answer"] = ""
                results[i]["generation_length"] = 0
                results[i]["model_answer"] = ""
        else:
            results[i]["generated_answer"] = ""
            results[i]["generation_length"] = 0
            results[i]["model_answer"] = ""

        elapsed = time.time() - t0
        logger.info(f"Sample {i} complete in {elapsed:.2f}s | Pages: {retrieved_pages_debug[:5]} | Chunks: {len(final_chunks_list)}")

    logger.info("\n" + "=" * 80)
    logger.info(f"PAGE-THEN-CHUNK COMPLETE: {len(results)} samples")
    logger.info("=" * 80)

    return results
