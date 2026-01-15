"""
Big2Small (Parent Document) Retrieval Runner - Custom Implementation.

Strategy:
1. Split documents into large "Parent" chunks (context).
2. Split Parents into small "Child" chunks (search index).
3. Retrieval: Search Children -> Identify Parent -> Return Parent (Text) + Child (Evidence).
"""
import os
import json
import logging
import hashlib
from typing import List, Dict, Any
from tqdm import tqdm

# Use PyMuPDFLoader for reliable metadata extraction
try:
    from langchain_community.document_loaders import PyMuPDFLoader
except ImportError:
    try:
        from langchain.document_loaders import PyMuPDFLoader
    except ImportError:
        PyMuPDFLoader = None

from .. import get_chroma_db_path, Chroma, populate_chroma_store, save_store_config

logger = logging.getLogger(__name__)

def _get_parent_map_path(experiment, db_path: str) -> str:
    """Path to the JSON file storing Parent ID -> {Text, Metadata}."""
    return os.path.join(db_path, "parent_map_with_meta.json")

def _load_parent_map(path: str) -> Dict[str, Dict[str, Any]]:
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load parent map from {path}: {e}")
            return {}
    return {}

def _save_parent_map(data: Dict[str, Dict[str, Any]], path: str):
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to save parent map to {path}: {e}")

def _generate_stable_id(text: str, source: str) -> str:
    """Generate a stable hash ID for a chunk."""
    raw = f"{source}:{text}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()

def run_big2small(experiment, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # 1. Validation
    if not experiment.child_splitter:
        raise ValueError(
            "Child splitter not initialized. You must run with chunking_strategy='hierarchical'"
        )
    if PyMuPDFLoader is None:
        raise RuntimeError("PyMuPDFLoader is not available. Install pymupdf and langchain.")

    # 2. Setup Paths
    # get_chroma_db_path uses the config hash (including parent/child sizes).
    # This ensures that if you change --parent-chunk-size, you get a NEW db_path.
    db_name, db_path = get_chroma_db_path(experiment, "all")
    parent_map_path = _get_parent_map_path(experiment, db_path)
    
    logger.info(f"Target Vector Store: {db_path}")
    logger.info(f"Target Parent Map: {parent_map_path}")

    # 3. Initialize Chroma
    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=experiment.embeddings,
        collection_name=f"b2s_children_{db_name}"
    )
    
    # 4. Check Index State
    collection_count = vectorstore._collection.count()
    parent_map = _load_parent_map(parent_map_path)
    
    # Rebuild if empty
    if collection_count == 0 or not parent_map:
        logger.info("Index or Parent Map missing/empty. Building Big2Small index...")
        
        if not experiment.use_all_pdfs:
            logger.warning("Big2Small requires a pre-built index. Forcing load of ALL PDFs.")

        # Load PDFs
        pdf_files = list(experiment.pdf_local_dir.glob("*.pdf"))
        if not pdf_files:
            logger.error(f"No PDF files found in {experiment.pdf_local_dir}")
            return []

        all_docs = []
        for pdf_path in tqdm(pdf_files, desc="Loading PDFs"):
            try:
                loader = PyMuPDFLoader(str(pdf_path))
                docs = loader.load()
                for d in docs:
                    d.metadata["doc_name"] = pdf_path.name
                    d.metadata["source"] = str(pdf_path.name)
                all_docs.extend(docs)
            except Exception as e:
                logger.error(f"Failed to load {pdf_path}: {e}")

        if not all_docs:
            return []

        # Create Hierarchy
        logger.info(f"Splitting documents into Parents (Size={experiment.parent_chunk_size})...")
        parent_docs = experiment.text_splitter.split_documents(all_docs)
        
        new_parent_map = {}
        child_docs_to_index = []

        logger.info(f"Generated {len(parent_docs)} Parents. Generating Children (Size={experiment.child_chunk_size})...")
        
        for parent in tqdm(parent_docs, desc="Processing Hierarchy"):
            # Generate Parent ID
            pid = _generate_stable_id(parent.page_content, parent.metadata.get("source", ""))
            
            # Store BOTH Text and Metadata
            new_parent_map[pid] = {
                "text": parent.page_content,
                "metadata": parent.metadata
            }
            
            # Split into Children
            children = experiment.child_splitter.split_documents([parent])
            for child in children:
                child.metadata["parent_id"] = pid
                # Copy useful metadata to child for debugging/filtering
                child.metadata["doc_name"] = parent.metadata.get("doc_name")
                child.metadata["page"] = parent.metadata.get("page")
                child_docs_to_index.append(child)

        # Index Children
        logger.info(f"Indexing {len(child_docs_to_index)} Children...")
        populate_chroma_store(experiment, vectorstore, child_docs_to_index, db_name)
        save_store_config(experiment, db_path)
        _save_parent_map(new_parent_map, parent_map_path)
        parent_map = new_parent_map
        logger.info("Indexing Complete.")
        
    else:
        logger.info(f"Using existing index ({collection_count} chunks) and Parent Map ({len(parent_map)} entries).")

    # 5. Run Inference
    results = experiment._create_skipped_results(
        data, "various", "various", "local_pdfs", experiment.experiment_type, start_id=0
    )
    
    # Retrieve more children than K to ensure we find K unique parents
    search_k = experiment.top_k * 3
    retriever = vectorstore.as_retriever(search_kwargs={"k": search_k})
    
    logger.info(f"Processing {len(data)} questions...")
    
    for i, item in enumerate(tqdm(data, desc="Inference")):
        question = item.get("question")
        if not question:
            continue
            
        # A. Retrieve Children
        try:
            child_docs = retriever.invoke(question)
        except AttributeError:
            child_docs = retriever.get_relevant_documents(question)
            
        # B. Map to Parents (Deduplicate)
        retrieved_parent_chunks = [] 
        retrieved_parent_texts = []
        seen_pids = set()
        
        for child in child_docs:
            pid = child.metadata.get("parent_id")
            if pid and pid in parent_map and pid not in seen_pids:
                # Get the full parent object
                parent_entry = parent_map[pid]
                
                # --- STRUCTURE FOR EVALUATION ---
                # We put the PARENT text in "text" so evaluation uses it.
                # We put the CHILD text in "matched_child" for debug.
                chunk_obj = {
                    "text": parent_entry["text"],         # <--- Main Chunk (Parent)
                    "metadata": parent_entry["metadata"], # <--- Main Metadata (Parent)
                    "parent_id": pid,
                    "matched_child": {                    # <--- Reference info (Child)
                        "text": child.page_content,
                        "metadata": child.metadata
                    }
                }
                
                retrieved_parent_chunks.append(chunk_obj)
                retrieved_parent_texts.append(parent_entry["text"])
                seen_pids.add(pid)
            
            if len(retrieved_parent_chunks) >= experiment.top_k:
                break
        
        # C. Construct Context for LLM (using PARENT text)
        context_text = "\n\n".join(retrieved_parent_texts)
        
        # D. Generate Answer
        answer, prompt = experiment._generate_answer(
            question, 
            context_text, 
            mode=experiment.experiment_type, 
            return_prompt=True
        )

        # --- FIX: Populate Gold Evidence ---
        # Normalize and store the gold evidence so evaluation scripts can read it
        gold_segments, gold_text = experiment._prepare_gold_evidence(item.get("evidence"))
        results[i]['gold_evidence'] = gold_text
        results[i]['gold_evidence_segments'] = gold_segments

        # E. Save Results
        results[i]['retrieved_chunks'] = retrieved_parent_chunks 
        results[i]['num_retrieved'] = len(retrieved_parent_chunks)
        results[i]['context_length'] = len(context_text)
        results[i]['generated_answer'] = answer
        results[i]['final_prompt'] = prompt
        
        experiment.notify_sample_complete(1)
        
    return results