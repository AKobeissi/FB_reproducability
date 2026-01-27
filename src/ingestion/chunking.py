import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

def get_splitters(experiment):
    """
    Factory function to return the appropriate splitters based on experiment config.
    """
    strategy = experiment.chunking_strategy
    unit = experiment.chunking_unit
    
    child_splitter = None
    parent_splitter = None

    # --- 1. Token-based Chunking ---
    if unit == "tokens":
        try:
            # FIX (Rec #4): Prioritize explicit tokenizer -> embedding model -> LLM
            # Embedding models (e.g. bge-m3) are safer defaults for tokenization than API model names.
            explicit_tokenizer = getattr(experiment, "chunk_tokenizer_name", None)
            embedding_model = getattr(experiment, "embedding_model", None)
            
            target_model = explicit_tokenizer or embedding_model or experiment.llm_model_name
            
            # Filter out non-HF strings (simple heuristic)
            if target_model and any(x in target_model.lower() for x in ["openai", "gpt-", "claude"]):
                # If the fallback is an API name, force a safe default like BERT or fallback to chars
                print(f"Warning: '{target_model}' looks like an API model. Using 'bert-base-uncased' for token counting.")
                target_model = "bert-base-uncased"

            tokenizer = AutoTokenizer.from_pretrained(target_model)
            
            parent_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                tokenizer,
                chunk_size=experiment.chunk_size,
                chunk_overlap=experiment.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        except Exception as e:
            print(f"Warning: Tokenizer load failed for '{target_model}' ({e}). Falling back to chars.")
            unit = "chars" # Fallback below

    # --- 2. Hierarchical Chunking ---
    if strategy == "hierarchical":
        child_size = experiment.child_chunk_size or 256
        child_overlap = experiment.child_chunk_overlap or 32
        
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size,
            chunk_overlap=child_overlap,
            length_function=len
        )
        
        if not parent_splitter:
            parent_size = experiment.parent_chunk_size or experiment.chunk_size or 1024
            parent_overlap = experiment.parent_chunk_overlap or experiment.chunk_overlap or 100
            
            parent_splitter = RecursiveCharacterTextSplitter(
                chunk_size=parent_size,
                chunk_overlap=parent_overlap,
                length_function=len
            )
            
    # --- 3. Standard Character Chunking ---
    elif not parent_splitter:
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=experiment.chunk_size,
            chunk_overlap=experiment.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    return parent_splitter, child_splitter


def chunk_docs_page_aware(docs, text_splitter, doc_name_override=None):
    """
    Chunks a list of LangChain Documents (pages) individually to ensure
    no chunk spans multiple pages.
    """
    all_chunks = []
    
    for page_idx, doc in enumerate(docs):
        page_chunks = text_splitter.split_text(doc.page_content)
        
        for chunk_idx, chunk_text in enumerate(page_chunks):
            # Resolve Metadata
            raw_page = doc.metadata.get("page", doc.metadata.get("page_number", page_idx))
            page_num = raw_page # 0-indexed
            
            doc_id = doc_name_override or doc.metadata.get("doc_name") or "unknown_doc"
            
            chunk_id_str = f"{doc_id}_p{page_num}_{chunk_idx}"
            chunk_id = hashlib.md5(chunk_id_str.encode()).hexdigest()

            from langchain.schema import Document
            
            new_meta = {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                # FIX (Rec #3): Ensure doc_name is present for downstream compatibility
                "doc_name": doc_id, 
                "page": page_num,
                "source": doc.metadata.get("source", "pdf"),
                # Preserve other existing meta, filtering out conflicts
                **{k:v for k,v in doc.metadata.items() if k not in ['page', 'source', 'doc_name', 'doc_id']}
            }

            all_chunks.append(Document(page_content=chunk_text, metadata=new_meta))
            
    return all_chunks