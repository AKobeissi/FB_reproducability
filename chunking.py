from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

def get_splitters(experiment):
    """
    Factory function to return the appropriate splitters based on experiment config.
    Returns: (primary_splitter, child_splitter)
             - primary_splitter: Used for main context or parent chunks
             - child_splitter: Used for search/indexing (only for hierarchical)
    """
    strategy = experiment.chunking_strategy
    unit = experiment.chunking_unit
    
    child_splitter = None
    parent_splitter = None

    # --- 1. Token-based Chunking ---
    if unit == "tokens":
        try:
            tokenizer = AutoTokenizer.from_pretrained(experiment.llm_model_name)
            parent_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                tokenizer,
                chunk_size=experiment.chunk_size,
                chunk_overlap=experiment.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        except Exception as e:
            print(f"Warning: Tokenizer load failed ({e}), falling back to chars.")
            unit = "chars" # Fallback below

    # --- 2. Hierarchical Chunking (The new logic) ---
    if strategy == "hierarchical":
        # Child Splitter (Small chunks for vector search)
        child_size = experiment.child_chunk_size or 256
        child_overlap = experiment.child_chunk_overlap or 32
        
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size,
            chunk_overlap=child_overlap,
            length_function=len
        )
        
        # Parent Splitter (Large chunks for context)
        # If we didn't already make a token splitter above, make a char one
        if not parent_splitter:
            parent_size = experiment.parent_chunk_size or experiment.chunk_size or 1024
            parent_overlap = experiment.parent_chunk_overlap or experiment.chunk_overlap or 100
            
            parent_splitter = RecursiveCharacterTextSplitter(
                chunk_size=parent_size,
                chunk_overlap=parent_overlap,
                length_function=len
            )
            
    # --- 3. Standard Character Chunking (Default) ---
    elif not parent_splitter:
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=experiment.chunk_size,
            chunk_overlap=experiment.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    return parent_splitter, child_splitter