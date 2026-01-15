"""
Logic for FinanceBench query expansion, rewriting, and decomposition experiments.
"""
import re

def expand_term_and_time(text: str) -> str:
    """
    Applies acronym expansion and time normalization.
    Useful for both Retrieval and Generation prompts.
    """
    # 1. Term Expansion (Case Insensitive)
    substitutions = [
        (r'(?i)\bcapex\b', 'CAPEX - capital expenditure'),
        (r'(?i)\bppe\b', 'PPE - property plant equipment'),
        (r'(?i)\bppne\b', 'PP&E - property plant equipment'),
        (r'(?i)\bsg&a\b', 'selling, general and administrative'),
        (r'(?i)\br&d\b', 'research and development'),
        (r'(?i)\bmd&a\b', 'management discussion and analysis'),
    ]
    
    expanded_text = text
    for pattern, replacement in substitutions:
        expanded_text = re.sub(pattern, replacement, expanded_text)
        
    # 2. Time Normalization
    # Matches FY2018, FY 2018, fy18 -> fiscal year 2018
    # Pattern: FY followed by optional space, then 2 or 4 digits
    expanded_text = re.sub(r'(?i)\bfy\s?(\d{2,4})\b', r'fiscal year \1', expanded_text)
    
    return expanded_text

def add_retrieval_anchors(text: str) -> str:
    """
    Adds numeric and unit anchors if the query appears to be about amounts.
    Best used ONLY for the retrieval query, not the generation prompt.
    """
    # Heuristic to detect amount-related questions
    amount_keywords = [
        "how much", "amount", "value", "total", "cost", "expense", "revenue", 
        "income", "profit", "margin", "expenditure", "spend", "balance",
        "asset", "liability", "equity", "million", "billion"
    ]
    
    # Check if any keyword exists in the lowercased text
    is_amount_question = any(k in text.lower() for k in amount_keywords)
    
    if is_amount_question:
        # Anchors to boost retrieval of tables and financial statements
        anchors = [
            "in millions", 
            "USD", 
            "$",
        ]
        # Append anchors to the query
        return f"{text} {' '.join(anchors)}"
    
    return text

def process_query_for_experiment(raw_query: str):
    """
    Returns a tuple of (retrieval_query, generation_query).
    
    - generation_query: Expanded terms + normalized time (clean for LLM).
    - retrieval_query: generation_query + anchors (optimized for search).
    """
    # Step 1: Base expansion (Terms + Time)
    base_expanded = expand_term_and_time(raw_query)
    
    # Step 2: Add anchors specifically for the retrieval step
    retrieval_query = add_retrieval_anchors(base_expanded)
    
    return retrieval_query, base_expanded