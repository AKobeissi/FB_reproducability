import re
import os
import json
import hashlib
from typing import List, Dict, Any, Tuple

class MetadataProcessor:
    """Handles metadata extraction, heuristics, and formatting."""

    @staticmethod
    def enrich_metadata(chunk) -> Dict[str, Any]:
        """
        Populates optional fields: year, filing_type, section, table_flag.
        """
        meta = chunk.metadata
        text = chunk.page_content
        doc_id = meta.get("doc_id", "").lower()

        # --- A2. Heuristics ---
        
        # 1. Year (Extract from 4-digit patterns in doc_id or common text headers)
        if "year" not in meta:
            year_match = re.search(r'(20\d{2})', doc_id)
            if year_match:
                meta["year"] = int(year_match.group(1))

        # 2. Filing Type (10-K, 10-Q, 8-K)
        if "filing_type" not in meta:
            if "10k" in doc_id or "10-k" in doc_id:
                meta["filing_type"] = "10-K"
            elif "10q" in doc_id or "10-q" in doc_id:
                meta["filing_type"] = "10-Q"
            elif "8k" in doc_id:
                meta["filing_type"] = "8-K"
            elif "earnings" in doc_id or "transcript" in doc_id:
                meta["filing_type"] = "Transcript"

        # 3. Section (Simple heuristic looking for "Item X")
        if "section" not in meta:
            section_match = re.search(r'(Item\s+\d+[A-Z]?)', text, re.IGNORECASE)
            if section_match:
                meta["section"] = section_match.group(1)

        # 4. Table Flag
        if "table_flag" not in meta:
            digit_count = sum(c.isdigit() for c in text)
            if len(text) > 0 and (digit_count / len(text) > 0.15): 
                meta["table_flag"] = True
            else:
                meta["table_flag"] = False
                
        chunk.metadata = meta
        return chunk

    @staticmethod
    def format_reranker_input(query: str, chunk, meta_fields: List[str]) -> Tuple[str, str]:
        """
        B2. Canonical formatting function.
        Returns (query, "[META ...] " + chunk_text)
        """
        meta = chunk.metadata
        parts = []
        ordered_fields = ["doc_id", "page", "year", "filing_type", "section"]
        
        for field in ordered_fields:
            if field in meta_fields and field in meta and meta[field] is not None:
                parts.append(f"{field}={meta[field]}")
        
        meta_str = f"[META {' '.join(parts)}]"
        if len(meta_str) > 200:
            meta_str = meta_str[:197] + "...]"
            
        text_b = f"{meta_str} {chunk.page_content}"
        return query, text_b

class RerankerCache:
    """B3. Disk-based caching system."""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def _generate_key(self, query: str, chunk_id: str, model_name: str, use_meta: bool) -> str:
        raw = f"{query}_{chunk_id}_{model_name}_{use_meta}"
        return hashlib.md5(raw.encode()).hexdigest()
    
    def get(self, query: str, chunk_id: str, model_name: str, use_meta: bool) -> float:
        key = self._generate_key(query, chunk_id, model_name, use_meta)
        path = os.path.join(self.cache_dir, key + ".json")
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)['score']
        return None
    
    def set(self, query: str, chunk_id: str, model_name: str, use_meta: bool, score: float):
        key = self._generate_key(query, chunk_id, model_name, use_meta)
        path = os.path.join(self.cache_dir, key + ".json")
        with open(path, 'w') as f:
            json.dump({'score': score}, f)