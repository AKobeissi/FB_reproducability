
import logging
import json
import argparse
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import Counter
import re
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RetrievalEvaluator:
    """
    Evaluator for Retrieval aspects of RAG systems.
    Computes Hit@k, Recall@k, MRR at Document, Page, and Chunk levels.
    """

    def __init__(self):
        pass

    def _normalize_text(self, text: str) -> str:
        """Normalize text for matching (lowercase, strip whitespace)."""
        if not text:
            return ""
        return " ".join(text.lower().split())

    def _check_match(self, retrieved_text: str, gold_text: str, threshold: float = 0.7) -> bool:
        """
        Check if retrieved text matches gold text using token overlap or LCS.
        Here we use a simple inclusion or high overlap check.
        """
        if not retrieved_text or not gold_text:
            return False
        
        norm_ret = self._normalize_text(retrieved_text)
        norm_gold = self._normalize_text(gold_text)

        # 1. Exact substring match (most reliable for "gold evidence in chunk")
        if norm_gold in norm_ret:
            return True
        
        # 2. Token Overlap (fallback for slight variations)
        ret_tokens = set(norm_ret.split())
        gold_tokens = set(norm_gold.split())
        
        if not gold_tokens:
            return False
            
        intersection = ret_tokens.intersection(gold_tokens)
        overlap = len(intersection) / len(gold_tokens)
        
        return overlap >= threshold

    def compute_metrics(self, samples: List[Dict[str, Any]], k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, Any]:
        """
        Compute retrieval metrics for a list of samples.
        
        Args:
            samples: List of sample dictionaries. Each sample must have:
                     - 'retrieved_chunks': List of dicts with 'text', 'metadata' -> {'doc_name', 'page'}
                     - 'gold_evidence_segments': List of dicts with 'text', 'doc_name', 'page'
                     - 'reference_answer': str (optional, for checking if answer is in chunk)
            k_values: List of k thresholds for Hit@k and Recall@k
        
        Returns:
            Dictionary of aggregated metrics.
        """
        metrics = {
            "doc_hit": {k: [] for k in k_values},
            "doc_recall": {k: [] for k in k_values},
            "page_hit": {k: [] for k in k_values},
            "page_recall": {k: [] for k in k_values},
            "chunk_hit": {k: [] for k in k_values}, # Matches gold evidence
            "chunk_recall": {k: [] for k in k_values},
            "ref_answer_hit": {k: [] for k in k_values}, # Matches reference answer
            "mrr": []
        }

        for sample in samples:
            retrieved = sample.get("retrieved_chunks", [])
            gold_segments = sample.get("gold_evidence_segments", [])
            
            # Handle case where gold_evidence_segments might be a single dict or other format
            if isinstance(gold_segments, dict):
                gold_segments = [gold_segments]
            elif not isinstance(gold_segments, list):
                # Try legacy format or skip
                if sample.get("gold_evidence"):
                     gold_segments = [{"text": sample["gold_evidence"]}]
                else:
                    gold_segments = []

            reference_answer = sample.get("reference_answer", "")

            # If no gold evidence, we can't evaluate retrieval (except maybe ref answer inclusion)
            if not gold_segments and not reference_answer:
                continue

            # Extract Gold Sets
            gold_docs = set()
            gold_pages = set()
            gold_texts = []

            for seg in gold_segments:
                if seg.get("doc_name"):
                    gold_docs.add(seg["doc_name"])
                if seg.get("doc_name") and seg.get("page") is not None:
                    gold_pages.add((seg["doc_name"], seg["page"]))
                if seg.get("text") or seg.get("evidence_text"):
                    gold_texts.append(seg.get("text") or seg.get("evidence_text"))

            # --- Evaluate at different K ---
            # Pre-compute matches for all retrieved chunks to avoid re-looping
            # Matches is a list of dicts: {'doc_match': bool, 'page_match': bool, 'chunk_match': bool, 'ref_match': bool}
            
            retrieval_metadata = []
            
            first_relevant_rank = 0

            for idx, chunk in enumerate(retrieved):
                meta = chunk.get("metadata", {})
                text = chunk.get("text", "")
                
                doc = meta.get("doc_name")
                page = meta.get("page")
                
                # Doc Match
                is_doc_match = doc in gold_docs
                
                # Page Match
                is_page_match = (doc, page) in gold_pages
                
                # Chunk Match (Gold Evidence)
                is_chunk_match = False
                for gold_text in gold_texts:
                    if self._check_match(text, gold_text):
                        is_chunk_match = True
                        break
                
                # Reference Answer Match
                is_ref_match = False
                if reference_answer:
                    # Check if reference answer is contained in the chunk (programmatic approach)
                    # We typically check if the reference answer is a substring of the chunk
                    if self._normalize_text(reference_answer) in self._normalize_text(text):
                        is_ref_match = True

                retrieval_metadata.append({
                    "doc_match": is_doc_match,
                    "page_match": is_page_match,
                    "chunk_match": is_chunk_match,
                    "ref_match": is_ref_match
                })
                
                # MRR Calculation (based on Chunk Match - finding the evidence)
                if first_relevant_rank == 0 and is_chunk_match:
                    first_relevant_rank = idx + 1

            # Calculate MRR score for this sample
            sample_mrr = 1.0 / first_relevant_rank if first_relevant_rank > 0 else 0.0
            metrics["mrr"].append(sample_mrr)
            
            # Store per-sample metrics
            sample_metrics = {"mrr": sample_mrr}

            # Calculate Hit@k and Recall@k
            for k in k_values:
                k_retrieved = retrieval_metadata[:k]
                
                # Document Level
                retrieved_docs = set()
                for i, item in enumerate(k_retrieved):
                     if retrieved[i].get("metadata", {}).get("doc_name"):
                         retrieved_docs.add(retrieved[i]["metadata"]["doc_name"])
                
                # DOC
                doc_hits = sum(1 for x in k_retrieved if x['doc_match'])
                doc_hit_val = 1 if doc_hits > 0 else 0
                metrics["doc_hit"][k].append(doc_hit_val)
                doc_recall_val = len(retrieved_docs.intersection(gold_docs)) / len(gold_docs) if gold_docs else 0
                metrics["doc_recall"][k].append(doc_recall_val)

                # PAGE
                page_hits = sum(1 for x in k_retrieved if x['page_match'])
                page_hit_val = 1 if page_hits > 0 else 0
                metrics["page_hit"][k].append(page_hit_val)
                
                retrieved_pages = set()
                for i, item in enumerate(k_retrieved):
                    m = retrieved[i].get("metadata", {})
                    if m.get("doc_name") and m.get("page") is not None:
                        retrieved_pages.add((m["doc_name"], m["page"]))
                
                page_recall_val = len(retrieved_pages.intersection(gold_pages)) / len(gold_pages) if gold_pages else 0
                metrics["page_recall"][k].append(page_recall_val)

                # CHUNK (Gold Evidence)
                chunk_hits = sum(1 for x in k_retrieved if x['chunk_match'])
                chunk_hit_val = 1 if chunk_hits > 0 else 0
                metrics["chunk_hit"][k].append(chunk_hit_val)
                
                found_segments = 0
                for gold_text in gold_texts:
                    found = False
                    for i in range(min(k, len(retrieved))):
                        if self._check_match(retrieved[i].get("text", ""), gold_text):
                            found = True
                            break
                    if found:
                        found_segments += 1
                
                chunk_recall_val = found_segments / len(gold_texts) if gold_texts else 0
                metrics["chunk_recall"][k].append(chunk_recall_val)

                # Reference Answer Inclusion
                ref_hits = sum(1 for x in k_retrieved if x['ref_match'])
                ref_hit_val = 1 if ref_hits > 0 else 0
                metrics["ref_answer_hit"][k].append(ref_hit_val)
                
                # Add to sample metrics
                sample_metrics[f"doc_hit@{k}"] = doc_hit_val
                sample_metrics[f"doc_recall@{k}"] = doc_recall_val
                sample_metrics[f"page_hit@{k}"] = page_hit_val
                sample_metrics[f"page_recall@{k}"] = page_recall_val
                sample_metrics[f"chunk_hit@{k}"] = chunk_hit_val
                sample_metrics[f"chunk_recall@{k}"] = chunk_recall_val
                sample_metrics[f"ref_answer_hit@{k}"] = ref_hit_val

            # Update sample in place
            sample["retrieval_metrics"] = sample_metrics

        # Aggregate Results
        aggregated = {}
        for metric_name, k_dict in metrics.items():
            if metric_name == "mrr":
                aggregated["mrr"] = np.mean(k_dict) if k_dict else 0.0
            else:
                for k, values in k_dict.items():
                    aggregated[f"{metric_name}@{k}"] = np.mean(values) if values else 0.0
        
        return aggregated

def main():
    parser = argparse.ArgumentParser(description="Run Retrieval Evaluation on RAG outputs.")
    parser.add_argument("input_file", help="Path to input JSON file with retrieval results.")
    parser.add_argument("--output-dir", help="Directory to save evaluation results.")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    with open(input_path, 'r') as f:
        data = json.load(f)

    # Support different JSON structures
    if isinstance(data, list):
        samples = data
    elif isinstance(data, dict) and "results" in data:
        samples = data["results"]
    else:
        logger.error("Invalid JSON format. Expected list or dict with 'results' key.")
        return

    evaluator = RetrievalEvaluator()
    metrics = evaluator.compute_metrics(samples)

    print("\nRetrieval Evaluation Results:")
    print(json.dumps(metrics, indent=2))

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{input_path.stem}_retrieval_eval.json"
        with open(out_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved evaluation results to {out_file}")

if __name__ == "__main__":
    main()
