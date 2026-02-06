#!/usr/bin/env python3
"""
Aggregate evaluation metrics from all experiment result files.

Creates 4 CSV files:
1. Overall experiment results (one line per experiment)
2. Results grouped by doc_type
3. Results grouped by question_type  
4. Results grouped by question_reasoning
"""

import json
import logging
import argparse
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict
from datetime import datetime
import csv
import numpy as np

# Import evaluation libraries
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    logger.warning("rouge_score not available, ROUGE metrics will be 0")

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False
    logger.warning("nltk not available, BLEU metrics will be 0")

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def extract_doc_type_from_name(doc_name: str) -> str:
    """Extract document type from doc_name using regex."""
    if not doc_name:
        return 'unknown'
    
    doc_lower = doc_name.lower()
    
    # Check for specific patterns
    if '10-k' in doc_lower or '10k' in doc_lower:
        return '10-K'
    elif '10-q' in doc_lower or '10q' in doc_lower:
        return '10-Q'
    elif '8-k' in doc_lower or '8k' in doc_lower:
        return '8-K'
    elif 'earnings' in doc_lower or 'earning' in doc_lower:
        return 'earnings'
    else:
        return 'other'


def compute_bleu_score(reference: str, hypothesis: str) -> float:
    """Compute BLEU score between reference and hypothesis."""
    if not BLEU_AVAILABLE or not reference or not hypothesis:
        return 0.0
    
    try:
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        
        if not hyp_tokens:
            return 0.0
        
        smoothing = SmoothingFunction().method1
        score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)
        return score
    except:
        return 0.0


def compute_rouge_scores(reference: str, hypothesis: str) -> Dict[str, float]:
    """Compute ROUGE scores between reference and hypothesis."""
    if not ROUGE_AVAILABLE or not reference or not hypothesis:
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    except:
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}


def extract_numbers(text: str) -> Set[str]:
    """Extract all numbers from text for numeric comparison."""
    if not text:
        return set()
    
    # Find all numbers including decimals, percentages, with commas
    pattern = r'\$?\d+(?:,\d{3})*(?:\.\d+)?%?'
    numbers = re.findall(pattern, text)
    # Normalize: remove $ and commas
    normalized = {n.replace('$', '').replace(',', '') for n in numbers}
    return normalized


def compute_numeric_match(reference: str, hypothesis: str) -> bool:
    """Check if any numbers in reference appear in hypothesis."""
    if not reference or not hypothesis:
        return False
    
    ref_numbers = extract_numbers(reference)
    hyp_numbers = extract_numbers(hypothesis)
    
    if not ref_numbers:
        return False
    
    # Check if any reference number appears in hypothesis
    return len(ref_numbers & hyp_numbers) > 0


def compute_recall(retrieved_items: List[str], evidence_items: List[str], k: int) -> bool:
    """Compute recall@k for retrieved vs evidence items."""
    if not evidence_items or not retrieved_items:
        return False
    
    retrieved_k = set(retrieved_items[:k])
    evidence_set = set(evidence_items)
    
    # Check if any evidence item is in top k retrieved
    return len(retrieved_k & evidence_set) > 0


def extract_retrieved_docs(result: Dict[str, Any]) -> List[str]:
    """Extract document names from retrieved chunks."""
    docs = []
    
    # Check retrieved_chunks
    if 'retrieved_chunks' in result and result['retrieved_chunks']:
        for chunk in result['retrieved_chunks']:
            if isinstance(chunk, dict):
                doc_name = chunk.get('doc_name') or chunk.get('metadata', {}).get('doc_name')
                if doc_name:
                    docs.append(doc_name)
    
    # Check retrieved_documents
    if 'retrieved_documents' in result and result['retrieved_documents']:
        for doc in result['retrieved_documents']:
            if isinstance(doc, dict):
                doc_name = doc.get('doc_name') or doc.get('metadata', {}).get('doc_name')
                if doc_name:
                    docs.append(doc_name)
            elif isinstance(doc, str):
                docs.append(doc)
    
    return docs


def extract_retrieved_pages(result: Dict[str, Any]) -> List[str]:
    """Extract page identifiers from retrieved chunks."""
    pages = []
    
    # Check retrieved_chunks
    if 'retrieved_chunks' in result and result['retrieved_chunks']:
        for chunk in result['retrieved_chunks']:
            if isinstance(chunk, dict):
                doc_name = chunk.get('doc_name') or chunk.get('metadata', {}).get('doc_name')
                page_num = chunk.get('page_number') or chunk.get('metadata', {}).get('page_number')
                
                if doc_name and page_num is not None:
                    pages.append(f"{doc_name}_{page_num}")
    
    # Check retrieved_pages
    if 'retrieved_pages' in result and result['retrieved_pages']:
        for page in result['retrieved_pages']:
            if isinstance(page, dict):
                doc_name = page.get('doc_name') or page.get('metadata', {}).get('doc_name')
                page_num = page.get('page_number') or page.get('metadata', {}).get('page_number')
                
                if doc_name and page_num is not None:
                    pages.append(f"{doc_name}_{page_num}")
    
    return pages


def extract_evidence_docs(result: Dict[str, Any]) -> List[str]:
    """Extract evidence document names."""
    docs = []
    
    # Check evidence_docs
    if 'evidence_docs' in result:
        evidence = result['evidence_docs']
        if isinstance(evidence, list):
            docs.extend([str(d) for d in evidence if d])
        elif evidence:
            docs.append(str(evidence))
    
    # Check doc_name as evidence
    if 'doc_name' in result and result['doc_name']:
        doc_name = result['doc_name']
        if doc_name not in docs:
            docs.append(doc_name)
    
    return docs


def extract_evidence_pages(result: Dict[str, Any]) -> List[str]:
    """Extract evidence page identifiers."""
    pages = []
    
    doc_name = result.get('doc_name', '')
    
    # Check evidence_pages
    if 'evidence_pages' in result:
        evidence = result['evidence_pages']
        if isinstance(evidence, list):
            for page_num in evidence:
                if page_num is not None and doc_name:
                    pages.append(f"{doc_name}_{page_num}")
        elif evidence is not None and doc_name:
            pages.append(f"{doc_name}_{evidence}")
    
    return pages


def compute_metrics_from_results(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute aggregated metrics from pre-computed result fields."""
    metrics = {
        'total_samples': 0,
        # Retrieval metrics
        'doc_recall_1': 0.0,
        'doc_recall_3': 0.0,
        'doc_recall_5': 0.0,
        'page_recall_1': 0.0,
        'page_recall_3': 0.0,
        'page_recall_5': 0.0,
        'retrieval_samples': 0,
        # Generation metrics
        'gen_bleu': 0.0,
        'gen_rouge1': 0.0,
        'gen_rouge2': 0.0,
        'gen_rougeL': 0.0,
        'numeric_correct': 0,
        'numeric_total': 0,
        'generation_samples': 0,
    }
    
    if not results:
        return metrics
    
    metrics['total_samples'] = len(results)
    
    # Collect retrieval metrics
    doc_recall_1 = []
    doc_recall_3 = []
    doc_recall_5 = []
    page_recall_1 = []
    page_recall_3 = []
    page_recall_5 = []
    
    # Collect generation metrics
    bleu_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    numeric_correct = 0
    numeric_total = 0
    
    for r in results:
        # === RETRIEVAL METRICS (from retrieval_metrics field) ===
        if 'retrieval_metrics' in r and r['retrieval_metrics']:
            rm = r['retrieval_metrics']
            
            # Doc recall - these are already float values (0.0 to 1.0), not binary
            if 'doc_recall@1' in rm and rm['doc_recall@1'] is not None:
                doc_recall_1.append(float(rm['doc_recall@1']))
            if 'doc_recall@3' in rm and rm['doc_recall@3'] is not None:
                doc_recall_3.append(float(rm['doc_recall@3']))
            if 'doc_recall@5' in rm and rm['doc_recall@5'] is not None:
                doc_recall_5.append(float(rm['doc_recall@5']))
            
            # Page recall - these are already float values (0.0 to 1.0), not binary
            if 'page_recall@1' in rm and rm['page_recall@1'] is not None:
                page_recall_1.append(float(rm['page_recall@1']))
            if 'page_recall@3' in rm and rm['page_recall@3'] is not None:
                page_recall_3.append(float(rm['page_recall@3']))
            if 'page_recall@5' in rm and rm['page_recall@5'] is not None:
                page_recall_5.append(float(rm['page_recall@5']))
        
        # Fallback: check top-level fields (convert to 0/1 only for binary fields like doc_recall_1)
        if not doc_recall_1 and 'doc_recall_1' in r and r['doc_recall_1'] is not None:
            # Top-level fields might be boolean or float, handle both
            val = r['doc_recall_1']
            doc_recall_1.append(float(val) if isinstance(val, (int, float)) else (1.0 if val else 0.0))
        if not doc_recall_3 and 'doc_recall_3' in r and r['doc_recall_3'] is not None:
            val = r['doc_recall_3']
            doc_recall_3.append(float(val) if isinstance(val, (int, float)) else (1.0 if val else 0.0))
        if not doc_recall_5 and 'doc_recall_5' in r and r['doc_recall_5'] is not None:
            val = r['doc_recall_5']
            doc_recall_5.append(float(val) if isinstance(val, (int, float)) else (1.0 if val else 0.0))
        
        if not page_recall_1 and 'page_recall_1' in r and r['page_recall_1'] is not None:
            val = r['page_recall_1']
            page_recall_1.append(float(val) if isinstance(val, (int, float)) else (1.0 if val else 0.0))
        if not page_recall_3 and 'page_recall_3' in r and r['page_recall_3'] is not None:
            val = r['page_recall_3']
            page_recall_3.append(float(val) if isinstance(val, (int, float)) else (1.0 if val else 0.0))
        if not page_recall_5 and 'page_recall_5' in r and r['page_recall_5'] is not None:
            val = r['page_recall_5']
            page_recall_5.append(float(val) if isinstance(val, (int, float)) else (1.0 if val else 0.0))
        
        # === GENERATION METRICS (from generative_metrics field) ===
        if 'generative_metrics' in r and r['generative_metrics']:
            gm = r['generative_metrics']
            
            if 'bleu_score' in gm and gm['bleu_score'] is not None:
                bleu_scores.append(gm['bleu_score'])
            
            if 'rouge_scores' in gm and gm['rouge_scores']:
                rs = gm['rouge_scores']
                if 'rouge1' in rs and rs['rouge1'] is not None:
                    rouge1_scores.append(rs['rouge1'])
                if 'rouge2' in rs and rs['rouge2'] is not None:
                    rouge2_scores.append(rs['rouge2'])
                if 'rougeL' in rs and rs['rougeL'] is not None:
                    rougeL_scores.append(rs['rougeL'])
            
            if 'numeric_match' in gm:
                q_type = r.get('question_type', '').lower()
                if q_type == 'metrics-generated':
                    numeric_total += 1
                    if gm['numeric_match']:
                        numeric_correct += 1
        
        # Fallback: check top-level fields
        if not bleu_scores and 'bleu_score' in r and r['bleu_score'] is not None:
            bleu_scores.append(r['bleu_score'])
        
        if not rouge1_scores and 'rouge_scores' in r and r['rouge_scores']:
            rs = r['rouge_scores']
            if 'rouge1' in rs and rs['rouge1'] is not None:
                rouge1_scores.append(rs['rouge1'])
            if 'rouge2' in rs and rs['rouge2'] is not None:
                rouge2_scores.append(rs['rouge2'])
            if 'rougeL' in rs and rs['rougeL'] is not None:
                rougeL_scores.append(rs['rougeL'])
        
        if not numeric_total and 'numeric_match' in r:
            q_type = r.get('question_type', '').lower()
            if q_type == 'metrics-generated':
                numeric_total += 1
                if r['numeric_match']:
                    numeric_correct += 1
    
    # Compute averages
    if doc_recall_1:
        metrics['doc_recall_1'] = np.mean(doc_recall_1)
        metrics['retrieval_samples'] = len(doc_recall_1)
    
    if doc_recall_3:
        metrics['doc_recall_3'] = np.mean(doc_recall_3)
    
    if doc_recall_5:
        metrics['doc_recall_5'] = np.mean(doc_recall_5)
    
    if page_recall_1:
        metrics['page_recall_1'] = np.mean(page_recall_1)
    
    if page_recall_3:
        metrics['page_recall_3'] = np.mean(page_recall_3)
    
    if page_recall_5:
        metrics['page_recall_5'] = np.mean(page_recall_5)
    
    if bleu_scores:
        metrics['gen_bleu'] = np.mean(bleu_scores)
        metrics['generation_samples'] = len(bleu_scores)
    
    if rouge1_scores:
        metrics['gen_rouge1'] = np.mean(rouge1_scores)
    
    if rouge2_scores:
        metrics['gen_rouge2'] = np.mean(rouge2_scores)
    
    if rougeL_scores:
        metrics['gen_rougeL'] = np.mean(rougeL_scores)
    
    metrics['numeric_correct'] = numeric_correct
    metrics['numeric_total'] = numeric_total
    
    return metrics


def extract_experiment_metadata(file_path: Path, data: Dict, results: List[Dict]) -> Dict[str, Any]:
    """Extract metadata from the top-level 'metadata' field in the JSON."""
    metadata = {
        'file_path': str(file_path),
        'file_name': file_path.name,
        'experiment_name': '',
        'date': '',
        'retrieval_method': '',
        'model_name': '',
        'embedding_model': '',
        'reranker_model': '',
    }
    
    # Extract date from filename (format: YYYYMMDD_HHMMSS)
    date_match = re.search(r'(\d{8})_(\d{6})', file_path.name)
    if date_match:
        date_str = date_match.group(1)
        time_str = date_match.group(2)
        try:
            dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
            metadata['date'] = dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            metadata['date'] = date_str
    elif date_match := re.search(r'(\d{8})', file_path.name):
        metadata['date'] = date_match.group(1)
    
    # Extract experiment name from path or filename
    parts = file_path.parts
    if 'results' in parts:
        idx = parts.index('results')
        if idx + 1 < len(parts):
            metadata['experiment_name'] = parts[idx + 1]
    elif 'outputs' in parts:
        # Use filename without extension and timestamp
        name = file_path.stem
        # Remove timestamp
        name = re.sub(r'_\d{8}_\d{6}', '', name)
        name = re.sub(r'_scored$', '', name)
        metadata['experiment_name'] = name
    
    # Extract from top-level 'metadata' field (priority)
    if isinstance(data, dict) and 'metadata' in data:
        meta = data['metadata']
        
        if 'experiment_type' in meta:
            metadata['retrieval_method'] = meta['experiment_type'].upper()
        
        if 'llm_model' in meta:
            metadata['model_name'] = meta['llm_model']
        
        if 'embedding_model' in meta:
            metadata['embedding_model'] = meta['embedding_model']
        
        if 'reranker_model' in meta:
            metadata['reranker_model'] = meta['reranker_model']
        
        if 'timestamp' in meta and not metadata['date']:
            try:
                ts = meta['timestamp']
                if isinstance(ts, str):
                    dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    metadata['date'] = dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                pass
    
    # Fallback: extract from filename if still empty
    if not metadata['retrieval_method']:
        filename_lower = file_path.name.lower()
        if 'bm25' in filename_lower:
            metadata['retrieval_method'] = 'BM25'
        elif 'hybrid' in filename_lower:
            metadata['retrieval_method'] = 'Hybrid'
        elif 'dense' in filename_lower:
            metadata['retrieval_method'] = 'Dense'
        elif 'splade' in filename_lower:
            metadata['retrieval_method'] = 'SPLADE'
        elif 'page_then_chunk' in filename_lower or 'page_chunk' in filename_lower:
            metadata['retrieval_method'] = 'Page-Then-Chunk'
        elif 'learned_page' in filename_lower:
            metadata['retrieval_method'] = 'Learned-Page'
        elif 'multi_hyde' in filename_lower:
            metadata['retrieval_method'] = 'Multi-HyDE'
    
    # Fallback: check config field
    if isinstance(data, dict) and 'config' in data:
        config = data['config']
        if 'retrieval_method' in config and not metadata['retrieval_method']:
            metadata['retrieval_method'] = config.get('retrieval_method', '')
        if 'model_name' in config and not metadata['model_name']:
            metadata['model_name'] = config.get('model_name', '')
        if 'embedding_model' in config and not metadata['embedding_model']:
            metadata['embedding_model'] = config.get('embedding_model', '')
    
    return metadata


def load_result_file(file_path: Path) -> Optional[tuple[Dict[str, Any], List[Dict[str, Any]]]]:
    """Load a result file and return metadata and results list."""
    # Skip non-result files
    skip_patterns = [
        'config.json', 'modules.json', 'tokenizer.json', 
        'tokenizer_config.json', 'sentence_bert_config.json',
        'config_sentence_transformers.json', 'training_metrics.json',
        'special_tokens_map.json', 'vocab.json', 'merges.txt'
    ]
    
    if file_path.name in skip_patterns:
        return None
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        logger.debug(f"Failed to load {file_path}: {e}")
        return None
    
    # Extract results list
    if isinstance(data, dict):
        if 'results' in data:
            results = data['results']
        elif 'samples' in data:
            results = data['samples']
        else:
            # Might be a dict where each key is a result
            if all(isinstance(v, dict) for v in data.values()):
                results = list(data.values())
            else:
                results = []
    elif isinstance(data, list):
        results = data
    else:
        return None
    
    # Ensure results is a list
    if not isinstance(results, list):
        results = list(results) if hasattr(results, '__iter__') else []
    
    # Filter: need at least 10 samples and must have 'question' field
    if len(results) < 10:
        return None
    
    # Check if samples have 'question' field
    sample_to_check = results[:min(5, len(results))]
    if not any('question' in r for r in sample_to_check if isinstance(r, dict)):
        return None
    
    # Extract metadata (pass results too)
    metadata = extract_experiment_metadata(file_path, data, results)
    
    return metadata, results


def aggregate_overall_results(result_dirs: List[Path]) -> List[Dict[str, Any]]:
    """Aggregate overall results from all experiment files."""
    all_experiments = []
    
    for results_dir in result_dirs:
        if not results_dir.exists():
            logger.warning(f"Directory not found: {results_dir}")
            continue
        
        logger.info(f"Processing {results_dir}...")
        
        # Find all JSON files
        json_files = list(results_dir.rglob("*.json"))
        logger.info(f"Found {len(json_files)} JSON files")
        
        for file_path in json_files:
            result = load_result_file(file_path)
            if result is None:
                continue
            
            metadata, results = result
            
            # Compute metrics
            metrics = compute_metrics_from_results(results)
            
            # Combine metadata and metrics
            experiment = {**metadata, **metrics}
            all_experiments.append(experiment)
            
            logger.info(f"  ✓ {file_path.name}: {metrics['total_samples']} samples")
    
    return all_experiments


def aggregate_by_group(result_dirs: List[Path], group_by: str) -> List[Dict[str, Any]]:
    """
    Aggregate results grouped by doc_type, question_type, or question_reasoning.
    
    Args:
        result_dirs: List of directories to process
        group_by: 'doc_type', 'question_type', or 'question_reasoning'
    """
    # First collect all results by experiment and group
    experiment_groups = defaultdict(lambda: defaultdict(list))
    
    for results_dir in result_dirs:
        if not results_dir.exists():
            continue
        
        json_files = list(results_dir.rglob("*.json"))
        
        for file_path in json_files:
            result = load_result_file(file_path)
            if result is None:
                continue
            
            metadata, results = result
            experiment_name = f"{metadata['experiment_name']}_{metadata['date']}"
            
            # Group results by the specified field
            for r in results:
                if group_by == 'doc_type':
                    # Use doc_type field or extract from doc_name
                    group_value = r.get('doc_type', '')
                    if not group_value:
                        doc_name = r.get('doc_name', '')
                        group_value = extract_doc_type_from_name(doc_name)
                elif group_by in ['question_type', 'question_reasoning']:
                    group_value = r.get(group_by, 'unknown')
                else:
                    group_value = 'unknown'
                
                experiment_groups[experiment_name][group_value].append(r)
    
    # Now compute metrics for each experiment-group combination
    aggregated_results = []
    
    for experiment_name, groups in experiment_groups.items():
        for group_value, group_results in groups.items():
            if not group_results:
                continue
            
            metrics = compute_metrics_from_results(group_results)
            
            result = {
                'experiment_name': experiment_name,
                'group_by': group_by,
                'group_value': group_value,
                **metrics
            }
            aggregated_results.append(result)
    
    return aggregated_results


def write_overall_csv(experiments: List[Dict[str, Any]], output_path: Path):
    """Write overall experiment results to CSV."""
    if not experiments:
        logger.warning("No experiments to write")
        return
    
    fieldnames = [
        'experiment_name', 'date', 'file_name', 'retrieval_method', 'model_name', 'embedding_model', 'reranker_model',
        'total_samples',
        # Retrieval metrics
        'retrieval_samples',
        'doc_recall_1', 'doc_recall_3', 'doc_recall_5',
        'page_recall_1', 'page_recall_3', 'page_recall_5',
        # Generation metrics
        'generation_samples',
        'gen_bleu', 'gen_rouge1', 'gen_rouge2', 'gen_rougeL',
        'numeric_correct', 'numeric_total',
        'file_path'
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(experiments)
    
    logger.info(f"✓ Wrote {len(experiments)} experiments to {output_path}")


def write_grouped_csv(results: List[Dict[str, Any]], output_path: Path):
    """Write grouped results to CSV."""
    if not results:
        logger.warning("No results to write")
        return
    
    fieldnames = [
        'experiment_name', 'group_by', 'group_value',
        'total_samples',
        # Retrieval metrics
        'retrieval_samples',
        'doc_recall_1', 'doc_recall_3', 'doc_recall_5',
        'page_recall_1', 'page_recall_3', 'page_recall_5',
        # Generation metrics
        'generation_samples',
        'gen_bleu', 'gen_rouge1', 'gen_rouge2', 'gen_rougeL',
        'numeric_correct', 'numeric_total',
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    
    logger.info(f"✓ Wrote {len(results)} grouped results to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate evaluation metrics from all experiment result files"
    )
    parser.add_argument(
        '--results-dirs',
        type=Path,
        nargs='+',
        default=[Path('outputs'), Path('results'), Path('evaluation_results')],
        help='Directories containing result files'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('aggregated_results'),
        help='Directory to write output CSV files'
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='',
        help='Prefix for output filenames'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{args.prefix}_" if args.prefix else ""
    
    logger.info("=" * 80)
    logger.info("AGGREGATING OVERALL RESULTS")
    logger.info("=" * 80)
    
    # 1. Overall results
    overall_experiments = aggregate_overall_results(args.results_dirs)
    overall_path = args.output_dir / f"{prefix}overall_results_{timestamp}.csv"
    write_overall_csv(overall_experiments, overall_path)
    
    logger.info("\n" + "=" * 80)
    logger.info("AGGREGATING BY DOC_TYPE")
    logger.info("=" * 80)
    
    # 2. Group by doc_type
    doc_type_results = aggregate_by_group(args.results_dirs, 'doc_type')
    doc_type_path = args.output_dir / f"{prefix}by_doc_type_{timestamp}.csv"
    write_grouped_csv(doc_type_results, doc_type_path)
    
    logger.info("\n" + "=" * 80)
    logger.info("AGGREGATING BY QUESTION_TYPE")
    logger.info("=" * 80)
    
    # 3. Group by question_type
    question_type_results = aggregate_by_group(args.results_dirs, 'question_type')
    question_type_path = args.output_dir / f"{prefix}by_question_type_{timestamp}.csv"
    write_grouped_csv(question_type_results, question_type_path)
    
    logger.info("\n" + "=" * 80)
    logger.info("AGGREGATING BY QUESTION_REASONING")
    logger.info("=" * 80)
    
    # 4. Group by question_reasoning
    reasoning_results = aggregate_by_group(args.results_dirs, 'question_reasoning')
    reasoning_path = args.output_dir / f"{prefix}by_question_reasoning_{timestamp}.csv"
    write_grouped_csv(reasoning_results, reasoning_path)
    
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Overall experiments: {len(overall_experiments)}")
    logger.info(f"Doc type groups: {len(doc_type_results)}")
    logger.info(f"Question type groups: {len(question_type_results)}")
    logger.info(f"Question reasoning groups: {len(reasoning_results)}")
    logger.info(f"\nOutput files written to: {args.output_dir}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
