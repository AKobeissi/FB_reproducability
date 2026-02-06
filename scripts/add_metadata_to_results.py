"""
Add metadata (question_type, question_reasoning, doc_type) to existing result files.
Uses FinanceBench dataset to lookup metadata by matching question text.
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List
import sys

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_financebench_metadata(data_file: Path) -> Dict[str, Dict[str, str]]:
    """
    Load FinanceBench metadata indexed by question text.
    Returns dict: {question_text: {question_type, question_reasoning, financebench_id, doc_name}}
    """
    metadata = {}
    
    with open(data_file, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            question = item.get('question', '')
            if question:
                metadata[question] = {
                    'question_type': item.get('question_type', ''),
                    'question_reasoning': item.get('question_reasoning', ''),
                    'financebench_id': item.get('financebench_id', ''),
                    'doc_name': item.get('doc_name', ''),
                }
    
    logger.info(f"Loaded metadata for {len(metadata)} questions from {data_file}")
    return metadata


def load_doc_type_mapping(doc_info_file: Path) -> Dict[str, str]:
    """
    Load document type information indexed by doc_name.
    Returns dict: {doc_name: doc_type}
    """
    doc_types = {}
    
    with open(doc_info_file, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            doc_name = item.get('doc_name', '')
            doc_type = item.get('doc_type', '')
            if doc_name:
                doc_types[doc_name] = doc_type
    
    logger.info(f"Loaded doc_type for {len(doc_types)} documents from {doc_info_file}")
    return doc_types


def update_result_file(
    file_path: Path, 
    question_metadata: Dict[str, Dict[str, str]],
    doc_type_mapping: Dict[str, str],
    dry_run: bool = False,
    min_samples: int = 100
) -> tuple[int, int, int]:
    """
    Update a single result file with metadata.
    Returns: (total_samples, updated_samples, missing_samples)
    """
    # Skip known non-result files
    skip_patterns = [
        'config.json', 'modules.json', 'tokenizer.json', 
        'tokenizer_config.json', 'sentence_bert_config.json',
        'config_sentence_transformers.json', 'training_metrics.json',
        'special_tokens_map.json', 'vocab.json', 'merges.txt'
    ]
    
    if file_path.name in skip_patterns:
        logger.debug(f"Skipping non-result file: {file_path.name}")
        return 0, 0, 0
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return 0, 0, 0
    
    # Handle both list and dict formats
    if isinstance(data, dict):
        if 'results' in data:
            results = data['results']
        elif 'samples' in data:
            results = data['samples']
        else:
            # Might be a dict where each key is a result
            results = list(data.values()) if all(isinstance(v, dict) for v in data.values()) else []
    elif isinstance(data, list):
        results = data
    else:
        logger.warning(f"Unknown format for {file_path}")
        return 0, 0, 0
    
    # Filter: Only process files with at least min_samples samples that look like experiment results
    if len(results) < min_samples:
        logger.debug(f"Skipping {file_path.name}: only {len(results)} samples (need at least {min_samples})")
        return 0, 0, 0
    
    # Check if samples have experiment-result structure (should have 'question' field)
    if results:
        sample_has_question = any(
            isinstance(r, dict) and 'question' in r 
            for r in results[:5]  # Check first 5 samples
        )
        if not sample_has_question:
            logger.debug(f"Skipping {file_path.name}: samples don't have 'question' field")
            return 0, 0, 0
    
    total = len(results)
    updated = 0
    missing = 0
    
    for result in results:
        if not isinstance(result, dict):
            continue
        
        question = result.get('question', '')
        doc_name = result.get('doc_name', '')
        
        # Check if already has all metadata
        has_metadata = all(
            result.get(field)
            for field in ['question_type', 'question_reasoning', 'doc_type']
        )
        
        if has_metadata:
            continue
        
        # Look up metadata by question
        if question in question_metadata:
            meta = question_metadata[question]
            result['question_type'] = meta['question_type']
            result['question_reasoning'] = meta['question_reasoning']
            if 'financebench_id' not in result or not result.get('financebench_id'):
                result['financebench_id'] = meta['financebench_id']
            updated += 1
        else:
            missing += 1
            logger.debug(f"No metadata found for question: {question[:100]}...")
        
        # Add doc_type
        if not result.get('doc_type') and doc_name in doc_type_mapping:
            result['doc_type'] = doc_type_mapping[doc_name]
        elif not result.get('doc_type') and doc_name:
            # Try without underscores/normalized
            normalized_doc_name = doc_name.replace('_', '').lower()
            for dn, dt in doc_type_mapping.items():
                if dn.replace('_', '').lower() == normalized_doc_name:
                    result['doc_type'] = dt
                    break
    
    if not dry_run and updated > 0:
        # Save updated file
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"✓ Updated {file_path.name}: {updated}/{total} samples updated, {missing} missing")
    else:
        logger.info(f"  {file_path.name}: {updated}/{total} samples would be updated, {missing} missing")
    
    return total, updated, missing


def process_directory(
    directory: Path,
    question_metadata: Dict[str, Dict[str, str]],
    doc_type_mapping: Dict[str, str],
    pattern: str = "*.json",
    recursive: bool = True,
    dry_run: bool = False,
    min_samples: int = 100
):
    """Process all JSON files in a directory."""
    
    if recursive:
        files = list(directory.rglob(pattern))
    else:
        files = list(directory.glob(pattern))
    
    logger.info(f"\nFound {len(files)} files matching '{pattern}' in {directory}")
    
    total_samples = 0
    total_updated = 0
    total_missing = 0
    
    for file_path in sorted(files):
        t, u, m = update_result_file(file_path, question_metadata, doc_type_mapping, dry_run, min_samples)
        total_samples += t
        total_updated += u
        total_missing += m
    
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Files processed: {len(files)}")
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Samples updated: {total_updated}")
    logger.info(f"Samples missing metadata: {total_missing}")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Add metadata (question_type, question_reasoning, doc_type) to result files"
    )
    parser.add_argument(
        '--data-file',
        type=Path,
        default=Path('data/financebench_open_source.jsonl'),
        help='Path to FinanceBench data file'
    )
    parser.add_argument(
        '--doc-info-file',
        type=Path,
        default=Path('data/financebench_document_information.jsonl'),
        help='Path to document information file'
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        default=Path('outputs'),
        help='Directory containing result files to update'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.json',
        help='File pattern to match (default: *.json)'
    )
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='Do not search recursively'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be updated without making changes'
    )
    parser.add_argument(
        '--file',
        type=Path,
        help='Update a single file instead of a directory'
    )
    parser.add_argument(
        '--min-samples',
        type=int,
        default=100,
        help='Minimum number of samples for a file to be processed (default: 100)'
    )
    
    args = parser.parse_args()
    
    # Load metadata
    logger.info("Loading FinanceBench metadata...")
    question_metadata = load_financebench_metadata(args.data_file)
    doc_type_mapping = load_doc_type_mapping(args.doc_info_file)
    
    if args.dry_run:
        logger.info("\n*** DRY RUN MODE - No files will be modified ***\n")
    
    logger.info(f"Minimum samples threshold: {args.min_samples}")
    
    # Process files
    if args.file:
        if not args.file.exists():
            logger.error(f"File not found: {args.file}")
            sys.exit(1)
        update_result_file(args.file, question_metadata, doc_type_mapping, args.dry_run, args.min_samples)
    else:
        if not args.results_dir.exists():
            logger.error(f"Directory not found: {args.results_dir}")
            sys.exit(1)
        process_directory(
            args.results_dir,
            question_metadata,
            doc_type_mapping,
            pattern=args.pattern,
            recursive=not args.no_recursive,
            dry_run=args.dry_run,
            min_samples=args.min_samples
        )


if __name__ == '__main__':
    main()
