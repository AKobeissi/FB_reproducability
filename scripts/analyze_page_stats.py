"""
Analyze page statistics for the 84 PDFs in FinanceBench dataset.

Provides statistics on page character counts:
- Max characters per page
- Mean characters per page
- Median characters per page
- Distribution across all pages
"""
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Set
import matplotlib.pyplot as plt
import seaborn as sns

from src.ingestion.data_loader import FinanceBenchLoader
from src.ingestion.page_processor import extract_pages_from_pdf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_unique_documents() -> Set[str]:
    """Get unique document names from FinanceBench dataset."""
    loader = FinanceBenchLoader()
    df = loader.load_data()
    unique_docs = set(df['doc_name'].unique())
    logger.info(f"Found {len(unique_docs)} unique documents in FinanceBench")
    return unique_docs


def analyze_page_statistics(pdf_dir: Path, doc_names: Set[str]) -> Dict:
    """
    Analyze page statistics for all documents.
    
    Returns:
        Dictionary with statistics and per-document data
    """
    all_page_lengths = []
    doc_stats = {}
    failed_docs = []
    
    logger.info(f"Analyzing {len(doc_names)} PDFs from {pdf_dir}")
    
    for doc_name in sorted(doc_names):
        pdf_path = pdf_dir / f"{doc_name}.pdf"
        
        if not pdf_path.exists():
            logger.warning(f"PDF not found: {pdf_path}")
            failed_docs.append(doc_name)
            continue
        
        try:
            pages = extract_pages_from_pdf(pdf_path, doc_name)
            page_lengths = [len(page['text']) for page in pages]
            
            doc_stats[doc_name] = {
                'num_pages': len(pages),
                'page_lengths': page_lengths,
                'max': max(page_lengths) if page_lengths else 0,
                'min': min(page_lengths) if page_lengths else 0,
                'mean': np.mean(page_lengths) if page_lengths else 0,
                'median': np.median(page_lengths) if page_lengths else 0,
                'std': np.std(page_lengths) if page_lengths else 0,
            }
            
            all_page_lengths.extend(page_lengths)
            
            logger.info(f"✓ {doc_name}: {len(pages)} pages, "
                       f"mean={doc_stats[doc_name]['mean']:.0f} chars, "
                       f"max={doc_stats[doc_name]['max']:.0f} chars")
            
        except Exception as e:
            logger.error(f"Failed to process {doc_name}: {e}")
            failed_docs.append(doc_name)
    
    # Overall statistics
    overall_stats = {
        'total_documents': len(doc_names),
        'successful_documents': len(doc_stats),
        'failed_documents': len(failed_docs),
        'failed_doc_names': failed_docs,
        'total_pages': len(all_page_lengths),
        'max_chars': int(np.max(all_page_lengths)) if all_page_lengths else 0,
        'min_chars': int(np.min(all_page_lengths)) if all_page_lengths else 0,
        'mean_chars': float(np.mean(all_page_lengths)) if all_page_lengths else 0,
        'median_chars': float(np.median(all_page_lengths)) if all_page_lengths else 0,
        'std_chars': float(np.std(all_page_lengths)) if all_page_lengths else 0,
        'percentiles': {
            '25th': float(np.percentile(all_page_lengths, 25)) if all_page_lengths else 0,
            '50th': float(np.percentile(all_page_lengths, 50)) if all_page_lengths else 0,
            '75th': float(np.percentile(all_page_lengths, 75)) if all_page_lengths else 0,
            '90th': float(np.percentile(all_page_lengths, 90)) if all_page_lengths else 0,
            '95th': float(np.percentile(all_page_lengths, 95)) if all_page_lengths else 0,
            '99th': float(np.percentile(all_page_lengths, 99)) if all_page_lengths else 0,
        }
    }
    
    return {
        'overall': overall_stats,
        'per_document': doc_stats,
        'all_page_lengths': all_page_lengths
    }


def print_statistics(stats: Dict):
    """Print formatted statistics."""
    overall = stats['overall']
    
    print("\n" + "="*80)
    print("PAGE CHARACTER STATISTICS - FINANCEBENCH 84 PDFs")
    print("="*80)
    print(f"\nDocuments Processed: {overall['successful_documents']}/{overall['total_documents']}")
    if overall['failed_documents'] > 0:
        print(f"Failed: {overall['failed_documents']} - {overall['failed_doc_names']}")
    print(f"Total Pages: {overall['total_pages']}")
    
    print("\n" + "-"*80)
    print("CHARACTER COUNT STATISTICS (per page)")
    print("-"*80)
    print(f"  Maximum:  {overall['max_chars']:,} chars")
    print(f"  Mean:     {overall['mean_chars']:,.0f} chars")
    print(f"  Median:   {overall['median_chars']:,.0f} chars")
    print(f"  Minimum:  {overall['min_chars']:,} chars")
    print(f"  Std Dev:  {overall['std_chars']:,.0f} chars")
    
    print("\n" + "-"*80)
    print("PERCENTILES")
    print("-"*80)
    for pct_name, pct_value in overall['percentiles'].items():
        print(f"  {pct_name:8s} {pct_value:,.0f} chars")
    
    print("\n" + "-"*80)
    print("TOP 10 DOCUMENTS BY MAX PAGE LENGTH")
    print("-"*80)
    doc_stats = stats['per_document']
    sorted_docs = sorted(doc_stats.items(), key=lambda x: x[1]['max'], reverse=True)[:10]
    for i, (doc_name, doc_stat) in enumerate(sorted_docs, 1):
        print(f"  {i:2d}. {doc_name:40s} {doc_stat['max']:6,} chars "
              f"(mean: {doc_stat['mean']:5,.0f}, pages: {doc_stat['num_pages']})")
    
    print("\n" + "-"*80)
    print("TOP 10 DOCUMENTS BY MEAN PAGE LENGTH")
    print("-"*80)
    sorted_docs = sorted(doc_stats.items(), key=lambda x: x[1]['mean'], reverse=True)[:10]
    for i, (doc_name, doc_stat) in enumerate(sorted_docs, 1):
        print(f"  {i:2d}. {doc_name:40s} {doc_stat['mean']:6,.0f} chars "
              f"(max: {doc_stat['max']:6,}, pages: {doc_stat['num_pages']})")
    
    print("\n" + "="*80)


def plot_distributions(stats: Dict, output_dir: Path):
    """Create visualizations of page length distributions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_lengths = stats['all_page_lengths']
    overall = stats['overall']
    
    # Set style
    sns.set_style("whitegrid")
    
    # Figure 1: Histogram + KDE
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Histogram
    ax = axes[0, 0]
    ax.hist(all_lengths, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(overall['mean_chars'], color='red', linestyle='--', linewidth=2, label=f"Mean: {overall['mean_chars']:.0f}")
    ax.axvline(overall['median_chars'], color='green', linestyle='--', linewidth=2, label=f"Median: {overall['median_chars']:.0f}")
    ax.set_xlabel('Characters per Page', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Page Lengths', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Box plot
    ax = axes[0, 1]
    ax.boxplot(all_lengths, vert=True)
    ax.set_ylabel('Characters per Page', fontsize=12)
    ax.set_title('Box Plot of Page Lengths', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Cumulative distribution
    ax = axes[1, 0]
    sorted_lengths = sorted(all_lengths)
    cumulative = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)
    ax.plot(sorted_lengths, cumulative, linewidth=2, color='steelblue')
    ax.axhline(0.5, color='green', linestyle='--', alpha=0.5, label='50th percentile')
    ax.axhline(0.95, color='red', linestyle='--', alpha=0.5, label='95th percentile')
    ax.set_xlabel('Characters per Page', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_title('Cumulative Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Log scale histogram
    ax = axes[1, 1]
    ax.hist(all_lengths, bins=50, color='coral', alpha=0.7, edgecolor='black')
    ax.set_yscale('log')
    ax.set_xlabel('Characters per Page', fontsize=12)
    ax.set_ylabel('Frequency (log scale)', fontsize=12)
    ax.set_title('Distribution (Log Scale)', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'page_length_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved distribution plot to {output_dir / 'page_length_distribution.png'}")
    
    # Figure 2: Per-document statistics
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    doc_stats = stats['per_document']
    doc_names = list(doc_stats.keys())
    doc_means = [doc_stats[d]['mean'] for d in doc_names]
    doc_maxs = [doc_stats[d]['max'] for d in doc_names]
    
    # Mean page lengths per document
    ax = axes[0]
    sorted_indices = np.argsort(doc_means)
    ax.barh(range(len(doc_names)), [doc_means[i] for i in sorted_indices], color='steelblue', alpha=0.7)
    ax.set_xlabel('Mean Characters per Page', fontsize=12)
    ax.set_ylabel('Document Index', fontsize=12)
    ax.set_title('Mean Page Length by Document', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, axis='x')
    
    # Max page lengths per document
    ax = axes[1]
    sorted_indices = np.argsort(doc_maxs)
    ax.barh(range(len(doc_names)), [doc_maxs[i] for i in sorted_indices], color='coral', alpha=0.7)
    ax.set_xlabel('Max Characters per Page', fontsize=12)
    ax.set_ylabel('Document Index', fontsize=12)
    ax.set_title('Max Page Length by Document', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'per_document_stats.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved per-document plot to {output_dir / 'per_document_stats.png'}")


def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze page statistics for FinanceBench PDFs")
    parser.add_argument('--pdf-dir', type=str, default='pdfs', help='Directory containing PDFs')
    parser.add_argument('--output-dir', type=str, default='analysis/page_stats', help='Output directory for results')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots')
    
    args = parser.parse_args()
    
    pdf_dir = Path(args.pdf_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get unique documents
    doc_names = get_unique_documents()
    
    # Analyze statistics
    stats = analyze_page_statistics(pdf_dir, doc_names)
    
    # Print results
    print_statistics(stats)
    
    # Save JSON results
    output_file = output_dir / 'page_statistics.json'
    # Remove raw page lengths for JSON (too large)
    stats_for_json = {
        'overall': stats['overall'],
        'per_document': stats['per_document']
    }
    with open(output_file, 'w') as f:
        json.dump(stats_for_json, f, indent=2)
    logger.info(f"Saved statistics to {output_file}")
    
    # Create visualizations
    if not args.no_plots:
        plot_distributions(stats, output_dir)
    
    print(f"\n✓ Analysis complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
