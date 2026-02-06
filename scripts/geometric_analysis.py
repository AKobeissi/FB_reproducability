"""
Geometric Analysis of Retrieval Difficulty: Curvature-Based Analysis
Computes local curvature in embedding space to predict retrieval difficulty.
"""
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from scipy import stats
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeometricAnalyzer:
    """Analyze local geometry of embedding space and its impact on retrieval."""
    
    def __init__(self, k_neighbors: int = 50):
        """
        Args:
            k_neighbors: Number of neighbors to consider for local curvature
        """
        self.k_neighbors = k_neighbors
        self.query_embeddings = None
        self.chunk_embeddings = None
        self.curvatures = None
        self.eigenvalue_spectra = None
        
    def compute_local_intrinsic_dimension(self, query_embedding: np.ndarray, corpus_embeddings: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute Local Intrinsic Dimensionality (LID) using MLE estimator.
        
        LID measures the effective dimensionality of the local neighborhood:
        - Low LID (e.g., 15) → neighbors in small subspace → easy retrieval
        - High LID (e.g., 120) → neighbors spread across many dims → hard retrieval
        
        Args:
            query_embedding: (d,) query vector
            corpus_embeddings: (N, d) all chunk embeddings
        
        Returns:
            lid: scalar, effective local dimensionality (higher = harder)
            distances: k nearest neighbor distances for analysis
        """
        # Find k nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors + 1, metric='cosine').fit(corpus_embeddings)
        distances, indices = nbrs.kneighbors([query_embedding])
        
        # Remove self (first neighbor is query itself with distance ~0)
        distances = distances[0][1:]  # k nearest neighbors
        
        # MLE estimator for intrinsic dimension
        # LID = k / sum(log(r_k / r_i)) where r_i is distance to i-th neighbor
        r_k = distances[-1]  # Distance to k-th neighbor
        
        if r_k < 1e-10:  # All neighbors at same distance
            return float(self.k_neighbors), distances
        
        # Compute log ratios
        log_ratios = np.log(r_k / (distances + 1e-10))
        lid = self.k_neighbors / np.sum(log_ratios)
        
        # Clip to reasonable range [1, embedding_dim]
        lid = np.clip(lid, 1.0, corpus_embeddings.shape[1])
        
        return float(lid), distances
    
    def compute_all_curvatures(self):
        """Compute Local Intrinsic Dimensionality for all queries."""
        if self.query_embeddings is None or self.chunk_embeddings is None:
            raise ValueError("Embeddings not loaded. Call load_embeddings first.")
        
        logger.info(f"Computing LID for {len(self.query_embeddings)} queries...")
        
        lids = []
        distance_profiles = []
        
        for i, q_emb in enumerate(tqdm(self.query_embeddings, desc="Computing LID")):
            lid, distances = self.compute_local_intrinsic_dimension(q_emb, self.chunk_embeddings)
            lids.append(lid)
            distance_profiles.append(distances)
        
        self.curvatures = np.array(lids)  # Store as curvatures for compatibility
        self.eigenvalue_spectra = np.array(distance_profiles)  # Store distances
        
        logger.info(f"LID statistics:")
        logger.info(f"  Mean: {self.curvatures.mean():.2f}")
        logger.info(f"  Std:  {self.curvatures.std():.2f}")
        logger.info(f"  Min:  {self.curvatures.min():.2f}")
        logger.info(f"  Max:  {self.curvatures.max():.2f}")
        
    def save_curvatures(self, output_dir: str):
        """Save curvature analysis results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        np.save(output_path / "query_curvatures.npy", self.curvatures)
        np.save(output_path / "eigenvalue_spectra.npy", self.eigenvalue_spectra)
        
        logger.info(f"Saved curvature results to {output_path}")
        
    def load_retrieval_results(self, results_file: str) -> pd.DataFrame:
        """
        Load retrieval results and add curvature information.
        
        Args:
            results_file: Path to results JSON file (e.g., *_scored.json)
        
        Returns:
            DataFrame with results and curvatures
        """
        logger.info(f"Loading retrieval results from {results_file}")
        
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        # Extract results
        if 'results' in data:
            results = data['results']
        else:
            results = data if isinstance(data, list) else [data]
        
        # Build DataFrame
        df_data = []
        for i, res in enumerate(results):
            row = {
                'sample_id': res.get('sample_id', i),
                'question': res.get('question', ''),
                'doc_name': res.get('doc_name', ''),
                'question_type': res.get('question_type', ''),
                'question_reasoning': res.get('question_reasoning', ''),
            }
            
            # Extract retrieval metrics
            if 'retrieval_metrics' in res:
                metrics = res['retrieval_metrics']
                row['page_rec@5'] = metrics.get('page_recall@5', 0.0)
                row['page_rec@10'] = metrics.get('page_recall@10', 0.0)
                row['context_bleu@5'] = metrics.get('context_bleu@5', 0.0)
                row['context_rougeL@5'] = metrics.get('context_rougeL@5', 0.0)
            
            # Extract generative metrics
            if 'generative_metrics' in res:
                gen_metrics = res['generative_metrics']
                row['answer_numeric_match'] = gen_metrics.get('numeric_match', 0.0)
                row['answer_bleu'] = gen_metrics.get('bleu', 0.0)
                row['answer_rougeL'] = gen_metrics.get('rougeL', 0.0)
            
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Add curvatures
        if self.curvatures is not None and len(self.curvatures) == len(df):
            df['curvature'] = self.curvatures
        else:
            logger.warning(f"Curvature count ({len(self.curvatures) if self.curvatures is not None else 0}) "
                          f"doesn't match results count ({len(df)})")
        
        return df
    
    def plot_curvature_vs_performance(self, results_df: pd.DataFrame, output_dir: str, 
                                     metric_col: str = 'page_rec@5', method_name: str = 'Method'):
        """Plot curvature vs retrieval performance."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if 'curvature' not in results_df.columns:
            logger.warning("No curvature data in results. Skipping plot.")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Scatter plot
        ax.scatter(results_df['curvature'], results_df[metric_col], alpha=0.6, s=50)
        
        # Trend line
        z = np.polyfit(results_df['curvature'].dropna(), results_df[metric_col].dropna(), 1)
        p = np.poly1d(z)
        ax.plot(results_df['curvature'], p(results_df['curvature']), 
                "r--", alpha=0.8, linewidth=2, label=f'Trend (slope={z[0]:.2f})')
        
        # Correlation
        corr, pval = stats.pearsonr(results_df['curvature'].dropna(), results_df[metric_col].dropna())
        
        ax.set_xlabel('Local Intrinsic Dimensionality (LID)', fontsize=12)
        ax.set_ylabel(f'{metric_col}', fontsize=12)
        ax.set_title(f'{method_name}\nLID vs Performance (r={corr:.3f}, p={pval:.4f})', fontsize=13)
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / f'lid_vs_{metric_col.replace("@", "_at_")}.pdf', 
                   dpi=300, bbox_inches='tight')
        plt.savefig(output_path / f'lid_vs_{metric_col.replace("@", "_at_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved plot: lid_vs_{metric_col}")
        logger.info(f"Correlation: r={corr:.3f}, p={pval:.4f}")
        
    def plot_curvature_by_question_type(self, results_df: pd.DataFrame, output_dir: str):
        """Plot curvature distribution by question type."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if 'curvature' not in results_df.columns or 'question_type' not in results_df.columns:
            logger.warning("Missing curvature or question_type data. Skipping plot.")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        question_types = results_df['question_type'].unique()
        data_to_plot = [results_df[results_df['question_type'] == qt]['curvature'].values 
                        for qt in question_types]
        
        bp = ax.boxplot(data_to_plot, labels=question_types, patch_artist=True)
        
        # Color boxes
        colors = plt.cm.Set3(range(len(question_types)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_ylabel('Local Intrinsic Dimensionality (LID)', fontsize=12)
        ax.set_xlabel('Question Type', fontsize=12)
        ax.set_title('LID Distribution by Question Type', fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=15, ha='right')
        
        plt.tight_layout()
        plt.savefig(output_path / 'lid_by_question_type.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(output_path / 'lid_by_question_type.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Saved plot: lid_by_question_type")
        
        # Statistical tests
        logger.info("\nStatistical tests (Mann-Whitney U):")
        for i, qt1 in enumerate(question_types):
            for qt2 in question_types[i+1:]:
                d1 = results_df[results_df['question_type'] == qt1]['curvature'].dropna()
                d2 = results_df[results_df['question_type'] == qt2]['curvature'].dropna()
                if len(d1) > 0 and len(d2) > 0:
                    stat, pval = stats.mannwhitneyu(d1, d2)
                    logger.info(f"  {qt1} vs {qt2}: p={pval:.4f}")
        
    def plot_eigenvalue_spectrum(self, output_dir: str):
        """Plot distance profiles for representative queries."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.curvatures is None or self.eigenvalue_spectra is None:
            logger.warning("No LID/distance data. Skipping plot.")
            return
        
        # Pick representative queries
        low_lid_idx = self.curvatures.argmin()
        high_lid_idx = self.curvatures.argmax()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot distance profiles (stored in eigenvalue_spectra)
        distances_low = self.eigenvalue_spectra[low_lid_idx]
        distances_high = self.eigenvalue_spectra[high_lid_idx]
        
        ax.plot(range(1, len(distances_low) + 1), distances_low, 
                'o-', linewidth=2, markersize=6, 
                label=f'Low LID (LID={self.curvatures[low_lid_idx]:.1f})')
        ax.plot(range(1, len(distances_high) + 1), distances_high, 
                's-', linewidth=2, markersize=6, 
                label=f'High LID (LID={self.curvatures[high_lid_idx]:.1f})')
        
        ax.set_xlabel('Neighbor Rank (k)', fontsize=12)
        ax.set_ylabel('Cosine Distance to Query', fontsize=12)
        ax.set_title('Distance Profile: Low vs High LID Queries', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'distance_profile.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(output_path / 'distance_profile.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Saved plot: distance_profile")
        
    def plot_curvature_heatmap_2d(self, output_dir: str):
        """Plot curvature heatmap on 2D t-SNE projection."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.query_embeddings is None or self.curvatures is None:
            logger.warning("No embeddings/curvature data. Skipping plot.")
            return
        
        logger.info("Computing t-SNE projection (this may take a minute)...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(self.query_embeddings) - 1))
        query_2d = tsne.fit_transform(self.query_embeddings)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Scatter plot colored by curvature
        scatter = ax.scatter(query_2d[:, 0], query_2d[:, 1], 
                            c=self.curvatures, cmap='RdYlBu_r', 
                            s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Annotate extremes
        low_curv_idx = self.curvatures.argmin()
        high_curv_idx = self.curvatures.argmax()
        
        for idx, label in [(low_curv_idx, 'Low'), (high_curv_idx, 'High')]:
            ax.annotate(f'{label}: κ={self.curvatures[idx]:.2f}', 
                       xy=(query_2d[idx, 0], query_2d[idx, 1]),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=10, bbox=dict(boxstyle='round', fc='white', alpha=0.8),
                       arrowprops=dict(arrowstyle='->', lw=1.5))
        
        plt.colorbar(scatter, label='Curvature κ(q)', ax=ax)
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.set_title('Query Embedding Space Colored by Local Curvature', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(output_path / 'curvature_heatmap_2d.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(output_path / 'curvature_heatmap_2d.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Saved plot: curvature_heatmap_2d")
        
    def generate_statistical_report(self, results_df: pd.DataFrame, output_dir: str):
        """Generate comprehensive statistical report."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("GEOMETRIC ANALYSIS REPORT")
        report_lines.append("="*80)
        report_lines.append("")
        
        # Curvature statistics
        report_lines.append("1. LOCAL INTRINSIC DIMENSIONALITY (LID) STATISTICS")
        report_lines.append("-" * 40)
        report_lines.append(f"Mean LID:          {results_df['curvature'].mean():.2f}")
        report_lines.append(f"Std deviation:     {results_df['curvature'].std():.2f}")
        report_lines.append(f"Min LID:           {results_df['curvature'].min():.2f}")
        report_lines.append(f"Max LID:           {results_df['curvature'].max():.2f}")
        report_lines.append(f"Median LID:        {results_df['curvature'].median():.2f}")
        report_lines.append(f"")
        report_lines.append(f"Interpretation:")
        report_lines.append(f"  Low LID (~10-20):  Neighbors in low-dimensional subspace → easy retrieval")
        report_lines.append(f"  High LID (~100+):  Neighbors spread across many dimensions → hard retrieval")
        report_lines.append("")
        
        # Correlation analysis
        report_lines.append("2. CORRELATION WITH RETRIEVAL METRICS")
        report_lines.append("-" * 40)
        
        metric_cols = [col for col in results_df.columns if 'rec' in col or 'bleu' in col or 'rouge' in col]
        for col in metric_cols:
            if col in results_df.columns:
                corr, pval = stats.pearsonr(results_df['curvature'].dropna(), 
                                           results_df[col].dropna())
                report_lines.append(f"{col:25s}: r = {corr:+.3f}, p = {pval:.4f}")
        report_lines.append("")
        
        # Performance by LID quartile
        report_lines.append("3. PERFORMANCE BY LID QUARTILE")
        report_lines.append("-" * 40)
        
        # Only do quartile analysis if LIDs vary
        if results_df['curvature'].std() > 1e-6:
            results_df['curv_quartile'] = pd.qcut(results_df['curvature'], q=4, 
                                                   labels=['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)'])
            
            if 'page_rec@5' in results_df.columns:
                quartile_stats = results_df.groupby('curv_quartile')['page_rec@5'].agg(['mean', 'std', 'count'])
                report_lines.append("\nPageRec@5 by LID Quartile:")
                report_lines.append(quartile_stats.to_string())
                report_lines.append("")
        else:
            report_lines.append("\\nCannot perform quartile analysis: all LIDs are constant")
            report_lines.append("")
        
        # Question type analysis
        if 'question_type' in results_df.columns:
            report_lines.append("4. LID BY QUESTION TYPE")
            report_lines.append("-" * 40)
            
            type_stats = results_df.groupby('question_type')['curvature'].agg(['mean', 'std', 'count'])
            report_lines.append(type_stats.to_string())
            report_lines.append("")
        
        # Save report
        report_text = "\n".join(report_lines)
        with open(output_path / 'geometric_analysis_report.txt', 'w') as f:
            f.write(report_text)
        
        logger.info("Saved statistical report")
        logger.info("\n" + report_text)
        
    def run_full_analysis(self, results_file: str, output_dir: str, method_name: str = "Method"):
        """Run complete geometric analysis pipeline."""
        logger.info("="*80)
        logger.info(f"RUNNING GEOMETRIC ANALYSIS FOR: {method_name}")
        logger.info("="*80)
        
        # Load results
        results_df = self.load_retrieval_results(results_file)
        
        # Save curvatures
        self.save_curvatures(output_dir)
        
        # Save results with curvature
        results_df.to_csv(Path(output_dir) / 'results_with_curvature.csv', index=False)
        logger.info(f"Saved results with curvature to {output_dir}/results_with_curvature.csv")
        
        # Generate all plots
        logger.info("\nGenerating visualizations...")
        
        # Plot 1: Curvature vs performance
        if 'page_rec@5' in results_df.columns:
            self.plot_curvature_vs_performance(results_df, output_dir, 
                                              metric_col='page_rec@5', method_name=method_name)
        
        # Plot 2: Curvature by question type
        if 'question_type' in results_df.columns:
            self.plot_curvature_by_question_type(results_df, output_dir)
        
        # Plot 3: Eigenvalue spectrum
        self.plot_eigenvalue_spectrum(output_dir)
        
        # Plot 4: 2D heatmap
        self.plot_curvature_heatmap_2d(output_dir)
        
        # Generate statistical report
        self.generate_statistical_report(results_df, output_dir)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Analysis complete! Results saved to: {output_dir}")
        logger.info(f"{'='*80}\n")


def main():
    """Main entry point for geometric analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Geometric analysis of retrieval difficulty")
    parser.add_argument('--results-file', type=str, required=True,
                       help='Path to retrieval results JSON file (*_scored.json)')
    parser.add_argument('--embeddings-dir', type=str, required=True,
                       help='Directory containing query_embeddings.npy and chunk_embeddings.npy')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save analysis results')
    parser.add_argument('--method-name', type=str, default='Method',
                       help='Name of retrieval method for plots')
    parser.add_argument('--k-neighbors', type=int, default=50,
                       help='Number of neighbors for curvature computation')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = GeometricAnalyzer(k_neighbors=args.k_neighbors)
    
    # Load embeddings
    logger.info(f"Loading embeddings from {args.embeddings_dir}")
    query_emb_path = Path(args.embeddings_dir) / "query_embeddings.npy"
    chunk_emb_path = Path(args.embeddings_dir) / "chunk_embeddings.npy"
    
    if not query_emb_path.exists() or not chunk_emb_path.exists():
        logger.error(f"Embeddings not found in {args.embeddings_dir}")
        logger.error("Please ensure query_embeddings.npy and chunk_embeddings.npy exist")
        return
    
    analyzer.query_embeddings = np.load(query_emb_path)
    analyzer.chunk_embeddings = np.load(chunk_emb_path)
    
    # Compute curvatures
    analyzer.compute_all_curvatures()
    
    # Run full analysis
    analyzer.run_full_analysis(
        results_file=args.results_file,
        output_dir=args.output_dir,
        method_name=args.method_name
    )


if __name__ == "__main__":
    main()
