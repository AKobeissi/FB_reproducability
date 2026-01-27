import json
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import logging

# Ensure this import works based on your project structure
# If running from root, this should work. 
# Otherwise, you can paste the RetrievalEvaluator class directly into this script.
from src.evaluation.retrieval_evaluator import RetrievalEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_sweep(input_path: str, output_dir: str):
    input_file = Path(input_path)
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return

    # 1. Load the Sweep JSON
    logger.info(f"Loading sweep results from {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Validate structure
    if "results" not in data or not isinstance(data["results"], dict):
        logger.error("Invalid JSON format. Expected 'results' key containing a dictionary of configurations.")
        return

    # 2. Initialize Evaluator
    evaluator = RetrievalEvaluator()
    
    sweep_metrics = {}
    csv_rows = []

    # 3. Iterate through each configuration in the sweep
    configs = data["results"]
    logger.info(f"Found {len(configs)} configurations to evaluate.")

    for config_name, samples in configs.items():
        logger.info(f"Evaluating config: {config_name} ({len(samples)} samples)")
        
        # Compute metrics for this specific configuration
        # The evaluator expects a list of samples, which matches your JSON structure per config
        metrics = evaluator.compute_metrics(samples)
        
        sweep_metrics[config_name] = metrics
        
        # Prepare row for CSV
        row = {"config": config_name}
        row.update(metrics)
        csv_rows.append(row)

    # 4. Save Results
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save detailed JSON
    json_out = out_path / f"{input_file.stem}_evaluated.json"
    with open(json_out, 'w') as f:
        json.dump(sweep_metrics, f, indent=2)
    logger.info(f"Saved JSON metrics to {json_out}")

    # Save CSV Summary (Best for comparing sweep parameters)
    if csv_rows:
        df = pd.DataFrame(csv_rows)
        
        # Sort by a key metric, e.g., MRR or Chunk Hit@5
        sort_col = "mrr" if "mrr" in df.columns else df.columns[1]
        df = df.sort_values(by=sort_col, ascending=False)
        
        csv_out = out_path / f"{input_file.stem}_summary.csv"
        df.to_csv(csv_out, index=False)
        logger.info(f"Saved CSV summary to {csv_out}")
        
        # Print top 3 configs
        print("\n--- Top 3 Configurations (sorted by MRR) ---")
        print(df[["config", "mrr", "chunk_hit@5", "chunk_recall@5"]].head(3).to_markdown(index=False))

def main():
    parser = argparse.ArgumentParser(description="Evaluate a Parameter Sweep JSON file.")
    parser.add_argument("input_file", help="Path to the input JSON file containing sweep results.")
    parser.add_argument("--output-dir", default="./evaluation_results", help="Directory to save output files.")
    
    args = parser.parse_args()
    
    evaluate_sweep(args.input_file, args.output_dir)

if __name__ == "__main__":
    main()