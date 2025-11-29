"""
Robust Data Loader for FinanceBench Dataset
"""
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datasets import load_dataset

logger = logging.getLogger(__name__)


class FinanceBenchLoader:
    """Loads and manages the FinanceBench dataset."""

    def __init__(self, dataset_name: str = "PatronusAI/financebench"):
        self.dataset_name = dataset_name
        self.data = None
        self.df: Optional[pd.DataFrame] = None

    def load_data(self, split: str = "train") -> pd.DataFrame:
        """
        Load FinanceBench dataset from HuggingFace.

        Args:
            split: Dataset split to load

        Returns:
            DataFrame with questions, answers, evidence, etc.
        """
        logger.info(f"Loading {self.dataset_name}, split: {split}")

        try:
            self.data = load_dataset(self.dataset_name, split=split)
            
            # Convert to DataFrame
            if hasattr(self.data, "to_pandas"):
                try:
                    self.df = self.data.to_pandas()
                except Exception:
                    self.df = pd.DataFrame(self.data)
            else:
                self.df = pd.DataFrame(self.data)

            if not isinstance(self.df, pd.DataFrame):
                self.df = pd.DataFrame(self.df)

            logger.info(f"Loaded {len(self.df)} examples")
            logger.info(f"Columns: {list(self.df.columns)}")

            self._log_dataset_statistics()
            
            return self.df

        except Exception as e:
            logger.error(f"Failed to load dataset: {e}", exc_info=True)
            raise

    def _log_dataset_statistics(self):
        """Log comprehensive dataset statistics."""
        if self.df is None:
            logger.warning("No data loaded")
            return

        logger.info("=" * 80)
        logger.info("DATASET STATISTICS")
        logger.info("=" * 80)
        logger.info(f"Total examples: {len(self.df)}")
        logger.info(f"Columns: {list(self.df.columns)}")

        # Check required columns
        required = ["question", "answer", "evidence", "doc_name"]
        for col in required:
            status = "✓" if col in self.df.columns else "✗"
            logger.info(f"{status} Column '{col}'")

        # Length statistics
        self._log_text_lengths("question")
        self._log_text_lengths("answer")
        self._log_evidence_lengths()
        self._log_missing_values()
        
        logger.info("=" * 80)

    def _log_text_lengths(self, col: str):
        """Log length statistics for a text column."""
        if col not in self.df.columns:
            return
            
        try:
            lengths = self.df[col].astype(str).str.len()
            logger.info(f"\n{col.capitalize()} lengths:")
            logger.info(f"  Mean: {lengths.mean():.1f}")
            logger.info(f"  Min: {lengths.min()}")
            logger.info(f"  Max: {lengths.max()}")
            logger.info(f"  Median: {lengths.median():.1f}")
        except Exception as e:
            logger.debug(f"Failed to compute {col} lengths: {e}")

    def _log_evidence_lengths(self):
        """Log evidence length statistics with safe handling."""
        if "evidence" not in self.df.columns:
            return
            
        try:
            lengths = self.df["evidence"].apply(_safe_evidence_len)
            logger.info("\nEvidence lengths:")
            logger.info(f"  Mean: {lengths.mean():.1f}")
            logger.info(f"  Min: {lengths.min()}")
            logger.info(f"  Max: {lengths.max()}")
            logger.info(f"  Median: {lengths.median():.1f}")
        except Exception as e:
            logger.debug(f"Failed to compute evidence lengths: {e}")

    def _log_missing_values(self):
        """Log missing value statistics."""
        try:
            logger.info("\nMissing values:")
            for col in self.df.columns:
                missing = self.df[col].isna().sum()
                if missing > 0:
                    pct = missing / len(self.df) * 100
                    logger.info(f"  {col}: {missing} ({pct:.1f}%)")
        except Exception as e:
            logger.debug(f"Failed to compute missing values: {e}")

    def get_sample(self, index: int = 0) -> Dict[str, Any]:
        """Get a single sample."""
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first")
        
        if index >= len(self.df):
            raise IndexError(
                f"Index {index} out of range. Dataset has {len(self.df)} examples"
            )

        sample = self.df.iloc[index].to_dict()
        
        logger.info(f"\nSample {index}:")
        logger.info(f"  Question: {str(sample.get('question',''))[:100]}...")
        logger.info(f"  Answer: {str(sample.get('answer',''))[:100]}...")
        logger.info(f"  Evidence: {_safe_evidence_len(sample.get('evidence',''))} chars")

        return sample

    def get_batch(
        self,
        indices: Optional[List[int]] = None,
        start: int = 0,
        end: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get a batch of samples."""
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first")

        if indices is not None:
            batch_df = self.df.iloc[indices]
        else:
            end = end or len(self.df)
            batch_df = self.df.iloc[start:end]

        logger.info(f"Retrieved batch of {len(batch_df)} examples")
        return batch_df.to_dict('records')

    def get_all_data(self) -> pd.DataFrame:
        """Get the entire dataset."""
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first")
        return self.df

    def filter_by_doc(self, doc_name: str) -> pd.DataFrame:
        """Filter dataset by document name."""
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first")

        if 'doc_name' not in self.df.columns:
            logger.warning("'doc_name' column not found")
            return pd.DataFrame()

        filtered = self.df[self.df['doc_name'] == doc_name]
        logger.info(f"Filtered to {len(filtered)} examples for: {doc_name}")
        
        return filtered


def _safe_evidence_len(x) -> int:
    """
    Safely compute evidence length handling various types.
    
    Avoids "truth value of array is ambiguous" error.
    """
    if x is None:
        return 0
    
    # Handle list/tuple/array types
    if isinstance(x, (list, tuple, np.ndarray)):
        return len(str(x))
    
    # Handle pandas NA / NaN
    if pd.isna(x):
        return 0
    
    # Default: convert to string
    try:
        return len(str(x))
    except Exception:
        return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    loader = FinanceBenchLoader()
    df = loader.load_data()
    sample = loader.get_sample(0)
    print("\nData loader test complete")