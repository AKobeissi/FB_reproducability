"""
RAG Experiments Main Module
Uses LangChain, Chroma, and HuggingFace models (Llama 3.2 3B, Qwen 2.5 7B)
Supports multiple experiment types: closed-book, single vector store, 
shared vector store, and open-book (evidence)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import torch
import argparse
import importlib.util

# Mandatory Imports - Fail fast if missing
import bitsandbytes as bnb
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings as LangchainOpenAIEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Local imports
from rag_dependencies import (
    RecursiveCharacterTextSplitter,
    HuggingFaceEmbeddings,
    HuggingFacePipeline,
)
from rag_experiment_mixins import (
    ComponentTrackingMixin,
    ChunkAndEvidenceMixin,
    PromptMixin,
    VectorstoreMixin,
    ResultsMixin,
)
from evaluate_outputs import run_scoring
from retrieval_evaluator import RetrievalEvaluator
from generative_evaluator import GenerativeEvaluator
from data_loader import FinanceBenchLoader

# Modular Runners
from rag_closed_book import run_closed_book as _run_closed_book
from rag_single_vector import run_single_vector as _run_single_vector
from rag_shared_vector import run_shared_vector as _run_shared_vector
from random_single_store import run_random_single_store as _run_random_single_store
from rag_open_book import run_open_book as _run_open_book


# Set up logging
def setup_logging(experiment_name: str, log_dir: Optional[str] = None):
    """Setup comprehensive logging"""
    if log_dir is None:
        base_dir = Path(__file__).resolve().parent
        log_dir = str(base_dir / "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/{experiment_name}_{timestamp}.log"
    
    # File handler with detailed formatting
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler with simpler formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Root logger - avoid adding duplicate handlers if already configured
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    if not root_logger.handlers:
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
    
    return log_file


class RAGExperiment(
    ComponentTrackingMixin,
    ChunkAndEvidenceMixin,
    PromptMixin,
    VectorstoreMixin,
    ResultsMixin,
):
    """Main RAG Experiment Runner using LangChain, FAISS, and HuggingFace LLMs"""
    
    # Experiment types
    CLOSED_BOOK = "closed_book"
    SINGLE_VECTOR = "single_vector"
    SHARED_VECTOR = "shared_vector"
    OPEN_BOOK = "open_book"
    RANDOM_SINGLE = "random_single"
    
    # Available LLMs
    LLAMA_3_2_3B = "meta-llama/Llama-3.2-3B-Instruct"
    QWEN_2_5_7B = "Qwen/Qwen2.5-7B-Instruct"
    
    OPENAI_EMBEDDING_PREFIXES = (
        "text-embedding",
        "text-search",
        "text-similarity",
        "text-moderation",
    )

    def __init__(self, 
                 experiment_type: str = CLOSED_BOOK,
                 llm_model: str = LLAMA_3_2_3B,
                 # FinanceBench-style defaults:
                 chunk_size: int = 1024,
                 chunk_overlap: int = 30,
                 top_k: int = 5,
                 embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
                 output_dir: Optional[str] = None,
                 vector_store_dir: Optional[str] = None,
                 pdf_local_dir: Optional[str] = None,
                 device: str = None,
                 load_in_8bit: bool = True,
                 max_new_tokens: int = 256,
                 use_api: bool = False,
                 api_base_url: str = "https://api.openai.com/v1",
                 api_key_env: str = "HF_TOKEN",
                 use_all_pdfs: bool = False,
                 eval_type: str = "both",
                 eval_mode: str = "static",
                 judge_model: str = "openai/gpt-4o"):
        """
        Initialize RAG Experiment
        """
        self.experiment_type = experiment_type
        self.llm_model_name = llm_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.embedding_model = embedding_model
        self.use_all_pdfs = use_all_pdfs
        
        # Evaluation config
        self.eval_type = eval_type
        self.eval_mode = eval_mode
        self.judge_model = judge_model
        
        base_dir = Path(__file__).resolve().parent
        if output_dir is None:
            output_root = base_dir / "outputs"
        else:
            output_root = Path(output_dir)
            
        if vector_store_dir is None:
            vector_store_dir = str(base_dir / "vector_stores")
            
        # Organize output directories: outputs/experiment_type/YYYYMMDD
        today_str = datetime.now().strftime("%Y%m%d")
        self.output_dir = str((output_root / experiment_type / today_str).resolve())
        
        # Organize results directories: results/experiment_type/YYYYMMDD
        # If output_root is named "outputs", result_root will be sibling "results"
        if output_root.name == "outputs":
            result_root = output_root.parent / "results"
        else:
            # Fallback: create results sibling to whatever output root was given
            result_root = output_root.parent / "results"
            
        self.results_dir = str((result_root / experiment_type / today_str).resolve())
        self.vector_store_dir = str(Path(vector_store_dir).resolve())
        
        # Local PDF directory: prefer this for PDF extraction if available
        # Default to the package-local `pdfs` directory so that
        # uploaded PDFs inside the package are preferred.
        if pdf_local_dir is None:
            # default to pdfs (inside the package)
            self.pdf_local_dir = Path(base_dir) / "pdfs"
        else:
            self.pdf_local_dir = Path(pdf_local_dir)
        self.load_in_8bit = load_in_8bit
        self.use_api = use_api
        self.api_base_url = api_base_url
        self.api_key_env = api_key_env
        self.api_client = None
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Initialize components
        self.logger = logging.getLogger(self.__class__.__name__)
        self.data_loader = FinanceBenchLoader()
        # LLM components (lazy loading)
        self.llm_tokenizer = None
        self.llm_model = None
        self.llm_pipeline = None
        
        # Initialize LangChain components
        self.embeddings = None
        self.text_splitter = None
        self.vector_stores = {}
        self.langchain_llm = None
        self.max_context_chars = 12000
        # Persist generation config on the instance
        self.max_new_tokens = max_new_tokens
        # Batch size for generation calls to the pipeline (helps GPU efficiency)
        self.generation_batch_size = 8

        # Results storage
        self.results = []
        self.component_usage: Dict[str, Dict[str, Any]] = {}
        self.progress_total: int = 0
        self.progress_completed: int = 0
        self.experiment_metadata = {
            'experiment_type': experiment_type,
            'llm_model': llm_model,
            'use_api': use_api,
            'api_base_url': api_base_url if use_api else None,
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap,
            'top_k': top_k,
            'embedding_model': embedding_model,
            'device': self.device,
            'load_in_8bit': load_in_8bit,
            'max_new_tokens': max_new_tokens,
            'pdf_local_dir': str(self.pdf_local_dir) if self.pdf_local_dir is not None else None,
            'use_all_pdfs': use_all_pdfs,
            'timestamp': datetime.now().isoformat(),
            'eval_type': eval_type,
            'eval_mode': eval_mode,
            'judge_model': judge_model
        }

        # Ensure output and vector store directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.vector_store_dir, exist_ok=True)
        
        self.logger.info("=" * 80)
        self.logger.info(f"INITIALIZING RAG EXPERIMENT: {experiment_type.upper()}")
        self.logger.info("=" * 80)
        self.logger.info("Configuration:")
        self.logger.info(f"  Experiment Type: {experiment_type}")
        self.logger.info(f"  LLM Model: {llm_model}")
        self.logger.info(f"  Device: {self.device}")
        self.logger.info(f"  8-bit Loading: {load_in_8bit}")
        self.logger.info(f"  Chunk Size: {chunk_size}")
        self.logger.info(f"  Chunk Overlap: {chunk_overlap}")
        self.logger.info(f"  Top-K Retrieval: {top_k}")
        self.logger.info(f"  Embedding Model: {embedding_model}")
        self.logger.info(f"  Max New Tokens: {max_new_tokens}")
        self.logger.info(f"  Use All PDFs: {use_all_pdfs}")
        self.logger.info(f"  Evaluation Type: {eval_type}")
        self.logger.info(f"  Evaluation Mode: {eval_mode}")
        if eval_mode == "semantic":
            self.logger.info(f"  Judge Model: {judge_model}")
        self.logger.info(f"  Output Dir: {self.output_dir}")
        self.logger.info(f"  Results Dir: {self.results_dir}")
        self.logger.info("  Using LangChain + FAISS + HuggingFace LLMs")
        if self.use_api:
            self.logger.info("  Using API-based LLM via OpenAI client (HF router)")
        self.logger.info("=" * 80)
        
        # Initialize LangChain components
        self._initialize_components()
        if self.use_api:
            # Ensure API client is ready up front so summaries include generator info
            self._initialize_llm()
        self._print_component_overview(stage="initial")

    
    def _initialize_components(self):
        """Initialize LangChain embeddings and text splitter"""
        self.logger.info("\nInitializing LangChain components...")
        
        self.embeddings = self._build_embeddings()
        self.register_component_usage(
            "embeddings",
            self.embeddings.__class__.__name__,
            {
                "package": self.embeddings.__class__.__module__,
                "model_name": self.embedding_model
            }
        )
        self.logger.info(f"✓ Embeddings loaded ({self.embeddings.__class__.__name__})")
        
        # Initialize text splitter using LangChain
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.register_component_usage(
            "chunker",
            self.text_splitter.__class__.__name__,
            {
                "package": self.text_splitter.__class__.__module__,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap
            }
        )
        self.logger.info(f"✓ Text splitter initialized (chunk_size={self.chunk_size}, overlap={self.chunk_overlap})")

    def _build_embeddings(self):
        """Create FinanceBench-style embeddings using HuggingFace or OpenAI models."""
        if self._is_openai_embedding_model(self.embedding_model):
            return self._build_openai_embeddings()
        return self._build_hf_embeddings()

    def _is_openai_embedding_model(self, model_name: str) -> bool:
        if not model_name:
            return False
        normalized = model_name.lower()
        return normalized.startswith(self.OPENAI_EMBEDDING_PREFIXES)

    def _build_openai_embeddings(self):
        api_key = os.environ.get(self.api_key_env) or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                f"Cannot initialize OpenAI embeddings ({self.embedding_model}). "
                f"Set the API key in '{self.api_key_env}' or OPENAI_API_KEY."
            )
        self.logger.info(f"Loading OpenAI embeddings: {self.embedding_model}")
        return LangchainOpenAIEmbeddings(
            model=self.embedding_model,
            api_key=api_key,
            base_url=self.api_base_url,
        )

    def _build_hf_embeddings(self):
        self.logger.info(f"Loading HuggingFace embeddings: {self.embedding_model}")
        return HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={'device': self.device},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def _initialize_llm(self):
        """Initialize HuggingFace LLM for generation"""
        # If configured to use API, initialize the API client and skip local model
        if self.use_api:
            if self.api_client is not None:
                return

            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                self.logger.error(f"API key environment variable '{self.api_key_env}' not set")
                raise RuntimeError(f"Missing API key in env var {self.api_key_env}")

            try:
                self.logger.info("Initializing OpenAI API client (HF router)...")
                self.api_client = OpenAI(base_url=self.api_base_url, api_key=api_key)
                self.logger.info("✓ API client initialized")
                self.register_component_usage(
                    "generator",
                    f"OpenAI Chat Completions ({self.llm_model_name})",
                    {
                        "base_url": self.api_base_url,
                        "key_env": self.api_key_env
                    }
                )
                return
            except Exception as e:
                self.logger.error(f"Failed to initialize API client: {e}")
                raise

        if self.llm_pipeline is not None:
            return  # Already initialized
        
        self.logger.info(f"\nInitializing LLM: {self.llm_model_name}")
        self.logger.info(f"  Device: {self.device}")
        self.logger.info(f"  8-bit: {self.load_in_8bit}")
        
        try:
            # Load tokenizer
            self.logger.info("Loading tokenizer...")
            self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
            
            # Load model
            self.logger.info("Loading model (this may take a moment)...")
            model_kwargs = {
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "device_map": "auto" if self.device == "cuda" else None,
            }

            if self.load_in_8bit and self.device == "cuda":
                # bitsandbytes must be available since we imported it
                model_kwargs["load_in_8bit"] = True
                self.logger.info("  Using 8-bit quantization for memory efficiency")
            
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                self.llm_model_name,
                **model_kwargs
            )
            
            # Create pipeline
            self.llm_pipeline = pipeline(
                "text-generation",
                model=self.llm_model,
                tokenizer=self.llm_tokenizer,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                return_full_text=False
            )
            
            self.logger.info("✓ LLM initialized successfully")
            # If LangChain's HuggingFacePipeline wrapper is available, create a
            # LangChain LLM wrapper so RetrievalQA can be used directly.
            self.langchain_llm = HuggingFacePipeline(pipeline=self.llm_pipeline)
            self.register_component_usage(
                "generator",
                f"HuggingFacePipeline ({self.llm_model_name})",
                {
                    "package": self.llm_pipeline.__class__.__module__,
                    "load_in_8bit": self.load_in_8bit,
                    "device": self.device
                }
            )
            self.logger.info("✓ Created LangChain HuggingFacePipeline wrapper for RetrievalQA")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {str(e)}")
            self.logger.error("Make sure you have sufficient GPU memory or try load_in_8bit=True")
            raise

    def ensure_langchain_llm(self):
        """Ensure a LangChain-compatible LLM wrapper exists."""
        if getattr(self, 'langchain_llm', None) is None:
            self._initialize_llm()

    
    def run_closed_book(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return _run_closed_book(self, data)
    
    def run_single_vector(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return _run_single_vector(self, data)
    
    def run_random_single(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return _run_random_single_store(self, data)
    
    def run_shared_vector(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return _run_shared_vector(self, data)
    
    def run_open_book(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return _run_open_book(self, data)
    
    def run_experiment(self, num_samples: int = None, sample_indices: List[int] = None):
        """
        Run the configured experiment
        
        Args:
            num_samples: Number of samples to process (None for all)
            sample_indices: Specific sample indices to process
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STARTING EXPERIMENT")
        self.logger.info("=" * 80)
        
        # Load data
        self.data_loader.load_data()
        
        # Select samples
        if sample_indices is not None:
            data = self.data_loader.get_batch(indices=sample_indices)
        elif num_samples is not None:
            data = self.data_loader.get_batch(start=0, end=num_samples)
        else:
            data = self.data_loader.get_batch()
        
        self.logger.info(f"Processing {len(data)} samples")
        self._set_progress_total(len(data))
        
        # Run appropriate experiment type
        if self.experiment_type == self.CLOSED_BOOK:
            results = self.run_closed_book(data)
        elif self.experiment_type == self.SINGLE_VECTOR:
            results = self.run_single_vector(data)
        elif self.experiment_type == self.RANDOM_SINGLE:
            results = self.run_random_single(data)
        elif self.experiment_type == self.SHARED_VECTOR:
            results = self.run_shared_vector(data)
        elif self.experiment_type == self.OPEN_BOOK:
            results = self.run_open_book(data)
        else:
            raise ValueError(f"Unknown experiment type: {self.experiment_type}")
        
        self.results = results
        
        # Compute aggregate statistics
        self._compute_aggregate_stats()
        
        # Save results
        output_file = self._save_results()
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("EXPERIMENT COMPLETE")
        self.logger.info("=" * 80)
        self._print_component_overview(stage="final")
        
        # Run automated evaluation
        if self.eval_type:
            self.run_evaluation(output_file)

    def _unload_model(self):
        """Unload generator model to free memory."""
        if self.llm_model is not None:
            self.logger.info("Unloading generator model to free memory for evaluation...")
            del self.llm_model
            del self.llm_pipeline
            if self.langchain_llm is not None:
                del self.langchain_llm
            
            self.llm_model = None
            self.llm_pipeline = None
            self.langchain_llm = None
            
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.logger.info("✓ Model unloaded")

    def run_evaluation(self, output_file: str):
        """Run automated evaluation on the experiment output using RetrievalEvaluator and GenerativeEvaluator."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STARTING AUTOMATED EVALUATION")
        self.logger.info("=" * 80)
        self.logger.info(f"Input file: {output_file}")
        self.logger.info(f"Results dir: {self.results_dir}")
        self.logger.info(f"Eval Type: {self.eval_type}")
        
        # Load results
        try:
            with open(output_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load output file: {e}")
            return

        if isinstance(data, dict) and "results" in data:
            samples = data["results"]
            metadata = data.get("metadata", {})
        else:
            samples = data
            metadata = {}

        # Free up memory
        self._unload_model()
        
        # 1. Retrieval Evaluation
        if self.eval_type in ["retrieval", "both"]:
            self.logger.info("Running Retrieval Evaluation...")
            try:
                ret_evaluator = RetrievalEvaluator()
                ret_metrics = ret_evaluator.compute_metrics(samples, k_values=[1, 3, 5, self.top_k])
                
                # Add summary to metadata/container
                if "evaluation_summary" not in data:
                    data["evaluation_summary"] = {}
                data["evaluation_summary"]["retrieval"] = ret_metrics
                
                self.logger.info("Retrieval Evaluation Complete.")
                self.logger.info(f"MRR: {ret_metrics.get('mrr', 0):.4f}")
            except Exception as e:
                self.logger.error(f"Retrieval evaluation failed: {e}", exc_info=True)

        # 2. Generative Evaluation
        if self.eval_type in ["generative", "both"]:
            self.logger.info("Running Generative Evaluation...")
            try:
                use_llm_judge = (self.eval_mode == "semantic")
                use_ragas = (self.eval_mode == "semantic")
                
                # Configure Judge Pipeline if needed
                judge_pipeline = None
                if use_llm_judge:
                    # We need to re-initialize a pipeline for the judge if it's a local model
                    # Check if judge_model is HF or OpenAI
                    judge_provider = "huggingface"
                    if "gpt" in self.judge_model.lower() or "openai" in self.judge_model.lower():
                        judge_provider = "openai"
                    
                    if judge_provider == "huggingface":
                         # Re-load model for judge if it's different or if we unloaded
                         self.logger.info(f"Loading Judge Model: {self.judge_model}")
                         try:
                            tokenizer = AutoTokenizer.from_pretrained(self.judge_model)
                            model = AutoModelForCausalLM.from_pretrained(
                                self.judge_model,
                                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                                device_map="auto" if self.device == "cuda" else None
                            )
                            judge_pipeline = pipeline(
                                "text-generation",
                                model=model,
                                tokenizer=tokenizer,
                                max_new_tokens=200,
                            )
                         except Exception as e:
                             self.logger.warning(f"Failed to load local judge model: {e}")
                    elif judge_provider == "openai" and self.use_api and self.api_client:
                         # We can use the existing API client or create a new one for the judge
                         # For now, GenerativeEvaluator expects a callable pipeline or similar.
                         # We can pass a wrapper.
                         pass # TODO: Implement OpenAI wrapper for GenerativeEvaluator if needed
                
                gen_evaluator = GenerativeEvaluator(
                    use_bertscore=(self.eval_mode == "static" or self.eval_mode == "semantic"),
                    use_llm_judge=use_llm_judge,
                    use_ragas=use_ragas,
                    judge_pipeline=judge_pipeline
                )
                
                evaluated_samples = []
                for sample in samples:
                    # RAGAS needs LangChain LLM. If we are in semantic mode, we might need it.
                    # self.langchain_llm might be None if we unloaded.
                    # For now, pass None.
                    metrics = gen_evaluator.evaluate_sample(sample, langchain_llm=None)
                    sample["generative_metrics"] = metrics
                    evaluated_samples.append(sample)
                
                samples = evaluated_samples
                
                # Compute averages for summary
                agg_gen = {}
                metric_keys = set()
                for s in samples:
                    metric_keys.update(s.get("generative_metrics", {}).keys())
                
                for k in metric_keys:
                    vals = [s["generative_metrics"][k] for s in samples if s["generative_metrics"].get(k) is not None]
                    if vals:
                         if all(isinstance(v, (int, float, bool)) for v in vals):
                            agg_gen[f"avg_{k}"] = sum(vals) / len(vals)
                
                if "evaluation_summary" not in data:
                    data["evaluation_summary"] = {}
                data["evaluation_summary"]["generative"] = agg_gen
                
                self.logger.info("Generative Evaluation Complete.")
            except Exception as e:
                self.logger.error(f"Generative evaluation failed: {e}", exc_info=True)

        # Save results to results_dir
        if "results" in data:
            data["results"] = samples
        else:
            data = samples # If it was a list

        # Create output path
        source_path = Path(output_file)
        out_filename = f"{source_path.stem}_scored{source_path.suffix}"
        out_path = Path(self.results_dir) / out_filename
        
        try:
            with open(out_path, 'w') as f:
                json.dump(data, f, indent=2)
            self.logger.info(f"Evaluation results saved to: {out_path}")
        except Exception as e:
            self.logger.error(f"Failed to save evaluation results: {e}")

    
def main():
    """
    CLI entry point for FinanceBench-style experiments with local Llama / Qwen.

    Examples:
        # Single-vector (singleStore) with local Llama 3.2 3B
        python rag_experiments.py llama3binstruct -e single

        # Shared-vector (sharedStore) with local Qwen 2.5 7B
        python rag_experiments.py qwen -e shared

        # Closed-book with both models (runs Llama then Qwen)
        python rag_experiments.py both -e closed -n 50
    """
    parser = argparse.ArgumentParser(
        description="FinanceBench-style RAG experiments with local Llama / Qwen models."
    )

    # Positional: which generator
    parser.add_argument(
        "llm",
        nargs="?",
        default="llama3binstruct",
        help=(
            "Which generator to use.\n"
            "Short names: 'llama3binstruct'/'llama' or 'qwen'.\n"
            "Use 'both' to run both local models, or pass a full HF model id."
        ),
    )

    # Experiment type (maps to FinanceBench eval modes)
    parser.add_argument(
        "-e",
        "--experiment",
        choices=["closed", "single", "random_single", "shared", "open"],
        default="single",
        help=(
            "Experiment type:\n"
            "  closed  = closed-book (no context)\n"
            "  single  = single-vector store per document (singleStore)\n"
            "  random_single = single store with random chunk selection baseline\n"
            "  shared  = shared vector store across docs (sharedStore)\n"
            "  open    = oracle / evidence open-book"
        ),
    )

    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (None = all).",
    )

    # Default paths should live under package directory so
    # running from different working directories still reads/writes inside
    # the package instead of the repository root.
    base_dir = Path(__file__).resolve().parent
    default_pdf_dir = str(base_dir / "pdfs")
    default_vector_store_dir = str(base_dir / "vector_stores")
    default_output_dir = str(base_dir / "outputs")

    parser.add_argument(
        "--pdf-dir",
        type=str,
        default=default_pdf_dir,
        help="Local directory containing FinanceBench PDFs (defaults to pdfs)",
    )

    parser.add_argument(
        "--vector-store-dir",
        type=str,
        default=default_vector_store_dir,
        help="Directory where vector indices will be created (defaults to vector_stores).",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=default_output_dir,
        help="Directory where JSON result files will be saved (defaults to outputs).",
    )

    # FinanceBench-style retrieval defaults, exposed as flags
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
        help="Chunk size for retrieval (FinanceBench default 1024).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=30,
        help="Chunk overlap for retrieval (FinanceBench default 30).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieved chunks per question.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="Embedding model identifier (HF sentence transformers or OpenAI text-embedding-*).",
    )

    parser.add_argument(
        "--use-api",
        action="store_true",
        help="Use HF Router / OpenAI API instead of local weights.",
    )
    parser.add_argument(
        "--api-base-url",
        type=str,
        default="https://api.openai.com/v1",
        help="Base URL for the OpenAI-compatible API endpoint (default: api.openai.com).",
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable containing the API key for --use-api.",
    )
    parser.add_argument(
        "--no-8bit",
        action="store_true",
        help="Disable 8-bit loading (load full-precision model).",
    )
    parser.add_argument(
        "--use-all-pdfs",
        action="store_true",
        help="Index all PDFs in the pdf-dir, not just those referenced in the dataset (for shared vector store).",
    )

    args = parser.parse_args()

    # Map experiment name to RAGExperiment constants
    exp_map = {
        "closed": RAGExperiment.CLOSED_BOOK,
        "single": RAGExperiment.SINGLE_VECTOR,
        "random_single": RAGExperiment.RANDOM_SINGLE,
        "shared": RAGExperiment.SHARED_VECTOR,
        "open": RAGExperiment.OPEN_BOOK,
    }
    experiment_type = exp_map[args.experiment]

    # Map LLM shorthand to HF model ids
    llm_arg = args.llm.lower()
    if llm_arg in {"llama", "llama3", "llama3b", "llama3binstruct"}:
        models_to_run = [("llama3binstruct", RAGExperiment.LLAMA_3_2_3B)]
    elif llm_arg in {"qwen", "qwen2", "qwen2.5", "qwen2.5-7b"}:
        models_to_run = [("qwen2.5-7b", RAGExperiment.QWEN_2_5_7B)]
    elif llm_arg in {"both", "all"}:
        models_to_run = [
            ("llama3binstruct", RAGExperiment.LLAMA_3_2_3B),
            ("qwen2.5-7b", RAGExperiment.QWEN_2_5_7B),
        ]
    else:
        # Treat as a full HF model id
        models_to_run = [(args.llm, args.llm)]

    for label, model_name in models_to_run:
        print("\n" + "=" * 80)
        print(f"Running experiment: {args.experiment} | model={label}")
        print("=" * 80)

        experiment = RAGExperiment(
            experiment_type=experiment_type,
            llm_model=model_name,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            top_k=args.top_k,
            embedding_model=args.embedding_model,
            output_dir=args.output_dir,
            vector_store_dir=args.vector_store_dir,
            pdf_local_dir=args.pdf_dir,
            load_in_8bit=not args.no_8bit,
            use_api=args.use_api,
            api_base_url=args.api_base_url,
            api_key_env=args.api_key_env,
            use_all_pdfs=args.use_all_pdfs,
        )

        experiment.run_experiment(num_samples=args.num_samples)


if __name__ == "__main__":
    main()