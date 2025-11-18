"""
RAG Experiments Main Module
Uses LangChain, Chroma, and HuggingFace models (Llama 3.2 3B, Qwen 2.5 7B)
Supports multiple experiment types: closed-book, single vector store, 
shared vector store, and open-book (evidence)
"""

from __future__ import annotations

import logging
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import numpy as np
from collections import defaultdict
import torch
import argparse

# bitsandbytes availability check (used for 8-bit quantization)
try:
    import bitsandbytes as bnb  # noqa: F401
    _BNB_AVAILABLE = True
except Exception:
    _BNB_AVAILABLE = False

# Optional OpenAI client (for HF router / nference style API)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# LangChain imports (try multiple possible packages / entrypoints and provide safe fallbacks)
_HAS_LANGCHAIN = False
from dataclasses import dataclass
Document = None
RecursiveCharacterTextSplitter = None
HuggingFaceEmbeddings = None
FAISS = None
BaseRetriever = None
Chroma = None
RetrievalQA = None
HuggingFacePipeline = None

_logger = logging.getLogger(__name__)

# 1) Text splitter: try langchain, then langchain_text_splitters
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    _HAS_LANGCHAIN = True
except Exception:
    try:
        # some installs expose splitters via this package
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        _HAS_LANGCHAIN = True
    except Exception:
        RecursiveCharacterTextSplitter = None

# 2) Embeddings: try langchain_community, then langchain.embeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    # from langchain_community.embeddings import HuggingFaceEmbeddings
    _HAS_LANGCHAIN = True
except Exception:
    try:
        from langchain.embeddings import HuggingFaceEmbeddings
        _HAS_LANGCHAIN = True
    except Exception:
        HuggingFaceEmbeddings = None

# 3) FAISS vectorstore: try langchain_community.vectorstores then langchain.vectorstores
try:
    from langchain_community.vectorstores import FAISS
    _HAS_LANGCHAIN = True
except Exception:
    try:
        from langchain.vectorstores import FAISS
        _HAS_LANGCHAIN = True
    except Exception:
        FAISS = None
# Try to import Chroma vector store if available
try:
    from langchain.vectorstores import Chroma
    Chroma = Chroma
    _HAS_LANGCHAIN = True
except Exception:
    try:
        from langchain_community.vectorstores import Chroma
        Chroma = Chroma
        _HAS_LANGCHAIN = True
    except Exception:
        Chroma = None

# 4) Document + BaseRetriever
try:
    from langchain.docstore.document import Document
    from langchain.schema import BaseRetriever
    _HAS_LANGCHAIN = True
except Exception:
    # If not available, we'll provide a tiny Document fallback below
    Document = None
    BaseRetriever = None
# RetrievalQA and HuggingFacePipeline (optional)
try:
    from langchain.chains import RetrievalQA
    RetrievalQA = RetrievalQA
    _HAS_LANGCHAIN = True
except Exception:
    RetrievalQA = None

try:
    from langchain.llms import HuggingFacePipeline
    HuggingFacePipeline = HuggingFacePipeline
    _HAS_LANGCHAIN = True
except Exception:
    HuggingFacePipeline = None

if not _HAS_LANGCHAIN:
    _logger.warning("langchain or langchain_community not available (or some subpackages missing); using minimal fallbacks. Install 'langchain' and 'langchain-community' for full functionality.")

# Vectorstore helpers (modularized)
try:
    from vectorstore import build_chroma_store, create_faiss_store, retrieve_faiss_chunks
except Exception:
    build_chroma_store = None
    create_faiss_store = None
    retrieve_faiss_chunks = None

    @dataclass
    class Document:
        page_content: str
        metadata: dict = None

    class _MinimalTextSplitter:
        """Minimal fallback for LangChain's RecursiveCharacterTextSplitter.

        Only implements create_documents(texts, metadatas) used in this repo.
        """
        def __init__(self, chunk_size=512, chunk_overlap=50, length_function=len, separators=None):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.length_function = length_function

        def _chunk_text(self, text: str) -> List[str]:
            if not text:
                return []
            chunks = []
            i = 0
            step = self.chunk_size - self.chunk_overlap if self.chunk_size > self.chunk_overlap else self.chunk_size
            while i < len(text):
                chunks.append(text[i:i + self.chunk_size])
                i += step
            return chunks

        def create_documents(self, texts: List[str], metadatas: List[dict]):
            docs = []
            for text, meta in zip(texts, metadatas):
                for chunk in self._chunk_text(text):
                    docs.append(Document(page_content=chunk, metadata=meta or {}))
            return docs

    RecursiveCharacterTextSplitter = _MinimalTextSplitter

    # Provide a placeholder FAISS class that raises a helpful error if used
    if FAISS is None:
        class FAISS:
            @classmethod
            def from_documents(cls, *args, **kwargs):
                raise RuntimeError("FAISS vector store not available. Install 'langchain-community' and a faiss package (faiss-cpu or faiss-gpu) to use vector stores.")


# HuggingFace transformers for LLM
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Import local modules (support running as package or script)
try:
    # Prefer package-relative imports when running as a package (python -m FB_reproducability.runner)
    from .data_loader import FinanceBenchLoader
    from .evaluator import Evaluator
except Exception:
    try:
        # Fall back to top-level imports when running as a script from package directory
        from data_loader import FinanceBenchLoader
        from evaluator import Evaluator
    except Exception as e:
        logging.getLogger(__name__).exception(
            "Failed to import local modules data_loader/evaluator.\n"
            "Make sure you're running this script from the project root or install the package."
        )
        # As a last resort, try to dynamically load the modules by file path
        try:
            import importlib.util
            base_dir = Path(__file__).resolve().parent
            dl_path = base_dir / 'data_loader.py'
            ev_path = base_dir / 'evaluator.py'

            def load_module_from_path(path: Path, module_name: str):
                spec = importlib.util.spec_from_file_location(module_name, str(path))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore
                return mod

            dl_mod = load_module_from_path(dl_path, 'data_loader_local')
            ev_mod = load_module_from_path(ev_path, 'evaluator_local')

            FinanceBenchLoader = getattr(dl_mod, 'FinanceBenchLoader')
            Evaluator = getattr(ev_mod, 'Evaluator')
        except Exception as e2:
            logging.getLogger(__name__).exception(f"Dynamic import fallback failed: {e2}")
            raise

# Import modular experiment runners (keeps main file small and easier to maintain)
try:
    from .rag_closed_book import run_closed_book as _run_closed_book
    from .rag_single_vector import run_single_vector as _run_single_vector
    from .rag_shared_vector import run_shared_vector as _run_shared_vector
    from .rag_open_book import run_open_book as _run_open_book
except Exception:
    # If package-style import fails (e.g., running as script), try absolute imports
    try:
        from rag_closed_book import run_closed_book as _run_closed_book
        from rag_single_vector import run_single_vector as _run_single_vector
        from rag_shared_vector import run_shared_vector as _run_shared_vector
        from rag_open_book import run_open_book as _run_open_book
    except Exception:
        # As a last resort, try dynamic import by file path so the script can be
        # executed both as a package and as a standalone script from the project root.
        try:
            import importlib.util
            base_dir = Path(__file__).resolve().parent

            def _load_runner(path: Path, name: str):
                spec = importlib.util.spec_from_file_location(name, str(path))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore
                return mod

            _run_closed_book = _load_runner(base_dir / 'rag_closed_book.py', 'rag_closed_book').run_closed_book
            _run_single_vector = _load_runner(base_dir / 'rag_single_vector.py', 'rag_single_vector').run_single_vector
            _run_shared_vector = _load_runner(base_dir / 'rag_shared_vector.py', 'rag_shared_vector').run_shared_vector
            _run_open_book = _load_runner(base_dir / 'rag_open_book.py', 'rag_open_book').run_open_book
        except Exception:
            # If even dynamic loading fails, set to None and let callers raise clearer errors
            _run_closed_book = None
            _run_single_vector = None
            _run_shared_vector = None
            _run_open_book = None


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


class RAGExperiment:
    """Main RAG Experiment Runner using LangChain, FAISS, and HuggingFace LLMs"""
    
    # Experiment types
    CLOSED_BOOK = "closed_book"
    SINGLE_VECTOR = "single_vector"
    SHARED_VECTOR = "shared_vector"
    OPEN_BOOK = "open_book"
    
    # Available LLMs
    LLAMA_3_2_3B = "meta-llama/Llama-3.2-3B-Instruct"
    QWEN_2_5_7B = "Qwen/Qwen2.5-7B-Instruct"
    MISTRAL_7B = "mistralai/Mistral-7B-Instruct-v0.3"  # For judge
    
    def __init__(self, 
                 experiment_type: str = CLOSED_BOOK,
                 llm_model: str = LLAMA_3_2_3B,
                 judge_model: str = MISTRAL_7B,
                 # FinanceBench-style defaults:
                 chunk_size: int = 1024,
                 chunk_overlap: int = 30,
                 top_k: int = 5,
                 embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
                 use_bertscore: bool = False,
                 use_llm_judge: bool = False,
                 output_dir: Optional[str] = None,
                 vector_store_dir: Optional[str] = None,
                 pdf_local_dir: Optional[str] = None,
                 device: str = None,
                 load_in_8bit: bool = True,
                 max_new_tokens: int = 256,
                 use_api: bool = False,
                 api_base_url: str = "https://router.huggingface.co/v1",
                 api_key_env: str = "HF_TOKEN"):
        """
        Initialize RAG Experiment
        
        Args:
            experiment_type: Type of experiment (closed_book, single_vector, shared_vector, open_book)
            llm_model: HuggingFace model for generation (Llama 3.2 3B or Qwen 2.5 7B)
            judge_model: HuggingFace model for LLM-as-judge (Mistral 7B)
            chunk_size: Size of text chunks for retrieval
            chunk_overlap: Overlap between chunks
            top_k: Number of chunks to retrieve
            embedding_model: Model for embeddings
            use_bertscore: Whether to use BERTScore
            use_llm_judge: Whether to use LLM as judge
            output_dir: Directory for outputs
            output_dir: Directory for outputs (default: "outputs")
            vector_store_dir: Directory for vector store persistence (default: "vector_stores")
            load_in_8bit: Whether to load models in 8-bit for memory efficiency
            max_new_tokens: Maximum tokens to generate
        """
        self.experiment_type = experiment_type
        self.llm_model_name = llm_model
        self.judge_model_name = judge_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.embedding_model = embedding_model
        self.output_dir = output_dir
        base_dir = Path(__file__).resolve().parent
        if output_dir is None:
            output_dir = str(base_dir / "outputs")
        if vector_store_dir is None:
            vector_store_dir = str(base_dir / "vector_stores")
        self.output_dir = str(Path(output_dir).resolve())
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
        self.evaluator = Evaluator(use_bertscore=use_bertscore, use_llm_judge=use_llm_judge)
        
        # LLM components (lazy loading)
        self.llm_tokenizer = None
        self.llm_model = None
        self.llm_pipeline = None
        
        # Judge LLM components (lazy loading)
        self.judge_tokenizer = None
        self.judge_model = None
        self.judge_pipeline = None
        
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
        self.experiment_metadata = {
            'experiment_type': experiment_type,
            'llm_model': llm_model,
            'use_api': use_api,
            'api_base_url': api_base_url if use_api else None,
            'judge_model': judge_model if use_llm_judge else None,
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap,
            'top_k': top_k,
            'embedding_model': embedding_model,
            'device': self.device,
            'load_in_8bit': load_in_8bit,
            'max_new_tokens': max_new_tokens,
            'pdf_local_dir': str(self.pdf_local_dir) if self.pdf_local_dir is not None else None,
            'timestamp': datetime.now().isoformat()
        }

        # Ensure output and vector store directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.vector_store_dir, exist_ok=True)
        
        self.logger.info("=" * 80)
        self.logger.info(f"INITIALIZING RAG EXPERIMENT: {experiment_type.upper()}")
        self.logger.info("=" * 80)
        self.logger.info(f"Configuration:")
        self.logger.info(f"  Experiment Type: {experiment_type}")
        self.logger.info(f"  LLM Model: {llm_model}")
        self.logger.info(f"  Judge Model: {judge_model if use_llm_judge else 'Disabled'}")
        self.logger.info(f"  Device: {self.device}")
        self.logger.info(f"  8-bit Loading: {load_in_8bit}")
        self.logger.info(f"  Chunk Size: {chunk_size}")
        self.logger.info(f"  Chunk Overlap: {chunk_overlap}")
        self.logger.info(f"  Top-K Retrieval: {top_k}")
        self.logger.info(f"  Embedding Model: {embedding_model}")
        self.logger.info(f"  Max New Tokens: {max_new_tokens}")
        self.logger.info(f"  Using LangChain + FAISS + HuggingFace LLMs")
        if self.use_api:
            self.logger.info("  Using API-based LLM via OpenAI client (HF router)")
        self.logger.info("=" * 80)
        
        # Initialize LangChain components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize LangChain embeddings and text splitter"""
        self.logger.info("\nInitializing LangChain components...")
        
        self.embeddings = self._build_embeddings()
        self.logger.info(f"✓ Embeddings loaded ({self.embeddings.__class__.__name__})")
        
        # Initialize text splitter using LangChain
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.logger.info(f"✓ Text splitter initialized (chunk_size={self.chunk_size}, overlap={self.chunk_overlap})")

    def _build_embeddings(self):
        """Create FinanceBench-style embeddings using HuggingFace sentence transformers."""
        if HuggingFaceEmbeddings is None:
            raise RuntimeError(
                "langchain-huggingface is required for embeddings. Install it before running experiments."
            )

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

            if OpenAI is None:
                self.logger.error("OpenAI client library not installed. Install 'openai' package to use API mode.")
                raise RuntimeError("OpenAI client not available")

            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                self.logger.error(f"API key environment variable '{self.api_key_env}' not set")
                raise RuntimeError(f"Missing API key in env var {self.api_key_env}")

            try:
                self.logger.info("Initializing OpenAI API client (HF router)...")
                self.api_client = OpenAI(base_url=self.api_base_url, api_key=api_key)
                self.logger.info("✓ API client initialized")
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
                if not _BNB_AVAILABLE:
                    # bitsandbytes not available — fall back to full precision to avoid exception
                    self.logger.warning("bitsandbytes not available or outdated; falling back to full precision (disabling 8-bit).")
                    # Don't set load_in_8bit; continue with default dtype/device_map
                    self.load_in_8bit = False
                else:
                    # If available, request 8-bit (older transformers use load_in_8bit; if your
                    # transformers requires BitsAndBytesConfig, you can upgrade or set quantization_config)
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
            
            self.logger.info(f"✓ LLM initialized successfully")
            # If LangChain's HuggingFacePipeline wrapper is available, create a
            # LangChain LLM wrapper so RetrievalQA can be used directly.
            if HuggingFacePipeline is not None:
                try:
                    self.langchain_llm = HuggingFacePipeline(pipeline=self.llm_pipeline)
                    self.logger.info("✓ Created LangChain HuggingFacePipeline wrapper for RetrievalQA")
                except Exception as exc:
                    self.logger.error(f"Failed to wrap pipeline for LangChain: {exc}")
                    raise
            else:
                raise RuntimeError(
                    "HuggingFacePipeline is not available. Install 'langchain' to enable RetrievalQA."
                )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {str(e)}")
            self.logger.error("Make sure you have sufficient GPU memory or try load_in_8bit=True")
            raise

    def ensure_langchain_llm(self):
        """Ensure a LangChain-compatible LLM wrapper exists."""
        if getattr(self, 'langchain_llm', None) is None:
            self._initialize_llm()

    def _build_vectorstore_chroma(self, docs, embeddings=None):
        """
        Build or load a Chroma vector store similar to the evaluation_playground notebook.

        Args:
            docs: 'all' or a single document name (str) or list of doc names
            embeddings: LangChain-compatible embedding function (optional)

        Returns:
            retriever, vectordb
        """
        # Delegate to vectorstore helper (Chroma preferred)
        if build_chroma_store is None:
            raise RuntimeError("Chroma vectorstore helper not available; ensure vectorstore.py is present and importable.")
        try:
            return build_chroma_store(self, docs, embeddings=embeddings)
        except Exception as e:
            self.logger.error(f"_build_vectorstore_chroma delegated to vectorstore failed: {e}")
            raise
    
    def _initialize_judge_llm(self):
        """Initialize HuggingFace LLM for judging (if enabled)"""
        if not self.evaluator.use_llm_judge:
            return
        
        if self.judge_pipeline is not None:
            return  # Already initialized
        
        self.logger.info(f"\nInitializing Judge LLM: {self.judge_model_name}")
        
        try:
            # Load tokenizer
            self.judge_tokenizer = AutoTokenizer.from_pretrained(self.judge_model_name)
            
            # Load model
            model_kwargs = {
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "device_map": "auto" if self.device == "cuda" else None,
            }
            
            if self.load_in_8bit and self.device == "cuda":
                model_kwargs["load_in_8bit"] = True
            
            self.judge_model = AutoModelForCausalLM.from_pretrained(
                self.judge_model_name,
                **model_kwargs
            )
            
            # Create pipeline
            self.judge_pipeline = pipeline(
                "text-generation",
                model=self.judge_model,
                tokenizer=self.judge_tokenizer,
                max_new_tokens=200,
                do_sample=False,
                temperature=0.1,
                return_full_text=False
            )
            
            self.logger.info(f"✓ Judge LLM initialized successfully")
            
            # Update evaluator with judge pipeline
            self.evaluator._judge_pipeline = self.judge_pipeline
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Judge LLM: {str(e)}")
            self.logger.warning("Continuing without LLM judge")
            self.evaluator.use_llm_judge = False
    
    def _chunk_text_langchain(self, text, metadata: Dict[str, Any] = None) -> List[Document]:
        """
        Chunk text using LangChain's RecursiveCharacterTextSplitter
        
        Args:
            text: Text to chunk (str) or list of LangChain Documents
            metadata: Metadata to attach to each chunk
            
        Returns:
            List of LangChain Document objects
        """
        metadata = metadata or {}

        if isinstance(text, list):
            documents_input = []
            for doc in text:
                doc_meta = dict(metadata)
                doc_meta.update(doc.metadata or {})
                doc.metadata = doc_meta
                documents_input.append(doc)
            documents = self.text_splitter.split_documents(documents_input)
        else:
            # Safely coerce bytes to string (some dataset fields may be bytes)
            if isinstance(text, (bytes, bytearray)):
                try:
                    text = text.decode('utf-8')
                except Exception:
                    text = text.decode('utf-8', errors='replace')

            if not text or len(text) == 0:
                return []
            
            documents = self.text_splitter.create_documents(
                texts=[str(text)],
                metadatas=[metadata]
            )
        
        # Log chunking statistics
        # Defensive: coerce any bytes page_content to str before computing lengths
        chunk_lengths = [
            len(doc.page_content)
            if not isinstance(doc.page_content, (bytes, bytearray))
            else len(doc.page_content.decode('utf-8', errors='replace'))
            for doc in documents
        ]

        self.logger.info(f"\nChunking Statistics (LangChain):")
        self.logger.info(f"  Total chunks: {len(documents)}")
        self.logger.info(f"  Avg chunk size: {np.mean(chunk_lengths):.2f} chars")
        self.logger.info(f"  Min chunk size: {np.min(chunk_lengths)} chars")
        self.logger.info(f"  Max chunk size: {np.max(chunk_lengths)} chars")
        self.logger.info(f"  Median chunk size: {np.median(chunk_lengths):.2f} chars")
        
        # Log first few chunks for inspection
        self.logger.debug(f"\nFirst 3 chunks preview:")
        for i, doc in enumerate(documents[:3]):
            content_preview = doc.page_content
            if isinstance(content_preview, (bytes, bytearray)):
                content_preview = content_preview.decode('utf-8', errors='replace')
            self.logger.debug(f"  Chunk {i}: {content_preview[:100]}...")
            self.logger.debug(f"    Metadata: {doc.metadata}")
        
        return documents

    def _normalize_evidence(self, evidence: Any) -> List[str]:
        """
        Normalize the dataset's `evidence` field into a list of text strings.

        Supported input types:
        - None -> []
        - str -> [str]
        - bytes/bytearray -> [decoded str]
        - list/tuple of str/bytes -> list of str
        - numpy arrays containing strings/bytes -> list of str
        - fallback: cast to str and return single-item list
        """
        parts: List[str] = []
        if evidence is None:
            return parts

        # bytes
        if isinstance(evidence, (bytes, bytearray)):
            try:
                parts.append(evidence.decode('utf-8'))
            except Exception:
                parts.append(evidence.decode('utf-8', errors='replace'))
            return parts

        # string
        if isinstance(evidence, str):
            return [evidence]

        # numpy arrays
        try:
            import numpy as _np
            if isinstance(evidence, _np.ndarray):
                for v in evidence.tolist():
                    if isinstance(v, (bytes, bytearray)):
                        try:
                            parts.append(v.decode('utf-8'))
                        except Exception:
                            parts.append(v.decode('utf-8', errors='replace'))
                    elif v is None:
                        continue
                    else:
                        parts.append(str(v))
                return [p for p in parts if p]
        except Exception:
            pass

        # iterable (list/tuple)
        if isinstance(evidence, (list, tuple)):
            for v in evidence:
                if v is None:
                    continue
                if isinstance(v, (bytes, bytearray)):
                    try:
                        parts.append(v.decode('utf-8'))
                    except Exception:
                        parts.append(v.decode('utf-8', errors='replace'))
                else:
                    parts.append(str(v))
            return [p for p in parts if p]

        # fallback: stringify
        try:
            return [str(evidence)]
        except Exception:
            return []
    
    def _create_vector_store_faiss(self, documents: List[Document], index_name: str = "default") -> FAISS:
        """
        Create FAISS vector store using LangChain
        
        Args:
            documents: List of LangChain Document objects
            index_name: Name for the vector store index
            
        Returns:
            FAISS vector store
        """
        # Delegate to vectorstore helper
        if create_faiss_store is None:
            raise RuntimeError("FAISS helper not available; ensure vectorstore.py is present and importable.")
        try:
            return create_faiss_store(self, documents, index_name=index_name)
        except Exception as e:
            self.logger.error(f"_create_vector_store_faiss delegated to vectorstore failed: {e}")
            raise

    def _retrieve_chunks_faiss(self, query: str, vector_store: FAISS, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve most relevant chunks using FAISS

        Args:
            query: Query text
            vector_store: FAISS vector store
            top_k: Number of chunks to retrieve

        Returns:
            List of retrieved chunks with scores and metadata
        """
        # Delegate retrieval to vectorstore helper if available
        if retrieve_faiss_chunks is None:
            # Last-resort: keep old behaviour if helper missing
            try:
                return retrieve_faiss_chunks(self, query, vector_store, top_k=top_k)
            except Exception:
                self.logger.warning("FAISS retrieval helper not available; no retrieval performed")
                return []
        try:
            return retrieve_faiss_chunks(self, query, vector_store, top_k=top_k)
        except Exception as e:
            self.logger.warning(f"FAISS retrieval helper failed: {e}")
            return []

    
    def _build_financebench_prompt(
        self,
        question: str,
        context: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> str:
        """
        Build prompts in the same style as FinanceBench's evaluation_playground:

        - Closed-book: only the question.
        - Open-book / RAG: question + an explicit context block delimited by markers.
        """
        question = (question or "").strip()
        context = (context or "").strip()
        prompt_mode = mode or self.experiment_type

        if context:
            context = context[: self.max_context_chars]

        # No context: closed-book behavior
        if not context:
            return f"Answer this question: {question}"

        # With context: oracle / in-context / RAG behavior
        if prompt_mode in {self.OPEN_BOOK, "oracle"}:
            header = "Here is the relevant evidence that you need to answer the question:"
        else:
            header = "Here is the relevant filing that you need to answer the question:"

        return (
            f"Answer this question: {question}\n"
            f"{header}\n"
            "[START OF FILING]\n"
            f"{context}\n"
            "[END OF FILING]"
        )

    def _generate_answer(
        self,
        question: str,
        context: Optional[str] = None,
        mode: Optional[str] = None,
        ) -> str:
        """
        Generate an answer using the configured LLM.

        This mirrors FinanceBench's get_answer() logic for prompts:
        - Closed-book: question only.
        - Open-book / RAG: question + [START OF FILING] ... [END OF FILING] context block.

        We still keep your local Llama / Qwen models and the existing HF Router API option.
        """
        # Build FinanceBench-style prompt
        prompt = self._build_financebench_prompt(question, context, mode=mode)

        # ---------------- API mode (HF Router / OpenAI) ---------------- #
        if self.use_api:
            # Initialize API client if needed
            self._initialize_llm()  # sets self.api_client

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful financial analyst assistant. "
                        "Answer strictly based on the question and the provided context "
                        "(if any) and avoid speculation."
                    ),
                },
                {"role": "user", "content": prompt},
            ]

            try:
                response = self.api_client.chat.completions.create(
                    model=self.llm_model_name,
                    messages=messages,
                    max_tokens=self.max_new_tokens,
                    temperature=0.0,
                )
                return (response.choices[0].message.content or "").strip()
            except Exception as e:
                self.logger.error(f"API generation failed: {e}")
                return ""

        # ---------------- Local HF pipeline mode (Llama / Qwen) ---------------- #
        # This reuses your existing _initialize_llm() and self.llm_pipeline
        self._initialize_llm()

        try:
            # text-generation pipeline (already configured with max_new_tokens, etc.)
            outputs = self.llm_pipeline(prompt)
        except TypeError:
            # Some pipeline versions require explicit kwargs
            outputs = self.llm_pipeline(
                prompt,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        if not outputs:
            return ""

        # HF text-generation pipeline returns a list of dicts with 'generated_text'
        text = outputs[0].get("generated_text", "")
        return (text or "").strip()

    
    def run_closed_book(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Delegate to modular closed-book runner
        if _run_closed_book is None:
            raise RuntimeError("Closed-book runner module not available")
        return _run_closed_book(self, data)
    
    def run_single_vector(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Delegate to modular single-vector runner
        if _run_single_vector is None:
            raise RuntimeError("Single-vector runner module not available")
        return _run_single_vector(self, data)
    
    def run_shared_vector(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Delegate to modular shared-vector runner
        if _run_shared_vector is None:
            raise RuntimeError("Shared-vector runner module not available")
        return _run_shared_vector(self, data)
    
    def run_open_book(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Delegate to modular open-book runner
        if _run_open_book is None:
            raise RuntimeError("Open-book runner module not available")
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
        df = self.data_loader.load_data()
        
        # Select samples
        if sample_indices is not None:
            data = self.data_loader.get_batch(indices=sample_indices)
        elif num_samples is not None:
            data = self.data_loader.get_batch(start=0, end=num_samples)
        else:
            data = self.data_loader.get_batch()
        
        self.logger.info(f"Processing {len(data)} samples")
        
        # Run appropriate experiment type
        if self.experiment_type == self.CLOSED_BOOK:
            results = self.run_closed_book(data)
        elif self.experiment_type == self.SINGLE_VECTOR:
            results = self.run_single_vector(data)
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
        self._save_results()
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("EXPERIMENT COMPLETE")
        self.logger.info("=" * 80)
    
    def _compute_aggregate_stats(self):
        """Compute and log aggregate statistics across all samples"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("AGGREGATE STATISTICS")
        self.logger.info("=" * 80)
        
        if not self.results:
            self.logger.warning("No results to aggregate")
            return
        
        # Generation length stats
        gen_lengths = [r['generation_length'] for r in self.results]
        self.logger.info(f"\nGenerated Answer Lengths:")
        self.logger.info(f"  Mean: {np.mean(gen_lengths):.2f} chars")
        self.logger.info(f"  Min: {np.min(gen_lengths)} chars")
        self.logger.info(f"  Max: {np.max(gen_lengths)} chars")
        self.logger.info(f"  Median: {np.median(gen_lengths):.2f} chars")
        
        # Retrieval stats (if applicable)
        if self.experiment_type in [self.SINGLE_VECTOR, self.SHARED_VECTOR]:
            context_lengths = [r['context_length'] for r in self.results]
            self.logger.info(f"\nContext Lengths:")
            self.logger.info(f"  Mean: {np.mean(context_lengths):.2f} chars")
            self.logger.info(f"  Min: {np.min(context_lengths)} chars")
            self.logger.info(f"  Max: {np.max(context_lengths)} chars")
            
            # Retrieval metrics: be defensive — some results may be skipped or have missing fields
            exact_matches = [
                r.get('retrieval_evaluation', {}).get('exact_match')
                for r in self.results
            ]
            exact_matches = [v for v in exact_matches if v is not None]

            max_overlaps = [
                r.get('retrieval_evaluation', {}).get('max_token_overlap')
                for r in self.results
            ]
            max_overlaps = [v for v in max_overlaps if v is not None]

            self.logger.info(f"\nRetrieval Performance:")
            if exact_matches:
                self.logger.info(f"  Exact Match Rate: {np.mean(exact_matches):.2%}")
            else:
                self.logger.info("  Exact Match Rate: n/a (no retrieval evals present)")

            if max_overlaps:
                self.logger.info(f"  Mean Max Token Overlap: {np.mean(max_overlaps):.4f}")
            else:
                self.logger.info("  Mean Max Token Overlap: n/a (no retrieval evals present)")
        
        # Generation metrics
        bleu_4_scores = []
        rouge_l_scores = []
        
        for r in self.results:
            if 'generation_evaluation' in r:
                eval_data = r['generation_evaluation']
                if 'bleu' in eval_data and 'bleu_4' in eval_data['bleu']:
                    bleu_4_scores.append(eval_data['bleu']['bleu_4'])
                if 'rouge' in eval_data and 'rouge_l_f1' in eval_data['rouge']:
                    rouge_l_scores.append(eval_data['rouge']['rouge_l_f1'])
        
        if bleu_4_scores:
            self.logger.info(f"\nGeneration Performance:")
            self.logger.info(f"  Mean BLEU-4: {np.mean(bleu_4_scores):.4f}")
            self.logger.info(f"  Mean ROUGE-L F1: {np.mean(rouge_l_scores):.4f}")
        
        self.logger.info("=" * 80)
    
    def _save_results(self):
        """Save results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/{self.experiment_type}_{timestamp}.json"
        
        output_data = {
            'metadata': self.experiment_metadata,
            'num_samples': len(self.results),
            'framework': 'LangChain + Chroma',
            'results': self.results
        }

        # Ensure JSON serializability (convert numpy types, ndarrays, tuples)
        def _to_json_serializable(obj):
            try:
                import numpy as _np
            except Exception:
                _np = None

            if obj is None:
                return None
            if isinstance(obj, (str, bool, int, float)):
                return obj
            if _np is not None and isinstance(obj, _np.ndarray):
                return obj.tolist()
            if _np is not None and isinstance(obj, _np.generic):
                try:
                    return obj.item()
                except Exception:
                    return str(obj)
            if isinstance(obj, dict):
                return {str(k): _to_json_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_to_json_serializable(v) for v in obj]
            if isinstance(obj, tuple):
                return [_to_json_serializable(v) for v in obj]
            # Fallback: try to cast to builtin types
            if hasattr(obj, '__dict__'):
                try:
                    return _to_json_serializable(obj.__dict__)
                except Exception:
                    pass
            try:
                return str(obj)
            except Exception:
                return None

        serializable = _to_json_serializable(output_data)

        with open(filename, 'w') as f:
            json.dump(serializable, f, indent=2)

        self.logger.info(f"\nResults saved to: {filename}")


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
        choices=["closed", "single", "shared", "open"],
        default="single",
        help=(
            "Experiment type:\n"
            "  closed  = closed-book (no context)\n"
            "  single  = single-vector store per document (singleStore)\n"
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
        "--use-api",
        action="store_true",
        help="Use HF Router / OpenAI API instead of local weights.",
    )
    parser.add_argument(
        "--no-8bit",
        action="store_true",
        help="Disable 8-bit loading (load full-precision model).",
    )

    args = parser.parse_args()

    # Map experiment name to RAGExperiment constants
    exp_map = {
        "closed": RAGExperiment.CLOSED_BOOK,
        "single": RAGExperiment.SINGLE_VECTOR,
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
            output_dir=args.output_dir,
            vector_store_dir=args.vector_store_dir,
            pdf_local_dir=args.pdf_dir,
            load_in_8bit=not args.no_8bit,
            use_api=args.use_api,
        )

        experiment.run_experiment(num_samples=args.num_samples)


if __name__ == "__main__":
    main()
