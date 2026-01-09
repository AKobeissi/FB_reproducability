"""
Shared dependency helpers for RAG experiments.

This module centralizes all of the optional LangChain / vectorstore imports so
that the main experiment logic can stay lean. It mirrors the previous fallback
behavior from `rag_experiments.py`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple


logger = logging.getLogger(__name__)

# LangChain imports (try multiple possible packages / entrypoints and provide safe fallbacks)
_HAS_LANGCHAIN = False
Document = None
RecursiveCharacterTextSplitter = None
HuggingFaceEmbeddings = None
FAISS = None
BaseRetriever = None
Chroma = None
RetrievalQA = None
HuggingFacePipeline = None
ParentDocumentRetriever = None
InMemoryStore = None
LocalFileStore = None

# 1) Text splitter: try langchain, then langchain_text_splitters
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    _HAS_LANGCHAIN = True
except Exception:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        _HAS_LANGCHAIN = True
    except Exception:
        RecursiveCharacterTextSplitter = None

# 2) Embeddings: try langchain_huggingface, then langchain.embeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
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
    Document = None
    BaseRetriever = None

from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore, LocalFileStore

# RetrievalQA and HuggingFacePipeline (optional)
try:
    from langchain.chains import RetrievalQA
    RetrievalQA = RetrievalQA
    _HAS_LANGCHAIN = True
except Exception:
    RetrievalQA = None

try:
    from langchain_community.llms import HuggingFacePipeline as _LC_HFP
    HuggingFacePipeline = _LC_HFP
    _HAS_LANGCHAIN = True
except Exception:
    try:
        from langchain.llms import HuggingFacePipeline as _LC_HFP
        HuggingFacePipeline = _LC_HFP
        _HAS_LANGCHAIN = True
    except Exception:
        HuggingFacePipeline = None

if not _HAS_LANGCHAIN:
    logger.warning(
        "langchain or langchain_community not available (or some subpackages missing); "
        "using minimal fallbacks. Install 'langchain' and 'langchain-community' for full functionality."
    )


# Provide dataclass fallback for Document if langchain import failed
if Document is None:

    @dataclass
    class Document:
        page_content: str
        metadata: dict = None



# Provide a placeholder FAISS class that raises a helpful error if used and FAISS import failed.
if FAISS is None:

    class FAISS:  # type: ignore
        @classmethod
        def from_documents(cls, *args, **kwargs):
            raise RuntimeError(
                "FAISS vector store not available. Install 'langchain-community' and a faiss package "
                "(faiss-cpu or faiss-gpu) to use vector stores."
            )

# Vectorstore helpers (modularized) with multi-strategy imports
build_chroma_store = None
create_faiss_store = None
retrieve_faiss_chunks = None
get_chroma_db_path = None
_vectorstore_import_errors: List[Tuple[str, Exception]] = []

try:
    from .vectorstore import build_chroma_store, create_faiss_store, retrieve_faiss_chunks, get_chroma_db_path  # type: ignore
    logger.info("Vectorstore helpers loaded via package-relative import.")
except Exception as e_pkg:
    _vectorstore_import_errors.append(("package-relative", e_pkg))
    try:
        from vectorstore import build_chroma_store, create_faiss_store, retrieve_faiss_chunks, get_chroma_db_path  # type: ignore
        logger.info("Vectorstore helpers loaded via absolute import.")
    except Exception as e_abs:
        _vectorstore_import_errors.append(("absolute", e_abs))
        logger.warning(
            "Vectorstore helpers not available; falling back to disabled vectorstore features."
        )
        build_chroma_store = None
        create_faiss_store = None
        retrieve_faiss_chunks = None
        get_chroma_db_path = None
        if _vectorstore_import_errors:
            for label, err in _vectorstore_import_errors:
                logger.warning("  [%s] import failed: %s", label, err)


__all__ = [
    "Document",
    "RecursiveCharacterTextSplitter",
    "HuggingFaceEmbeddings",
    "FAISS",
    "Chroma",
    "BaseRetriever",
    "RetrievalQA",
    "HuggingFacePipeline",
    "build_chroma_store",
    "create_faiss_store",
    "retrieve_faiss_chunks",
    "get_chroma_db_path",
    "ParentDocumentRetriever",
    "InMemoryStore",
    "LocalFileStore",
]
