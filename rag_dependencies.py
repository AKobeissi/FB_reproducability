"""
Shared dependency helpers for RAG experiments.

This module centralizes all of the optional LangChain / vectorstore imports so
that the main experiment logic can stay lean. It mirrors the previous fallback
behavior from `rag_experiments.py`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

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


class _MinimalTextSplitter:
    """Minimal fallback for LangChain's RecursiveCharacterTextSplitter."""

    def __init__(self, chunk_size=512, chunk_overlap=50, length_function=len, separators=None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function

    def _coerce_text(self, text: Any) -> str:
        if text is None:
            return ""
        if isinstance(text, (bytes, bytearray)):
            try:
                return text.decode("utf-8")
            except Exception:
                return text.decode("utf-8", errors="replace")
        return str(text)

    def _chunk_text(self, text: str) -> List[str]:
        normalized = self._coerce_text(text)
        if not normalized:
            return []
        chunks = []
        i = 0
        step = self.chunk_size - self.chunk_overlap if self.chunk_size > self.chunk_overlap else self.chunk_size
        while i < len(normalized):
            chunks.append(normalized[i:i + self.chunk_size])
            i += step
        return chunks

    def create_documents(self, texts: List[str], metadatas: List[dict]):
        docs = []
        for text, meta in zip(texts, metadatas):
            for chunk in self._chunk_text(text):
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata=dict(meta or {}),
                    )
                )
        return docs

    def split_documents(self, documents: List[Document]):
        """Mimic LangChain's split_documents for Document inputs."""
        split_docs: List[Document] = []
        for doc in documents or []:
            content = getattr(doc, "page_content", "") or ""
            base_metadata = dict(getattr(doc, "metadata", None) or {})
            for chunk in self._chunk_text(content):
                split_docs.append(
                    Document(
                        page_content=chunk,
                        metadata=dict(base_metadata),
                    )
                )
        return split_docs


# Provide dataclass fallback for Document if langchain import failed
if Document is None:

    @dataclass
    class Document:
        page_content: str
        metadata: dict = None

# Ensure we always have a text splitter available
if RecursiveCharacterTextSplitter is None:
    RecursiveCharacterTextSplitter = _MinimalTextSplitter

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
_vectorstore_import_errors: List[Tuple[str, Exception]] = []

try:
    from .vectorstore import build_chroma_store, create_faiss_store, retrieve_faiss_chunks  # type: ignore
    logger.info("Vectorstore helpers loaded via package-relative import.")
except Exception as e_pkg:
    _vectorstore_import_errors.append(("package-relative", e_pkg))
    try:
        from vectorstore import build_chroma_store, create_faiss_store, retrieve_faiss_chunks  # type: ignore
        logger.info("Vectorstore helpers loaded via absolute import.")
    except Exception as e_abs:
        _vectorstore_import_errors.append(("absolute", e_abs))
        logger.warning(
            "Vectorstore helpers not available; falling back to disabled vectorstore features."
        )
        build_chroma_store = None
        create_faiss_store = None
        retrieve_faiss_chunks = None
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
]
