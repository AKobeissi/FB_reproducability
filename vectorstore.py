"""Vector store helpers (Chroma preferred, optional FAISS fallback).

These helpers operate on an `experiment` object that exposes the same
interface as `RAGExperiment` in this repo, i.e. it should provide:

  - experiment.vector_store_dir : base directory for persisted stores
  - experiment.embeddings       : LangChain embeddings object
  - experiment._chunk_text_langchain(text, metadata) -> List[Document]
  - experiment.top_k            : (optional) int for retrieval
"""
from __future__ import annotations

from typing import List, Any, Optional
import os
import logging

logger = logging.getLogger(__name__)

# --- Try to import Chroma & FAISS from langchain / langchain_community --------

Chroma = None
FAISS = None

# Try community package first (newer style)
try:  # pragma: no cover - import robustness
    from langchain_community.vectorstores import Chroma as _Chroma  # type: ignore
    Chroma = _Chroma
except Exception:
    try:
        from langchain.vectorstores import Chroma as _Chroma  # type: ignore
        Chroma = _Chroma
    except Exception:
        Chroma = None

try:  # pragma: no cover - import robustness
    from langchain_community.vectorstores import FAISS as _FAISS  # type: ignore
    FAISS = _FAISS
except Exception:
    try:
        from langchain.vectorstores import FAISS as _FAISS  # type: ignore
        FAISS = _FAISS
    except Exception:
        FAISS = None


def build_chroma_store(
    experiment,
    docs,
    embeddings=None,
    documents: Optional[List[Any]] = None,
):
    """Build / load a Chroma store and return ``(retriever, vectordb)``.

    Parameters
    ----------
    experiment:
        RAGExperiment-like object.
    docs:
        Either ``"all"``, a single document identifier (str), or an iterable
        of identifiers. These identifiers determine the persisted DB path.
    embeddings:
        Optional embeddings object. Defaults to ``experiment.embeddings``.
    documents:
        Optional list of *already chunked* LangChain ``Document`` objects.
    """
    if Chroma is None:
        raise RuntimeError(
            "Chroma vectorstore is not available in this environment. "
            "Install langchain and chromadb support."
        )

    if embeddings is None:
        embeddings = getattr(experiment, "embeddings", None)

    # --- Normalise docs argument & derive DB name --------------------------------
    if docs == "all":
        docs_list = None
        db_name = "shared"
    elif isinstance(docs, str):
        docs_list = [docs]
        db_name = docs
    elif isinstance(docs, (list, tuple, set)):
        docs_list = list(docs)
        db_name = "_".join(docs_list) if len(docs_list) <= 3 else docs_list[0]
    else:
        docs_list = None
        db_name = "shared"

    db_path = os.path.join(experiment.vector_store_dir, "chroma", db_name)
    os.makedirs(db_path, exist_ok=True)

    exp_logger = getattr(experiment, "logger", logger)

    try:
        vectordb = Chroma(persist_directory=db_path, embedding_function=embeddings)
    except Exception as e:  # pragma: no cover - environment dependent
        exp_logger.error(f"Failed to initialise Chroma DB at {db_path}: {e}")
        raise

    # --- Detect whether store is effectively empty --------------------------------
    try:
        collection_files = os.listdir(db_path)
        is_empty = len(collection_files) <= 1
    except Exception:  # pragma: no cover
        is_empty = False

    # --- Populate the store if empty --------------------------------------------
    if is_empty:
        if documents:
            exp_logger.info(
                f"Populating Chroma DB '{db_name}' from "
                f"{len(documents)} precomputed chunks"
            )
            try:
                vectordb.add_documents(list(documents))
                vectordb.persist()
            except Exception as e:  # pragma: no cover
                exp_logger.error(
                    f"Failed to populate Chroma from precomputed documents: {e}"
                )
        else:
            raise RuntimeError(
                f"Chroma DB '{db_name}' is empty and no documents were provided "
                "to populate it."
            )

    top_k = getattr(experiment, "top_k", 5)
    retriever = vectordb.as_retriever(search_kwargs={"k": top_k})
    return retriever, vectordb


def create_faiss_store(
    experiment,
    documents: List[Any],
    index_name: str = "default",
):
    """Create a FAISS vector store using LangChain (if available).

    Returns the FAISS store instance. Raises a RuntimeError if FAISS
    is not installed.
    """
    if FAISS is None:
        raise RuntimeError(
            "FAISS vectorstore is not available; install FAISS + langchain."
        )

    exp_logger = getattr(experiment, "logger", logger)
    exp_logger.info(
        f"Creating FAISS vector store '{index_name}' with {len(documents)} documents..."
    )
    vector_store = FAISS.from_documents(
        documents,
        getattr(experiment, "embeddings", None),
    )
    try:  # pragma: no cover - FAISS internals
        ntotal = getattr(vector_store.index, "ntotal", None)
        if ntotal is not None:
            exp_logger.info(f"✓ FAISS vector store created with {ntotal} vectors")
    except Exception:
        pass
    return vector_store


def retrieve_faiss_chunks(
    experiment,
    query: str,
    vector_store,
    top_k: Optional[int] = None,
):
    """Retrieve most relevant chunks from a FAISS vector store.

    Returns a list of dictionaries with ``text``, ``score``, ``length``,
    and ``metadata`` keys.
    """
    if FAISS is None:
        raise RuntimeError(
            "FAISS vectorstore is not available; cannot perform FAISS retrieval."
        )

    if top_k is None:
        top_k = getattr(experiment, "top_k", 5)

    exp_logger = getattr(experiment, "logger", logger)

    try:
        docs_and_scores = vector_store.similarity_search_with_score(query, k=top_k)
        results = []
        for rank, (doc, score) in enumerate(docs_and_scores):
            text = getattr(doc, "page_content", None) or getattr(doc, "content", None)
            if text is None:
                text = str(doc)
            results.append(
                {
                    "rank": rank + 1,
                    "text": text,
                    "score": float(score) if score is not None else None,
                    "length": len(text),
                    "metadata": getattr(doc, "metadata", {}),
                }
            )
        return results
    except Exception as e:  # pragma: no cover
        exp_logger.warning(f"FAISS retrieval failed: {e}")
        return []
