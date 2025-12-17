"""Vector store helpers (Chroma preferred, optional FAISS fallback).

These helpers operate on an `experiment` object that exposes the same
interface as `RAGExperiment` in this repo, i.e. it should provide:

  - experiment.vector_store_dir : base directory for persisted stores
  - experiment.embeddings       : LangChain embeddings object
  - experiment._chunk_text_langchain(text, metadata) -> List[Document]
  - experiment.top_k            : (optional) int for retrieval
"""
from __future__ import annotations

from typing import List, Any, Optional, Iterable
import os
import logging
import json
import shutil
import re
import hashlib
from uuid import uuid4

logger = logging.getLogger(__name__)

# --- Try to import Chroma & FAISS from langchain / langchain_community --------

Chroma = None
FAISS = None

# Try langchain-chroma package first (newest distribution)
try:  # pragma: no cover - import robustness
    from langchain_chroma import Chroma as _Chroma  # type: ignore
    Chroma = _Chroma
except Exception:
    # Try community package next
    try:
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


def ensure_documents_have_ids(documents: Iterable[Any], namespace: str) -> List[Any]:
    """
    Ensure every document exposes an `id` attribute required by recent Chroma versions.
    """
    safe_docs: List[Any] = []

    for idx, doc in enumerate(documents or []):
        if doc is None:
            continue

        metadata = getattr(doc, "metadata", None)
        if metadata is None:
            metadata = {}
        elif not isinstance(metadata, dict):
            metadata = dict(metadata)

        doc_id = (
            getattr(doc, "id", None)
            or metadata.get("id")
            or metadata.get("doc_id")
            or metadata.get("chunk_id")
        )
        if not doc_id:
            doc_name = metadata.get("doc_name", "doc")
            doc_id = f"{namespace}-{doc_name}-{idx}-{uuid4().hex[:8]}"

        try:
            setattr(doc, "id", doc_id)
            safe_docs.append(doc)
            continue
        except Exception:
            pass

        page_content = getattr(doc, "page_content", None) or getattr(doc, "content", "")
        if isinstance(page_content, (bytes, bytearray)):
            page_content = page_content.decode("utf-8", errors="replace")

        class _DocShim:
            __slots__ = ("page_content", "metadata", "id")

            def __init__(self, content, meta, doc_id_value):
                self.page_content = content
                self.metadata = meta
                self.id = doc_id_value

        safe_docs.append(_DocShim(page_content, metadata, doc_id))

    return safe_docs


DEFAULT_CHROMA_BATCH_SIZE = 2000


def _detect_chroma_batch_limit(vectordb) -> Optional[int]:
    """
    Best-effort detection of the maximum batch size enforced by the underlying
    Chroma client/settings object.
    """
    candidates = [
        getattr(vectordb, "_client_settings", None),
        getattr(vectordb, "_client", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        settings = getattr(candidate, "_settings", candidate)
        max_batch = getattr(settings, "max_batch_size", None)
        if isinstance(max_batch, int) and max_batch > 0:
            return max_batch
    return None


def _batched_docs(documents: List[Any], batch_size: int) -> Iterable[List[Any]]:
    for idx in range(0, len(documents), batch_size):
        yield documents[idx : idx + batch_size]


def _sanitize_path_segment(name: str) -> str:
    """Sanitize a string to be safe for use as a filename/directory name."""
    if not name:
        return "unknown"
    # Replace non-alphanumeric characters (except - and _) with underscores
    cleaned = re.sub(r'[^a-zA-Z0-9\-_]', '_', name)
    # Collapse multiple underscores
    cleaned = re.sub(r'_+', '_', cleaned)
    return cleaned.strip('_') or "unknown"


def populate_chroma_store(experiment, vectordb, documents: List[Any], db_name: str = "shared") -> int:
    """
    Populate a ChromaDB instance with documents.
    Returns the number of chunks added.
    """
    exp_logger = getattr(experiment, "logger", logger)
    if not documents:
        return 0

    safe_documents = ensure_documents_have_ids(documents, db_name)
    batch_limit = _detect_chroma_batch_limit(vectordb)
    override_batch = getattr(experiment, "chroma_batch_size", None)
    if isinstance(override_batch, int) and override_batch > 0:
        batch_limit = override_batch if batch_limit is None else min(
            batch_limit, override_batch
        )
    if not batch_limit or batch_limit <= 0:
        batch_limit = DEFAULT_CHROMA_BATCH_SIZE
    
    if batch_limit < len(safe_documents):
        exp_logger.info(
            "Chroma batch limit detected at %s – ingesting in %s batches",
            batch_limit,
            (len(safe_documents) + batch_limit - 1) // batch_limit,
        )
    
    added = 0
    for chunk in _batched_docs(safe_documents, batch_limit):
        vectordb.add_documents(chunk)
        added += len(chunk)
    
    return added


def save_store_config(experiment, db_path: str):
    """Save the current experiment configuration to the vector store directory."""
    current_config = {
        "chunk_size": getattr(experiment, "chunk_size", None),
        "chunk_overlap": getattr(experiment, "chunk_overlap", None),
        "embedding_model": getattr(experiment, "embedding_model", None),
    }
    config_path = os.path.join(db_path, "config.json")
    with open(config_path, 'w') as f:
        json.dump(current_config, f)


def build_chroma_store(
    experiment,
    docs,
    embeddings=None,
    documents: Optional[List[Any]] = None,
    lazy_load: bool = False,
):
    """Build / load a Chroma store and return ``(retriever, vectordb)`` or ``(retriever, vectordb, is_new)``.

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
    lazy_load:
        If True, returns (retriever, vectordb, is_new) tuple. 
        If is_new is True, the caller is responsible for populating the store.
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
        db_name = _sanitize_path_segment(docs)
    elif isinstance(docs, (list, tuple, set)):
        docs_list = list(docs)
        if len(docs_list) <= 3:
            # Join them, but sanitize the result to avoid weird chars from individual names
            # or from the join itself if we used weird separators (though we use _)
            # Using _ as separator is safe if we sanitize segments first.
            sanitized_list = [_sanitize_path_segment(d) for d in docs_list]
            db_name = "_".join(sanitized_list)
        else:
            # Use a hash of the sorted document names to ensure uniqueness for large sets
            sorted_docs = sorted([str(d) for d in docs_list])
            doc_hash = hashlib.md5("_".join(sorted_docs).encode("utf-8")).hexdigest()
            db_name = f"custom_set_{doc_hash[:8]}"
    else:
        docs_list = None
        db_name = "shared"

    db_path = os.path.join(experiment.vector_store_dir, "chroma", db_name)
    exp_logger = getattr(experiment, "logger", logger)
    
    # --- Check for stale vector store configuration ------------------------------
    current_config = {
        "chunk_size": getattr(experiment, "chunk_size", None),
        "chunk_overlap": getattr(experiment, "chunk_overlap", None),
        "embedding_model": getattr(experiment, "embedding_model", None),
    }
    
    config_path = os.path.join(db_path, "config.json")
    should_clean = False
    
    if os.path.exists(db_path) and os.listdir(db_path):
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    stored_config = json.load(f)
                
                # Check for mismatch in critical parameters
                if stored_config != current_config:
                    exp_logger.warning(
                        f"Vector store '{db_name}' configuration mismatch.\n"
                        f"Stored: {stored_config}\n"
                        f"Current: {current_config}\n"
                        "Rebuilding vector store..."
                    )
                    should_clean = True
            except Exception as e:
                exp_logger.warning(f"Failed to read config from '{db_path}': {e}. Rebuilding...")
                should_clean = True
        else:
            # If the directory exists but config.json is missing, it might be an old store.
            # To be safe against stale data (different chunk sizes), we rebuild it.
            # Only do this if we actually have documents to populate it with or if lazy_load is requested
            # otherwise we might just be loading a valid legacy store.
            if documents or lazy_load:
                exp_logger.warning(f"Vector store '{db_name}' missing config.json. Rebuilding...")
                should_clean = True
    
    if should_clean:
        try:
            if os.path.exists(db_path):
                shutil.rmtree(db_path)
        except Exception as e:
            exp_logger.error(f"Failed to clear stale vector store at {db_path}: {e}")
            raise

    os.makedirs(db_path, exist_ok=True)

    try:
        vectordb = Chroma(persist_directory=db_path, embedding_function=embeddings)
    except Exception as e:  # pragma: no cover - environment dependent
        exp_logger.error(f"Failed to initialise Chroma DB at {db_path}: {e}")
        raise

    # --- Detect whether store is effectively empty --------------------------------
    try:
        collection_files = os.listdir(db_path)
        # If we just cleaned it, it is definitely empty (or just has what Chroma init created).
        is_empty = len(collection_files) <= 1 or should_clean
    except Exception:  # pragma: no cover
        is_empty = False

    top_k = getattr(experiment, "top_k", 5)
    retriever = vectordb.as_retriever(search_kwargs={"k": top_k})

    if lazy_load:
        return retriever, vectordb, is_empty

    # --- Populate the store if empty --------------------------------------------
    if is_empty:
        if documents:
            total_docs = len(documents)
            exp_logger.info(
                f"Populating Chroma DB '{db_name}' from {total_docs} precomputed chunks"
            )
            try:
                added = populate_chroma_store(experiment, vectordb, documents, db_name)
                exp_logger.info("Persisting Chroma DB '%s' with %s chunks", db_name, added)
                
                # Write config.json after successful population
                save_store_config(experiment, db_path)
                    
            except Exception as e:  # pragma: no cover
                exp_logger.error(
                    f"Failed to populate Chroma from precomputed documents: {e}"
                )
        else:
            if should_clean:
                 raise RuntimeError(
                    f"Vector store '{db_name}' was stale and cleared, but no documents were provided to rebuild it."
                )
            raise RuntimeError(
                f"Chroma DB '{db_name}' is empty and no documents were provided "
                "to populate it."
            )

    if hasattr(experiment, "register_component_usage"):
        experiment.register_component_usage(
            "vector_store",
            f"Chroma ({Chroma.__module__})" if Chroma is not None else "Chroma",
            {
                "persist_directory": db_path,
                "db_name": db_name,
                "config": current_config
            }
        )
        experiment.register_component_usage(
            "retriever",
            retriever.__class__.__name__,
            {
                "backend": "Chroma",
                "top_k": top_k
            }
        )
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
    if hasattr(experiment, "register_component_usage"):
        experiment.register_component_usage(
            "vector_store",
            f"FAISS ({FAISS.__module__})" if FAISS is not None else "FAISS",
            {
                "index_name": index_name,
                "num_documents": len(documents)
            }
        )
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
