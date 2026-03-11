import hashlib
import logging
from typing import Any, Dict, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from src.ingestion.advanced_chunking import FixedWindowSplitter, SentenceChunker, SemanticChunker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CustomChunkerAdapter
# Wraps the strategy functions from src/experiments/chunking_strategies.py
# into a LangChain-compatible splitter interface (split_documents / split_text
# / create_documents).  Each call receives page-level LangChain Documents for
# ONE document and converts them to the (page_num, text) format expected by the
# custom chunkers, then converts the resulting Chunk objects back to Documents.
# ---------------------------------------------------------------------------

class CustomChunkerAdapter:
    """Thin adapter making custom Chunk-returning functions look like a LangChain splitter."""

    def __init__(self, chunker_fn, doc_id_key: str = "doc_name", **chunker_kwargs):
        self.chunker_fn = chunker_fn
        self.doc_id_key = doc_id_key
        self.chunker_kwargs = chunker_kwargs
        # Expose fake size attributes used by logging in _initialize_components
        self._chunk_size = chunker_kwargs.get("chunk_size", chunker_kwargs.get("child_chunk_size", 1024))
        self._chunk_overlap = chunker_kwargs.get("chunk_overlap", chunker_kwargs.get("child_overlap", 128))

    # ------------------------------------------------------------------
    def split_documents(self, documents: List[Any]) -> List[Any]:
        """
        Convert LangChain Documents (one per page of one PDF) into chunks
        via the wrapped custom strategy, then return as LangChain Documents.
        """
        from langchain.schema import Document as LCDocument

        if not documents:
            return []

        # Extract doc_id from first document's metadata
        first_meta = getattr(documents[0], "metadata", {}) or {}
        doc_id = (
            first_meta.get(self.doc_id_key)
            or first_meta.get("doc_id")
            or first_meta.get("source", "unknown")
        )

        # Build (page_num, text) list – PyMuPDF pages are 0-indexed, convert to 1-indexed
        pages = []
        for doc in documents:
            meta = getattr(doc, "metadata", {}) or {}
            raw_page = meta.get("page", meta.get("page_number", 0))
            try:
                page_num = int(raw_page) + 1  # convert 0-index → 1-index
            except (TypeError, ValueError):
                page_num = 1
            text = getattr(doc, "page_content", "") or ""
            pages.append((page_num, text))

        # Sort by page number so the doc is in order
        pages.sort(key=lambda x: x[0])

        # Base metadata to propagate to child documents
        base_meta = dict(first_meta)

        # Call the custom chunker
        try:
            chunks = self.chunker_fn(pages=pages, doc_id=doc_id, **self.chunker_kwargs)
        except Exception as e:
            logger.warning(f"CustomChunkerAdapter: chunker_fn failed ({e}). Returning empty list.")
            return []

        # Convert Chunk objects → LangChain Documents, skipping parent-only chunks
        result = []
        for chunk in chunks:
            if getattr(chunk, "strategy", "") == "parent_child_parent":
                continue  # index children only for parent_child strategy

            text = chunk.text  # includes context prefix if contextual strategy
            page_nums = chunk.page_nums or []

            meta = dict(base_meta)
            meta.update({
                "doc_name": chunk.doc_id,
                "doc_id": chunk.doc_id,
                # Use the first page for the "page" field (used for page_recall evaluation)
                "page": (page_nums[0] - 1) if page_nums else 0,  # back to 0-index for consistency
                "source": base_meta.get("source", "pdf"),
                "chunk_strategy": chunk.strategy,
                "chunk_index": chunk.chunk_index,
            })
            if page_nums:
                meta["all_page_nums"] = page_nums
            if chunk.parent_chunk_index >= 0:
                meta["parent_chunk_index"] = chunk.parent_chunk_index
            # Store parent text in metadata so generation context can be expanded
            parent_text = chunk.metadata.get("parent_text")
            if parent_text:
                meta["parent_text"] = parent_text

            result.append(LCDocument(page_content=text, metadata=meta))

        return result

    # ------------------------------------------------------------------
    def split_text(self, text: str) -> List[str]:
        """Fallback: single-text split (no page awareness)."""
        pages = [(1, text)]
        try:
            chunks = self.chunker_fn(pages=pages, doc_id="unknown", **self.chunker_kwargs)
        except Exception:
            return [text]
        return [
            c.text for c in chunks
            if getattr(c, "strategy", "") != "parent_child_parent"
        ]

    # ------------------------------------------------------------------
    def create_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Any]:
        """Create documents from plain texts (used by _chunk_text_langchain for str input)."""
        from langchain.schema import Document as LCDocument

        metadatas = metadatas or [{}] * len(texts)
        out = []
        for text, meta in zip(texts, metadatas):
            doc_id = meta.get("doc_name", meta.get("doc_id", "unknown"))
            page_num = meta.get("page", 0)
            try:
                page_num = int(page_num) + 1
            except (TypeError, ValueError):
                page_num = 1
            pages = [(page_num, text)]
            try:
                chunks = self.chunker_fn(pages=pages, doc_id=doc_id, **self.chunker_kwargs)
            except Exception:
                out.append(LCDocument(page_content=text, metadata=dict(meta)))
                continue
            for chunk in chunks:
                if getattr(chunk, "strategy", "") == "parent_child_parent":
                    continue
                new_meta = dict(meta)
                new_meta["chunk_strategy"] = chunk.strategy
                out.append(LCDocument(page_content=chunk.text, metadata=new_meta))
        return out


# ---------------------------------------------------------------------------
# get_splitters – factory that returns (parent_splitter, child_splitter)
# ---------------------------------------------------------------------------

def get_splitters(experiment):
    """
    Factory function to return the appropriate splitters based on experiment config.
    """
    strategy = experiment.chunking_strategy
    unit = experiment.chunking_unit

    child_splitter = None
    parent_splitter = None

    # --- 1. Fixed Window Chunking (Char or Token) ---
    if strategy == "fixed":
        tokenizer_name = getattr(experiment, "chunk_tokenizer_name", None)
        parent_splitter = FixedWindowSplitter(
            chunk_size=experiment.chunk_size,
            chunk_overlap=experiment.chunk_overlap,
            unit=unit,
            tokenizer_name=tokenizer_name,
        )
        return parent_splitter, None

    # --- 2. Sentence Chunking ---
    if strategy == "sentence":
        sentence_size = getattr(experiment, "sentence_chunk_size", None) or experiment.chunk_size
        sentence_overlap = getattr(experiment, "sentence_overlap", None) or experiment.chunk_overlap
        max_chars = getattr(experiment, "sentence_max_chars", None)
        parent_splitter = SentenceChunker(
            sentence_chunk_size=sentence_size,
            sentence_overlap=sentence_overlap,
            max_chars=max_chars,
        )
        return parent_splitter, None

    # --- 3. Semantic Chunking ---
    if strategy == "semantic":
        embedder = None
        if hasattr(experiment, "embeddings") and experiment.embeddings is not None:
            if hasattr(experiment.embeddings, "embed_documents"):
                embedder = experiment.embeddings.embed_documents
        parent_splitter = SemanticChunker(
            embedder=embedder,
            similarity_threshold=getattr(experiment, "semantic_similarity_threshold", 0.6),
            min_sentences=getattr(experiment, "semantic_min_sentences", 1),
            max_sentences=getattr(experiment, "semantic_max_sentences", 12),
            max_chars=getattr(experiment, "semantic_max_chunk_chars", None),
        )
        return parent_splitter, None

    # --- 4. Adaptive Chunking ---
    # Varies chunk size by content density (tabular/numeric = smaller, prose = larger).
    if strategy == "adaptive":
        try:
            from src.experiments.chunking_strategies import chunk_adaptive
            parent_splitter = CustomChunkerAdapter(
                chunk_adaptive,
                base_chunk_size=experiment.chunk_size,
                chunk_overlap=experiment.chunk_overlap,
                min_chunk_size=getattr(experiment, "adaptive_min_chunk_size",
                                       getattr(experiment, "adaptive_min", 256)),
                max_chunk_size=getattr(experiment, "adaptive_max_chunk_size",
                                       getattr(experiment, "adaptive_max", 2048)),
                tokenizer_name=getattr(experiment, "chunk_tokenizer_name", None),
            )
            return parent_splitter, None
        except ImportError as e:
            logger.warning(f"Could not import chunk_adaptive ({e}). Falling through to default.")

    # --- 5. Parent-Child (Hierarchical) Chunking ---
    # Index small child chunks; parent text is stored in metadata for context expansion.
    if strategy == "parent_child":
        try:
            from src.experiments.chunking_strategies import chunk_parent_child
            p_size = getattr(experiment, "parent_chunk_size", None) or 2048
            p_overlap = getattr(experiment, "parent_chunk_overlap", None) or 256
            c_size = getattr(experiment, "child_chunk_size", None) or experiment.chunk_size or 512
            c_overlap = getattr(experiment, "child_chunk_overlap", None) or experiment.chunk_overlap or 64
            parent_splitter = CustomChunkerAdapter(
                chunk_parent_child,
                parent_chunk_size=p_size,
                parent_overlap=p_overlap,
                child_chunk_size=c_size,
                child_overlap=c_overlap,
                tokenizer_name=getattr(experiment, "chunk_tokenizer_name", None),
            )
            return parent_splitter, None
        except ImportError as e:
            logger.warning(f"Could not import chunk_parent_child ({e}). Falling through to default.")

    # --- 6. Table-Aware Chunking ---
    # Keeps detected table regions as atomic chunks; non-table text uses naive chunking.
    if strategy == "table_aware":
        try:
            from src.experiments.chunking_strategies import chunk_table_aware
            parent_splitter = CustomChunkerAdapter(
                chunk_table_aware,
                chunk_size=experiment.chunk_size,
                chunk_overlap=experiment.chunk_overlap,
                tokenizer_name=getattr(experiment, "chunk_tokenizer_name", None),
            )
            return parent_splitter, None
        except ImportError as e:
            logger.warning(f"Could not import chunk_table_aware ({e}). Falling through to default.")

    # --- 7. Contextual Chunking ---
    # Prepends doc/page/section context prefix to each chunk before embedding.
    if strategy == "contextual":
        try:
            from src.experiments.chunking_strategies import chunk_contextual
            parent_splitter = CustomChunkerAdapter(
                chunk_contextual,
                chunk_size=experiment.chunk_size,
                chunk_overlap=experiment.chunk_overlap,
                context_budget=getattr(experiment, "context_budget", 128),
                tokenizer_name=getattr(experiment, "chunk_tokenizer_name", None),
            )
            return parent_splitter, None
        except ImportError as e:
            logger.warning(f"Could not import chunk_contextual ({e}). Falling through to default.")

    # --- 8. Metadata-Enriched Chunking ---
    # Same fixed-size chunks as naive but with rich structured metadata fields.
    if strategy == "metadata":
        try:
            from src.experiments.chunking_strategies import chunk_metadata
            parent_splitter = CustomChunkerAdapter(
                chunk_metadata,
                chunk_size=experiment.chunk_size,
                chunk_overlap=experiment.chunk_overlap,
                tokenizer_name=getattr(experiment, "chunk_tokenizer_name", None),
            )
            return parent_splitter, None
        except ImportError as e:
            logger.warning(f"Could not import chunk_metadata ({e}). Falling through to default.")

    # --- 9. Token-based Chunking (default for recursive and unrecognised strategies) ---
    if unit == "tokens":
        try:
            explicit_tokenizer = getattr(experiment, "chunk_tokenizer_name", None)

            # Get resolved model name from embeddings if available
            if hasattr(experiment, 'embeddings') and experiment.embeddings:
                if hasattr(experiment.embeddings, 'model_name'):
                    resolved_embedding_model = experiment.embeddings.model_name
                elif hasattr(experiment.embeddings, 'model'):
                    resolved_embedding_model = getattr(experiment.embeddings.model, 'name_or_path', None)
                else:
                    resolved_embedding_model = None
            else:
                resolved_embedding_model = None

            embedding_model = getattr(experiment, "embedding_model", None)
            target_model = (
                explicit_tokenizer
                or resolved_embedding_model
                or embedding_model
                or experiment.llm_model_name
            )

            if target_model and any(x in target_model.lower() for x in ["openai", "gpt-", "claude"]):
                print(f"Warning: '{target_model}' looks like an API model. Using 'bert-base-uncased' for token counting.")
                target_model = "bert-base-uncased"

            tokenizer = AutoTokenizer.from_pretrained(target_model)

            parent_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                tokenizer,
                chunk_size=experiment.chunk_size,
                chunk_overlap=experiment.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        except Exception as e:
            print(f"Warning: Tokenizer load failed for '{target_model}' ({e}). Falling back to chars.")
            unit = "chars"

    # --- 10. Late Chunking (fixed boundaries; embedding handled by long-context model) ---
    if strategy == "late":
        tokenizer_name = getattr(experiment, "chunk_tokenizer_name", None)
        parent_splitter = FixedWindowSplitter(
            chunk_size=experiment.chunk_size,
            chunk_overlap=experiment.chunk_overlap,
            unit="tokens",
            tokenizer_name=tokenizer_name,
        )
        return parent_splitter, None

    # --- 11. Hierarchical Chunking (legacy "hierarchical" strategy name) ---
    if strategy == "hierarchical":
        child_size = experiment.child_chunk_size or 256
        child_overlap = experiment.child_chunk_overlap or 32

        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size,
            chunk_overlap=child_overlap,
            length_function=len
        )

        if not parent_splitter:
            parent_size = experiment.parent_chunk_size or experiment.chunk_size or 1024
            parent_overlap = experiment.parent_chunk_overlap or experiment.chunk_overlap or 100

            parent_splitter = RecursiveCharacterTextSplitter(
                chunk_size=parent_size,
                chunk_overlap=parent_overlap,
                length_function=len
            )

    # --- 12. Standard Character Chunking (final fallback) ---
    elif not parent_splitter:
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=experiment.chunk_size,
            chunk_overlap=experiment.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    return parent_splitter, child_splitter


def chunk_docs_page_aware(docs, text_splitter, doc_name_override=None):
    """
    Chunks a list of LangChain Documents (pages) individually to ensure
    no chunk spans multiple pages.
    """
    all_chunks = []

    for page_idx, doc in enumerate(docs):
        page_chunks = text_splitter.split_text(doc.page_content)

        for chunk_idx, chunk_text in enumerate(page_chunks):
            # Resolve Metadata
            raw_page = doc.metadata.get("page", doc.metadata.get("page_number", page_idx))
            page_num = raw_page  # 0-indexed

            doc_id = doc_name_override or doc.metadata.get("doc_name") or "unknown_doc"

            chunk_id_str = f"{doc_id}_p{page_num}_{chunk_idx}"
            chunk_id = hashlib.md5(chunk_id_str.encode()).hexdigest()

            from langchain.schema import Document

            new_meta = {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "doc_name": doc_id,
                "page": page_num,
                "source": doc.metadata.get("source", "pdf"),
                **{k: v for k, v in doc.metadata.items() if k not in ['page', 'source', 'doc_name', 'doc_id']}
            }

            all_chunks.append(Document(page_content=chunk_text, metadata=new_meta))

    return all_chunks