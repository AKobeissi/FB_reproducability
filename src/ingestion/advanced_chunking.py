import logging
import re
from typing import List, Dict, Any, Optional

import numpy as np
from transformers import AutoTokenizer

from src.core.rag_dependencies import Document

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p and p.strip()]


class BaseTextSplitter:
    def split_text(self, text: str) -> List[str]:
        raise NotImplementedError

    def split_documents(self, documents: List[Any]) -> List[Document]:
        out: List[Document] = []
        for doc in documents or []:
            content = getattr(doc, "page_content", None) or getattr(doc, "content", "")
            meta = getattr(doc, "metadata", {}) or {}
            for chunk in self.split_text(content):
                out.append(Document(page_content=chunk, metadata=dict(meta)))
        return out

    def create_documents(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[Document]:
        out: List[Document] = []
        if metadatas is None:
            metadatas = [{} for _ in texts]
        for text, meta in zip(texts, metadatas):
            for chunk in self.split_text(text):
                out.append(Document(page_content=chunk, metadata=dict(meta)))
        return out


class FixedWindowSplitter(BaseTextSplitter):
    def __init__(self, chunk_size: int, chunk_overlap: int, unit: str = "chars", tokenizer_name: Optional[str] = None):
        self.chunk_size = max(1, int(chunk_size))
        self.chunk_overlap = max(0, int(chunk_overlap))
        self.unit = unit
        self.tokenizer = None

        if self.unit == "tokens":
            target = tokenizer_name or "bert-base-uncased"
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(target)
            except Exception as exc:
                logger.warning("Tokenizer load failed for '%s' (%s). Falling back to chars.", target, exc)
                self.unit = "chars"

    def split_text(self, text: str) -> List[str]:
        if not text:
            return []

        if self.unit == "tokens" and self.tokenizer is not None:
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            if not token_ids:
                return []
            chunks = []
            step = max(1, self.chunk_size - self.chunk_overlap)
            for start in range(0, len(token_ids), step):
                end = min(start + self.chunk_size, len(token_ids))
                chunk_ids = token_ids[start:end]
                chunk_text = self.tokenizer.decode(chunk_ids, skip_special_tokens=True)
                if chunk_text.strip():
                    chunks.append(chunk_text)
                if end >= len(token_ids):
                    break
            return chunks

        # Char-based fixed windows
        chunks = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        for start in range(0, len(text), step):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            if end >= len(text):
                break
        return chunks


class SentenceChunker(BaseTextSplitter):
    def __init__(self, sentence_chunk_size: int, sentence_overlap: int = 0, max_chars: Optional[int] = None):
        self.sentence_chunk_size = max(1, int(sentence_chunk_size))
        self.sentence_overlap = max(0, int(sentence_overlap))
        self.max_chars = max_chars

    def split_text(self, text: str) -> List[str]:
        sentences = _split_sentences(text)
        if not sentences:
            return []

        chunks: List[str] = []
        step = max(1, self.sentence_chunk_size - self.sentence_overlap)
        for start in range(0, len(sentences), step):
            end = min(start + self.sentence_chunk_size, len(sentences))
            chunk = " ".join(sentences[start:end]).strip()
            if self.max_chars and len(chunk) > self.max_chars:
                chunk = chunk[: self.max_chars]
            if chunk:
                chunks.append(chunk)
            if end >= len(sentences):
                break
        return chunks


class SemanticChunker(BaseTextSplitter):
    def __init__(
        self,
        embedder,
        similarity_threshold: float = 0.6,
        min_sentences: int = 1,
        max_sentences: int = 12,
        max_chars: Optional[int] = None,
    ):
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.min_sentences = max(1, int(min_sentences))
        self.max_sentences = max(2, int(max_sentences))
        self.max_chars = max_chars

    def split_text(self, text: str) -> List[str]:
        sentences = _split_sentences(text)
        if not sentences:
            return []
        if self.embedder is None:
            logger.warning("Semantic chunker has no embedder; falling back to sentence grouping.")
            fallback = SentenceChunker(self.max_sentences, 0, self.max_chars)
            return fallback.split_text(text)

        embeddings = self.embedder(sentences)
        vectors = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        vectors = vectors / norms

        chunks: List[str] = []
        current: List[str] = []

        def flush():
            if not current:
                return
            chunk = " ".join(current).strip()
            if self.max_chars and len(chunk) > self.max_chars:
                chunk = chunk[: self.max_chars]
            if chunk:
                chunks.append(chunk)

        for idx, sentence in enumerate(sentences):
            if not current:
                current.append(sentence)
                continue

            sim = float(np.dot(vectors[idx - 1], vectors[idx]))
            if sim < self.similarity_threshold and len(current) >= self.min_sentences:
                flush()
                current = [sentence]
                continue

            current.append(sentence)
            if len(current) >= self.max_sentences:
                flush()
                current = []

        flush()
        return chunks
