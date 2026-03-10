import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None

from transformers import AutoModel, AutoTokenizer

from src.core.rag_dependencies import Document
from src.retrieval.bm25 import _compute_corpus_fingerprint

logger = logging.getLogger(__name__)


def _sanitize(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in (name or ""))
    safe = "_".join([p for p in safe.split("_") if p])
    return safe or "unknown"


def _hash_config(config: Dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:8]


def _index_paths(experiment, scope: str, fingerprint: str) -> Tuple[str, str, Dict[str, Any]]:
    config = {
        "scope": scope,
        "fingerprint": fingerprint,
        "chunk_size": getattr(experiment, "chunk_size", None),
        "chunk_overlap": getattr(experiment, "chunk_overlap", None),
        "chunking_strategy": getattr(experiment, "chunking_strategy", None),
        "embedding_model": getattr(experiment, "embedding_model", None),
        "late_model": getattr(experiment, "late_model", None),
        "late_max_tokens": getattr(experiment, "late_max_tokens", None),
        "late_window_stride": getattr(experiment, "late_window_stride", None),
        "late_pooling": getattr(experiment, "late_pooling", None),
    }
    config_hash = _hash_config(config)
    name = f"{_sanitize(scope)}_late_{config_hash}"
    base_dir = Path(getattr(experiment, "vector_store_dir", "."))
    index_dir = base_dir / "late_chunks" / name
    return name, str(index_dir), config


def _mean_pool(token_embeddings: torch.Tensor) -> torch.Tensor:
    if token_embeddings.numel() == 0:
        return torch.zeros((token_embeddings.shape[-1],), device=token_embeddings.device)
    return token_embeddings.mean(dim=0)


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    norm = np.where(norm == 0.0, 1.0, norm)
    return vec / norm


def _token_windows(token_ids: List[int], max_tokens: int, stride: int) -> Iterable[Tuple[int, List[int]]]:
    if max_tokens <= 0:
        yield 0, token_ids
        return
    step = max(1, max_tokens - stride)
    for start in range(0, len(token_ids), step):
        end = min(start + max_tokens, len(token_ids))
        yield start, token_ids[start:end]
        if end >= len(token_ids):
            break


def _late_chunk_page(
    text: str,
    metadata: Dict[str, Any],
    tokenizer,
    model,
    chunk_size: int,
    chunk_overlap: int,
    max_tokens: int,
    stride: int,
    device: str,
) -> Tuple[List[Document], np.ndarray]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return [], np.empty((0, model.config.hidden_size), dtype=np.float32)

    all_chunks: List[Document] = []
    all_vectors: List[np.ndarray] = []

    for window_idx, (window_start, window_tokens) in enumerate(_token_windows(token_ids, max_tokens, stride)):
        input_ids = torch.tensor([window_tokens], device=device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            token_embeddings = outputs.last_hidden_state.squeeze(0)

        step = max(1, chunk_size - chunk_overlap)
        for start in range(0, len(window_tokens), step):
            end = min(start + chunk_size, len(window_tokens))
            span_tokens = window_tokens[start:end]
            if not span_tokens:
                continue

            chunk_text = tokenizer.decode(span_tokens, skip_special_tokens=True).strip()
            if not chunk_text:
                continue

            span_emb = _mean_pool(token_embeddings[start:end])
            vector = span_emb.detach().cpu().numpy().astype(np.float32)
            all_vectors.append(vector)

            chunk_meta = dict(metadata)
            chunk_meta.update({
                "late_window_index": window_idx,
                "late_window_start_token": window_start,
                "late_chunk_start": start,
                "late_chunk_end": end,
            })
            chunk_id_source = f"{chunk_meta.get('doc_name','doc')}_{chunk_meta.get('page','x')}_{window_idx}_{start}_{end}"
            chunk_meta["chunk_id"] = hashlib.md5(chunk_id_source.encode("utf-8")).hexdigest()

            all_chunks.append(Document(page_content=chunk_text, metadata=chunk_meta))

        torch.cuda.empty_cache() if device.startswith("cuda") else None

    if not all_vectors:
        return [], np.empty((0, model.config.hidden_size), dtype=np.float32)

    vectors = np.stack(all_vectors, axis=0)
    return all_chunks, vectors


class LateChunkIndex:
    def __init__(
        self,
        index,
        chunks: List[Document],
        tokenizer,
        model,
        device: str,
    ):
        self.index = index
        self.chunks = chunks
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.dim = getattr(model.config, "hidden_size", None)

    def embed_query(self, query: str) -> np.ndarray:
        input_ids = self.tokenizer.encode(query, add_special_tokens=False)
        if not input_ids:
            return np.zeros((self.dim,), dtype=np.float32)
        ids = torch.tensor([input_ids], device=self.device)
        mask = torch.ones_like(ids)
        with torch.no_grad():
            outputs = self.model(input_ids=ids, attention_mask=mask)
            token_embeddings = outputs.last_hidden_state.squeeze(0)
        pooled = _mean_pool(token_embeddings).detach().cpu().numpy().astype(np.float32)
        return _normalize(pooled)

    def search(self, query: str, k: int) -> List[Document]:
        query_vec = self.embed_query(query)
        return self.search_by_vector(query_vec, k)

    def search_by_vector(self, query_vec: np.ndarray, k: int) -> List[Document]:
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        query_vec = _normalize(query_vec.astype(np.float32))
        scores, indices = self.index.search(query_vec, k)
        out: List[Document] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            doc = self.chunks[idx]
            doc.metadata = dict(doc.metadata or {})
            doc.metadata["score"] = float(score)
            out.append(doc)
        return out


class LateChunkRetriever:
    def __init__(self, index: LateChunkIndex, top_k: int):
        self.index = index
        self.k = top_k

    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.index.search(query, self.k)

    def invoke(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)

    def similarity_search(self, query: str, k: Optional[int] = None) -> List[Document]:
        return self.index.search(query, k or self.k)


def build_late_chunk_index(
    experiment,
    docs: List[Any],
    scope: str,
    fingerprint: str,
) -> LateChunkIndex:
    if faiss is None:
        raise RuntimeError("faiss is required for late chunking.")

    name, index_dir, config = _index_paths(experiment, scope, fingerprint)
    os.makedirs(index_dir, exist_ok=True)
    index_path = os.path.join(index_dir, "index.faiss")
    chunks_path = os.path.join(index_dir, "chunks.json")
    config_path = os.path.join(index_dir, "config.json")

    if os.path.exists(index_path) and os.path.exists(chunks_path) and os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                stored = json.load(f)
            if stored == config:
                index = faiss.read_index(index_path)
                with open(chunks_path, "r") as f:
                    payload = json.load(f)
                chunks = [Document(page_content=c["text"], metadata=c["metadata"]) for c in payload]
                tokenizer, model, device = _load_late_model(experiment)
                return LateChunkIndex(index=index, chunks=chunks, tokenizer=tokenizer, model=model, device=device)
        except Exception as exc:
            logger.warning("Failed to load late index '%s': %s. Rebuilding.", name, exc)

    tokenizer, model, device = _load_late_model(experiment)
    chunk_size = int(getattr(experiment, "chunk_size", 512))
    chunk_overlap = int(getattr(experiment, "chunk_overlap", 64))
    max_tokens = int(getattr(experiment, "late_max_tokens", 2048))
    stride = int(getattr(experiment, "late_window_stride", 128))

    all_chunks: List[Document] = []
    all_vectors: List[np.ndarray] = []

    for doc in docs:
        text = getattr(doc, "page_content", None) or getattr(doc, "content", "")
        meta = getattr(doc, "metadata", {}) or {}
        page_chunks, page_vectors = _late_chunk_page(
            text=text,
            metadata=meta,
            tokenizer=tokenizer,
            model=model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_tokens=max_tokens,
            stride=stride,
            device=device,
        )
        if page_chunks:
            all_chunks.extend(page_chunks)
            all_vectors.append(page_vectors)

    if not all_chunks:
        raise RuntimeError("Late chunking produced no chunks.")

    vectors = np.vstack(all_vectors).astype(np.float32)
    vectors = _normalize(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, index_path)
    with open(chunks_path, "w") as f:
        json.dump([
            {"text": c.page_content, "metadata": c.metadata} for c in all_chunks
        ], f)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    return LateChunkIndex(index=index, chunks=all_chunks, tokenizer=tokenizer, model=model, device=device)


def _resolve_model_name(experiment, model_name: str) -> str:
    aliases = getattr(experiment, "EMBEDDING_ALIASES", {}) or {}
    if not isinstance(aliases, dict):
        return model_name
    resolved = aliases.get(model_name.lower(), model_name)
    if resolved != model_name:
        logger.info("Resolved late model alias '%s' -> '%s'.", model_name, resolved)
    return resolved


def _load_late_model(experiment) -> Tuple[Any, Any, str]:
    model_name = getattr(experiment, "late_model", None) or getattr(experiment, "embedding_model", None)
    if not model_name:
        raise RuntimeError("Late chunking requires a model name.")
    model_name = _resolve_model_name(experiment, model_name)

    pooling = getattr(experiment, "late_pooling", "mean")
    if pooling != "mean":
        logger.warning("Late pooling '%s' not supported; defaulting to mean.", pooling)

    # Jina models (v2 and v3) load remote code that imports transformers.onnx,
    # which was removed in transformers >= 4.30. Inject a stub so it doesn't crash.
    import sys, types
    if "transformers.onnx" not in sys.modules:
        _stub = types.ModuleType("transformers.onnx")
        class OnnxConfig: pass
        _stub.OnnxConfig = OnnxConfig
        sys.modules["transformers.onnx"] = _stub

    # transformers bug on Python 3.13: dot_natural_key returns mixed str/int lists
    # which can't be compared with < in Python 3.13. Patch to use type-safe tuples.
    try:
        import transformers.core_model_loading as _cml
        _orig_dnk = _cml.dot_natural_key
        def _safe_dot_natural_key(key):
            return [(0, p) if isinstance(p, int) else (1, p) for p in _orig_dnk(key)]
        _cml.dot_natural_key = _safe_dot_natural_key
    except Exception:
        pass

    device = getattr(experiment, "device", "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def load_late_index_for_scope(experiment, docs: List[Any], scope: str, pdf_dir: Optional[Path] = None) -> LateChunkIndex:
    if pdf_dir is None:
        pdf_dir = Path(getattr(experiment, "pdf_local_dir", "."))
    fingerprint = _compute_corpus_fingerprint(pdf_dir)
    return build_late_chunk_index(experiment, docs=docs, scope=scope, fingerprint=fingerprint)
