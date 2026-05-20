"""Dense retrieval using BGE-M3 embeddings + FAISS index."""
import pathlib
import pickle
import sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

from src.utils.logging import get_logger

logger = get_logger("dense_bge")


class DenseRetriever:
    def __init__(self, model_name: str = "BAAI/bge-m3", batch_size: int = 32, device: str | None = None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.model = None
        self.index = None
        self.corpus: list[dict] = []
        self.embeddings: np.ndarray | None = None

    def _get_model(self):
        if self.model is not None:
            return self.model
        from sentence_transformers import SentenceTransformer
        import torch
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(self.model_name, device=device)
        logger.info(f"Loaded {self.model_name} on {device}")
        return self.model

    def _embed(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        model = self._get_model()
        embs = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embs

    def build(self, pages: list[dict]) -> None:
        import faiss
        self.corpus = pages
        texts = [p["text"][:2048] for p in pages]
        logger.info(f"Encoding {len(texts)} pages with {self.model_name}...")
        self.embeddings = self._embed(texts)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings.astype(np.float32))
        logger.info(f"FAISS index built: {self.index.ntotal} vectors, dim={dim}")

    def retrieve(self, query: str, top_k: int = 100, doc_id: str | None = None) -> list[dict]:
        import faiss
        if self.index is None:
            raise RuntimeError("Index not built.")
        q_emb = self._embed([query], show_progress=False).astype(np.float32)
        if doc_id is not None:
            # Filter to specific doc pages
            doc_indices = [i for i, p in enumerate(self.corpus) if p["doc_id"] == doc_id]
            if not doc_indices:
                return []
            doc_embs = self.embeddings[doc_indices].astype(np.float32)
            scores = (q_emb @ doc_embs.T).squeeze(0)
            ranked = sorted(zip(doc_indices, scores.tolist()), key=lambda x: x[1], reverse=True)
            results = []
            for rank, (idx, score) in enumerate(ranked[:top_k]):
                p = self.corpus[idx]
                results.append({**p, "dense_score": float(score), "rank": rank + 1, "retriever": "bge_m3"})
            return results
        else:
            scores, indices = self.index.search(q_emb, min(top_k, self.index.ntotal))
            results = []
            for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
                if idx < 0:
                    continue
                p = self.corpus[idx]
                results.append({**p, "dense_score": float(score), "rank": rank + 1, "retriever": "bge_m3"})
            return results

    def embed_query(self, query: str) -> np.ndarray:
        return self._embed([query], show_progress=False)[0]

    def save(self, path: str | pathlib.Path) -> None:
        import faiss
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "faiss.index"))
        np.save(str(path / "embeddings.npy"), self.embeddings)
        with open(path / "corpus.pkl", "wb") as f:
            pickle.dump({"corpus": self.corpus, "model_name": self.model_name}, f)
        logger.info(f"Saved dense index to {path}")

    @classmethod
    def load(cls, path: str | pathlib.Path, batch_size: int = 32) -> "DenseRetriever":
        import faiss
        path = pathlib.Path(path)
        with open(path / "corpus.pkl", "rb") as f:
            state = pickle.load(f)
        obj = cls(model_name=state["model_name"], batch_size=batch_size)
        obj.corpus = state["corpus"]
        obj.index = faiss.read_index(str(path / "faiss.index"))
        obj.embeddings = np.load(str(path / "embeddings.npy"))
        logger.info(f"Loaded dense index from {path} ({len(obj.corpus)} pages)")
        return obj
