"""BM25 retrieval over page corpus."""
import pathlib
import pickle
import re
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

from rank_bm25 import BM25Okapi
from src.utils.logging import get_logger

logger = get_logger("bm25")


def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.corpus: list[dict] = []

    def build(self, pages: list[dict]) -> None:
        self.corpus = pages
        tokenized = [tokenize(p["text"]) for p in pages]
        self.bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)
        logger.info(f"BM25 index built over {len(pages)} pages")

    def retrieve(self, query: str, top_k: int = 100, doc_id: str | None = None) -> list[dict]:
        if self.bm25 is None:
            raise RuntimeError("Index not built. Call build() first.")
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        # Filter to specific doc if requested
        if doc_id is not None:
            candidates = [
                (i, scores[i]) for i, p in enumerate(self.corpus) if p["doc_id"] == doc_id
            ]
        else:
            candidates = list(enumerate(scores))
        candidates.sort(key=lambda x: x[1], reverse=True)
        results = []
        for rank, (idx, score) in enumerate(candidates[:top_k]):
            p = self.corpus[idx]
            results.append({
                **p,
                "bm25_score": float(score),
                "rank": rank + 1,
                "retriever": "bm25",
            })
        return results

    def save(self, path: str | pathlib.Path) -> None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "corpus": self.corpus, "k1": self.k1, "b": self.b}, f)
        logger.info(f"Saved BM25 index to {path}")

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "BM25Retriever":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(k1=state["k1"], b=state["b"])
        obj.bm25 = state["bm25"]
        obj.corpus = state["corpus"]
        logger.info(f"Loaded BM25 index from {path} ({len(obj.corpus)} pages)")
        return obj
