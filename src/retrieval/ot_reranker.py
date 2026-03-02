"""
Sentence-level Optimal Transport reranker.

Designed for second-stage reranking on a small candidate set (top-N).
This implementation uses entropic Sinkhorn on cosine-distance costs.
"""

from __future__ import annotations

import re
from typing import List, Sequence

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


class OTReranker:
    """Entropic OT reranker for query-document pairs."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str | None = None,
        query_max_sentences: int = 8,
        doc_max_sentences: int = 24,
        sinkhorn_reg: float = 0.05,
        sinkhorn_iters: int = 40,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.query_max_sentences = max(1, query_max_sentences)
        self.doc_max_sentences = max(1, doc_max_sentences)
        self.sinkhorn_reg = max(1e-4, sinkhorn_reg)
        self.sinkhorn_iters = max(5, sinkhorn_iters)

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        if not text:
            return []
        raw_parts = re.split(r"(?<=[.!?])\s+|\n+", text)
        parts = [part.strip() for part in raw_parts if part and part.strip()]
        return parts

    def _select_sentences(self, text: str, max_sentences: int) -> List[str]:
        parts = self._split_sentences(text)
        if not parts:
            clean = (text or "").strip()
            return [clean] if clean else [" "]
        return parts[:max_sentences]

    def _encode_sentences(self, sentences: Sequence[str]) -> torch.Tensor:
        embs = self.model.encode(
            list(sentences),
            convert_to_tensor=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        if embs.dim() == 1:
            embs = embs.unsqueeze(0)
        return embs.to(self.device)

    @staticmethod
    def _sinkhorn_distance(cost: torch.Tensor, reg: float, n_iters: int) -> torch.Tensor:
        m, n = cost.shape
        a = torch.full((m,), 1.0 / m, device=cost.device, dtype=cost.dtype)
        b = torch.full((n,), 1.0 / n, device=cost.device, dtype=cost.dtype)

        k_mat = torch.exp(-cost / reg).clamp_min(1e-9)
        u = torch.ones_like(a)
        v = torch.ones_like(b)

        for _ in range(n_iters):
            kv = torch.matmul(k_mat, v).clamp_min(1e-9)
            u = a / kv
            ktu = torch.matmul(k_mat.transpose(0, 1), u).clamp_min(1e-9)
            v = b / ktu

        transport = u.unsqueeze(1) * k_mat * v.unsqueeze(0)
        return torch.sum(transport * cost)

    @torch.no_grad()
    def score(self, query: str, docs: Sequence[str]) -> List[float]:
        """
        Returns OT similarity scores (higher is better).
        Score is negative OT distance.
        """
        if not docs:
            return []

        query_sents = self._select_sentences(query, self.query_max_sentences)
        query_embs = self._encode_sentences(query_sents)

        scores: List[float] = []
        for doc_text in docs:
            doc_sents = self._select_sentences(doc_text, self.doc_max_sentences)
            doc_embs = self._encode_sentences(doc_sents)

            cosine_sim = torch.matmul(query_embs, doc_embs.transpose(0, 1))
            cost = (1.0 - cosine_sim).clamp(min=0.0, max=2.0)
            distance = self._sinkhorn_distance(cost, reg=self.sinkhorn_reg, n_iters=self.sinkhorn_iters)
            scores.append(float(-distance.item()))

        return scores
