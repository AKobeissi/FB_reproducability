"""
Standard RAG pipeline orchestrator.

This provides a single, explicit "pipeline" abstraction that composes existing
modules (ingestion/vectorstore/retrieval/query-expansion/reranking/generation).
It is intended to be the primary mental model for the repo.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import numpy as np

from src.core.rag_dependencies import BM25Retriever, Document
from src.experiments.hybrid_retrieval import (
    _get_docs,
    _perform_rrf,
    _load_or_build_bm25_chunks,
    finance_preprocess_func,
)
from src.experiments.rag_hyde_shared import _generate_hypotheticals
from src.pipeline.shared_index import ensure_shared_chroma_index
from src.retrieval.bm25 import _compute_corpus_fingerprint

logger = logging.getLogger(__name__)

RetrievalMode = Literal["dense", "sparse", "hybrid"]


@dataclass(frozen=True)
class StandardPipelineConfig:
    # Query expansion (HyDE)
    use_hyde: bool = False
    hyde_k: int = 1

    # Retrieval
    retrieval_mode: RetrievalMode = "dense"
    rrf_k: int = 60
    dense_weight: float = 1.0
    sparse_weight: float = 1.0

    # Reranking
    use_rerank: bool = False
    reranker_model_id: str = "BAAI/bge-reranker-v2-m3"
    candidate_multiplier_if_rerank: int = 4
    rerank_batch_size: int = 8


def _safe_import_cross_encoder():
    try:
        from sentence_transformers import CrossEncoder  # type: ignore

        return CrossEncoder
    except Exception:
        return None


def _format_retrieved_docs(question: str, docs: List[Document]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rank, d in enumerate(docs):
        score = d.metadata.get("rerank_score")
        if score is None:
            score = d.metadata.get("score", 0.0)
        out.append(
            {
                "rank": rank + 1,
                "text": d.page_content,
                "score": float(score) if score is not None else 0.0,
                "metadata": d.metadata or {},
            }
        )
    return out


def _dense_retrieve(vectordb: Any, query: str, k: int) -> List[Document]:
    if not vectordb:
        return []
    try:
        return vectordb.similarity_search(query, k=k) or []
    except Exception as e:
        logger.warning("Dense retrieval failed: %s", e)
        return []


def _dense_retrieve_by_vector(vectordb: Any, query_vector: List[float], k: int) -> List[Document]:
    if not vectordb:
        return []
    try:
        docs_scores = vectordb.similarity_search_by_vector_with_score(query_vector, k=k)
        dense_docs: List[Document] = []
        for doc, score in docs_scores:
            doc.metadata = doc.metadata or {}
            doc.metadata["score"] = float(score)
            dense_docs.append(doc)
        return dense_docs
    except Exception as e:
        logger.warning("Dense vector retrieval failed: %s", e)
        return []


def _maybe_rerank(
    *,
    reranker: Any,
    question: str,
    docs: List[Document],
    top_k: int,
    batch_size: int,
) -> List[Document]:
    if not reranker or not docs:
        return docs[:top_k]

    pairs: List[List[str]] = []
    valid_docs: List[Document] = []
    for d in docs:
        text = d.page_content or ""
        if not text.strip():
            continue
        pairs.append([question, text])
        valid_docs.append(d)

    if not pairs:
        return docs[:top_k]

    try:
        scores = reranker.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        for d, s in zip(valid_docs, scores):
            d.metadata = d.metadata or {}
            d.metadata["rerank_score"] = float(s)
        valid_docs.sort(key=lambda x: x.metadata.get("rerank_score", 0.0), reverse=True)
        return valid_docs[:top_k]
    except Exception as e:
        logger.warning("Reranking failed: %s", e)
        return docs[:top_k]


def run_standard_pipeline(
    experiment: Any,
    data: List[Dict[str, Any]],
    *,
    config: Optional[StandardPipelineConfig] = None,
) -> List[Dict[str, Any]]:
    """
    Run the standard pipeline for a batch of FinanceBench samples.

    Returns a results list compatible with the repo's saved JSON format.
    """
    if config is None:
        config = StandardPipelineConfig()

    logger.info("\n" + "=" * 80)
    logger.info("RUNNING STANDARD PIPELINE")
    logger.info("=" * 80)

    top_k = int(getattr(experiment, "top_k", 5))
    candidate_k = top_k
    if config.use_rerank:
        candidate_k = max(top_k, top_k * int(config.candidate_multiplier_if_rerank))

    logger.info("Pipeline configuration:")
    logger.info("  HyDE: %s (k=%s)", config.use_hyde, config.hyde_k)
    logger.info("  Retrieval: %s (candidate_k=%s, top_k=%s)", config.retrieval_mode, candidate_k, top_k)
    logger.info("  Rerank: %s", config.use_rerank)

    # Decide what stores we need.
    need_dense_store = config.retrieval_mode in ("dense", "hybrid") or config.use_hyde

    vectordb = None
    if need_dense_store:
        try:
            shared = ensure_shared_chroma_index(experiment, data)
            vectordb = shared.vectordb
        except Exception as e:
            logger.error("Failed to ensure shared Chroma index: %s", e, exc_info=True)
            if config.retrieval_mode == "dense":
                return []

    sparse_retriever = None
    if config.retrieval_mode in ("sparse", "hybrid"):
        try:
            pdf_dir = getattr(experiment, "pdf_local_dir", None)
            if not pdf_dir:
                raise RuntimeError("pdf_local_dir is not set; cannot initialize sparse retriever.")
            fingerprint = _compute_corpus_fingerprint(pdf_dir)
            chunks = _load_or_build_bm25_chunks(experiment, fingerprint)
            if chunks:
                sparse_retriever = BM25Retriever.from_documents(
                    chunks, preprocess_func=finance_preprocess_func
                )
                sparse_retriever.k = candidate_k
            else:
                logger.warning("BM25 chunks missing; falling back to dense.")
                if config.retrieval_mode == "sparse":
                    return []
        except Exception as e:
            logger.warning("Sparse retriever init failed (%s); falling back to dense.", e)
            if config.retrieval_mode == "sparse":
                return []

    # Optional reranker
    reranker = None
    if config.use_rerank:
        CrossEncoder = _safe_import_cross_encoder()
        if CrossEncoder is None:
            logger.warning("sentence-transformers not available; disabling reranking.")
        else:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
                reranker = CrossEncoder(
                    config.reranker_model_id,
                    model_kwargs={
                        "torch_dtype": torch.float16 if device == "cuda" else torch.float32
                    },
                    device=device,
                    trust_remote_code=True,
                )
            except Exception as e:
                logger.warning("Failed to load reranker; continuing without rerank (%s)", e)
                reranker = None

    results = experiment._create_skipped_results(
        data, "pipeline", "pipeline", "pdf", "unified", start_id=0
    )

    for i, sample in enumerate(data):
        question = sample.get("question") or ""
        results[i]["doc_name"] = sample.get("doc_name")
        results[i]["doc_link"] = sample.get("doc_link")
        if not question.strip():
            continue

        # Stage 1: HyDE
        search_query_vector = None
        hypotheticals: List[str] = []
        if config.use_hyde:
            hypotheticals = _generate_hypotheticals(experiment, question, n=max(1, int(config.hyde_k)))
            results[i]["hypotheticals"] = hypotheticals
            try:
                if getattr(experiment, "embeddings", None) and hypotheticals:
                    emb = experiment.embeddings.embed_documents(hypotheticals)
                    if len(emb) > 1:
                        search_query_vector = np.mean(emb, axis=0).tolist()
                    elif emb:
                        search_query_vector = emb[0]
            except Exception as e:
                logger.warning("HyDE embedding failed: %s", e)
                search_query_vector = None

        # Stage 2: Retrieval
        dense_docs: List[Document] = []
        sparse_docs: List[Any] = []

        if need_dense_store and vectordb:
            if search_query_vector is not None:
                dense_docs = _dense_retrieve_by_vector(vectordb, search_query_vector, k=candidate_k)
            else:
                dense_docs = _dense_retrieve(vectordb, question, k=candidate_k)

        if sparse_retriever is not None:
            sparse_docs = _get_docs(sparse_retriever, question) or []

        final_docs: List[Document] = []
        if config.retrieval_mode == "hybrid":
            fused = _perform_rrf(
                dense_docs,
                sparse_docs,
                rrf_k=int(config.rrf_k),
                top_k=candidate_k,
                dense_weight=float(config.dense_weight),
                sparse_weight=float(config.sparse_weight),
            )
            # Convert fused dicts back to Document
            final_docs = []
            for item in fused:
                d = Document(page_content=item["text"], metadata=item.get("metadata") or {})
                d.metadata["score"] = float(item.get("score", 0.0))
                final_docs.append(d)
        elif config.retrieval_mode == "sparse":
            # BM25 returns Documents; SPLADE wrapper (if used elsewhere) returns dicts.
            final_docs = []
            for it in sparse_docs:
                if isinstance(it, dict):
                    final_docs.append(
                        Document(
                            page_content=it.get("page_content") or it.get("text") or "",
                            metadata=it.get("metadata") or {},
                        )
                    )
                    if "score" in it:
                        final_docs[-1].metadata["score"] = float(it["score"])
                else:
                    final_docs.append(it)
        else:
            final_docs = list(dense_docs or [])

        # Stage 3: Reranking
        if config.use_rerank:
            final_docs = _maybe_rerank(
                reranker=reranker,
                question=question,
                docs=final_docs,
                top_k=top_k,
                batch_size=int(config.rerank_batch_size),
            )
        else:
            final_docs = final_docs[:top_k]

        # Persist retrieved chunks
        formatted_chunks = _format_retrieved_docs(question, final_docs)
        results[i]["retrieved_chunks"] = formatted_chunks
        results[i]["num_retrieved"] = len(formatted_chunks)

        # Stage 4: Generation
        context = "\n\n".join([c["text"] for c in formatted_chunks])
        ans, prompt = experiment._generate_answer(question, context, return_prompt=True)
        results[i]["generated_answer"] = ans
        results[i]["final_prompt"] = prompt

        gold_segs, gold_str = experiment._prepare_gold_evidence(sample.get("evidence", ""))
        results[i]["gold_evidence"] = gold_str
        results[i]["gold_evidence_segments"] = gold_segs

        experiment.notify_sample_complete(1)

    return results

