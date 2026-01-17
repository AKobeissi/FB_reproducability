"""
SPLADE Retrieval Runner (sparse lexical retrieval).

FIXES:
- Canonical FinanceBench doc_name handling (Path(pdf).stem)
- Retrieved chunks explicitly include doc_name
- Prevents "pdfs/XYZ.pdf" vs "XYZ" evaluation mismatch
- Optional recursive PDF discovery (safe)
"""

import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.retrieval.bm25 import _compute_corpus_fingerprint

try:
    from langchain_community.document_loaders import PyMuPDFLoader
except ImportError:
    try:
        from langchain.document_loaders import PyMuPDFLoader
    except ImportError:
        PyMuPDFLoader = None

logger = logging.getLogger(__name__)

SPLADE_MODEL_ID = "naver/splade-cocondenser-ensembledistil"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _normalize_device(device: str) -> str:
    d = (str(device) if device else "cpu").lower()
    if d.startswith("cuda") and torch.cuda.is_available():
        return device
    return "cpu"


def _canonical_doc_name(path: str | Path) -> str:
    """
    FinanceBench canonical document identity.
    """
    return Path(path).stem


# ---------------------------------------------------------------------------
# Index data structure
# ---------------------------------------------------------------------------

@dataclass
class SpladeIndex:
    model_id: str
    fingerprint: str
    chunk_size: int
    chunk_overlap: int
    top_n_terms: int
    texts: List[str]
    metadatas: List[dict]
    postings: Dict[int, Tuple[np.ndarray, np.ndarray]]


# ---------------------------------------------------------------------------
# SPLADE encoder
# ---------------------------------------------------------------------------

class SpladeEncoder:
    def __init__(self, model_id: str = SPLADE_MODEL_ID, device: str = "cpu", max_length: int = 512):
        self.model_id = model_id
        self.device = _normalize_device(device)
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_topk(
        self,
        texts: List[str],
        top_n_terms: int = 256,
        batch_size: int = 8,
        min_value: float = 0.0,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        SPLADE encoding:
        max over tokens of log(1 + ReLU(logits))
        """
        results: List[Tuple[np.ndarray, np.ndarray]] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)

            logits = self.model(**inputs).logits
            values = torch.log1p(F.relu(logits))

            # mask padding
            values = values * inputs.attention_mask.unsqueeze(-1)

            pooled, _ = torch.max(values, dim=1)

            k = min(top_n_terms, pooled.size(-1))
            top_vals, top_ids = torch.topk(pooled, k=k, dim=-1)

            for ids, vals in zip(top_ids.cpu().numpy(), top_vals.cpu().numpy()):
                if min_value > 0:
                    keep = vals > min_value
                    ids, vals = ids[keep], vals[keep]

                results.append(
                    (
                        ids.astype(np.int32, copy=False),
                        vals.astype(np.float32, copy=False),
                    )
                )

        return results


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def _index_cache_path(experiment, fingerprint: str, top_n_terms: int) -> str:
    chunk_overlap = int(getattr(experiment, "chunk_overlap", 0) or 0)
    splitter_id = type(experiment.text_splitter).__name__

    base = os.path.join(experiment.vector_store_dir, "splade_cache")
    os.makedirs(base, exist_ok=True)

    fname = (
        f"splade_index__model={SPLADE_MODEL_ID.replace('/', '_')}"
        f"__splitter={splitter_id}"
        f"__chunk={int(experiment.chunk_size)}"
        f"__overlap={chunk_overlap}"
        f"__topn={int(top_n_terms)}"
        f"__fp={fingerprint}.pkl"
    )
    return os.path.join(base, fname)


def _load_pdfs_as_documents(pdf_dir: Path):
    if PyMuPDFLoader is None:
        raise ImportError("PyMuPDFLoader not available")

    # SAFE: recursive discovery (works even if flat)
    pdf_files = sorted(pdf_dir.rglob("*.pdf"))

    if not pdf_files:
        raise RuntimeError(f"[SPLADE] No PDFs found under {pdf_dir}")

    raw_docs = []

    for pdf_path in tqdm(pdf_files, desc="Loading PDFs for SPLADE"):
        try:
            loader = PyMuPDFLoader(str(pdf_path))
            docs = loader.load()

            canonical = _canonical_doc_name(pdf_path)

            for d in docs:
                # 🔑 CRITICAL: canonical FinanceBench identity
                d.metadata = dict(d.metadata or {})
                d.metadata["doc_name"] = canonical
                d.metadata["source"] = canonical
                d.metadata["file_name"] = canonical

            raw_docs.extend(docs)

        except Exception as e:
            logger.warning(f"Failed to load {pdf_path}: {e}")

    return raw_docs


def _build_splade_index(
    experiment,
    encoder: SpladeEncoder,
    pdf_dir: Path,
    fingerprint: str,
    top_n_terms: int = 256,
) -> SpladeIndex:
    logger.info("[SPLADE] Building index...")

    raw_docs = _load_pdfs_as_documents(pdf_dir)
    chunks = experiment.text_splitter.split_documents(raw_docs)

    if not chunks:
        raise RuntimeError("[SPLADE] No chunks created")

    texts = [c.page_content for c in chunks]
    metadatas = [dict(c.metadata) for c in chunks]

    sparse_docs = encoder.encode_topk(texts, top_n_terms=top_n_terms)

    postings_acc: Dict[int, List[Tuple[int, float]]] = {}

    for doc_id, (term_ids, term_weights) in enumerate(tqdm(sparse_docs, desc="Indexing")):
        for t, w in zip(term_ids.tolist(), term_weights.tolist()):
            if w <= 0:
                continue
            postings_acc.setdefault(int(t), []).append((doc_id, float(w)))

    postings = {
        term: (
            np.array([d for d, _ in pairs], dtype=np.int32),
            np.array([w for _, w in pairs], dtype=np.float32),
        )
        for term, pairs in postings_acc.items()
    }

    return SpladeIndex(
        model_id=SPLADE_MODEL_ID,
        fingerprint=fingerprint,
        chunk_size=int(experiment.chunk_size),
        chunk_overlap=int(getattr(experiment, "chunk_overlap", 0) or 0),
        top_n_terms=int(top_n_terms),
        texts=texts,
        metadatas=metadatas,
        postings=postings,
    )


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def _score_query(q_terms, q_weights, index: SpladeIndex, top_k: int):
    scores = np.zeros(len(index.texts), dtype=np.float32)

    for t, qw in zip(q_terms.tolist(), q_weights.tolist()):
        posting = index.postings.get(int(t))
        if posting is None:
            continue
        doc_ids, doc_ws = posting
        scores[doc_ids] += qw * doc_ws

    k = min(top_k, len(scores))
    idx = np.argpartition(scores, -k)[-k:]
    idx = idx[np.argsort(scores[idx])[::-1]]

    return scores[idx], idx


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def run_splade(experiment, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    logger.info("=" * 80)
    logger.info("RUNNING SPLADE EXPERIMENT")
    logger.info("=" * 80)

    encoder = SpladeEncoder(device=experiment.device)

    pdf_dir = Path(experiment.pdf_local_dir)
    fingerprint = _compute_corpus_fingerprint(pdf_dir)
    top_n_terms = int(getattr(experiment, "splade_top_n_terms", 256))

    cache_path = _index_cache_path(experiment, fingerprint, top_n_terms)

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            index = pickle.load(f)
    else:
        index = _build_splade_index(experiment, encoder, pdf_dir, fingerprint, top_n_terms)
        with open(cache_path, "wb") as f:
            pickle.dump(index, f)

    results = experiment._create_skipped_results(
        data, "splade", "splade", "pdf", "splade", start_id=0
    )

    for i, sample in enumerate(tqdm(data, desc="Inference")):
        q = sample["question"]

        q_terms, q_weights = encoder.encode_topk([q], top_n_terms, batch_size=1)[0]
        scores, ids = _score_query(q_terms, q_weights, index, experiment.top_k)

        retrieved = []
        for s, idx in zip(scores, ids):
            md = index.metadatas[idx]
            retrieved.append(
                {
                    "text": index.texts[idx],
                    "metadata": md,
                    "doc_name": md["doc_name"],  # 🔑 evaluator-safe
                    "score": float(s),
                }
            )

        context = "\n\n".join(r["text"] for r in retrieved)
        answer, prompt = experiment._generate_answer(q, context, return_prompt=True)

        gold_segments, gold_text = experiment._prepare_gold_evidence(sample["evidence"])

        results[i].update(
            {
                "doc_name": sample["doc_name"],
                "doc_link": sample["doc_link"],
                "retrieved_chunks": retrieved,
                "num_retrieved": len(retrieved),
                "gold_evidence": gold_text,
                "gold_evidence_segments": gold_segments,
                "generated_answer": answer,
                "final_prompt": prompt,
            }
        )

        experiment.notify_sample_complete(1)

    return results
