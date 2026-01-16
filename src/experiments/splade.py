"""
SPLADE Retrieval Runner (sparse lexical retrieval).
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

def _normalize_device(device: str) -> str:
    d = (str(device) if device else "cpu").lower()
    if d.startswith("cuda") and torch.cuda.is_available():
        return device
    return "cpu"

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
    def encode_topk(self, texts: List[str], top_n_terms: int = 256, batch_size: int = 8, min_value: float = 0.0) -> List[Tuple[np.ndarray, np.ndarray]]:
        assert top_n_terms > 0, "top_n_terms must be > 0"
        results: List[Tuple[np.ndarray, np.ndarray]] = []
        
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            # Ensure we pass token_type_ids if the model expects them (BERT usually does)
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(self.device)
            
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # SPLADE formula: log(1 + ReLU(logits)) * AttentionMask
            values = torch.log1p(F.relu(logits))
            attn = inputs.attention_mask.unsqueeze(-1)
            values = values * attn
            
            # Max pooling over sequence length
            pooled, _ = torch.max(values, dim=1)
            
            # Top K
            k = min(top_n_terms, pooled.size(-1))
            top_vals, top_ids = torch.topk(pooled, k=k, dim=-1)
            
            top_vals = top_vals.detach().float().cpu().numpy()
            top_ids = top_ids.detach().int().cpu().numpy()
            
            for row_ids, row_vals in zip(top_ids, top_vals):
                if min_value > 0.0:
                    keep = row_vals > min_value
                    row_ids = row_ids[keep]
                    row_vals = row_vals[keep]
                results.append((row_ids.astype(np.int32, copy=False), row_vals.astype(np.float32, copy=False)))
        return results

def _index_cache_path(experiment, fingerprint: str, top_n_terms: int) -> str:
    chunk_overlap = int(getattr(experiment, "chunk_overlap", 0) or 0)
    splitter = getattr(experiment, "text_splitter", None)
    splitter_id = type(splitter).__name__ if splitter else "unknown"
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
        raise ImportError("PyMuPDFLoader not available.")
    
    if not pdf_dir.exists():
        logger.error(f"PDF Directory does not exist: {pdf_dir}")
        return []
        
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDFs found in {pdf_dir}")
        return []

    raw_docs = []
    for p in tqdm(pdf_files, desc="Loading PDFs for SPLADE"):
        try:
            loader = PyMuPDFLoader(str(p))
            raw_docs.extend(loader.load())
        except Exception as e:
            logger.warning(f"Failed to load {p}: {e}")
            
    return raw_docs

def _build_splade_index(experiment, encoder: SpladeEncoder, pdf_dir: Path, fingerprint: str, top_n_terms: int = 256, encode_batch_size: Optional[int] = None) -> SpladeIndex:
    logger.info("Building SPLADE index...")
    raw_docs = _load_pdfs_as_documents(pdf_dir)
    
    chunks = experiment.text_splitter.split_documents(raw_docs)
    logger.info(f"Chunked corpus into {len(chunks)} chunks.")
    
    if len(chunks) == 0:
        logger.error("No chunks created! Check PDF directory and splitter.")
    
    texts = [c.page_content for c in chunks]
    metadatas = [dict(c.metadata) for c in chunks]
    
    if encode_batch_size is None:
        encode_batch_size = 4 if str(encoder.device).startswith("cuda") else 8
        
    logger.info(f"Encoding chunks with SPLADE (top_n_terms={top_n_terms})...")
    sparse_docs = encoder.encode_topk(texts, top_n_terms=top_n_terms, batch_size=encode_batch_size)
    
    # Build Inverted Index
    postings_acc: Dict[int, List[Tuple[int, float]]] = {}
    for doc_id, (term_ids, term_weights) in enumerate(tqdm(sparse_docs, desc="Indexing")):
        for t, w in zip(term_ids.tolist(), term_weights.tolist()):
            if w <= 0.0: continue
            # Ensure keys are standard Python ints, not numpy ints
            postings_acc.setdefault(int(t), []).append((int(doc_id), float(w)))
            
    postings: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for term, pairs in postings_acc.items():
        doc_ids = np.fromiter((p[0] for p in pairs), dtype=np.int32, count=len(pairs))
        weights = np.fromiter((p[1] for p in pairs), dtype=np.float32, count=len(pairs))
        postings[int(term)] = (doc_ids, weights)
        
    chunk_overlap = int(getattr(experiment, "chunk_overlap", 0) or 0)
    logger.info(f"SPLADE Index built. Vocabulary size (active terms): {len(postings)}")
    
    return SpladeIndex(model_id=SPLADE_MODEL_ID, fingerprint=fingerprint, chunk_size=int(experiment.chunk_size), chunk_overlap=chunk_overlap, top_n_terms=int(top_n_terms), texts=texts, metadatas=metadatas, postings=postings)

def _score_query_against_index(q_terms: np.ndarray, q_weights: np.ndarray, index: SpladeIndex, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    n_docs = len(index.texts)
    if n_docs == 0: 
        return np.array([], dtype=np.float32), np.array([], dtype=np.int32)
        
    scores = np.zeros(n_docs, dtype=np.float32)
    
    for t, qw in zip(q_terms.tolist(), q_weights.tolist()):
        # Lookup using standard int
        posting = index.postings.get(int(t))
        if posting is None: continue
        
        doc_ids, doc_ws = posting
        # doc_ids is int32, doc_ws is float32
        scores[doc_ids] += np.float32(qw) * doc_ws
        
    k = min(int(top_k), n_docs)
    if k <= 0: return np.array([], dtype=np.float32), np.array([], dtype=np.int32)
    
    # Efficient top-k
    idx_part = np.argpartition(scores, -k)[-k:]
    idx_sorted = idx_part[np.argsort(scores[idx_part])[::-1]]
    
    return scores[idx_sorted], idx_sorted.astype(np.int32, copy=False)

def run_splade(experiment, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING SPLADE EXPERIMENT")
    logger.info("=" * 80)

    # 1) Setup
    device = getattr(experiment, "device", "cpu")
    logger.info(f"Loading SPLADE model ({SPLADE_MODEL_ID}) on {device}...")
    encoder = SpladeEncoder(device=device)

    pdf_dir = Path(experiment.pdf_local_dir)
    fingerprint = _compute_corpus_fingerprint(pdf_dir)
    top_n_terms = int(getattr(experiment, "splade_top_n_terms", 256) or 256)
    cache_path = _index_cache_path(experiment, fingerprint=fingerprint, top_n_terms=top_n_terms)

    # 2) Load or build index
    index = None
    if os.path.exists(cache_path):
        logger.info(f"Loading SPLADE index cache: {cache_path}")
        try:
            with open(cache_path, "rb") as f:
                index = pickle.load(f)
            # Basic validation
            if index.chunk_size != int(experiment.chunk_size):
                logger.warning("Cache chunk size mismatch. Rebuilding.")
                index = None
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            index = None

    if index is None:
        index = _build_splade_index(experiment, encoder, pdf_dir, fingerprint, top_n_terms)
        with open(cache_path, "wb") as f:
            pickle.dump(index, f)

    # 3) Initialize Results with Placeholder
    results = experiment._create_skipped_results(
        data, "splade_placeholder", "splade_placeholder", "pdf", "splade", start_id=0
    )

    # 4) Retrieval Loop
    logger.info("Running SPLADE retrieval...")
    
    for i, sample in enumerate(tqdm(data, desc="Inference")):
        question = sample.get("question")
        
        results[i]['doc_name'] = sample.get('doc_name')
        results[i]['doc_link'] = sample.get('doc_link')
        
        if not question:
            continue

        # Encode & Search
        # encode_topk returns list of tuples, we take [0] for the single query
        q_sparse = encoder.encode_topk([question], top_n_terms=top_n_terms, batch_size=1)[0]
        q_terms, q_weights = q_sparse
        
        top_scores, top_ids = _score_query_against_index(q_terms, q_weights, index, int(experiment.top_k))

        retrieved = []
        for score, doc_id in zip(top_scores.tolist(), top_ids.tolist()):
            # Filter out zero scores if desirable (though SPLADE can return 0 if no overlap)
            if score <= 0: continue
            retrieved.append({
                "text": index.texts[doc_id],
                "metadata": index.metadatas[doc_id],
                "score": float(score),
            })
            
        if not retrieved:
            # Fallback for empty retrieval to prevent context errors
            logger.debug(f"No results for query: {question[:30]}")

        context = "\n\n".join([r["text"] for r in retrieved])
        answer, prompt = experiment._generate_answer(question, context, return_prompt=True)

        # Save
        gold_segments, gold_text = experiment._prepare_gold_evidence(sample.get("evidence"))
        results[i]["gold_evidence"] = gold_text
        results[i]["gold_evidence_segments"] = gold_segments
        results[i]["retrieved_chunks"] = retrieved
        results[i]["num_retrieved"] = len(retrieved)
        results[i]["generated_answer"] = answer
        results[i]["final_prompt"] = prompt

        experiment.notify_sample_complete(1)

    return results