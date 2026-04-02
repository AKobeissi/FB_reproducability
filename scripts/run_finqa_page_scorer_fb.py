#!/usr/bin/env python3
"""
scripts/run_finqa_page_scorer_fb.py
=====================================
Evaluates the FinQA-trained page scorer on FinanceBench 150 questions
over a global index of all 84 PDFs.

max_seq_length=2048 for page scoring
--------------------------------------
Consistent with training. Financial pages average ~500 tokens (p95 ~1500),
so 2048 covers virtually all pages. The previous 4096 caused OOM during
training due to O(n²) attention scaling.

At inference, BGE-M3 runs one page at a time in a batch (no MNR constraint),
so the memory pressure is much lower — but we keep 2048 for training/inference
consistency, which matters for embedding quality.

Pipeline
--------
Stage 1  Global bi-encoder (fine-tuned BGE-M3, max_seq=2048)
         top page_k=30 pages across all 84 documents

Stage 2  Cross-encoder page reranking (BGE-reranker-v2-m3)
         top rerank_k=20

Stage 3  Token-based chunk retrieval (1024 tok / 128 overlap, Paper 1 def)
         top chunk_k=5 → local Qwen2.5-7B-Instruct generation
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from pypdf import PdfReader
except ImportError:
    logger.error("pypdf not found — pip install pypdf"); sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
except ImportError:
    logger.error("sentence-transformers not found"); sys.exit(1)

try:
    from peft import PeftModel
except ImportError:
    logger.error("peft not found — pip install peft"); sys.exit(1)

try:
    import faiss
except ImportError:
    logger.error("faiss not found — pip install faiss-cpu"); sys.exit(1)

try:
    from src.ingestion.data_loader import FinanceBenchLoader
    from src.evaluation.retrieval_evaluator import RetrievalEvaluator
    from src.evaluation.generative_evaluator import GenerativeEvaluator
except ImportError as e:
    logger.error(f"Project import failed: {e}\nRun from repo root."); sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Constants — must match training
# ─────────────────────────────────────────────────────────────────────────────

BGE_MAX_SEQ_LENGTH = 2048   # matches training; covers ~99% of financial pages

_WS = re.compile(r"\s+")


def normalise(text: str) -> str:
    return _WS.sub(" ", (text or "").strip())


# ─────────────────────────────────────────────────────────────────────────────
# 1.  PDF page extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_pdf_pages(pdf_path: Path) -> List[str]:
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as e:
        logger.warning(f"Cannot open {pdf_path.name}: {e}")
        return []
    pages = []
    for page in reader.pages:
        try:
            raw = page.extract_text() or ""
        except Exception:
            raw = ""
        pages.append(normalise(raw))
    return pages


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Token-based chunking (Paper 1: 1024 tokens, 128 overlap)
# ─────────────────────────────────────────────────────────────────────────────

def chunk_text_by_tokens(
    text: str,
    tokenizer,
    chunk_tokens: int = 1024,
    overlap_tokens: int = 128,
) -> List[str]:
    if not text.strip():
        return []
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return []
    step   = max(1, chunk_tokens - overlap_tokens)
    chunks = []
    for start in range(0, len(token_ids), step):
        decoded = tokenizer.decode(
            token_ids[start : start + chunk_tokens],
            skip_special_tokens=True,
        )
        if decoded.strip():
            chunks.append(decoded)
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Load fine-tuned scorer (base BGE-M3 + LoRA merged)
# ─────────────────────────────────────────────────────────────────────────────

def load_scorer_model(model_path: Path, device: str) -> SentenceTransformer:
    adapter_path = model_path / "adapter"
    if not adapter_path.exists():
        logger.info(f"No adapter/ found — loading as full ST model: {model_path}")
        model = SentenceTransformer(str(model_path), device=device)
        model.max_seq_length = BGE_MAX_SEQ_LENGTH
        return model

    logger.info(f"Loading BGE-M3 + LoRA from {adapter_path} ...")
    base_st    = SentenceTransformer("BAAI/bge-m3", device=device)
    base_hf    = base_st[0].auto_model
    peft_model = PeftModel.from_pretrained(base_hf, str(adapter_path))
    merged     = peft_model.merge_and_unload()
    base_st[0].auto_model = merged

    # Must match training
    base_st.max_seq_length = BGE_MAX_SEQ_LENGTH
    logger.info(f"✓ LoRA merged. max_seq_length={BGE_MAX_SEQ_LENGTH}")
    return base_st


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Global page index (FAISS flat-IP)
# ─────────────────────────────────────────────────────────────────────────────

def build_or_load_page_index(
    docs: List[Dict],
    model: SentenceTransformer,
    cache_dir: Path,
    model_tag: str = "finqa_scorer",
    batch_size: int = 64,
) -> Tuple[faiss.Index, List[Dict]]:
    """
    Cached FAISS index over all documents.  Delete cache if model changes.
    Page texts stored full (no truncation) for chunking.
    Embeddings use max_seq_length=2048 (consistent with training).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / f"page_index_{model_tag}.faiss"
    meta_path  = cache_dir / f"page_meta_{model_tag}.json"

    if index_path.exists() and meta_path.exists():
        logger.info(f"Loading cached page index from {cache_dir} ...")
        index = faiss.read_index(str(index_path))
        with open(meta_path, encoding="utf-8") as f:
            page_meta = json.load(f)
        n_docs = len({m["doc_name"] for m in page_meta})
        logger.info(f"  {index.ntotal} pages ({n_docs} docs) loaded from cache")
        return index, page_meta

    logger.info(f"Building global page index over {len(docs)} documents ...")
    page_meta:  List[Dict] = []
    page_texts: List[str]  = []

    for entry in tqdm(docs, desc="Extracting pages"):
        doc_name = entry["doc_name"]
        pages    = extract_pdf_pages(Path(entry["pdf_path"]))
        for page_idx, text in enumerate(pages):
            page_meta.append({"doc_name": doc_name, "page": page_idx, "text": text})
            page_texts.append(text)

    logger.info(
        f"Encoding {len(page_texts)} pages "
        f"(max_seq_length={model.max_seq_length}) ..."
    )
    embeddings = model.encode(
        page_texts,
        batch_size           = batch_size,
        show_progress_bar    = True,
        normalize_embeddings = True,
        convert_to_numpy     = True,
    ).astype(np.float32)

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(index_path))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(page_meta, f)

    logger.info(f"✓ Page index: {index.ntotal} pages, dim={dim}")
    return index, page_meta


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Stage 1 — Global bi-encoder retrieval
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_pages_global(
    query: str,
    model: SentenceTransformer,
    index: faiss.Index,
    page_meta: List[Dict],
    page_k: int = 30,
    query_prefix: str = "",
) -> List[Dict]:
    encoded_query = f"{query_prefix}{query}" if query_prefix else query
    q_emb = model.encode(
        [encoded_query], normalize_embeddings=True, convert_to_numpy=True,
    ).astype(np.float32)
    scores, indices = index.search(q_emb, page_k)
    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
        if idx < 0:
            continue
        meta = page_meta[idx]
        results.append({
            "doc_name": meta["doc_name"],
            "page":     meta["page"],
            "text":     meta["text"],
            "score":    float(score),
            "rank":     rank + 1,
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Stage 2 — Cross-encoder page reranking
# ─────────────────────────────────────────────────────────────────────────────

def rerank_pages(
    query: str,
    pages: List[Dict],
    reranker: CrossEncoder,
    rerank_k: int = 20,
    batch_size: int = 32,
) -> List[Dict]:
    if not pages:
        return pages
    # Cross-encoder processes text independently — cap at 4000 chars for safety
    pairs  = [[query, p["text"][:4000]] for p in pages]
    scores = reranker.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    for page, score in zip(pages, scores):
        page["rerank_score"] = float(score)
    reranked = sorted(pages, key=lambda x: x["rerank_score"], reverse=True)
    for i, p in enumerate(reranked):
        p["rank"] = i + 1
    return reranked[:rerank_k]


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Stage 3 — Token-based chunk retrieval
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_chunks_from_pages(
    query: str,
    pages: List[Dict],
    model: SentenceTransformer,
    tokenizer,
    chunk_tokens: int = 1024,
    overlap_tokens: int = 128,
    chunk_k: int = 5,
) -> List[Dict]:
    all_chunks: List[Dict] = []
    for page in pages:
        parts = chunk_text_by_tokens(page["text"], tokenizer, chunk_tokens, overlap_tokens)
        for chunk_idx, part in enumerate(parts):
            all_chunks.append({
                "text":       part,
                "doc_name":   page["doc_name"],
                "page":       page["page"],
                "chunk_idx":  chunk_idx,
                "page_score": page.get("rerank_score") or page.get("score", 0.0),
            })

    if not all_chunks:
        return []

    chunk_embs = model.encode(
        [c["text"] for c in all_chunks],
        batch_size=256, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    ).astype(np.float32)

    q_emb  = model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True,
    ).astype(np.float32)

    scores = (chunk_embs @ q_emb.T).squeeze()
    if scores.ndim == 0:
        scores = np.array([float(scores)])

    top_idx = np.argsort(scores)[::-1][:chunk_k]
    results = []
    for rank, idx in enumerate(top_idx):
        chunk = dict(all_chunks[idx])
        chunk["score"] = float(scores[idx])
        chunk["rank"]  = rank + 1
        results.append(chunk)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Gold evidence extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_gold_evidence(ev_list) -> str:
    if hasattr(ev_list, "tolist"):
        ev_list = ev_list.tolist()
    texts = []
    for ev in (ev_list or []):
        if not isinstance(ev, dict):
            continue
        t = (
            ev.get("evidence_text_full_page")
            or ev.get("evidence_text")
            or ev.get("text")
            or ""
        )
        if str(t).strip():
            texts.append(str(t).strip())
    return " ".join(texts)


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Metrics — identical to Paper 1 definition
# ─────────────────────────────────────────────────────────────────────────────

def compute_aggregate_metrics(results: List[Dict], k_values: List[int]) -> Dict:
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        smoother  = SmoothingFunction().method1
        _has_bleu = True
    except ImportError:
        _has_bleu = False

    try:
        from rouge_score import rouge_scorer as rouge_lib
        rscorer    = rouge_lib.RougeScorer(["rougeL"], use_stemmer=False)
        _has_rouge = True
    except ImportError:
        _has_rouge = False

    doc_hits    = defaultdict(list)
    page_hits   = defaultdict(list)
    page_recall = defaultdict(list)
    mrr_vals    = defaultdict(list)
    chunk_bleu  = defaultdict(list)
    chunk_rouge = defaultdict(list)

    for r in results:
        gold_doc   = r.get("doc_name", "")
        gold_pgs   = set(r.get("gold_pages", []))
        gold_ev    = r.get("gold_evidence", "").strip()
        retrieved  = r.get("retrieved_pages", [])
        chunks     = r.get("retrieved_chunks", [])
        gold_pairs = {(gold_doc, gp) for gp in gold_pgs}

        for k in k_values:
            top_k_pages  = retrieved[:k]
            top_k_chunks = chunks[:k]

            # DocRec@k
            doc_hits[k].append(
                int(any(p.get("doc_name") == gold_doc for p in top_k_pages))
            )

            # PageRec@k
            ret_pairs = {(p.get("doc_name"), p.get("page")) for p in top_k_pages}
            matched   = gold_pairs & ret_pairs
            page_hits[k].append(1 if matched else 0)
            page_recall[k].append(
                len(matched) / len(gold_pairs) if gold_pairs else 0.0
            )

            # MRR@k
            mrr = 0.0
            for rank, p in enumerate(top_k_pages, 1):
                if (p.get("doc_name"), p.get("page")) in gold_pairs:
                    mrr = 1.0 / rank
                    break
            mrr_vals[k].append(mrr)

            # MaxBLEU / MaxROUGE-L: max over top-k chunks vs gold evidence
            # This matches Paper 1 Fig 1 / Table 3 exactly.
            if gold_ev and top_k_chunks:
                ref      = gold_ev.lower()
                b_scores = []
                r_scores = []
                for c in top_k_chunks:
                    hyp = c.get("text", "").lower()
                    if not hyp:
                        continue
                    if _has_bleu:
                        b_scores.append(sentence_bleu(
                            [ref.split()], hyp.split(),
                            smoothing_function=smoother,
                        ))
                    if _has_rouge:
                        r_scores.append(
                            rscorer.score(ref, hyp)["rougeL"].fmeasure
                        )
                if b_scores:
                    chunk_bleu[k].append(max(b_scores))
                if r_scores:
                    chunk_rouge[k].append(max(r_scores))

    agg: Dict = {}
    for k in k_values:
        agg[f"DocRec@{k}"]       = float(np.mean(doc_hits[k]))    if doc_hits[k]    else 0.0
        agg[f"PageHit@{k}"]      = float(np.mean(page_hits[k]))   if page_hits[k]   else 0.0
        agg[f"PageRec@{k}"]      = float(np.mean(page_recall[k])) if page_recall[k] else 0.0
        agg[f"MRR@{k}"]          = float(np.mean(mrr_vals[k]))    if mrr_vals[k]    else 0.0
        if chunk_bleu[k]:
            agg[f"MaxBLEU@{k}"]    = float(np.mean(chunk_bleu[k]))
        if chunk_rouge[k]:
            agg[f"MaxROUGE-L@{k}"] = float(np.mean(chunk_rouge[k]))

    # Generation metrics
    if _has_bleu and _has_rouge:
        gen_bleu, gen_rouge, gen_num = [], [], []
        from rouge_score import rouge_scorer as rouge_lib
        rsc = rouge_lib.RougeScorer(["rougeL"], use_stemmer=False)
        sm  = SmoothingFunction().method1
        for r in results:
            pred = r.get("generated_answer", "").lower().strip()
            ref  = r.get("reference_answer", "").lower().strip()
            if pred and ref:
                gen_bleu.append(sentence_bleu([ref.split()], pred.split(), smoothing_function=sm))
                gen_rouge.append(rsc.score(ref, pred)["rougeL"].fmeasure)
                mp = re.search(r"([\d,]+\.?\d*)\s*%?", pred)
                mr = re.search(r"([\d,]+\.?\d*)\s*%?", ref)
                if mp and mr:
                    try:
                        pn = float(mp.group(1).replace(",", ""))
                        rn = float(mr.group(1).replace(",", ""))
                        if rn != 0:
                            gen_num.append(int(abs(pn - rn) / abs(rn) <= 0.05))
                    except (ValueError, ZeroDivisionError):
                        pass
        if gen_bleu:
            agg["Gen_BLEU-4"]   = float(np.mean(gen_bleu))
            agg["Gen_ROUGE-L"]  = float(np.mean(gen_rouge))
        if gen_num:
            agg["NumericMatch"] = float(np.mean(gen_num))

    return agg


# ─────────────────────────────────────────────────────────────────────────────
# 10.  Local HF generator
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(question: str, context: str, max_context_chars: int = 16000) -> str:
    return (
        f"Answer this question: {question}\n"
        f"Here is the relevant filing that you need to answer the question:\n"
        f"[START OF FILING]\n{context[:max_context_chars]}\n[END OF FILING]"
    )


def init_local_generator(llm_model: str, device: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    logger.info(f"Loading local LLM: {llm_model} ...")
    tok = AutoTokenizer.from_pretrained(llm_model, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        llm_model,
        torch_dtype       = torch.float16 if device == "cuda" else torch.float32,
        device_map        = "auto" if device == "cuda" else None,
        trust_remote_code = True,
    )
    pipe = pipeline(
        "text-generation", model=mdl, tokenizer=tok,
        max_new_tokens=256, do_sample=False, return_full_text=False,
    )
    logger.info("✓ Local LLM loaded")

    def generate(prompt: str) -> str:
        try:
            out = pipe(prompt)
            return (out[0].get("generated_text", "") if out else "").strip()
        except Exception as e:
            logger.warning(f"Generation error: {e}")
            return ""

    return generate, mdl, tok


# ─────────────────────────────────────────────────────────────────────────────
# 11.  Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Device: {device}")

    scorer    = load_scorer_model(Path(args.model_path), device)
    tokenizer = scorer.tokenizer

    reranker = None
    if args.use_reranker:
        logger.info(f"Loading cross-encoder: {args.reranker_model} ...")
        reranker = CrossEncoder(args.reranker_model, device=device, trust_remote_code=True)
        logger.info("✓ Cross-encoder loaded")

    logger.info("Loading FinanceBench dataset ...")
    df = FinanceBenchLoader().load_data()

    data = []
    for idx, (_, row) in enumerate(df.iterrows()):
        ev_list = row.get("evidence", [])
        if hasattr(ev_list, "tolist"):
            ev_list = ev_list.tolist()
        gold_pages = set()
        for ev in (ev_list or []):
            if isinstance(ev, dict):
                p = ev.get("evidence_page_num") or ev.get("page_ix") or ev.get("page")
                if p is not None:
                    gold_pages.add(int(p))
        data.append({
            "sample_id":          idx,
            "doc_name":           row["doc_name"],
            "question":           row["question"],
            "reference_answer":   row.get("answer", ""),
            "question_type":      row.get("question_type", ""),
            "question_reasoning": row.get("question_reasoning", ""),
            "gold_pages":         sorted(gold_pages),
            "gold_evidence":      extract_gold_evidence(ev_list),
        })
    logger.info(f"  {len(data)} questions loaded")

    pdf_dir     = Path(args.pdf_dir)
    unique_docs = list({d["doc_name"] for d in data})
    docs        = []
    for doc_name in unique_docs:
        p = pdf_dir / f"{doc_name}.pdf"
        if p.exists():
            docs.append({"doc_name": doc_name, "pdf_path": str(p)})
        else:
            logger.warning(f"  PDF not found: {doc_name}")
    logger.info(f"  {len(docs)} PDFs found")

    cache_dir  = Path(args.output_dir) / "index_cache"
    page_index, page_meta = build_or_load_page_index(
        docs, scorer, cache_dir,
        model_tag  = Path(args.model_path).name,
        batch_size = args.index_batch_size,
    )

    generator, llm_obj, llm_tok = init_local_generator(args.llm, device)

    results = []
    for sample in tqdm(data, desc="Running FinanceBench"):
        question = sample["question"]

        top_pages = retrieve_pages_global(
            question, scorer, page_index, page_meta, args.page_k,
            query_prefix=args.query_prefix,
        )

        if reranker and top_pages:
            top_pages = rerank_pages(question, top_pages, reranker, args.rerank_k)
        else:
            top_pages = top_pages[:args.rerank_k]

        top_chunks = retrieve_chunks_from_pages(
            question, top_pages, scorer, tokenizer,
            chunk_tokens=args.chunk_tokens, overlap_tokens=args.overlap_tokens,
            chunk_k=args.chunk_k,
        )

        context = "\n\n".join(c["text"] for c in top_chunks)
        prompt  = build_prompt(question, context, args.max_context_chars)
        answer  = generator(prompt)

        results.append({
            "sample_id":          sample["sample_id"],
            "doc_name":           sample["doc_name"],
            "question":           question,
            "reference_answer":   sample["reference_answer"],
            "question_type":      sample["question_type"],
            "question_reasoning": sample["question_reasoning"],
            "gold_evidence":      sample["gold_evidence"],
            "gold_pages":         sample["gold_pages"],
            "generated_answer":   answer,
            "final_prompt":       prompt,
            "context_length":     len(context),
            "generation_length":  len(answer),
            "num_retrieved":      len(top_chunks),
            "experiment_type":    "finqa_learned_page_scorer",
            "vector_store_type":  "faiss_global",
            "pdf_source":         str(pdf_dir),
            "retrieved_pages": [
                {"doc_name": p["doc_name"], "page": p["page"],
                 "score": p.get("rerank_score") or p.get("score", 0.0),
                 "rank": p.get("rank", 0)}
                for p in top_pages
            ],
            "retrieved_chunks": [
                {"rank": c["rank"], "text": c["text"], "score": c["score"],
                 "metadata": {"doc_name": c["doc_name"], "page": c["page"]}}
                for c in top_chunks
            ],
        })

    k_values = [1, 3, 5, 10, 20]
    agg      = compute_aggregate_metrics(results, k_values)

    logger.info("\n" + "=" * 66)
    logger.info("  AGGREGATE RESULTS — FinanceBench 150")
    logger.info("=" * 66)
    for metric, val in sorted(agg.items()):
        logger.info(f"  {metric:<26}: {val:.4f}")
    logger.info("=" * 66)

    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"finqa_scorer_{timestamp}.json"

    payload = {
        "metadata": {
            "experiment_type":  "finqa_learned_page_scorer",
            "model_path":       args.model_path,
            "max_seq_length":   BGE_MAX_SEQ_LENGTH,
            "llm":              args.llm,
            "page_k":           args.page_k,
            "rerank_k":         args.rerank_k,
            "use_reranker":     args.use_reranker,
            "reranker_model":   args.reranker_model,
            "chunk_k":          args.chunk_k,
            "chunk_tokens":     args.chunk_tokens,
            "overlap_tokens":   args.overlap_tokens,
            "n_questions":      len(results),
            "n_docs":           len(docs),
            "created_at":       timestamp,
        },
        "aggregate_stats": agg,
        "num_samples":     len(results),
        "framework": (
            "FinQA LoRA scorer (BGE-M3, max_seq=2048) + "
            "cross-encoder reranking + token-based chunks"
        ),
        "results": results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"\n✓ Raw results: {out_path}")

    try:
        ret_eval    = RetrievalEvaluator()
        ret_metrics = ret_eval.compute_metrics(results, k_values=k_values)
        payload["evaluation_summary"] = {"retrieval": ret_metrics}
    except Exception as e:
        logger.warning(f"RetrievalEvaluator: {e}")

    try:
        gen_eval = GenerativeEvaluator(use_bertscore=False)
        for r in results:
            r["generative_metrics"] = gen_eval.evaluate_sample(r)
        gen_agg = {}
        keys = {k for r in results for k in r.get("generative_metrics", {})}
        for k in keys:
            vals = [r["generative_metrics"][k] for r in results
                    if isinstance(r.get("generative_metrics", {}).get(k), (int, float))]
            if vals:
                gen_agg[f"avg_{k}"] = float(np.mean(vals))
        payload.setdefault("evaluation_summary", {})["generative"] = gen_agg
    except Exception as e:
        logger.warning(f"GenerativeEvaluator: {e}")

    scored_path = out_path.with_name(out_path.stem + "_scored.json")
    with open(scored_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"✓ Scored results: {scored_path}")

    del llm_obj, llm_tok
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# 12.  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path",        default="models/finqa_page_scorer_lora")
    parser.add_argument("--pdf-dir",           default="pdfs")
    parser.add_argument("--output-dir",        default="outputs/finqa_page_scorer")
    parser.add_argument("--page-k",            type=int, default=30)
    parser.add_argument("--rerank-k",          type=int, default=20)
    parser.add_argument("--use-reranker",      action="store_true", default=True)
    parser.add_argument("--no-reranker",       dest="use_reranker", action="store_false")
    parser.add_argument("--reranker-model",    default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--chunk-k",           type=int, default=5)
    parser.add_argument("--chunk-tokens",      type=int, default=1024)
    parser.add_argument("--overlap-tokens",    type=int, default=128)
    parser.add_argument("--max-context-chars", type=int, default=16000)
    parser.add_argument("--llm",               default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--index-batch-size",  type=int, default=64)
    parser.add_argument("--query-prefix",      type=str, default="",
                        help="Instruction prefix prepended to queries before encoding. "
                             "Must match the prefix used during training. "
                             "BGE-M3 recommended: 'Represent this sentence for searching relevant passages: '")
    args = parser.parse_args()

    if not Path(args.pdf_dir).exists():
        logger.error(f"--pdf-dir not found: {args.pdf_dir}"); sys.exit(1)
    run(args)


if __name__ == "__main__":
    main()