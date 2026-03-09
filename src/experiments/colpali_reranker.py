"""
colpali_reranker.py
===================
ColPali-as-reranker experiment for FinanceBench.

Pipeline
--------
Stage 1 (BGE-M3 dense):   All pages  →  top-M candidates   (high recall, fast)
Stage 2 (ColPali visual):  top-M pages →  reranked top-K    (visual precision)
Stage 3 (Generation):      top-K pages text  →  Qwen answer

Key improvements over flat ColPali retrieval
---------------------------------------------
- Only renders/embeds ~20 pages per query instead of 4000  → 200× faster
- ColPali VRAM freed before generation                     → no OOM
- Vectorised ColBERT scoring (einsum, no python loop)      → ms not hours
- Weighted RRF fusion (BGE-M3 + ColPali)                   → best of both
- Page-complete context truncation                         → no mid-sentence cuts
- Full page-recall tracking @1/3/5                         → oracle framework compatible
- FinanceBench evidence_page_num offset fix (1-idx → 0-idx)
"""

from __future__ import annotations

import gc
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    import fitz
    if not hasattr(fitz, "open"):
        raise ImportError
except Exception:
    import pymupdf as fitz

try:
    from PIL import Image
except ImportError:
    Image = None

from src.ingestion.pdf_utils import _find_local_pdf
from src.ingestion.page_processor import extract_pages_from_pdf, make_page_digest

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_BGE_TOP_M   = 20    # pages fed to ColPali after BGE-M3 first-stage
DEFAULT_COLPALI_DPI = 150   # render DPI for ColPali image input
DEFAULT_TOP_K       = 5     # final pages passed to the LLM
DEFAULT_ALPHA       = 0.35  # fusion weight: 0 = pure ColPali, 1 = pure BGE-M3
IMAGE_BATCH_SIZE    = 8     # images per ColPali forward pass
QUERY_BATCH_SIZE    = 8     # queries per ColPali forward pass
MAX_CONTEXT_CHARS   = 16000 # generous limit; page-complete truncation used


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _get_device() -> Tuple[torch.device, str, torch.dtype]:
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda:0", torch.bfloat16
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps"), "mps", torch.float32
    return torch.device("cpu"), "cpu", torch.float32


def _load_colpali_engine():
    try:
        from colpali_engine.models import ColPali, ColPaliProcessor
        return ColPali, ColPaliProcessor
    except ImportError as e:
        raise RuntimeError(
            "ColPali not installed. Run: pip install 'colpali-engine>=0.3.0,<0.4.0'"
        ) from e


def _render_page_image(pdf_path: str, page_idx: int, dpi: int) -> Optional[Image.Image]:
    """Render a single PDF page to a PIL RGB image."""
    if Image is None:
        raise RuntimeError("Pillow is required for ColPali page rendering.")
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_idx]
        pix = page.get_pixmap(dpi=dpi)
        mode = "RGBA" if pix.alpha else "RGB"
        img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        doc.close()
        if mode == "RGBA":
            img = img.convert("RGB")
        return img
    except Exception as exc:
        logger.warning("Failed to render page %d of %s: %s", page_idx, pdf_path, exc)
        return None


def _load_page_text_structured(pdf_path: str, page_idx: int) -> str:
    """
    Extract page text with basic table structure preservation.
    Falls back to raw PyMuPDF text if pdfplumber is unavailable.
    """
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            if page_idx >= len(pdf.pages):
                return ""
            p = pdf.pages[page_idx]
            raw = p.extract_text() or ""
            tables = p.extract_tables() or []
            if not tables:
                return raw
            # Format tables as markdown-style grids
            table_strs = []
            for tbl in tables:
                rows = []
                for row in tbl:
                    cells = [str(c or "").strip() for c in row]
                    rows.append(" | ".join(cells))
                table_strs.append("\n".join(rows))
            return raw + "\n\n" + "\n\n".join(table_strs)
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("pdfplumber failed on page %d: %s", page_idx, exc)

    # Fallback: PyMuPDF
    try:
        doc = fitz.open(pdf_path)
        if page_idx >= len(doc):
            return ""
        text = doc[page_idx].get_text("text") or ""
        doc.close()
        return text
    except Exception as exc:
        logger.warning("Failed to extract text from page %d of %s: %s", page_idx, pdf_path, exc)
        return ""


def _collect_target_pdfs(
    experiment, data: List[Dict[str, Any]]
) -> List[Tuple[str, Path]]:
    """Return deduplicated list of (doc_name, pdf_path) relevant to data."""
    pdf_dir = Path(getattr(experiment, "pdf_local_dir", "pdfs"))
    targets: List[Tuple[str, Path]] = []
    seen: set = set()

    if getattr(experiment, "use_all_pdfs", False):
        for path in sorted(pdf_dir.glob("**/*.pdf")):
            key = path.stem.lower()
            if key not in seen:
                targets.append((path.stem, path))
                seen.add(key)
        return targets

    for sample in data:
        doc_name = sample.get("doc_name") or sample.get("document") or ""
        if not doc_name:
            continue
        key = doc_name.lower()
        if key in seen:
            continue
        local = _find_local_pdf(doc_name, str(pdf_dir))
        if local is None:
            logger.warning("PDF not found for doc_name='%s'", doc_name)
            continue
        targets.append((doc_name, local))
        seen.add(key)

    return targets


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1 — BGE-M3 dense page ranking
# ──────────────────────────────────────────────────────────────────────────────

def _bge_score_pages(
    experiment,
    question: str,
    page_records: List[Dict[str, Any]],
    top_m: int,
) -> List[Tuple[int, float]]:
    """
    Score all page records against the question using the experiment's
    existing embedding model (BGE-M3 or any SentenceTransformer).

    Returns list of (page_record_index, cosine_score) sorted descending,
    length = min(top_m, len(page_records)).
    """
    embeddings = getattr(experiment, "embeddings", None)
    if embeddings is None:
        # No embedding model — return all pages with equal score
        n = min(top_m, len(page_records))
        return [(i, 1.0) for i in range(n)]

    # Embed question
    try:
        q_vec = embeddings.embed_query(question)
    except Exception:
        try:
            q_vec = embeddings.encode(question, normalize_embeddings=True)
        except Exception as exc:
            logger.warning("embed_query failed: %s — using uniform scores", exc)
            return [(i, 1.0) for i in range(min(top_m, len(page_records)))]

    q_vec = np.asarray(q_vec, dtype=np.float32)
    norm = np.linalg.norm(q_vec)
    if norm > 1e-8:
        q_vec /= norm

    # Embed page digests in batches
    EMBED_BATCH = 32
    digests = [p.get("digest") or make_page_digest(p.get("text", "")) for p in page_records]
    all_doc_vecs = []

    for start in range(0, len(digests), EMBED_BATCH):
        batch = digests[start : start + EMBED_BATCH]
        try:
            vecs = embeddings.embed_documents(batch)
        except Exception:
            try:
                vecs = embeddings.encode(batch, normalize_embeddings=True)
            except Exception as exc:
                logger.warning("embed_documents failed: %s", exc)
                vecs = np.zeros((len(batch), len(q_vec)), dtype=np.float32)
        all_doc_vecs.append(np.asarray(vecs, dtype=np.float32))

    doc_matrix = np.vstack(all_doc_vecs)  # [N, dim]

    # Normalise rows
    norms = np.linalg.norm(doc_matrix, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    doc_matrix /= norms

    scores = doc_matrix @ q_vec  # [N]
    top_n = min(top_m, len(scores))
    top_indices = np.argpartition(scores, -top_n)[-top_n:]
    top_indices = top_indices[np.argsort(-scores[top_indices])]

    return [(int(i), float(scores[i])) for i in top_indices]


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 — ColPali visual reranking
# ──────────────────────────────────────────────────────────────────────────────

def _vectorised_colbert_scores(
    query_emb: torch.Tensor,          # [Q_tok, dim]
    page_embs: torch.Tensor,          # [N, P_tok, dim]
) -> torch.Tensor:                    # [N]
    """
    Vectorised ColBERT max-sim: for each query token find its best matching
    page token, then sum across query tokens.
    No Python loops over pages.
    """
    # Ensure consistent dtype (ColPali model outputs BFloat16)
    query_emb = query_emb.to(dtype=page_embs.dtype)
    # sim: [Q_tok, N, P_tok]
    sim = torch.einsum("qd,npd->qnp", query_emb, page_embs)
    # max over page tokens → [Q_tok, N]
    max_sim = sim.max(dim=-1).values
    # sum over query tokens → [N]
    return max_sim.sum(dim=0)


def _embed_images_batched(
    model, processor, images: List[Image.Image], device: torch.device, batch_size: int = IMAGE_BATCH_SIZE
) -> List[torch.Tensor]:
    """Embed a list of PIL images; returns list of CPU tensors [P_tok, dim]."""
    from torch.utils.data import DataLoader

    loader = DataLoader(
        images,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: processor.process_images(x),
    )
    result: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            embs = model(**batch)
            result.extend(list(torch.unbind(embs.cpu())))
    return result


def _embed_queries_batched(
    model, processor, queries: List[str], device: torch.device, batch_size: int = QUERY_BATCH_SIZE
) -> List[torch.Tensor]:
    """Embed a list of query strings; returns list of CPU tensors [Q_tok, dim]."""
    from torch.utils.data import DataLoader

    loader = DataLoader(
        queries,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: processor.process_queries(x),
    )
    result: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            embs = model(**batch)
            result.extend(list(torch.unbind(embs.cpu())))
    return result


def _minmax_norm(t: torch.Tensor) -> torch.Tensor:
    mn, mx = t.min(), t.max()
    if (mx - mn).item() < 1e-8:
        return torch.zeros_like(t)
    return (t - mn) / (mx - mn)


def _colpali_rerank_candidates(
    model,
    processor,
    device: torch.device,
    dpi: int,
    question: str,
    candidates: List[Dict[str, Any]],   # each has pdf_path, page, bge_score
    top_k: int,
    alpha: float,
) -> List[Dict[str, Any]]:
    """
    Visually rerank a small candidate set using ColPali.
    Fuses ColPali score with BGE-M3 score via weighted sum after MinMax normalisation.
    Returns top_k candidates with score fields attached.
    """
    if not candidates:
        return candidates

    # Render pages
    images, valid_candidates = [], []
    for cand in candidates:
        img = _render_page_image(cand["pdf_path"], cand["page"], dpi)
        if img is not None:
            images.append(img)
            valid_candidates.append(cand)
        else:
            logger.debug("Skipping page %d (render failed)", cand["page"])

    if not images:
        logger.warning("ColPali reranker: no images rendered, returning BGE-M3 ranking")
        return candidates[:top_k]

    # Embed the single query
    q_embs = _embed_queries_batched(model, processor, [question], device)
    q_emb = q_embs[0].to(device)  # [Q_tok, dim]

    # Embed candidate pages — small batch, fits in VRAM easily
    page_embs_list = _embed_images_batched(model, processor, images, device)

    # Pad to uniform P_tok length for vectorised scoring
    max_p = max(e.shape[0] for e in page_embs_list)
    dim   = page_embs_list[0].shape[1]
    N     = len(page_embs_list)

    page_stack = torch.zeros(N, max_p, dim)
    for i, e in enumerate(page_embs_list):
        page_stack[i, :e.shape[0]] = e
    page_stack = page_stack.to(device)  # [N, max_p, dim]

    with torch.no_grad():
        colpali_scores = _vectorised_colbert_scores(q_emb, page_stack).cpu()  # [N]

    # Fuse with BGE-M3 scores
    bge_scores = torch.tensor(
        [c.get("bge_score", 0.0) for c in valid_candidates], dtype=torch.float32
    )
    colpali_norm = _minmax_norm(colpali_scores)
    bge_norm     = _minmax_norm(bge_scores)
    fused        = alpha * bge_norm + (1.0 - alpha) * colpali_norm  # [N]

    # Attach scores and sort
    ranked = []
    for i, cand in enumerate(valid_candidates):
        c = dict(cand)
        c["colpali_score"]      = float(colpali_scores[i].item())
        c["colpali_score_norm"] = float(colpali_norm[i].item())
        c["bge_score_norm"]     = float(bge_norm[i].item())
        c["fused_score"]        = float(fused[i].item())
        ranked.append(c)

    ranked.sort(key=lambda x: -x["fused_score"])
    return ranked[:top_k]


# ──────────────────────────────────────────────────────────────────────────────
# Page recall helpers
# ──────────────────────────────────────────────────────────────────────────────

def _extract_gold_pages_0indexed(gold_segments: List[Dict[str, Any]]) -> set:
    """
    Pull 0-indexed page numbers out of gold evidence segments.
    FinanceBench evidence_page_num is 1-indexed, so we subtract 1.
    """
    pages = set()
    for seg in gold_segments:
        p = seg.get("page")
        if p is None:
            continue
        try:
            p_int = int(p)
            # FinanceBench stores 1-indexed page numbers
            pages.add(p_int - 1)
        except (ValueError, TypeError):
            pass
    return pages


def _compute_page_recall(
    retrieved: List[Dict[str, Any]],
    gold_pages: set,
    k_values: List[int] = (1, 3, 5),
) -> Dict[str, Any]:
    """Compute page hit@k for multiple k values."""
    if not gold_pages:
        return {f"page_hit@{k}": None for k in k_values}

    results = {}
    for k in k_values:
        top_k_pages = {r["page"] for r in retrieved[:k]}
        results[f"page_hit@{k}"] = bool(top_k_pages & gold_pages)

    results["gold_pages_0indexed"]    = sorted(gold_pages)
    results["retrieved_page_nums"]    = [r["page"] for r in retrieved]
    results["page_recall_hit"]        = results.get("page_hit@5", False)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Context assembly
# ──────────────────────────────────────────────────────────────────────────────

def _assemble_context(
    pages: List[Dict[str, Any]],
    max_chars: int = MAX_CONTEXT_CHARS,
) -> str:
    """
    Concatenate full page texts up to budget, respecting page boundaries.
    Never cuts mid-page. If a single page exceeds the budget, truncates only that page.
    """
    parts, running = [], 0
    for p in pages:
        text = p.get("text", "")
        if not text:
            continue
        if running + len(text) > max_chars:
            remaining = max_chars - running
            if remaining > 200:                   # worth adding a partial page
                parts.append(text[:remaining])
            break
        parts.append(text)
        running += len(text)

    return "\n\n---\n\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Prompt & generation
# ──────────────────────────────────────────────────────────────────────────────

def _build_prompt(question: str, context: str) -> str:
    return (
        "You are a precise financial analyst. Answer using ONLY information "
        "in the context below. If the answer is not present, say 'Not found in context'.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def _generate(experiment, prompt: str) -> str:
    if getattr(experiment, "use_api", False):
        experiment._initialize_llm()
        messages = [
            {"role": "system", "content": "You are a helpful financial analyst. Answer strictly based on the context."},
            {"role": "user",   "content": prompt},
        ]
        try:
            resp = experiment.api_client.chat.completions.create(
                model=experiment.llm_model_name,
                messages=messages,
                max_tokens=getattr(experiment, "max_new_tokens", 512),
                temperature=0.0,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.error("API generation failed: %s", exc)
            return ""

    experiment._initialize_llm()
    try:
        outputs = experiment.llm_pipeline(
            prompt,
            max_new_tokens=getattr(experiment, "max_new_tokens", 512),
            do_sample=False,
        )
    except TypeError:
        outputs = experiment.llm_pipeline(prompt)

    if not outputs:
        return ""
    raw = outputs[0].get("generated_text", "")
    # Strip the prompt prefix if the pipeline echoes it
    if raw.startswith(prompt):
        raw = raw[len(prompt):]
    return raw.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_colpali_reranker(
    experiment,
    data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Full two-stage retrieval: BGE-M3 page recall → ColPali visual reranking.

    Parameters read from experiment object
    ----------------------------------------
    colpali_model      str   vidore/colpali-v1.2
    colpali_dpi        int   150 (use 200 for better table recall)
    colpali_top_m      int   20  (BGE-M3 candidates fed to ColPali)
    colpali_alpha      float 0.35 (0 = pure ColPali, 1 = pure BGE-M3)
    top_k              int   5   (final pages sent to LLM)
    max_context_chars  int   16000
    """
    logger.info("=" * 80)
    logger.info("RUNNING COLPALI RERANKER (BGE-M3 → ColPali)")
    logger.info("=" * 80)

    # ── Config ──────────────────────────────────────────────────────────────
    model_name  = getattr(experiment, "colpali_model", "vidore/colpali-v1.2")
    dpi         = int(getattr(experiment, "colpali_dpi",   DEFAULT_COLPALI_DPI))
    top_m       = int(getattr(experiment, "colpali_top_m", DEFAULT_BGE_TOP_M))
    alpha       = float(getattr(experiment, "colpali_alpha", DEFAULT_ALPHA))
    top_k       = int(getattr(experiment, "top_k",          DEFAULT_TOP_K))
    max_ctx     = int(getattr(experiment, "max_context_chars", MAX_CONTEXT_CHARS))

    logger.info("Config: model=%s  dpi=%d  top_m=%d  alpha=%.2f  top_k=%d",
                model_name, dpi, top_m, alpha, top_k)

    device, device_map, dtype = _get_device()

    # ── Load all page text (needed for both BGE-M3 scoring and generation) ──
    targets = _collect_target_pdfs(experiment, data)
    if not targets:
        logger.error("No PDFs found.")
        return experiment._create_skipped_results(
            data, "colpali_rerank", "colpali_rerank", "pdf", "colpali_rerank", start_id=0
        )

    logger.info("Extracting text from %d PDFs …", len(targets))
    # page_pool: list of {text, digest, page (0-idx), doc_name, source (pdf_path)}
    page_pool: List[Dict[str, Any]] = []
    for doc_name, pdf_path in targets:
        pages = extract_pages_from_pdf(pdf_path, doc_name)
        # Attach pdf_path so we can render later
        for p in pages:
            p["pdf_path"] = str(pdf_path)
        page_pool.extend(pages)
        logger.debug("  %s → %d pages", doc_name, len(pages))

    logger.info("Total pages in pool: %d", len(page_pool))

    # Build a fast lookup: (doc_name.lower(), page_idx) → pool index
    page_lookup: Dict[Tuple[str, int], int] = {}
    for pool_idx, p in enumerate(page_pool):
        key = (p["doc_name"].lower(), p["page"])
        page_lookup[key] = pool_idx

    # ── Load ColPali ─────────────────────────────────────────────────────────
    ColPali, ColPaliProcessor = _load_colpali_engine()
    logger.info("Loading ColPali: %s (dtype=%s)", model_name, dtype)
    colpali_model = ColPali.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
    ).eval()
    colpali_proc = ColPaliProcessor.from_pretrained(model_name)
    logger.info("ColPali loaded.")

    # ── Per-sample retrieval ─────────────────────────────────────────────────
    results: List[Dict[str, Any]] = []

    for idx, sample in enumerate(data):
        question   = sample.get("question", "")
        doc_name   = sample.get("doc_name") or ""
        gold_seg, gold_evidence_str = experiment._prepare_gold_evidence(
            sample.get("evidence", "")
        )
        gold_pages = _extract_gold_pages_0indexed(gold_seg)

        logger.info("[%d/%d] %s", idx + 1, len(data), question[:80])

        # ── Stage 1: BGE-M3 dense ranking over ALL pages in pool ────────────
        bge_ranked = _bge_score_pages(experiment, question, page_pool, top_m=top_m)
        # bge_ranked: [(pool_idx, score), ...]

        candidates = []
        for pool_idx, bge_score in bge_ranked:
            rec = page_pool[pool_idx]
            candidates.append({
                "pool_idx":  pool_idx,
                "doc_name":  rec["doc_name"],
                "page":      rec["page"],       # 0-indexed
                "pdf_path":  rec["pdf_path"],
                "text":      rec.get("text", ""),
                "bge_score": bge_score,
                "rank_bge":  len(candidates) + 1,
            })

        logger.debug("  BGE-M3 stage: %d candidates", len(candidates))

        # ── Stage 2: ColPali visual reranking over top-M candidates ─────────
        reranked = _colpali_rerank_candidates(
            model=colpali_model,
            processor=colpali_proc,
            device=device,
            dpi=dpi,
            question=question,
            candidates=candidates,
            top_k=top_k,
            alpha=alpha,
        )

        logger.debug("  ColPali reranked → top %d pages", len(reranked))

        # ── Page recall metrics ──────────────────────────────────────────────
        recall_metrics = _compute_page_recall(reranked, gold_pages, k_values=[1, 3, 5])

        # ── Context assembly ─────────────────────────────────────────────────
        context = _assemble_context(reranked, max_chars=max_ctx)
        prompt  = _build_prompt(question, context)

        # ── Format retrieved_chunks for RetrievalEvaluator ───────────────────
        retrieved_chunks = [
            {
                "text": r["text"],
                "metadata": {
                    "doc_name":     r["doc_name"],
                    "page":         r["page"],
                    "pdf_path":     r["pdf_path"],
                    "rank":         rank + 1,
                    "score":        r.get("fused_score", r.get("bge_score", 0.0)),
                    "bge_score":    r.get("bge_score", 0.0),
                    "colpali_score":r.get("colpali_score", 0.0),
                    "retrieval":    "colpali_rerank",
                },
            }
            for rank, r in enumerate(reranked)
        ]

        results.append({
            "sample_id":               idx,
            "doc_name":                doc_name,
            "doc_link":                sample.get("doc_link"),
            "question":                question,
            "reference_answer":        sample.get("answer"),
            "question_type":           sample.get("question_type"),
            "question_reasoning":      sample.get("question_reasoning"),
            "gold_evidence":           gold_evidence_str,
            "gold_evidence_segments":  gold_seg,
            "retrieved_chunks":        retrieved_chunks,
            "num_retrieved":           len(retrieved_chunks),
            "context_length":          len(context),
            "final_prompt":            prompt,
            # Page recall — filled before generation so it's always present
            **recall_metrics,
            # Generation filled below
            "generated_answer":        None,
            "generation_length":       0,
            "experiment_type":         "colpali_rerank",
            "vector_store_type":       "colpali_rerank",
            "retrieval_config": {
                "bge_top_m":  top_m,
                "colpali_top_k": top_k,
                "alpha":      alpha,
                "dpi":        dpi,
            },
        })

        if hasattr(experiment, "notify_sample_complete"):
            experiment.notify_sample_complete()

    # ── Free ColPali BEFORE loading Qwen ─────────────────────────────────────
    # NOTE: avoid .cpu() — models loaded with device_map="auto" may have meta
    # tensors that raise NotImplementedError on any explicit device transfer.
    # Simply deleting the reference and clearing the cache is sufficient.
    logger.info("Freeing ColPali from GPU before generation …")
    del colpali_model, colpali_proc
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("ColPali freed.")

    # ── Generation pass ───────────────────────────────────────────────────────
    logger.info("Generating answers with %s …", getattr(experiment, "llm_model_name", "LLM"))
    for result in results:
        answer = _generate(experiment, result["final_prompt"])
        result["generated_answer"]  = answer
        result["generation_length"] = len(answer)

    # ── Summary stats ─────────────────────────────────────────────────────────
    _log_recall_summary(results)

    return results


def _log_recall_summary(results: List[Dict[str, Any]]) -> None:
    """Log page recall@1/3/5 to console so you see it before scored JSON."""
    totals = {k: 0 for k in ("page_hit@1", "page_hit@3", "page_hit@5")}
    valid  = {k: 0 for k in totals}

    for r in results:
        for k in totals:
            v = r.get(k)
            if v is not None:
                valid[k]  += 1
                totals[k] += int(v)

    logger.info("=" * 60)
    logger.info("PAGE RECALL SUMMARY")
    for k in ("page_hit@1", "page_hit@3", "page_hit@5"):
        n = valid[k]
        if n:
            logger.info("  %s: %.1f%%  (%d/%d)", k, 100 * totals[k] / n, totals[k], n)
        else:
            logger.info("  %s: N/A (no gold pages found)", k)
    logger.info("=" * 60)