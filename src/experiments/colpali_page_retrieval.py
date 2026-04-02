"""
ColQwen2 page retrieval experiment with document-aware candidate selection,
neighbor-page expansion, and lightweight text reranking.

Pipeline:
1) Render PDF pages to images.
2) Embed pages with ColQwen2.
3) Retrieve a broad first-stage set of pages.
4) Aggregate first-stage evidence at the document level.
5) Keep top documents, expand top pages with neighbors.
6) Rerank candidate pages with a hybrid visual + text score.
7) Build grounded context from reranked page text and generate an answer.

This module intentionally keeps the public entry point name
`run_colpali_page_retrieval(...)` so it can replace the older ColPali file
without requiring experiment-registry changes.
"""

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    import fitz  # PyMuPDF
    if not hasattr(fitz, "open"):
        raise ImportError("fitz missing open")
except Exception:  # pragma: no cover
    import pymupdf as fitz

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

from src.ingestion.pdf_utils import _find_local_pdf

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------
def _load_colqwen2() -> Tuple[Any, Any, Any]:
    try:
        from transformers import ColQwen2ForRetrieval, ColQwen2Processor
        from transformers.utils.import_utils import is_flash_attn_2_available
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "ColQwen2 dependencies missing. Install a recent transformers version "
            "that includes ColQwen2 support."
        ) from exc

    return ColQwen2ForRetrieval, ColQwen2Processor, is_flash_attn_2_available


# -----------------------------------------------------------------------------
# PDF + text helpers
# -----------------------------------------------------------------------------
def _render_pdf_pages(pdf_path: Path, doc_name: str, dpi: int) -> List[Dict[str, Any]]:
    if Image is None:
        raise RuntimeError("Pillow is required to render PDF pages for ColQwen2.")

    pages: List[Dict[str, Any]] = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        logger.warning("Failed to open PDF %s: %s", pdf_path, exc)
        return pages

    try:
        for page_idx in range(len(doc)):
            try:
                page = doc[page_idx]
                pix = page.get_pixmap(dpi=dpi)
                mode = "RGBA" if pix.alpha else "RGB"
                img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                if mode == "RGBA":
                    img = img.convert("RGB")
                pages.append(
                    {
                        "image": img,
                        "doc_name": doc_name,
                        "page": page_idx,
                        "pdf_path": str(pdf_path),
                    }
                )
            except Exception as exc:
                logger.warning(
                    "Failed to render page %s from %s: %s", page_idx, pdf_path, exc
                )
    finally:
        doc.close()

    return pages


def _load_page_text(pdf_path: str, page_idx: int) -> str:
    doc = None
    try:
        doc = fitz.open(pdf_path)
        if page_idx < 0 or page_idx >= len(doc):
            return ""
        page = doc[page_idx]
        text = page.get_text("text")
        return text or ""
    except Exception as exc:
        logger.warning("Failed to load page text %s from %s: %s", page_idx, pdf_path, exc)
        return ""
    finally:
        if doc is not None:
            doc.close()


def _collect_target_pdfs(experiment, data: List[Dict[str, Any]]) -> List[Tuple[str, Path]]:
    pdf_dir = Path(getattr(experiment, "pdf_local_dir", "pdfs"))
    targets: List[Tuple[str, Path]] = []
    seen = set()

    if getattr(experiment, "use_all_pdfs", False):
        for path in pdf_dir.glob("**/*.pdf"):
            doc_name = path.stem
            key = doc_name.lower()
            if key in seen:
                continue
            targets.append((doc_name, path))
            seen.add(key)
        return targets

    for sample in data:
        doc_name = sample.get("doc_name") or sample.get("document") or ""
        if not doc_name:
            continue
        key = doc_name.lower()
        if key in seen:
            continue
        local_path = _find_local_pdf(doc_name, str(pdf_dir))
        if local_path is None:
            continue
        targets.append((doc_name, local_path))
        seen.add(key)

    return targets


# -----------------------------------------------------------------------------
# Prompting / generation helpers
# -----------------------------------------------------------------------------
def _build_colpali_prompt(question: str, context: str) -> str:
    question = (question or "").strip()
    context = (context or "").strip()
    return (
        "You are a careful financial analyst. Use ONLY the context from the retrieved pages. "
        "If the answer is not in the context, say you do not know. Prefer short, direct, "
        "numerically precise answers when applicable.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def _generate_with_prompt(experiment, prompt: str) -> str:
    if experiment.use_api:
        experiment._initialize_llm()
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful financial analyst assistant. "
                    "Answer strictly based on the provided context and avoid speculation."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        try:
            response = experiment.api_client.chat.completions.create(
                model=experiment.llm_model_name,
                messages=messages,
                max_tokens=experiment.max_new_tokens,
                temperature=0.0,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            experiment.logger.error("API generation failed: %s", exc)
            return ""

    experiment._initialize_llm()
    try:
        outputs = experiment.llm_pipeline(prompt)
    except TypeError:
        outputs = experiment.llm_pipeline(
            prompt,
            max_new_tokens=experiment.max_new_tokens,
            do_sample=False,
        )

    if not outputs:
        return ""
    text = outputs[0].get("generated_text", "")
    return (text or "").strip()


# -----------------------------------------------------------------------------
# Retrieval / reranking helpers
# -----------------------------------------------------------------------------
def _tokenize_for_rerank(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9$%.,/\-]+", (text or "").lower())


def _extract_numbers(text: str) -> List[str]:
    return re.findall(r"\d[\d,\.\-]*%?", text or "")


def _simple_text_rerank_score(query: str, page_text: str) -> float:
    q_tokens = set(_tokenize_for_rerank(query))
    text_tokens = _tokenize_for_rerank(page_text)

    if not q_tokens or not text_tokens:
        return 0.0

    text_set = set(text_tokens[:4000])
    overlap = len(q_tokens & text_set) / max(1, len(q_tokens))

    # Numeric match: score proportional to fraction of query numbers found on
    # this page.  Financial queries are highly specific — a page that contains
    # none of the queried numbers is almost certainly wrong.
    q_nums = set(_extract_numbers(query))
    t_nums = set(_extract_numbers(page_text))
    if q_nums:
        num_overlap_frac = len(q_nums & t_nums) / len(q_nums)
        num_bonus = 0.35 * num_overlap_frac  # up to 0.35 based on how many numbers match
    else:
        num_bonus = 0.0

    finance_terms = {
        "revenue",
        "income",
        "net",
        "loss",
        "assets",
        "liabilities",
        "cash",
        "debt",
        "equity",
        "ppe",
        "inventory",
        "operating",
        "segment",
        "tax",
        "million",
        "billion",
        "fiscal",
        "year",
        "percent",
        "%",
        "table",
        "note",
        "expenses",
        "sales",
        "earnings",
        "share",
    }
    finance_q = q_tokens & finance_terms
    finance_overlap = len(finance_q & text_set) / max(1, len(finance_q)) if finance_q else 0.0
    finance_bonus = 0.15 * finance_overlap

    return float(overlap + num_bonus + finance_bonus)


def _minmax_norm(values: List[float]) -> Dict[int, float]:
    if not values:
        return {}
    vmin = min(values)
    vmax = max(values)
    if abs(vmax - vmin) < 1e-8:
        return {i: 1.0 for i in range(len(values))}
    return {i: (v - vmin) / (vmax - vmin) for i, v in enumerate(values)}


def _build_page_lookup(page_records: List[Dict[str, Any]]) -> Dict[Tuple[str, int], int]:
    return {(p["doc_name"], int(p["page"])): i for i, p in enumerate(page_records)}


def _group_context_chunks(chunks: List[Dict[str, Any]], max_context_chars: int) -> str:
    blocks: List[str] = []
    total = 0
    seen = set()

    for chunk in chunks:
        meta = chunk["metadata"]
        key = (meta["doc_name"], int(meta["page"]))
        if key in seen:
            continue
        seen.add(key)

        block = (
            f"[Document: {meta['doc_name']} | Page: {meta['page'] + 1} | "
            f"Rank: {meta['rank']} | Score: {meta['score']:.4f}]\n"
            f"{chunk['text'].strip()}\n"
        )
        if total + len(block) > max_context_chars:
            break
        blocks.append(block)
        total += len(block)

    return "\n\n".join(blocks)


def _build_model_inputs(batch_feature: Any, device: torch.device) -> Dict[str, torch.Tensor]:
    if hasattr(batch_feature, "to"):
        batch_feature = batch_feature.to(device)
    return dict(batch_feature)


def _trim_zero_rows(embedding: torch.Tensor) -> torch.Tensor:
    """
    ColQwen2 uses late-interaction multi-vector embeddings. When pages are embedded
    in batches, shorter pages may be padded up to the max sequence length in that
    batch. We trim rows that are entirely zero so each page keeps only its effective
    vectors before downstream scoring.
    """
    if embedding.ndim != 2:
        return embedding

    keep = torch.any(embedding != 0, dim=-1)
    if torch.any(keep):
        return embedding[keep]
    return embedding


def _score_query_against_pages(
    processor: Any,
    query_embedding: torch.Tensor,
    doc_embeddings: List[torch.Tensor],
    device: torch.device,
    doc_batch_size: int,
) -> np.ndarray:
    """
    Score one query against a list of ColQwen2 page embeddings.

    Important: ColQwen2 page embeddings are variable-length late-interaction
    tensors (num_vectors x dim). They must not be stacked across pages.
    Instead, pass a single query embedding and a list of page embeddings to
    processor.score_retrieval(...), chunked for memory efficiency.
    """
    scores: List[np.ndarray] = []
    q_batch = [_trim_zero_rows(query_embedding).to(device)]

    with torch.inference_mode():
        for start in range(0, len(doc_embeddings), doc_batch_size):
            batch_docs = [
                _trim_zero_rows(emb).to(device)
                for emb in doc_embeddings[start : start + doc_batch_size]
            ]

            batch_scores = processor.score_retrieval(q_batch, batch_docs)

            if isinstance(batch_scores, torch.Tensor):
                batch_scores = batch_scores.detach().float().cpu().numpy().reshape(-1)
            else:
                batch_scores = np.asarray(batch_scores).reshape(-1)

            scores.append(batch_scores)

    if not scores:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(scores, axis=0).astype(np.float32)


# -----------------------------------------------------------------------------
# Main experiment entry point
# -----------------------------------------------------------------------------
def run_colpali_page_retrieval(experiment, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING COLQWEN2 PAGE RETRIEVAL")
    logger.info("=" * 80)

    ColQwen2ForRetrieval, ColQwen2Processor, is_flash_attn_2_available = _load_colqwen2()

    model_name = getattr(experiment, "colqwen2_model", None) or getattr(experiment, "colpali_model", None) or "vidore/colqwen2-v1.0-hf"
    dpi = int(getattr(experiment, "colqwen2_dpi", getattr(experiment, "colpali_dpi", 200)))

    page_batch_size = int(getattr(experiment, "colqwen2_page_batch_size", 4))
    query_batch_size = int(getattr(experiment, "colqwen2_query_batch_size", 1))
    doc_score_batch_size = int(getattr(experiment, "colqwen2_score_batch_size", 64))

    first_stage_k = int(getattr(experiment, "colqwen2_first_stage_k", 50))
    top_docs_keep = int(getattr(experiment, "colqwen2_top_docs", 3))
    pages_per_doc = int(getattr(experiment, "colqwen2_pages_per_doc", 8))
    neighbor_window = int(getattr(experiment, "colqwen2_neighbor_window", 1))

    visual_weight = float(getattr(experiment, "colqwen2_visual_weight", 0.50))
    text_weight = float(getattr(experiment, "colqwen2_text_weight", 0.40))
    doc_bonus_weight = float(getattr(experiment, "colqwen2_doc_bonus_weight", 0.10))

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model_dtype = torch.bfloat16
        attn_impl = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
        device_map = "auto"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
        model_dtype = torch.float32
        attn_impl = "sdpa"
        device_map = None
    else:
        device = torch.device("cpu")
        model_dtype = torch.float32
        attn_impl = "sdpa"
        device_map = None

    logger.info("Loading ColQwen2 model: %s", model_name)
    model_kwargs = {
        "attn_implementation": attn_impl,
    }
    if device_map is not None:
        model_kwargs["device_map"] = device_map

    try:
        model = ColQwen2ForRetrieval.from_pretrained(
            model_name,
            dtype=model_dtype,
            **model_kwargs,
        ).eval()
    except TypeError:
        model = ColQwen2ForRetrieval.from_pretrained(
            model_name,
            torch_dtype=model_dtype,
            **model_kwargs,
        ).eval()

    processor = ColQwen2Processor.from_pretrained(model_name)

    targets = _collect_target_pdfs(experiment, data)
    if not targets:
        logger.error("No PDFs found for ColQwen2 retrieval.")
        return experiment._create_skipped_results(
            data,
            "colpali_page",
            "colpali_page",
            "pdf",
            "colqwen2",
            start_id=0,
        )

    # Render and embed one document at a time so that only one doc's page images
    # are ever in RAM simultaneously (prevents OOM on large PDF corpora).
    logger.info("Rendering and embedding %s PDFs at %s DPI", len(targets), dpi)
    page_records: List[Dict[str, Any]] = []
    doc_embeddings: List[torch.Tensor] = []

    with torch.inference_mode():
        for doc_name, path in targets:
            pages = _render_pdf_pages(path, doc_name, dpi=dpi)
            if not pages:
                continue
            for start in range(0, len(pages), page_batch_size):
                batch = pages[start : start + page_batch_size]
                batch_images = [p["image"] for p in batch]
                batch_inputs = processor(images=batch_images, return_tensors="pt")
                batch_inputs = _build_model_inputs(batch_inputs, device)
                batch_embeddings = model(**batch_inputs).embeddings.detach().cpu()
                for i, emb in enumerate(torch.unbind(batch_embeddings, dim=0)):
                    doc_embeddings.append(_trim_zero_rows(emb))
                    p = batch[i]
                    page_records.append(
                        {
                            "doc_name": p["doc_name"],
                            "page": p["page"],
                            "pdf_path": p["pdf_path"],
                        }
                    )
            # pages (and their PIL images) go out of scope here → freed

    if not page_records:
        logger.error("No pages rendered for ColQwen2 retrieval.")
        return experiment._create_skipped_results(
            data,
            "colpali_page",
            "colpali_page",
            "pdf",
            "colqwen2",
            start_id=0,
        )

    page_lookup = _build_page_lookup(page_records)

    # Pre-build full doc → all-page-index mapping so that after top-doc selection
    # we can rank ALL pages of those docs (not just the subset that happened to
    # appear in the first-stage global top-k).
    all_doc_to_page_ids: DefaultDict[str, List[int]] = defaultdict(list)
    for _i, _p in enumerate(page_records):
        all_doc_to_page_ids[_p["doc_name"]].append(_i)

    logger.info("Embedding %s queries with ColQwen2", len(data))
    query_texts = [sample.get("question", "") for sample in data]
    query_embeddings: List[torch.Tensor] = []

    # Query batch size defaults to 1 to avoid any ambiguity around padding effects.
    with torch.inference_mode():
        for start in range(0, len(query_texts), query_batch_size):
            batch_queries = query_texts[start : start + query_batch_size]
            batch_inputs = processor(text=batch_queries, return_tensors="pt")
            batch_inputs = _build_model_inputs(batch_inputs, device)
            batch_embeddings = model(**batch_inputs).embeddings.detach().cpu()
            for emb in torch.unbind(batch_embeddings, dim=0):
                query_embeddings.append(_trim_zero_rows(emb))

    logger.info("Scoring %s queries against %s pages", len(query_embeddings), len(doc_embeddings))
    all_scores = np.zeros((len(query_embeddings), len(doc_embeddings)), dtype=np.float32)

    for q_idx, query_embedding in enumerate(query_embeddings):
        all_scores[q_idx] = _score_query_against_pages(
            processor=processor,
            query_embedding=query_embedding,
            doc_embeddings=doc_embeddings,
            device=device,
            doc_batch_size=doc_score_batch_size,
        )

    results: List[Dict[str, Any]] = []
    max_context = int(getattr(experiment, "max_context_chars", 16000))
    page_text_cache: Dict[Tuple[str, int], str] = {}

    for idx, sample in enumerate(data):
        question = sample.get("question", "")
        doc_scores = all_scores[idx]
        if len(doc_scores) == 0:
            retrieved_chunks = []
            context = ""
        else:
            first_stage = min(first_stage_k, len(doc_scores))
            first_stage_idx = np.argsort(doc_scores)[::-1][:first_stage]

            doc_best_score: Dict[str, float] = {}
            doc_to_page_ids: DefaultDict[str, List[int]] = defaultdict(list)
            for page_idx in first_stage_idx:
                page = page_records[int(page_idx)]
                doc_name = page["doc_name"]
                score = float(doc_scores[int(page_idx)])
                doc_best_score[doc_name] = max(doc_best_score.get(doc_name, -1e9), score)
                doc_to_page_ids[doc_name].append(int(page_idx))

            top_docs = sorted(doc_best_score.keys(), key=lambda d: doc_best_score[d], reverse=True)[:top_docs_keep]

            candidate_ids = set()
            for doc_name in top_docs:
                # Use ALL pages of the top doc (scores already computed), not
                # just the subset that appeared in first_stage_idx.  This is the
                # key fix: first-stage only gives us the best pages globally, but
                # the correct page for a specific query may not rank highly across
                # the entire corpus even though it is the best page in its doc.
                ranked_doc_pages = sorted(
                    all_doc_to_page_ids[doc_name],
                    key=lambda i: float(doc_scores[i]),
                    reverse=True,
                )[:pages_per_doc]

                for pid in ranked_doc_pages:
                    candidate_ids.add(pid)
                    base_page_no = int(page_records[pid]["page"])
                    for delta in range(-neighbor_window, neighbor_window + 1):
                        neighbor_key = (doc_name, base_page_no + delta)
                        if neighbor_key in page_lookup:
                            candidate_ids.add(page_lookup[neighbor_key])

            candidate_ids = sorted(candidate_ids)

            candidate_visual_scores = [float(doc_scores[i]) for i in candidate_ids]
            visual_norm = _minmax_norm(candidate_visual_scores)

            candidate_rows: List[Dict[str, Any]] = []
            for local_idx, global_page_idx in enumerate(candidate_ids):
                page = page_records[global_page_idx]
                cache_key = (page["pdf_path"], int(page["page"]))
                if cache_key not in page_text_cache:
                    page_text_cache[cache_key] = _load_page_text(page["pdf_path"], int(page["page"]))
                page_text = page_text_cache[cache_key]

                text_score = _simple_text_rerank_score(question, page_text)
                # Rank-graded doc bonus: pages from the top-ranked doc get the
                # full bonus; lower-ranked docs get proportionally less.
                # (Previously this was a binary in/out check, but all candidates
                # are already from top_docs, so it was adding a constant to all.)
                doc_rank = top_docs.index(page["doc_name"]) if page["doc_name"] in top_docs else len(top_docs)
                rank_bonus = 1.0 - doc_rank / max(1, len(top_docs))
                final_score = (
                    visual_weight * visual_norm.get(local_idx, 0.0)
                    + text_weight * text_score
                    + doc_bonus_weight * rank_bonus
                )

                candidate_rows.append(
                    {
                        "global_page_idx": global_page_idx,
                        "page_text": page_text,
                        "final_score": float(final_score),
                        "visual_score": float(doc_scores[global_page_idx]),
                        "text_score": float(text_score),
                    }
                )

            final_top_k = min(int(getattr(experiment, "top_k", 5)), len(candidate_rows))
            candidate_rows = sorted(candidate_rows, key=lambda x: x["final_score"], reverse=True)[:final_top_k]

            retrieved_chunks = []
            for rank, row in enumerate(candidate_rows, start=1):
                page_idx = int(row["global_page_idx"])
                page = page_records[page_idx]
                retrieved_chunks.append(
                    {
                        "text": row["page_text"],
                        "metadata": {
                            "doc_name": page["doc_name"],
                            "page": int(page["page"]),
                            "pdf_path": page["pdf_path"],
                            "rank": rank,
                            "score": float(row["final_score"]),
                            "visual_score": float(row["visual_score"]),
                            "text_score": float(row["text_score"]),
                            "retrieval": "colqwen2_hybrid_reranked",
                        },
                    }
                )

            context = _group_context_chunks(retrieved_chunks, max_context)

        prompt = _build_colpali_prompt(question, context)
        generated_answer = _generate_with_prompt(experiment, prompt)
        gold_segments, gold_evidence_str = experiment._prepare_gold_evidence(sample.get("evidence", ""))

        results.append(
            {
                "sample_id": idx,
                "doc_name": sample.get("doc_name"),
                "doc_link": sample.get("doc_link"),
                "question": question,
                "reference_answer": sample.get("answer"),
                "question_type": sample.get("question_type"),
                "question_reasoning": sample.get("question_reasoning"),
                "gold_evidence": gold_evidence_str,
                "gold_evidence_segments": gold_segments,
                "retrieved_chunks": retrieved_chunks,
                "num_retrieved": len(retrieved_chunks),
                "context_length": len(context),
                "generated_answer": generated_answer,
                "generation_length": len(generated_answer),
                "experiment_type": "colpali_page",
                "vector_store_type": "colqwen2",
                "final_prompt": prompt,
            }
        )

        if hasattr(experiment, "notify_sample_complete"):
            experiment.notify_sample_complete()

    return results
