"""
ColPali page retrieval experiment.

Pipeline:
1) Render PDF pages to images.
2) Embed pages with ColPali (multi-vector ColBERT-style).
3) Embed queries and retrieve top-k pages.
4) Load text for retrieved pages and generate an answer.
"""
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def _load_colpali() -> Tuple[Any, Any]:
    try:
        from colpali_engine.models import ColPali, ColPaliProcessor
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "ColPali dependencies missing. Install with: pip install 'colpali-engine>=0.3.0,<0.4.0'"
        ) from exc

    return ColPali, ColPaliProcessor


def _colbert_score(query: torch.Tensor, doc: torch.Tensor) -> float:
    """Compute ColBERT max-sim score between query and document embeddings."""
    if query.ndim != 2 or doc.ndim != 2:
        return 0.0
    sim = torch.matmul(query, doc.transpose(0, 1))
    per_query = sim.max(dim=1).values
    return float(per_query.sum().item())


def _render_pdf_pages(pdf_path: Path, doc_name: str, dpi: int) -> List[Dict[str, Any]]:
    if Image is None:
        raise RuntimeError("Pillow is required to render PDF pages for ColPali.")

    pages: List[Dict[str, Any]] = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        logger.warning("Failed to open PDF %s: %s", pdf_path, exc)
        return pages

    for page_idx in range(len(doc)):
        try:
            page = doc[page_idx]
            pix = page.get_pixmap(dpi=dpi)
            mode = "RGBA" if pix.alpha else "RGB"
            img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
            if mode == "RGBA":
                img = img.convert("RGB")
            pages.append({
                "image": img,
                "doc_name": doc_name,
                "page": page_idx,
                "pdf_path": str(pdf_path),
            })
        except Exception as exc:
            logger.warning("Failed to render page %s from %s: %s", page_idx, pdf_path, exc)

    doc.close()
    return pages


def _load_page_text(pdf_path: str, page_idx: int) -> str:
    try:
        doc = fitz.open(pdf_path)
        if page_idx < 0 or page_idx >= len(doc):
            return ""
        page = doc[page_idx]
        text = page.get_text("text")
        doc.close()
        return text or ""
    except Exception as exc:
        logger.warning("Failed to load page text %s from %s: %s", page_idx, pdf_path, exc)
        return ""


def _build_colpali_prompt(question: str, context: str) -> str:
    question = (question or "").strip()
    context = (context or "").strip()
    return (
        "You are a careful financial analyst. Use ONLY the context from the retrieved pages. "
        "If the answer is not in the context, say you do not know.\n\n"
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
                    "Answer strictly based on the question and the provided context "
                    "(if any) and avoid speculation."
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


def run_colpali_page_retrieval(experiment, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING COLPALI PAGE RETRIEVAL")
    logger.info("=" * 80)

    ColPali, ColPaliProcessor = _load_colpali()

    model_name = getattr(experiment, "colpali_model", None) or "vidore/colpali-v1.2"
    dpi = int(getattr(experiment, "colpali_dpi", 150))

    if getattr(experiment, "colpali_base_model", None):
        logger.warning("colpali_base_model is ignored for colpali_engine>=0.3; using %s", model_name)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_map = "cuda:0"
        dtype = torch.bfloat16
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
        device_map = "mps"
        dtype = torch.float32
    else:
        device = torch.device("cpu")
        device_map = "cpu"
        dtype = torch.float32

    logger.info("Loading ColPali model: %s", model_name)
    model = ColPali.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
    ).eval()
    processor = ColPaliProcessor.from_pretrained(model_name)

    targets = _collect_target_pdfs(experiment, data)
    if not targets:
        logger.error("No PDFs found for ColPali retrieval.")
        return experiment._create_skipped_results(data, "colpali_page", "colpali_page", "pdf", "colpali", start_id=0)

    logger.info("Rendering %s PDFs at %s DPI", len(targets), dpi)
    page_records: List[Dict[str, Any]] = []
    for doc_name, path in targets:
        page_records.extend(_render_pdf_pages(path, doc_name, dpi=dpi))

    if not page_records:
        logger.error("No pages rendered for ColPali retrieval.")
        return experiment._create_skipped_results(data, "colpali_page", "colpali_page", "pdf", "colpali", start_id=0)

    logger.info("Embedding %s page images with ColPali", len(page_records))
    from torch.utils.data import DataLoader

    images = [p["image"] for p in page_records]
    doc_embeddings: List[torch.Tensor] = []

    image_loader = DataLoader(
        images,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: processor.process_images(x),
    )

    with torch.no_grad():
        for batch in image_loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            embeddings = model(**batch)
            doc_embeddings.extend(list(torch.unbind(embeddings.to("cpu"))))

    logger.info("Embedding %s queries with ColPali", len(data))
    query_texts = [sample.get("question", "") for sample in data]
    query_embeddings: List[torch.Tensor] = []

    query_loader = DataLoader(
        query_texts,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: processor.process_queries(x),
    )

    with torch.no_grad():
        for batch in query_loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            embeddings = model(**batch)
            query_embeddings.extend(list(torch.unbind(embeddings.to("cpu"))))

    logger.info("Scoring queries against %s pages", len(doc_embeddings))
    scores = np.zeros((len(query_embeddings), len(doc_embeddings)), dtype=np.float32)

    for q_idx, query in enumerate(query_embeddings):
        q = query.to(device)
        for d_idx, doc in enumerate(doc_embeddings):
            d = doc.to(device)
            scores[q_idx, d_idx] = _colbert_score(q, d)

    results: List[Dict[str, Any]] = []
    max_context = getattr(experiment, "max_context_chars", 12000)

    for idx, sample in enumerate(data):
        question = sample.get("question", "")
        doc_scores = scores[idx]
        top_k = min(getattr(experiment, "top_k", 5), len(doc_scores))
        top_idx = np.argsort(doc_scores)[::-1][:top_k]

        retrieved_chunks = []
        for rank, page_idx in enumerate(top_idx):
            page = page_records[int(page_idx)]
            page_text = _load_page_text(page["pdf_path"], int(page["page"]))
            retrieved_chunks.append({
                "text": page_text,
                "metadata": {
                    "doc_name": page["doc_name"],
                    "page": int(page["page"]),
                    "pdf_path": page["pdf_path"],
                    "rank": rank + 1,
                    "score": float(doc_scores[int(page_idx)]),
                    "retrieval": "colpali",
                },
            })

        context = "\n\n".join(chunk["text"] for chunk in retrieved_chunks)
        context = context[:max_context]
        prompt = _build_colpali_prompt(question, context)
        generated_answer = _generate_with_prompt(experiment, prompt)
        gold_segments, gold_evidence_str = experiment._prepare_gold_evidence(sample.get("evidence", ""))

        results.append({
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
            "vector_store_type": "colpali",
            "final_prompt": prompt,
        })

        if hasattr(experiment, "notify_sample_complete"):
            experiment.notify_sample_complete()

    return results
