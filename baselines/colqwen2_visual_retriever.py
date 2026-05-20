from __future__ import annotations

import json
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

logger = logging.getLogger(__name__)


def _load_colqwen2():
    try:
        from transformers import ColQwen2ForRetrieval, ColQwen2Processor
        from transformers.utils.import_utils import is_flash_attn_2_available
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "ColQwen2 support is unavailable. Install a transformers build "
            "that includes ColQwen2ForRetrieval and ColQwen2Processor."
        ) from exc

    return ColQwen2ForRetrieval, ColQwen2Processor, is_flash_attn_2_available


def _normalize_doc_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (name or "").lower())


def _trim_zero_rows(embedding: torch.Tensor) -> torch.Tensor:
    if embedding.ndim != 2:
        return embedding
    keep = torch.any(embedding != 0, dim=-1)
    if torch.any(keep):
        return embedding[keep]
    return embedding


def _tokenize_for_rerank(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9$%.,/\-]+", (text or "").lower())


def _extract_numbers(text: str) -> List[str]:
    return re.findall(r"\d[\d,.\-]*%?", text or "")


def _simple_text_rerank_score(query: str, page_text: str) -> float:
    q_tokens = set(_tokenize_for_rerank(query))
    text_tokens = _tokenize_for_rerank(page_text)
    if not q_tokens or not text_tokens:
        return 0.0

    text_set = set(text_tokens[:4000])
    overlap = len(q_tokens & text_set) / max(1, len(q_tokens))

    q_nums = set(_extract_numbers(query))
    t_nums = set(_extract_numbers(page_text))
    if q_nums:
        num_bonus = 0.35 * (len(q_nums & t_nums) / len(q_nums))
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
    finance_bonus = 0.15 * (
        len(finance_q & text_set) / max(1, len(finance_q))
    ) if finance_q else 0.0

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


def _score_query_against_pages(
    processor: Any,
    query_embedding: torch.Tensor,
    doc_embeddings: List[torch.Tensor],
    device: torch.device,
    doc_batch_size: int,
) -> np.ndarray:
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


class ColQwen2VisualRetriever:
    """Page-level visual retriever for FinanceBench using ColQwen2."""

    def __init__(
        self,
        pdf_dir: str,
        cache_dir: str,
        model_name: str = "vidore/colqwen2-v1.0-hf",
        dpi: int = 150,
        page_batch_size: int = 4,
        score_batch_size: int = 64,
        first_stage_k: int = 50,
        top_docs: int = 3,
        pages_per_doc: int = 8,
        neighbor_window: int = 1,
        visual_weight: float = 0.50,
        text_weight: float = 0.40,
        doc_bonus_weight: float = 0.10,
    ) -> None:
        self.pdf_dir = Path(pdf_dir)
        self.cache_dir = Path(cache_dir)
        self.model_name = model_name
        self.dpi = dpi
        self.page_batch_size = page_batch_size
        self.score_batch_size = score_batch_size
        self.first_stage_k = first_stage_k
        self.top_docs = top_docs
        self.pages_per_doc = pages_per_doc
        self.neighbor_window = neighbor_window
        self.visual_weight = visual_weight
        self.text_weight = text_weight
        self.doc_bonus_weight = doc_bonus_weight

        self._model = None
        self._processor = None
        self._device = None
        self._page_records: List[Dict[str, Any]] = []
        self._page_embeddings: List[torch.Tensor] = []
        self._page_lookup: Dict[Tuple[str, int], int] = {}
        self._doc_to_page_ids: DefaultDict[str, List[int]] = defaultdict(list)
        self._page_text_cache: Dict[Tuple[str, int], str] = {}

    @property
    def cache_path(self) -> Path:
        model_tag = re.sub(r"[^a-z0-9]+", "_", self.model_name.lower().split("/")[-1]).strip("_")
        return self.cache_dir / f"{model_tag}_dpi{self.dpi}.pt"

    def build_or_load_index(self, overwrite: bool = False) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if self.cache_path.exists() and not overwrite:
            logger.info("Loading cached ColQwen2 index from %s", self.cache_path)
            state = torch.load(self.cache_path, map_location="cpu", weights_only=False)
            self._page_records = state["page_records"]
            self._page_embeddings = state["page_embeddings"]
            self._rebuild_lookups()
            logger.info(
                "ColQwen2 cache loaded: %d pages across %d docs",
                len(self._page_records),
                len({p['doc_name'] for p in self._page_records}),
            )
            return

        self._load_model()
        pdf_files = sorted(self.pdf_dir.glob("**/*.pdf"))
        logger.info("Building ColQwen2 page index from %d PDFs", len(pdf_files))

        page_records: List[Dict[str, Any]] = []
        page_embeddings: List[torch.Tensor] = []

        with torch.inference_mode():
            for pdf_path in pdf_files:
                rendered_pages = self._render_pdf_pages(pdf_path)
                if not rendered_pages:
                    continue

                for start in range(0, len(rendered_pages), self.page_batch_size):
                    batch = rendered_pages[start : start + self.page_batch_size]
                    batch_images = [row["image"] for row in batch]
                    batch_inputs = self._processor(images=batch_images, return_tensors="pt")
                    if hasattr(batch_inputs, "to"):
                        batch_inputs = batch_inputs.to(self._device)
                    batch_embeddings = self._model(**dict(batch_inputs)).embeddings.detach().cpu()

                    for i, emb in enumerate(torch.unbind(batch_embeddings, dim=0)):
                        page_embeddings.append(_trim_zero_rows(emb))
                        page_records.append(batch[i]["record"])

                done = len(page_records)
                if done and done % 1000 < len(rendered_pages):
                    logger.info("  Embedded %d pages so far", done)

        if not page_records:
            raise RuntimeError(
                f"No pages were embedded from PDFs under {self.pdf_dir}."
            )

        torch.save(
            {
                "page_records": page_records,
                "page_embeddings": page_embeddings,
            },
            self.cache_path,
        )

        self._page_records = page_records
        self._page_embeddings = page_embeddings
        self._rebuild_lookups()
        logger.info("Saved ColQwen2 index to %s", self.cache_path)

    def search(
        self,
        query: str,
        top_k: int = 5,
        doc_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        if not self._page_records or not self._page_embeddings:
            raise RuntimeError("ColQwen2 index is not loaded. Call build_or_load_index() first.")

        self._load_model()

        batch_inputs = self._processor(text=[query], return_tensors="pt")
        if hasattr(batch_inputs, "to"):
            batch_inputs = batch_inputs.to(self._device)
        query_embedding = self._model(**dict(batch_inputs)).embeddings.detach().cpu()[0]

        doc_scores = _score_query_against_pages(
            processor=self._processor,
            query_embedding=query_embedding,
            doc_embeddings=self._page_embeddings,
            device=self._device,
            doc_batch_size=self.score_batch_size,
        )
        if len(doc_scores) == 0:
            return []

        allowed = None
        if doc_filter:
            allowed = {_normalize_doc_name(d) for d in doc_filter if d}

        if allowed:
            first_stage_pool = [
                i for i, page in enumerate(self._page_records)
                if _normalize_doc_name(page["doc_name"]) in allowed
            ]
        else:
            first_stage_pool = list(range(len(self._page_records)))

        if not first_stage_pool:
            return []

        first_stage_count = min(self.first_stage_k, len(first_stage_pool))
        first_stage_idx = sorted(
            first_stage_pool,
            key=lambda i: float(doc_scores[i]),
            reverse=True,
        )[:first_stage_count]

        doc_best_score: Dict[str, float] = {}
        for page_idx in first_stage_idx:
            page = self._page_records[page_idx]
            doc_name = page["doc_name"]
            score = float(doc_scores[page_idx])
            doc_best_score[doc_name] = max(doc_best_score.get(doc_name, -1e9), score)

        top_docs = sorted(doc_best_score, key=lambda d: doc_best_score[d], reverse=True)[: self.top_docs]
        if not top_docs:
            return []

        candidate_ids = set()
        for doc_name in top_docs:
            ranked_doc_pages = sorted(
                self._doc_to_page_ids.get(doc_name, []),
                key=lambda i: float(doc_scores[i]),
                reverse=True,
            )[: self.pages_per_doc]

            for page_id in ranked_doc_pages:
                candidate_ids.add(page_id)
                base_page = int(self._page_records[page_id]["page"])
                for delta in range(-self.neighbor_window, self.neighbor_window + 1):
                    neighbor_key = (doc_name, base_page + delta)
                    if neighbor_key in self._page_lookup:
                        candidate_ids.add(self._page_lookup[neighbor_key])

        candidate_ids = sorted(candidate_ids)
        candidate_visual_scores = [float(doc_scores[i]) for i in candidate_ids]
        visual_norm = _minmax_norm(candidate_visual_scores)

        rows: List[Dict[str, Any]] = []
        for local_idx, global_page_idx in enumerate(candidate_ids):
            page = self._page_records[global_page_idx]
            page_text = self._load_page_text(page["pdf_rel_path"], int(page["page"]))
            text_score = _simple_text_rerank_score(query, page_text)

            doc_rank = top_docs.index(page["doc_name"]) if page["doc_name"] in top_docs else len(top_docs)
            doc_bonus = 1.0 - doc_rank / max(1, len(top_docs))
            final_score = (
                self.visual_weight * visual_norm.get(local_idx, 0.0)
                + self.text_weight * text_score
                + self.doc_bonus_weight * doc_bonus
            )

            rows.append(
                {
                    "page": page,
                    "page_text": page_text,
                    "final_score": float(final_score),
                    "visual_score": float(doc_scores[global_page_idx]),
                    "text_score": float(text_score),
                }
            )

        rows = sorted(rows, key=lambda row: row["final_score"], reverse=True)[: min(top_k, len(rows))]

        results: List[Dict[str, Any]] = []
        for rank, row in enumerate(rows, start=1):
            page = row["page"]
            results.append(
                {
                    "text": row["page_text"],
                    "metadata": {
                        "doc_name": page["doc_name"],
                        "page": int(page["page"]),
                        "pdf_rel_path": page["pdf_rel_path"],
                        "retrieval": "colqwen2_page_visual",
                    },
                    "_score": row["final_score"],
                    "_visual_score": row["visual_score"],
                    "_text_score": row["text_score"],
                    "rank": rank,
                }
            )

        return results

    def free(self) -> None:
        self._model = None
        self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _rebuild_lookups(self) -> None:
        self._page_lookup = _build_page_lookup(self._page_records)
        self._doc_to_page_ids = defaultdict(list)
        for idx, page in enumerate(self._page_records):
            self._doc_to_page_ids[page["doc_name"]].append(idx)

    def _load_model(self) -> None:
        if self._model is not None and self._processor is not None and self._device is not None:
            return

        ColQwen2ForRetrieval, ColQwen2Processor, is_flash_attn_2_available = _load_colqwen2()
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
            model_dtype = torch.bfloat16
            attn_impl = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
            device_map = "auto"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self._device = torch.device("mps")
            model_dtype = torch.float32
            attn_impl = "sdpa"
            device_map = None
        else:
            self._device = torch.device("cpu")
            model_dtype = torch.float32
            attn_impl = "sdpa"
            device_map = None

        logger.info("Loading ColQwen2 model: %s", self.model_name)
        kwargs = {"attn_implementation": attn_impl}
        if device_map is not None:
            kwargs["device_map"] = device_map

        try:
            self._model = ColQwen2ForRetrieval.from_pretrained(
                self.model_name,
                dtype=model_dtype,
                **kwargs,
            ).eval()
        except TypeError:
            self._model = ColQwen2ForRetrieval.from_pretrained(
                self.model_name,
                torch_dtype=model_dtype,
                **kwargs,
            ).eval()

        self._processor = ColQwen2Processor.from_pretrained(self.model_name)

    def _render_pdf_pages(self, pdf_path: Path) -> List[Dict[str, Any]]:
        if Image is None:
            raise RuntimeError("Pillow is required for ColQwen2 PDF rendering.")

        rel_path = pdf_path.relative_to(self.pdf_dir)
        doc_name = pdf_path.stem
        pages: List[Dict[str, Any]] = []
        doc = None
        try:
            doc = fitz.open(pdf_path)
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                pix = page.get_pixmap(dpi=self.dpi)
                mode = "RGBA" if pix.alpha else "RGB"
                img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                if mode == "RGBA":
                    img = img.convert("RGB")
                pages.append(
                    {
                        "image": img,
                        "record": {
                            "doc_name": doc_name,
                            "page": page_idx,
                            "pdf_rel_path": str(rel_path),
                        },
                    }
                )
        except Exception as exc:
            logger.warning("Failed to render PDF %s: %s", pdf_path, exc)
        finally:
            if doc is not None:
                doc.close()

        return pages

    def _load_page_text(self, pdf_rel_path: str, page_idx: int) -> str:
        cache_key = (pdf_rel_path, int(page_idx))
        if cache_key in self._page_text_cache:
            return self._page_text_cache[cache_key]

        pdf_path = self.pdf_dir / pdf_rel_path
        text = ""
        doc = None
        try:
            doc = fitz.open(pdf_path)
            if 0 <= page_idx < len(doc):
                text = doc[page_idx].get_text("text") or ""
        except Exception as exc:
            logger.warning("Failed to read page text from %s page %s: %s", pdf_path, page_idx, exc)
        finally:
            if doc is not None:
                doc.close()

        self._page_text_cache[cache_key] = text
        return text
