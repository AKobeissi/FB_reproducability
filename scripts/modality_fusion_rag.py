#!/usr/bin/env python3
"""
Modality-Fusion Hierarchical RAG
=================================

Two new retrieval variants for FinanceBench that extract and index
tables and text separately, then fuse ranks via Reciprocal Rank Fusion (RRF).

Variant A: hier_marker_fusion  (Open-Source Markdown extraction)
  - Extract SEC PDFs with marker-pdf (falls back to pdfplumber if unavailable)
  - Tables → BM25 sparse index
  - Text → Fine-tuned BGE-M3 / ChromaDB dense index
  - Query both, apply RRF, map back to parent pages → top-5

Variant B: hier_vlm_fusion  (Vision-Language Model extraction)
  - Render PDF pages to PNG using PyMuPDF (no files written to disk)
  - Qwen2-VL-7B-Instruct extracts <table> blocks and plain text per page
  - Tables → BM25, Text → Fine-tuned BGE-M3 / ChromaDB
  - Same RRF pipeline as Variant A

VRAM plan for NVIDIA L40S (48 GB):
  Phase 1 (VLM only): Qwen2-VL-7B bf16 ≈ 14 GB
  Phase 2 (BGE-M3 only): ≈ 2 GB
  → VLM is deleted and CUDA cache cleared before loading BGE-M3.

Results are saved in the same format as baselines/ and hierarchical_rag/:
  modality_fusion_rag/results/predictions/{variant}_retrieval.json
  modality_fusion_rag/results/metrics/{variant}_metrics.json
  modality_fusion_rag/results/metrics/all_variants_metrics.json
  modality_fusion_rag/results/metrics/fusion_table_k5.tex
  modality_fusion_rag/results/metrics/by_question_type.csv
  modality_fusion_rag/results/plots/*.{pdf,png}
"""

import argparse
import copy
import csv
import gc
import io
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fusion_rag")

# ─── Constants ────────────────────────────────────────────────────────────────

EMBED_MODEL   = "BAAI/bge-m3"
FT_MODEL_PATH = str(PROJECT_ROOT / "models" / "fin_adapted_biencoder_bge_m3")
VLM_MODEL     = "Qwen/Qwen2-VL-7B-Instruct"

PAGE_MAX_TOKENS = 2048
CHUNK_SIZE      = 512
CHUNK_OVERLAP   = 64

K_VALUES   = [1, 3, 5, 10, 20]
MAIN_K     = 5
K_TEXT     = 100   # text chunks retrieved before RRF (per-doc after doc-filter)
K_TABLE    = 100   # table chunks retrieved before RRF (per-doc)
RRF_K      = 60    # RRF constant (standard value)
K_RERANK   = 20    # top-K RRF pages fed to cross-encoder reranker
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"  # cross-encoder reranker

QUESTION_TYPES = ["metrics-generated", "domain-relevant", "novel-generated"]

VARIANT_LABELS = {
    "hier_marker_fusion": "Fusion: Marker/PDF + BM25 (RRF)",
    "hier_vlm_fusion":    "Fusion: Qwen2-VL + BM25 (RRF)",
}

VLM_EXTRACTION_PROMPT = (
    "Analyze this financial document page. "
    "Extract all standard text as plain paragraphs. "
    "Extract all financial tables and output them strictly enclosed in <table> HTML tags."
)

# ─── 1. Data loading (same as hierarchical_rag.py) ────────────────────────────

def load_financebench_samples(fb_path: str, doc_info_path: str) -> Tuple[List[Dict], Dict]:
    samples = []
    with open(fb_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            gold_segs = []
            for ev in raw.get("evidence", []):
                gold_segs.append({
                    "text":     ev.get("evidence_text", ""),
                    "doc_name": ev.get("doc_name", raw.get("doc_name", "")),
                    "page":     ev.get("evidence_page_num", -1),
                })
            samples.append({
                "financebench_id":        raw.get("financebench_id", ""),
                "question":               raw.get("question", ""),
                "reference_answer":       raw.get("answer", ""),
                "question_type":          raw.get("question_type", "unknown"),
                "doc_name":               raw.get("doc_name", ""),
                "doc_link":               raw.get("doc_link", ""),
                "gold_evidence_segments": gold_segs,
                "retrieved_chunks":       [],
                "generated_answer":       "",
            })

    doc_info: Dict[str, Dict] = {}
    if doc_info_path and os.path.exists(doc_info_path):
        with open(doc_info_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                doc_info[d["doc_name"]] = d

    counts = {qt: sum(1 for s in samples if s["question_type"] == qt)
              for qt in QUESTION_TYPES}
    logger.info(f"Loaded {len(samples)} FinanceBench samples — {counts}")
    return samples, doc_info


def load_all_pages(samples: List[Dict], pdf_dir: str) -> Dict[str, List[Dict]]:
    """Load pages for all FinanceBench documents. Returns {doc_name → [{text,page_num,doc_name}]}."""
    from src.ingestion.pdf_utils import load_pdf_with_fallback

    doc_pages: Dict[str, List[Dict]] = {}
    unique_docs = {(s["doc_name"], s["doc_link"]) for s in samples}

    for doc_name, doc_link in tqdm(unique_docs, desc="Loading PDFs"):
        if doc_name in doc_pages:
            continue
        try:
            pages, _ = load_pdf_with_fallback(doc_name, doc_link, pdf_dir)
            doc_pages[doc_name] = [
                {
                    "text":     p.page_content,
                    "page_num": p.metadata.get("page", i),
                    "doc_name": doc_name,
                }
                for i, p in enumerate(pages or [])
            ]
        except Exception as e:
            logger.warning(f"Could not load {doc_name}: {e}")
            doc_pages[doc_name] = []

    total = sum(len(v) for v in doc_pages.values())
    logger.info(f"Loaded {len(doc_pages)} docs, {total:,} pages")
    return doc_pages


# ─── 2. Tokenizer & chunking helpers ─────────────────────────────────────────

_tokenizer_cache: Dict = {}


def _get_tokenizer(model_name: str = EMBED_MODEL):
    if model_name not in _tokenizer_cache:
        from transformers import AutoTokenizer
        _tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(
            model_name, use_fast=True
        )
    return _tokenizer_cache[model_name]


def chunk_text_tokens(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    model_name: str = EMBED_MODEL,
) -> List[str]:
    tokenizer = _get_tokenizer(model_name)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return []
    stride = max(1, chunk_size - chunk_overlap)
    chunks = []
    start = 0
    while start < len(token_ids):
        end = min(start + chunk_size, len(token_ids))
        chunk_ids = token_ids[start:end]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
        if chunk_text:
            chunks.append(chunk_text)
        if end >= len(token_ids):
            break
        start += stride
    return chunks


# ─── 3. PDF path resolver (mirrors pdf_utils._find_local_pdf) ────────────────

def _normalize_name(name: str) -> str:
    name = (name or "").lower()
    if "." in name:
        name = name.rsplit(".", 1)[0]
    return re.sub(r"[^a-z0-9]+", "", name)


def find_pdf_path(doc_name: str, pdf_dir: str) -> Optional[Path]:
    """Return Path to a local PDF matching doc_name, or None."""
    pdf_dir_path = Path(pdf_dir)
    if not pdf_dir_path.is_dir():
        return None
    norm_target = _normalize_name(doc_name)
    candidates = []
    try:
        for p in pdf_dir_path.iterdir():
            if p.is_file() and p.suffix.lower() == ".pdf":
                if _normalize_name(p.stem) == norm_target or norm_target in _normalize_name(p.stem):
                    candidates.append(p)
    except Exception:
        return None
    if candidates:
        return sorted(candidates)[0]
    direct = pdf_dir_path / f"{doc_name}.pdf"
    return direct if direct.exists() else None


# ─── 4. Markdown table helpers ────────────────────────────────────────────────

def _pdfplumber_table_to_markdown(table: List[List]) -> str:
    """Convert a pdfplumber table (list of rows, each row is list of cell strings) to Markdown."""
    if not table:
        return ""
    rows = []
    for i, row in enumerate(table):
        cells = [str(c or "").replace("|", "\\|").replace("\n", " ") for c in row]
        rows.append("| " + " | ".join(cells) + " |")
        if i == 0:
            rows.append("| " + " | ".join(["---"] * len(cells)) + " |")
    return "\n".join(rows)


def _extract_page_content_pdfplumber(pdf_path: str) -> List[Dict]:
    """
    Extract text and tables from each page of a PDF using pdfplumber.
    Returns: [{page_num, text_chunks, table_strings}]
    """
    import pdfplumber

    results = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            # ── Tables ──────────────────────────────────────────────────────
            table_strings = []
            try:
                tables = page.extract_tables()
                for t in tables:
                    md = _pdfplumber_table_to_markdown(t)
                    if md and len(md) > 20:
                        table_strings.append(md)
            except Exception as e:
                logger.debug(f"pdfplumber table extraction failed page {page_num}: {e}")

            # ── Text (excluding table bounding boxes) ────────────────────────
            text = ""
            try:
                table_finder = page.find_tables()
                table_bboxes = [t.bbox for t in table_finder]

                if table_bboxes:
                    def not_in_table(obj):
                        for bbox in table_bboxes:
                            if (obj.get("x0", 0) >= bbox[0] and
                                    obj.get("x1", 0) <= bbox[2] and
                                    obj.get("top", 0) >= bbox[1] and
                                    obj.get("bottom", 0) <= bbox[3]):
                                return False
                        return True
                    text = page.filter(not_in_table).extract_text() or ""
                else:
                    text = page.extract_text() or ""
            except Exception as e:
                logger.debug(f"pdfplumber text extraction failed page {page_num}: {e}")
                try:
                    text = page.extract_text() or ""
                except Exception:
                    text = ""

            text_chunks = chunk_text_tokens(text.strip()) if text.strip() else []
            results.append({
                "page_num":     page_num,
                "text_chunks":  text_chunks,
                "table_strings": table_strings,
            })
    return results


def _extract_page_content_marker(pdf_path: str) -> Optional[List[Dict]]:
    """
    Try to extract text and tables from a PDF using marker-pdf.
    Returns None if marker is not installed.
    """
    # Attempt marker-pdf (may not be installed)
    try:
        # marker >= 1.x API
        try:
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
            converter = PdfConverter(artifact_path=None)
            rendered = converter(pdf_path)
            markdown = rendered.markdown
        except (ImportError, AttributeError):
            # marker 0.2.x API
            from marker.convert import convert_single_pdf
            from marker.models import load_all_models
            models = load_all_models()
            markdown, _, _ = convert_single_pdf(pdf_path, models)

    except ImportError:
        return None

    except Exception as e:
        logger.warning(f"marker-pdf failed for {pdf_path}: {e}")
        return None

    # marker returns a single markdown string for the whole document.
    # We split by page separator (marker uses \n\n---\n\n or form-feed).
    pages_md = re.split(r"\n\n---\n\n|\f", markdown)

    results = []
    for page_num, page_md in enumerate(pages_md):
        # Extract markdown tables: lines that start and end with |
        table_pattern = re.compile(
            r"(\|.+\|\n(?:\|[-:| ]+\|\n)?(?:\|.+\|\n)*)",
            re.MULTILINE,
        )
        table_strings = []
        for m in table_pattern.finditer(page_md):
            t = m.group(0).strip()
            if t and len(t) > 20:
                table_strings.append(t)

        # Remove table blocks to get plain text
        text = table_pattern.sub("", page_md).strip()
        # Also remove image links/HTML
        text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
        text = re.sub(r"<[^>]+>", "", text)
        text_chunks = chunk_text_tokens(text) if text else []

        results.append({
            "page_num":      page_num,
            "text_chunks":   text_chunks,
            "table_strings": table_strings,
        })
    return results


# ─── 5. VLM extraction (Qwen2-VL-7B) ─────────────────────────────────────────

def _render_page_to_pil(pdf_path: str, page_num: int, dpi: int = 144):
    """Render a single PDF page to a PIL Image using PyMuPDF. No file is written."""
    import pymupdf as fitz
    from PIL import Image

    doc = fitz.open(pdf_path)
    if page_num >= len(doc):
        doc.close()
        return None
    page = doc[page_num]
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img_bytes = pix.tobytes("png")
    doc.close()
    del pix
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return pil_img


def _parse_vlm_response(response: str) -> Tuple[str, List[str]]:
    """
    Parse Qwen2-VL response into (plain_text, table_strings).
    Table content is enclosed in <table>...</table> tags.
    """
    table_pattern = re.compile(r"<table>(.*?)</table>", re.DOTALL | re.IGNORECASE)
    tables = [m.strip() for m in table_pattern.findall(response) if m.strip()]
    plain_text = table_pattern.sub("", response).strip()
    # Clean up extra whitespace
    plain_text = re.sub(r"\n{3,}", "\n\n", plain_text).strip()
    return plain_text, tables


def extract_all_content_vlm(
    doc_pages: Dict[str, List[Dict]],
    pdf_dir: str,
    cache_dir: str,
    batch_size: int = 4,
) -> Dict[str, List[Dict]]:
    """
    Phase 1: Run Qwen2-VL-7B on every PDF page to extract text and tables.
    Stores results in cache_dir/{doc_name}.json.
    Returns {doc_name → [{page_num, text_chunks, table_strings}]}.

    The VLM model is loaded once, all PDFs are processed, then freed.
    """
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    os.makedirs(cache_dir, exist_ok=True)
    extraction_results: Dict[str, List[Dict]] = {}

    # Determine which docs need processing
    docs_to_process = []
    for doc_name in doc_pages:
        cache_path = os.path.join(cache_dir, f"{doc_name}.json")
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                cached = json.load(f)
            # Validate: reject cache entries where every page is empty (corrupted/failed run)
            has_content = any(
                p.get("text_chunks") or p.get("table_strings")
                for p in cached
            )
            if cached and has_content:
                extraction_results[doc_name] = cached
                logger.info(f"Cache hit: {doc_name}")
            else:
                logger.warning(f"Cache for {doc_name} is empty/corrupted — re-processing.")
                os.remove(cache_path)
                docs_to_process.append(doc_name)
        else:
            docs_to_process.append(doc_name)

    if not docs_to_process:
        logger.info("All VLM extractions cached — skipping VLM load.")
        return extraction_results

    logger.info(f"Loading Qwen2-VL-7B-Instruct (bfloat16) for {len(docs_to_process)} docs…")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        VLM_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    processor = AutoProcessor.from_pretrained(VLM_MODEL)
    logger.info("Qwen2-VL-7B loaded.")

    def _process_single_page(pil_img) -> str:
        """Run VLM on one image, return raw text response."""
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": VLM_EXTRACTION_PROMPT},
            ],
        }]
        try:
            text_input = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text_input],
                images=[pil_img],
                padding=True,
                return_tensors="pt",
            ).to(model.device)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )
            response = processor.decode(
                output_ids[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True,
            ).strip()
            del inputs, output_ids
            return response
        except Exception as e:
            import traceback
            logger.warning(f"VLM inference failed: {e}\n{traceback.format_exc()}")
            return ""

    for doc_name in tqdm(docs_to_process, desc="VLM extraction (docs)"):
        pdf_path = find_pdf_path(doc_name, pdf_dir)
        if pdf_path is None:
            logger.warning(f"PDF not found for {doc_name}, skipping VLM extraction.")
            extraction_results[doc_name] = []
            continue

        num_pages = len(doc_pages.get(doc_name, []))
        if num_pages == 0:
            extraction_results[doc_name] = []
            continue

        doc_results: List[Dict] = []

        for page_num in tqdm(range(num_pages), desc=f"  {doc_name}", leave=False):
            try:
                pil_img = _render_page_to_pil(str(pdf_path), page_num, dpi=144)
                if pil_img is None:
                    doc_results.append({"page_num": page_num, "text_chunks": [], "table_strings": []})
                    continue

                response = _process_single_page(pil_img)
                del pil_img  # free immediately

                plain_text, tables = _parse_vlm_response(response)
                text_chunks = chunk_text_tokens(plain_text) if plain_text else []

                doc_results.append({
                    "page_num":      page_num,
                    "text_chunks":   text_chunks,
                    "table_strings": tables,
                })

                # Free GPU cache periodically
                if page_num % 20 == 0:
                    torch.cuda.empty_cache()

            except Exception as e:
                import traceback
                logger.warning(f"Failed page {page_num} of {doc_name}: {e}\n{traceback.format_exc()}")
                doc_results.append({"page_num": page_num, "text_chunks": [], "table_strings": []})

        extraction_results[doc_name] = doc_results
        cache_path = os.path.join(cache_dir, f"{doc_name}.json")
        with open(cache_path, "w") as f:
            json.dump(doc_results, f)
        logger.info(f"Cached VLM extraction: {doc_name} ({num_pages} pages)")

    # ── Free VLM before returning ─────────────────────────────────────────────
    logger.info("Freeing Qwen2-VL-7B from GPU memory…")
    del model, processor
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("Qwen2-VL freed. CUDA cache cleared.")

    return extraction_results


def extract_all_content_marker(
    doc_pages: Dict[str, List[Dict]],
    pdf_dir: str,
    cache_dir: str,
) -> Dict[str, List[Dict]]:
    """
    Extract text + tables from all PDFs using marker-pdf (with pdfplumber fallback).
    Results are cached to cache_dir/{doc_name}.json.
    Returns {doc_name → [{page_num, text_chunks, table_strings}]}.
    """
    os.makedirs(cache_dir, exist_ok=True)
    extraction_results: Dict[str, List[Dict]] = {}

    for doc_name in tqdm(doc_pages, desc="Marker extraction"):
        cache_path = os.path.join(cache_dir, f"{doc_name}.json")
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                extraction_results[doc_name] = json.load(f)
            continue

        pdf_path = find_pdf_path(doc_name, pdf_dir)
        if pdf_path is None:
            logger.warning(f"PDF not found for {doc_name}")
            extraction_results[doc_name] = []
            continue

        # Try marker-pdf first, fall back to pdfplumber
        page_data = _extract_page_content_marker(str(pdf_path))
        if page_data is None:
            logger.debug(f"marker-pdf unavailable or failed for {doc_name}, using pdfplumber")
            try:
                page_data = _extract_page_content_pdfplumber(str(pdf_path))
            except Exception as e:
                logger.warning(f"pdfplumber also failed for {doc_name}: {e}")
                page_data = []

        extraction_results[doc_name] = page_data
        with open(cache_path, "w") as f:
            json.dump(page_data, f)

    total_text  = sum(sum(len(p["text_chunks"])  for p in v) for v in extraction_results.values())
    total_table = sum(sum(len(p["table_strings"]) for p in v) for v in extraction_results.values())
    logger.info(
        f"Extraction complete: {total_text:,} text chunks, {total_table:,} table strings"
    )
    return extraction_results


# ─── 6. Build dual index ──────────────────────────────────────────────────────

def build_dual_index(
    extraction_results: Dict[str, List[Dict]],
    ft_model_path: str,
    vs_dir: str,
    collection_name: str,
) -> Tuple[object, List[Dict], object, callable]:
    """
    Build:
      - ChromaDB collection (BGE-M3 embeddings) over all text chunks
      - BM25 index over all table strings

    Returns (text_collection, table_flat, bm25_index, bm25_tokenize).
    """
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    from rank_bm25 import BM25Okapi

    # ── Text index (ChromaDB + BGE-M3) ────────────────────────────────────────
    os.makedirs(vs_dir, exist_ok=True)
    ef = SentenceTransformerEmbeddingFunction(
        model_name=ft_model_path, device="cuda", trust_remote_code=True
    )
    client = chromadb.PersistentClient(path=vs_dir)
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    existing_ids = set(collection.get(include=[])["ids"])
    docs_to_add, metas_to_add, ids_to_add = [], [], []

    for doc_name, pages in tqdm(extraction_results.items(), desc=f"Indexing text → {collection_name}"):
        for page_data in pages:
            page_num = page_data["page_num"]
            for chunk_idx, chunk_text in enumerate(page_data.get("text_chunks", [])):
                chunk_id = f"{doc_name}__p{page_num}__c{chunk_idx}"
                if chunk_id in existing_ids:
                    continue
                docs_to_add.append(chunk_text[:4096])
                metas_to_add.append({"doc_name": doc_name, "page": int(page_num)})
                ids_to_add.append(chunk_id)

                if len(docs_to_add) >= 512:
                    collection.add(documents=docs_to_add, metadatas=metas_to_add, ids=ids_to_add)
                    docs_to_add, metas_to_add, ids_to_add = [], [], []

    if docs_to_add:
        collection.add(documents=docs_to_add, metadatas=metas_to_add, ids=ids_to_add)
    logger.info(f"Text index '{collection_name}': {collection.count():,} chunks")

    # ── Table index (BM25) ────────────────────────────────────────────────────
    table_flat: List[Dict] = []
    for doc_name, pages in extraction_results.items():
        for page_data in pages:
            page_num = page_data["page_num"]
            for table_str in page_data.get("table_strings", []):
                table_flat.append({
                    "text":     table_str,
                    "metadata": {"doc_name": doc_name, "page": int(page_num)},
                })

    def _bm25_tokenize(text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^\w\s$.,%-]", " ", text)
        return text.split()

    corpus_tokens = [_bm25_tokenize(t["text"]) for t in table_flat]
    bm25_index = BM25Okapi(corpus_tokens) if corpus_tokens else None
    logger.info(f"Table BM25 index: {len(table_flat):,} table strings")

    return collection, table_flat, bm25_index, _bm25_tokenize


# ─── 7. RRF retrieval ─────────────────────────────────────────────────────────

def retrieve_modality_fusion(
    sample: Dict,
    embed_model,
    text_collection,
    table_flat: List[Dict],
    bm25_index,
    bm25_tokenize,
    doc_pages: Dict[str, List[Dict]],
    reranker=None,
    k: int = MAIN_K,
    k_text: int = K_TEXT,
    k_table: int = K_TABLE,
    rrf_k: int = RRF_K,
    k_rerank: int = K_RERANK,
    ner_filter=None,
    oracle_doc_name: Optional[str] = None,
) -> List[Dict]:
    """
    Three-stage pipeline:
      1. Doc-filter: restrict both retrievers to predicted documents.
         - If ner_filter (DocContentFilter) is provided: year regex + BM25
           over first-page content predicts the target document(s).
           No oracle labels used; handles vocabulary asymmetry naturally.
         - If oracle_doc_name is set: use the gold label (ablation only).
         - Otherwise: search globally across all documents.
      2. Dual-retrieval + RRF: BGE-M3 text chunks + BM25 table strings,
         ranks aggregated at page level → top-k_rerank candidate pages.
      3. Cross-encoder reranker: re-scores the top-k_rerank pages with
         BAAI/bge-reranker-v2-m3, returns top-k final pages.
    """
    question = sample["question"]

    # ── Stage 0: Determine document filter ────────────────────────────────────
    if oracle_doc_name:
        # Ablation mode: use gold label (cheating)
        predicted_docs = [oracle_doc_name]
    elif ner_filter is not None:
        # NER-based prediction (proper approach)
        top_k = getattr(ner_filter, "_default_top_k", 3)
        predicted_docs = ner_filter.predict_target_docs(question, top_k=top_k)
    else:
        predicted_docs = []  # global search

    # Build ChromaDB where-filter from predicted docs
    if len(predicted_docs) == 1:
        where_filter: Optional[Dict] = {"doc_name": predicted_docs[0]}
    elif len(predicted_docs) > 1:
        where_filter = {"doc_name": {"$in": predicted_docs}}
    else:
        where_filter = None

    # ── Stage 1: Dense text retrieval (doc-filtered) ──────────────────────────
    q_emb = embed_model.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    rrf_page_scores: Dict[Tuple, float] = {}
    rrf_page_best_chunk: Dict[Tuple, Dict] = {}

    try:
        query_kwargs = dict(
            query_embeddings=[q_emb.tolist()],
            n_results=k_text,
            include=["documents", "metadatas", "distances"],
        )
        if where_filter:
            query_kwargs["where"] = where_filter

        text_res = text_collection.query(**query_kwargs)

        for rank, (text, meta, dist) in enumerate(
            zip(text_res["documents"][0], text_res["metadatas"][0], text_res["distances"][0]),
            start=1,
        ):
            page_key = (meta["doc_name"], int(meta["page"]))
            score = 1.0 / (rrf_k + rank)
            rrf_page_scores[page_key] = rrf_page_scores.get(page_key, 0.0) + score
            if page_key not in rrf_page_best_chunk:
                rrf_page_best_chunk[page_key] = {
                    "text":     text,
                    "metadata": {"doc_name": meta["doc_name"], "page": int(meta["page"])},
                    "_source":  "text",
                }
    except Exception as e:
        logger.warning(f"Dense retrieval failed (predicted_docs={predicted_docs}): {e}")

    # ── Stage 2: Sparse table retrieval (BM25, doc-filtered) ──────────────────
    if bm25_index is not None and table_flat:
        query_tokens = bm25_tokenize(question)
        bm25_scores = bm25_index.get_scores(query_tokens)

        # Apply doc-filter: zero out scores for documents not in predicted set
        if predicted_docs:
            predicted_set = set(predicted_docs)
            for i, chunk in enumerate(table_flat):
                if chunk["metadata"]["doc_name"] not in predicted_set:
                    bm25_scores[i] = 0.0

        top_table_idx = np.argsort(bm25_scores)[::-1][:k_table]
        for rank, idx in enumerate(top_table_idx, start=1):
            if bm25_scores[int(idx)] <= 0.0:
                break  # no more relevant results after zeroing
            chunk = table_flat[int(idx)]
            meta = chunk["metadata"]
            page_key = (meta["doc_name"], int(meta["page"]))
            score = 1.0 / (rrf_k + rank)
            rrf_page_scores[page_key] = rrf_page_scores.get(page_key, 0.0) + score
            if page_key not in rrf_page_best_chunk:
                rrf_page_best_chunk[page_key] = {
                    "text":     chunk["text"],
                    "metadata": {"doc_name": meta["doc_name"], "page": int(meta["page"])},
                    "_source":  "table",
                }

    # ── Stage 3: RRF aggregation → top-k_rerank candidate pages ──────────────
    sorted_pages = sorted(
        rrf_page_scores.keys(), key=lambda pk: rrf_page_scores[pk], reverse=True
    )
    candidates = sorted_pages[:k_rerank]

    # Build full-page text for each candidate (used for reranking and output)
    candidate_pages = []
    for page_key in candidates:
        pname, pnum = page_key
        full_text = rrf_page_best_chunk[page_key]["text"]
        for p in doc_pages.get(pname, []):
            if int(p["page_num"]) == pnum:
                full_text = p["text"]
                break
        candidate_pages.append({
            "text":      full_text,
            "metadata":  {"doc_name": pname, "page": pnum},
            "_rrf_score": rrf_page_scores[page_key],
        })

    if not candidate_pages:
        return []

    # ── Stage 4: Cross-encoder reranking ─────────────────────────────────────
    if reranker is not None and len(candidate_pages) > k:
        try:
            pairs = [[question, p["text"][:2048]] for p in candidate_pages]
            ce_scores = reranker.predict(pairs, show_progress_bar=False)
            ranked_idx = np.argsort(ce_scores)[::-1]
            candidate_pages = [candidate_pages[i] for i in ranked_idx]
        except Exception as e:
            logger.warning(f"Reranker failed, using RRF order: {e}")

    # Return up to max(K_VALUES) results so metrics at all k levels are meaningful
    max_k = max(K_VALUES)
    results = []
    for rank, page in enumerate(candidate_pages[:max_k], start=1):
        results.append({
            "text":     page["text"],
            "metadata": page["metadata"],
            "rank":     rank,
            "_score":   page.get("_rrf_score", 0.0),
        })
    return results


# ─── 8. Embedding model helper ────────────────────────────────────────────────

_embed_model_cache: Dict = {}


def load_embed_model(model_path: str, device: str = "cuda"):
    if model_path not in _embed_model_cache:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {model_path}")
        _embed_model_cache[model_path] = SentenceTransformer(
            model_path, device=device, trust_remote_code=True
        )
    return _embed_model_cache[model_path]


def load_reranker(model_name: str = RERANK_MODEL, device: str = "cuda"):
    """Load a cross-encoder reranker (sentence-transformers CrossEncoder)."""
    try:
        from sentence_transformers import CrossEncoder
        logger.info(f"Loading reranker: {model_name}")
        return CrossEncoder(model_name, device=device, trust_remote_code=True)
    except Exception as e:
        logger.warning(f"Could not load reranker {model_name}: {e}. Reranking disabled.")
        return None


# ─── 9. Evaluation ────────────────────────────────────────────────────────────

def compute_retrieval_metrics(samples: List[Dict], k_values: List[int] = K_VALUES) -> Dict:
    from src.evaluation.retrieval_evaluator import RetrievalEvaluator
    ev = RetrievalEvaluator()
    return ev.compute_metrics(samples, k_values=k_values)


def _numeric_match(pred: str, ref: str, rtol: float = 0.03) -> float:
    def _extract(text: str) -> Optional[float]:
        text = re.sub(r"[$,€£%]", "", text)
        matches = re.findall(r"-?\d[\d,]*\.?\d*", text)
        if not matches:
            return None
        try:
            return float(matches[0].replace(",", ""))
        except ValueError:
            return None
    pn, rn = _extract(pred), _extract(ref)
    if pn is None or rn is None:
        return 0.0
    if rn == 0:
        return 1.0 if pn == 0 else 0.0
    return 1.0 if abs(pn - rn) / abs(rn) <= rtol else 0.0


def compute_generative_metrics(samples: List[Dict]) -> Dict:
    try:
        from rouge_score import rouge_scorer as rs_module
        scorer = rs_module.RougeScorer(["rougeL"], use_stemmer=True)
    except ImportError:
        scorer = None

    rougeL_scores = []
    numeric_matches = []
    for s in samples:
        gen = s.get("generated_answer", "")
        ref = s.get("reference_answer", "")
        if gen and ref and scorer:
            score = scorer.score(ref, gen)["rougeL"].fmeasure
        else:
            score = 0.0
        rougeL_scores.append(score)
        if s.get("question_type") == "metrics-generated":
            numeric_matches.append(_numeric_match(gen, ref))

    return {
        "answer_rougeL": float(np.mean(rougeL_scores)) if rougeL_scores else 0.0,
        "numeric_match":  float(np.mean(numeric_matches)) if numeric_matches else 0.0,
        "n_samples":      len(samples),
    }


def aggregate_by_group(samples: List[Dict], key_fn, k_values=K_VALUES) -> Dict:
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for s in samples:
        groups[key_fn(s)].append(s)
    return {
        grp: {**compute_retrieval_metrics(gs, k_values), **compute_generative_metrics(gs)}
        for grp, gs in groups.items()
    }


# ─── 10. Saving results ───────────────────────────────────────────────────────

def save_predictions(samples: List[Dict], variant: str, results_dir: str) -> None:
    pred_dir = os.path.join(results_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    path = os.path.join(pred_dir, f"{variant}_retrieval.json")
    with open(path, "w") as f:
        json.dump(samples, f, indent=2)
    logger.info(f"Predictions saved: {path}")


def save_metrics(metrics: Dict, variant: str, results_dir: str) -> None:
    met_dir = os.path.join(results_dir, "metrics")
    os.makedirs(met_dir, exist_ok=True)
    path = os.path.join(met_dir, f"{variant}_metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved: {path}")


def save_metrics_table(all_results: Dict[str, Dict], results_dir: str) -> None:
    met_dir = os.path.join(results_dir, "metrics")
    os.makedirs(met_dir, exist_ok=True)

    with open(os.path.join(met_dir, "all_variants_metrics.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    headers = (
        ["Method"]
        + [f"DocRec@{k}" for k in K_VALUES]
        + [f"PageRec@{k}" for k in K_VALUES]
        + [f"BLEU@{MAIN_K}", f"ROUGE-L@{MAIN_K}"]
        + ["AnsROUGE-L", "NumericMatch"]
    )
    csv_path = os.path.join(met_dir, "fusion_table.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for name, label in VARIANT_LABELS.items():
            if name not in all_results:
                continue
            r = all_results[name]
            row = [label]
            row += [f"{r.get(f'doc_recall@{k}', float('nan')):.3f}"  for k in K_VALUES]
            row += [f"{r.get(f'page_recall@{k}', float('nan')):.3f}" for k in K_VALUES]
            row += [
                f"{r.get(f'context_bleu@{MAIN_K}',   float('nan')):.3f}",
                f"{r.get(f'context_rougeL@{MAIN_K}', float('nan')):.3f}",
                f"{r.get('answer_rougeL', float('nan')):.3f}",
                f"{r.get('numeric_match', float('nan')):.3f}",
            ]
            w.writerow(row)
    logger.info(f"CSV table: {csv_path}")

    tex_path = os.path.join(met_dir, "fusion_table_k5.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{tabular}{lcccccc}\n\\toprule\n")
        f.write("\\textbf{Method} & \\textbf{DocRec@5} & \\textbf{PageRec@5} "
                "& \\textbf{BLEU@5} & \\textbf{ROUGE-L@5} "
                "& \\textbf{AnsROUGE-L} & \\textbf{NumMatch} \\\\\n\\midrule\n")
        for name, label in VARIANT_LABELS.items():
            if name not in all_results:
                continue
            r = all_results[name]
            f.write(
                f"{label} & "
                f"{r.get('doc_recall@5',      0):.3f} & "
                f"{r.get('page_recall@5',     0):.3f} & "
                f"{r.get('context_bleu@5',    0):.3f} & "
                f"{r.get('context_rougeL@5',  0):.3f} & "
                f"{r.get('answer_rougeL',     0):.3f} & "
                f"{r.get('numeric_match',     0):.3f} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n")
    logger.info(f"LaTeX table: {tex_path}")


def save_by_type_table(by_type_all: Dict[str, Dict], results_dir: str) -> None:
    met_dir = os.path.join(results_dir, "metrics")
    os.makedirs(met_dir, exist_ok=True)
    path = os.path.join(met_dir, "by_question_type.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Method", "QuestionType", "N",
                    f"DocRec@{MAIN_K}", f"PageRec@{MAIN_K}",
                    f"BLEU@{MAIN_K}", f"ROUGE-L@{MAIN_K}",
                    "AnsROUGE-L", "NumericMatch"])
        for name, label in VARIANT_LABELS.items():
            if name not in by_type_all:
                continue
            for qt, m in by_type_all[name].items():
                w.writerow([
                    label, qt, m.get("n_samples", 0),
                    f"{m.get(f'doc_recall@{MAIN_K}',    float('nan')):.3f}",
                    f"{m.get(f'page_recall@{MAIN_K}',   float('nan')):.3f}",
                    f"{m.get(f'context_bleu@{MAIN_K}',  float('nan')):.3f}",
                    f"{m.get(f'context_rougeL@{MAIN_K}',float('nan')):.3f}",
                    f"{m.get('answer_rougeL', float('nan')):.3f}",
                    f"{m.get('numeric_match', float('nan')):.3f}",
                ])
    logger.info(f"By-type table: {path}")


# ─── 11. Plots ────────────────────────────────────────────────────────────────

def _save_fig(fig, path: str) -> None:
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"Plot: {path}")


def plot_recall_at_k(all_results: Dict[str, Dict], results_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    plots_dir = os.path.join(results_dir, "plots")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = cm.Set1(np.linspace(0, 1, max(len(VARIANT_LABELS), 2)))

    for ax, metric_prefix, ylabel in [
        (axes[0], "page_recall", "PageRec@k"),
        (axes[1], "doc_recall",  "DocRec@k"),
    ]:
        for (name, label), color in zip(VARIANT_LABELS.items(), colors):
            if name not in all_results:
                continue
            vals = [all_results[name].get(f"{metric_prefix}@{k}", 0) for k in K_VALUES]
            ax.plot(K_VALUES, vals, marker="o", label=label, color=color, linewidth=2)
        ax.set_xlabel("k")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.set_xticks(K_VALUES)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Modality-Fusion RAG — Recall Curves", fontsize=11)
    fig.tight_layout()
    _save_fig(fig, os.path.join(plots_dir, "recall_at_k_curves.pdf"))


def plot_bar_k5(all_results: Dict[str, Dict], results_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = os.path.join(results_dir, "plots")
    names = [label for name, label in VARIANT_LABELS.items() if name in all_results]
    keys  = [name  for name in VARIANT_LABELS          if name in all_results]
    if not keys:
        return

    metric_map = {
        f"doc_recall@{MAIN_K}":     "DocRec@5",
        f"page_recall@{MAIN_K}":    "PageRec@5",
        f"context_bleu@{MAIN_K}":   "BLEU@5",
        f"context_rougeL@{MAIN_K}": "ROUGE-L@5",
    }
    x = np.arange(len(keys))
    width = 0.20
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

    fig, ax = plt.subplots(figsize=(max(8, len(keys) * 2), 5))
    for i, (mk, ml) in enumerate(metric_map.items()):
        vals = [all_results[k].get(mk, 0) for k in keys]
        ax.bar(x + (i - 1.5) * width, vals, width, label=ml, color=colors[i], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(f"Modality-Fusion RAG @ k={MAIN_K}")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, os.path.join(plots_dir, "retrieval_bar_k5.pdf"))


def plot_heatmap_by_type(by_type_all: Dict[str, Dict], results_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = os.path.join(results_dir, "plots")
    method_names = [n for n in VARIANT_LABELS if n in by_type_all]
    if not method_names:
        return

    types  = QUESTION_TYPES
    matrix = np.zeros((len(method_names), len(types)))
    for i, name in enumerate(method_names):
        for j, qt in enumerate(types):
            matrix[i, j] = by_type_all[name].get(qt, {}).get(f"page_recall@{MAIN_K}", 0)

    fig, ax = plt.subplots(figsize=(8, max(3, len(method_names) * 0.8)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels([t.replace("-", "\n") for t in types], fontsize=9)
    ax.set_yticks(range(len(method_names)))
    ax.set_yticklabels([VARIANT_LABELS[n] for n in method_names], fontsize=9)
    for i in range(len(method_names)):
        for j in range(len(types)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax, label="PageRec@5")
    ax.set_title(f"PageRec@{MAIN_K} by Question Type — Modality Fusion")
    fig.tight_layout()
    _save_fig(fig, os.path.join(plots_dir, "heatmap_by_question_type.pdf"))


# ─── 12. Main ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Modality-Fusion Hierarchical RAG for FinanceBench")
    p.add_argument("--data-path",      default="data/financebench_open_source.jsonl")
    p.add_argument("--doc-info-path",  default="data/financebench_document_information.jsonl")
    p.add_argument("--pdf-dir",        default="pdfs")
    p.add_argument("--ft-model",       default=FT_MODEL_PATH)
    p.add_argument("--results-dir",    default="modality_fusion_rag/results")
    p.add_argument("--vs-dir",         default="modality_fusion_rag/vector_store")
    p.add_argument("--cache-dir",      default="modality_fusion_rag/cache")
    p.add_argument("--variants",       nargs="+",
                   default=list(VARIANT_LABELS.keys()),
                   choices=list(VARIANT_LABELS.keys()))
    p.add_argument("--k",              type=int, default=MAIN_K)
    p.add_argument("--k-text",         type=int, default=K_TEXT,
                   help="Text chunks retrieved per query before RRF")
    p.add_argument("--k-table",        type=int, default=K_TABLE,
                   help="Table chunks retrieved per query before RRF")
    p.add_argument("--rrf-k",          type=int, default=RRF_K,
                   help="RRF smoothing constant")
    p.add_argument("--no-rerank",      action="store_true",
                   help="Disable cross-encoder reranking (faster, lower recall)")
    p.add_argument("--rerank-model",   default=RERANK_MODEL,
                   help="HuggingFace cross-encoder model for reranking")
    p.add_argument("--k-rerank",       type=int, default=K_RERANK,
                   help="Number of RRF candidate pages passed to reranker")
    p.add_argument("--skip-extract",   action="store_true",
                   help="Skip extraction phase (use cache only; error if cache missing)")
    p.add_argument("--rebuild-index",  action="store_true",
                   help="Delete and rebuild ChromaDB collection from scratch")
    p.add_argument("--oracle-doc-filter", action="store_true",
                   help="Use gold doc_name label for filtering (oracle / ablation only)")
    p.add_argument("--no-ner-filter",  action="store_true",
                   help="Disable NER doc filtering (retrieve globally across all docs)")
    p.add_argument("--ner-top-k",      type=int, default=3,
                   help="Number of candidate documents returned by NER filter (default 3)")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    def _p(path):
        return path if os.path.isabs(path) else str(PROJECT_ROOT / path)

    fb_path       = _p(args.data_path)
    doc_info_path = _p(args.doc_info_path)
    pdf_dir       = _p(args.pdf_dir)
    ft_model_path = args.ft_model if os.path.exists(args.ft_model) else _p(args.ft_model)
    results_dir   = _p(args.results_dir)
    vs_dir        = _p(args.vs_dir)
    cache_dir     = _p(args.cache_dir)

    for d in [results_dir, vs_dir, cache_dir]:
        os.makedirs(d, exist_ok=True)

    # ── Phase 1: Load FinanceBench data & PDFs ────────────────────────────────
    logger.info("=== Phase 1: Loading data & PDFs ===")
    samples, doc_info = load_financebench_samples(fb_path, doc_info_path)
    doc_pages = load_all_pages(samples, pdf_dir)

    # ── Doc filter (replaces oracle doc_name lookup) ─────────────────────────
    # Uses BM25 over first-page content (company + year; no alias maps).
    ner_filter = None
    if args.oracle_doc_filter:
        logger.info("Doc filter: ORACLE (gold label — for ablation only)")
    elif args.no_ner_filter:
        logger.info("Doc filter: DISABLED (global search across all docs)")
    else:
        from ner_doc_filter import DocContentFilter
        logger.info(
            "Doc filter: content-based BM25 over first-page text "
            "(company+year, no oracle labels)"
        )
        ner_filter = DocContentFilter(
            doc_pages=doc_pages,
            doc_info=doc_info,
            n_header_pages=3,
            year_window=1,
        )
        ner_filter._default_top_k = args.ner_top_k
        logger.info(f"Doc filter ready (top_k={args.ner_top_k})")

    all_results:  Dict[str, Dict] = {}
    by_type_all:  Dict[str, Dict] = {}

    # ── Load reranker once (shared across both variants) ──────────────────────
    reranker = None if args.no_rerank else load_reranker(args.rerank_model)

    # ── Phase 2a: hier_marker_fusion ──────────────────────────────────────────
    if "hier_marker_fusion" in args.variants:
        variant = "hier_marker_fusion"
        logger.info(f"=== Phase 2a: {variant} ===")

        marker_cache  = os.path.join(cache_dir, "marker")
        marker_vs     = os.path.join(vs_dir, "marker")
        collection_nm = "fusion_marker_text"

        if args.skip_extract:
            # Load only from cache
            extraction_results: Dict[str, List[Dict]] = {}
            for doc_name in doc_pages:
                cp = os.path.join(marker_cache, f"{doc_name}.json")
                if not os.path.exists(cp):
                    logger.warning(f"Cache missing for {doc_name} — will have empty index.")
                    extraction_results[doc_name] = []
                else:
                    with open(cp) as f:
                        extraction_results[doc_name] = json.load(f)
        else:
            extraction_results = extract_all_content_marker(
                doc_pages, pdf_dir, marker_cache
            )

        # Optionally wipe and rebuild the vector store
        if args.rebuild_index:
            import shutil
            if os.path.exists(marker_vs):
                shutil.rmtree(marker_vs)
                logger.info(f"Deleted existing vector store: {marker_vs}")

        text_collection, table_flat, bm25_index, bm25_tokenize = build_dual_index(
            extraction_results, ft_model_path, marker_vs, collection_nm
        )
        embed_model = load_embed_model(ft_model_path)

        var_samples = copy.deepcopy(samples)
        for s in tqdm(var_samples, desc=variant):
            s["retrieved_chunks"] = retrieve_modality_fusion(
                s, embed_model, text_collection, table_flat,
                bm25_index, bm25_tokenize, doc_pages,
                reranker=reranker,
                k=args.k, k_text=args.k_text, k_table=args.k_table,
                rrf_k=args.rrf_k, k_rerank=args.k_rerank,
                ner_filter=ner_filter,
                oracle_doc_name=s.get("doc_name", "") if args.oracle_doc_filter else None,
            )

        save_predictions(var_samples, variant, results_dir)
        ret_metrics = compute_retrieval_metrics(var_samples)
        gen_metrics = compute_generative_metrics(var_samples)
        by_type     = aggregate_by_group(var_samples, lambda s: s["question_type"])

        full_metrics = {"overall": {**ret_metrics, **gen_metrics}, "by_question_type": by_type}
        if doc_info:
            full_metrics["by_doc_type"] = aggregate_by_group(
                var_samples,
                lambda s: doc_info.get(s["doc_name"], {}).get("doc_type", "unknown"),
            )
        save_metrics(full_metrics, variant, results_dir)
        all_results[variant] = full_metrics["overall"]
        by_type_all[variant] = by_type

        logger.info(
            f"{VARIANT_LABELS[variant]}: "
            f"page_recall@5={ret_metrics.get('page_recall@5', 0):.3f}  "
            f"doc_recall@5={ret_metrics.get('doc_recall@5', 0):.3f}"
        )

        # Free embed model cache between variants to save VRAM
        _embed_model_cache.clear()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # ── Phase 2b: hier_vlm_fusion ─────────────────────────────────────────────
    if "hier_vlm_fusion" in args.variants:
        variant = "hier_vlm_fusion"
        logger.info(f"=== Phase 2b: {variant} ===")

        vlm_cache  = os.path.join(cache_dir, "vlm")
        vlm_vs     = os.path.join(vs_dir, "vlm")
        collection_nm = "fusion_vlm_text"

        if args.skip_extract:
            vlm_extraction: Dict[str, List[Dict]] = {}
            for doc_name in doc_pages:
                cp = os.path.join(vlm_cache, f"{doc_name}.json")
                if not os.path.exists(cp):
                    logger.warning(f"VLM cache missing for {doc_name}.")
                    vlm_extraction[doc_name] = []
                else:
                    with open(cp) as f:
                        vlm_extraction[doc_name] = json.load(f)
        else:
            # ── VLM extraction: loads Qwen2-VL, processes all PDFs, then frees it ──
            vlm_extraction = extract_all_content_vlm(
                doc_pages, pdf_dir, vlm_cache
            )
            # By the time extract_all_content_vlm returns, the VLM has been freed.

        # Optionally wipe and rebuild
        if args.rebuild_index:
            import shutil
            if os.path.exists(vlm_vs):
                shutil.rmtree(vlm_vs)
                logger.info(f"Deleted existing VLM vector store: {vlm_vs}")

        text_collection, table_flat, bm25_index, bm25_tokenize = build_dual_index(
            vlm_extraction, ft_model_path, vlm_vs, collection_nm
        )
        embed_model = load_embed_model(ft_model_path)

        var_samples = copy.deepcopy(samples)
        for s in tqdm(var_samples, desc=variant):
            s["retrieved_chunks"] = retrieve_modality_fusion(
                s, embed_model, text_collection, table_flat,
                bm25_index, bm25_tokenize, doc_pages,
                reranker=reranker,
                k=args.k, k_text=args.k_text, k_table=args.k_table,
                rrf_k=args.rrf_k, k_rerank=args.k_rerank,
                ner_filter=ner_filter,
                oracle_doc_name=s.get("doc_name", "") if args.oracle_doc_filter else None,
            )

        save_predictions(var_samples, variant, results_dir)
        ret_metrics = compute_retrieval_metrics(var_samples)
        gen_metrics = compute_generative_metrics(var_samples)
        by_type     = aggregate_by_group(var_samples, lambda s: s["question_type"])

        full_metrics = {"overall": {**ret_metrics, **gen_metrics}, "by_question_type": by_type}
        if doc_info:
            full_metrics["by_doc_type"] = aggregate_by_group(
                var_samples,
                lambda s: doc_info.get(s["doc_name"], {}).get("doc_type", "unknown"),
            )
        save_metrics(full_metrics, variant, results_dir)
        all_results[variant] = full_metrics["overall"]
        by_type_all[variant] = by_type

        logger.info(
            f"{VARIANT_LABELS[variant]}: "
            f"page_recall@5={ret_metrics.get('page_recall@5', 0):.3f}  "
            f"doc_recall@5={ret_metrics.get('doc_recall@5', 0):.3f}"
        )

    # ── Phase 3: Tables & plots ───────────────────────────────────────────────
    logger.info("=== Phase 3: Saving tables & plots ===")
    save_metrics_table(all_results, results_dir)
    save_by_type_table(by_type_all, results_dir)

    try:
        plot_recall_at_k(all_results, results_dir)
        plot_bar_k5(all_results, results_dir)
        plot_heatmap_by_type(by_type_all, results_dir)
    except Exception as e:
        logger.warning(f"Plotting failed (non-fatal): {e}")

    elapsed = time.time() - t0
    logger.info(f"Done in {elapsed / 60:.1f} min")

    print("\n" + "=" * 100)
    print(f"{'Method':<40} " + "  ".join(f"PR@{k:>2}" for k in K_VALUES) + "  " +
          "  ".join(f"DR@{k:>2}" for k in K_VALUES))
    print("-" * 100)
    for name, label in VARIANT_LABELS.items():
        if name not in all_results:
            continue
        r = all_results[name]
        pr_str = "  ".join(f"{r.get(f'page_recall@{k}', 0):>5.3f}" for k in K_VALUES)
        dr_str = "  ".join(f"{r.get(f'doc_recall@{k}',  0):>5.3f}" for k in K_VALUES)
        print(f"{label:<40}  {pr_str}  {dr_str}")
    print("=" * 100)


if __name__ == "__main__":
    main()
