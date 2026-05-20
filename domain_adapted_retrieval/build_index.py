"""
Build ChromaDB vector indexes of the FinanceBench PDF corpus.

Two indexes are built:
  1. PAGE-LEVEL index  — one vector per page (used by standard dense retrieval)
  2. CHUNK-LEVEL index — one vector per chunk (~400 chars) with parent-page
     metadata (used by hierarchical retrieval)

Key design decisions:
  • Raw page text only in embed_text — NO "Document: X  Page N" prefix.
    The previous version included a doc/page prefix which caused a train/test
    mismatch (training used raw text; inference queried prefix-enriched embeddings).
    Since doc-filtered retrieval uses a ChromaDB `where` clause to restrict
    search to the target document, the prefix is also unnecessary.
  • Cosine similarity throughout (BGE models are cosine-trained).
  • Chunk-level index enables hierarchical retrieval: retrieve fine-grained
    chunks, then aggregate to parent pages for consistent @page evaluation.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pdfplumber
from tqdm import tqdm

logger = logging.getLogger(__name__)

MAX_PAGE_CHARS = 2000
EMBED_BATCH_SIZE = 256
CHROMA_INSERT_BATCH = 500


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def _parse_pdf_name(doc_name: str):
    """
    Parse structured metadata from a PDF filename stem.

    FinanceBench PDF names follow the pattern: TICKER_YEAR_DOCTYPE
    e.g. "AAPL_2022_10K" → company="AAPL", year="2022", doc_type="10K"
    Falls back gracefully for non-standard names.
    """
    parts = doc_name.split("_")
    company  = parts[0] if len(parts) >= 1 else doc_name
    year     = parts[1] if len(parts) >= 2 else ""
    doc_type = parts[2] if len(parts) >= 3 else ""
    return company, year, doc_type


def extract_pdf_pages(
    pdf_path: str,
    doc_name: str,
    max_chars: int = MAX_PAGE_CHARS,
) -> List[Dict]:
    """
    Extract all pages from a PDF file.

    Returns a list of dicts with keys:
        text        : raw page text (stored in ChromaDB; used for chunking)
        embed_text  : text for embedding — prefixed with PDF-derived identifier
                      so global cross-document retrieval has company/year context
        page_num    : 0-indexed page number (matches FinanceBench evidence_page_num)
        doc_name    : document identifier (PDF filename stem)
        pdf_path    : absolute path to the source PDF
        company     : ticker/company extracted from filename
        year        : fiscal year extracted from filename
        doc_type    : document type extracted from filename (10K, 10Q, …)
    """
    company, year, doc_type = _parse_pdf_name(doc_name)
    # Build a short doc-context prefix from the PDF filename.
    # This is derived entirely from the PDF path — no FinanceBench metadata used.
    # Including it in embed_text helps the bi-encoder distinguish pages across
    # companies/years during global (non-filtered) retrieval.
    doc_context = " ".join(filter(None, [company, year, doc_type]))
    pages = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                raw_text = (page.extract_text() or "").strip()
                if not raw_text:
                    continue

                truncated = raw_text[:max_chars]
                # Prefix embed_text with the PDF-filename-derived identifier.
                embed_text = f"{doc_context}: {truncated}" if doc_context else truncated

                pages.append({
                    "text": truncated,
                    "embed_text": embed_text,
                    "page_num": page_idx,      # 0-indexed
                    "doc_name": doc_name,
                    "pdf_path": pdf_path,
                    "company":  company,
                    "year":     year,
                    "doc_type": doc_type,
                })
    except Exception as e:
        logger.error(f"Could not extract pages from {pdf_path}: {e}")

    return pages


def _chunk_page(
    page: Dict,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Dict]:
    """
    Split a page into overlapping text chunks.

    Each chunk dict has:
        text, embed_text : chunk text
        page_num         : parent page (0-indexed)
        doc_name, pdf_path : inherited from parent page
        chunk_idx        : chunk index within the page
    """
    text = page["text"]
    if not text:
        return []

    chunks = []
    start = 0
    idx = 0

    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunk_text = text[start:]
        else:
            ws = text.rfind(" ", start, end)
            end = ws if ws > start else end
            chunk_text = text[start:end]

        chunk_text = chunk_text.strip()
        if chunk_text:
            # Build embed prefix from PDF-filename metadata (same as page-level)
            company  = page.get("company", "")
            year     = page.get("year", "")
            doc_type = page.get("doc_type", "")
            doc_context = " ".join(filter(None, [company, year, doc_type]))
            embed_text = f"{doc_context}: {chunk_text}" if doc_context else chunk_text
            chunks.append({
                "text": chunk_text,
                "embed_text": embed_text,
                "page_num": page["page_num"],
                "doc_name": page["doc_name"],
                "pdf_path": page["pdf_path"],
                "company":  company,
                "year":     year,
                "doc_type": doc_type,
                "chunk_idx": idx,
            })
            idx += 1

        next_start = end - chunk_overlap
        if next_start <= start or next_start >= len(text):
            break
        start = next_start

    return chunks


def _get_all_pdfs(pdf_dir: str) -> Dict[str, str]:
    """Return {doc_name: pdf_path} for all *.pdf files in pdf_dir."""
    mapping = {}
    for p in Path(pdf_dir).glob("*.pdf"):
        mapping[p.stem] = str(p)
    logger.info(f"Found {len(mapping)} PDFs in {pdf_dir}")
    return mapping


# ---------------------------------------------------------------------------
# ChromaDB helpers
# ---------------------------------------------------------------------------

def _get_or_create_client(persist_dir: str):
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        raise ImportError("chromadb not installed. Run: pip install chromadb")

    os.makedirs(persist_dir, exist_ok=True)
    return chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )


def _embed_and_insert(
    collection,
    items: List[Dict],
    model,
    id_fn,
    doc_fn,
    meta_fn,
    embed_fn,
) -> None:
    """Generic batch embed + insert into a ChromaDB collection."""
    logger.info(f"Embedding {len(items)} items …")
    all_embeddings = []
    for start in tqdm(range(0, len(items), EMBED_BATCH_SIZE), desc="Embedding"):
        batch = [embed_fn(it) for it in items[start: start + EMBED_BATCH_SIZE]]
        embs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.extend(embs.tolist())

    logger.info("Inserting into ChromaDB …")
    for start in tqdm(range(0, len(items), CHROMA_INSERT_BATCH), desc="Inserting"):
        end = min(start + CHROMA_INSERT_BATCH, len(items))
        batch = items[start:end]
        collection.add(
            ids=[id_fn(it) for it in batch],
            embeddings=all_embeddings[start:end],
            documents=[doc_fn(it) for it in batch],
            metadatas=[meta_fn(it) for it in batch],
        )


# ---------------------------------------------------------------------------
# Page-level index
# ---------------------------------------------------------------------------

def build_chroma_index(
    config,
    model,
    collection_name: str = None,
    force_rebuild: bool = False,
) -> "chromadb.Collection":
    """
    Build (or load) a page-level ChromaDB index of the FinanceBench PDFs.

    One vector per page; metadata includes doc_name and page number so that
    doc-filtered queries can use `where={"doc_name": ...}`.
    """
    dc = config.data
    persist_dir = dc.chroma_persist_dir
    if collection_name is None:
        collection_name = dc.collection_name

    client = _get_or_create_client(persist_dir)
    existing = [c.name for c in client.list_collections()]

    if collection_name in existing and not force_rebuild:
        col = client.get_collection(collection_name)
        if col.count() > 0:
            logger.info(
                f"Using existing page-level collection '{collection_name}' "
                f"({col.count()} pages). Pass force_rebuild=True to regenerate."
            )
            return col
        else:
            logger.info("Existing collection is empty — rebuilding.")
            client.delete_collection(collection_name)
    elif collection_name in existing and force_rebuild:
        logger.info(f"force_rebuild=True: deleting '{collection_name}'")
        client.delete_collection(collection_name)

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Extract pages
    pdf_files = _get_all_pdfs(dc.financebench_pdf_dir)
    if not pdf_files:
        raise ValueError(f"No PDFs found in {dc.financebench_pdf_dir}")

    all_pages: List[Dict] = []
    logger.info("Extracting page texts from FinanceBench PDFs …")
    for doc_name, pdf_path in tqdm(pdf_files.items(), desc="Extracting"):
        all_pages.extend(extract_pdf_pages(pdf_path, doc_name))

    logger.info(f"Total pages to index: {len(all_pages)}")

    _embed_and_insert(
        collection, all_pages, model,
        id_fn=lambda p: f"{p['doc_name']}_page_{p['page_num']}",
        doc_fn=lambda p: p["text"],
        meta_fn=lambda p: {
            "doc_name":  p["doc_name"],
            "page":      p["page_num"],
            "pdf_path":  p["pdf_path"],
            "company":   p["company"],
            "year":      p["year"],
            "doc_type":  p["doc_type"],
        },
        embed_fn=lambda p: p["embed_text"],
    )

    final_count = collection.count()
    logger.info(f"Page-level index built: {final_count} pages from {len(pdf_files)} docs")

    # Save metadata
    meta_path = os.path.join(persist_dir, f"{collection_name}_metadata.json")
    with open(meta_path, "w") as f:
        json.dump({
            "collection_name": collection_name,
            "type": "page",
            "num_pages": final_count,
            "num_docs": len(pdf_files),
            "doc_names": sorted(pdf_files.keys()),
        }, f, indent=2)

    return collection


# ---------------------------------------------------------------------------
# Chunk-level index (for hierarchical retrieval)
# ---------------------------------------------------------------------------

def build_chunk_index(
    config,
    model,
    collection_name: str = None,
    force_rebuild: bool = False,
) -> "chromadb.Collection":
    """
    Build (or load) a chunk-level ChromaDB index.

    Each page is split into overlapping chunks (~400 chars).  Chunks embed more
    precisely to their specific content (tables, paragraphs) than full pages,
    improving recall for questions that target a specific table row or sentence.

    During retrieval, chunks are fetched first, then aggregated back to their
    parent pages.
    """
    dc = config.data
    rc = config.retrieval
    persist_dir = dc.chroma_persist_dir
    if collection_name is None:
        collection_name = dc.chunk_collection_name

    client = _get_or_create_client(persist_dir)
    existing = [c.name for c in client.list_collections()]

    if collection_name in existing and not force_rebuild:
        col = client.get_collection(collection_name)
        if col.count() > 0:
            logger.info(
                f"Using existing chunk-level collection '{collection_name}' "
                f"({col.count()} chunks). Pass force_rebuild=True to regenerate."
            )
            return col
        else:
            client.delete_collection(collection_name)
    elif collection_name in existing and force_rebuild:
        logger.info(f"force_rebuild=True: deleting '{collection_name}'")
        client.delete_collection(collection_name)

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Extract and chunk pages
    pdf_files = _get_all_pdfs(dc.financebench_pdf_dir)
    if not pdf_files:
        raise ValueError(f"No PDFs found in {dc.financebench_pdf_dir}")

    all_chunks: List[Dict] = []
    logger.info("Extracting and chunking pages for hierarchical index …")
    for doc_name, pdf_path in tqdm(pdf_files.items(), desc="Extracting"):
        pages = extract_pdf_pages(pdf_path, doc_name)
        for page in pages:
            chunks = _chunk_page(page, rc.hier_chunk_size, rc.hier_chunk_overlap)
            all_chunks.extend(chunks)

    logger.info(f"Total chunks to index: {len(all_chunks)}")

    _embed_and_insert(
        collection, all_chunks, model,
        id_fn=lambda c: f"{c['doc_name']}_page_{c['page_num']}_chunk_{c['chunk_idx']}",
        doc_fn=lambda c: c["text"],
        meta_fn=lambda c: {
            "doc_name":  c["doc_name"],
            "page":      c["page_num"],
            "chunk_idx": c["chunk_idx"],
            "pdf_path":  c["pdf_path"],
            "company":   c.get("company", ""),
            "year":      c.get("year", ""),
            "doc_type":  c.get("doc_type", ""),
        },
        embed_fn=lambda c: c["embed_text"],
    )

    final_count = collection.count()
    logger.info(f"Chunk-level index built: {final_count} chunks from {len(pdf_files)} docs")

    meta_path = os.path.join(persist_dir, f"{collection_name}_metadata.json")
    with open(meta_path, "w") as f:
        json.dump({
            "collection_name": collection_name,
            "type": "chunk",
            "chunk_size": rc.hier_chunk_size,
            "chunk_overlap": rc.hier_chunk_overlap,
            "num_chunks": final_count,
            "num_docs": len(pdf_files),
        }, f, indent=2)

    return collection


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def load_chroma_index(config, collection_name: str) -> "chromadb.Collection":
    """Load an existing ChromaDB collection (raises if missing or empty)."""
    client = _get_or_create_client(config.data.chroma_persist_dir)
    col = client.get_collection(collection_name)
    count = col.count()
    if count == 0:
        raise RuntimeError(
            f"Collection '{collection_name}' exists but is empty. "
            "Run build_chroma_index() / build_chunk_index() first."
        )
    logger.info(f"Loaded collection '{collection_name}' with {count} items")
    return col


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from domain_adapted_retrieval.config import ExperimentConfig
    from sentence_transformers import SentenceTransformer

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--collection-name", type=str, default=None)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--baseline", action="store_true",
                        help="Use BGE-M3 baseline model")
    parser.add_argument("--chunks", action="store_true",
                        help="Build chunk-level index instead of page-level")
    args = parser.parse_args()

    cfg = ExperimentConfig()

    if args.baseline:
        model_path = "BAAI/bge-m3"
        col_name = args.collection_name or (
            cfg.data.baseline_chunk_collection_name if args.chunks
            else cfg.data.baseline_collection_name
        )
    else:
        model_path = args.model_path or cfg.training.output_model_path
        col_name = args.collection_name or (
            cfg.data.chunk_collection_name if args.chunks
            else cfg.data.collection_name
        )

    logger.info(f"Loading embedding model: {model_path}")
    emb_model = SentenceTransformer(model_path)

    if args.chunks:
        col = build_chunk_index(cfg, emb_model, collection_name=col_name,
                                force_rebuild=args.force_rebuild)
        print(f"\nChunk index ready: '{col_name}' — {col.count()} chunks")
    else:
        col = build_chroma_index(cfg, emb_model, collection_name=col_name,
                                 force_rebuild=args.force_rebuild)
        print(f"\nPage index ready: '{col_name}' — {col.count()} pages")
