#!/usr/bin/env python3
"""
extraction_study.py
===================
Compares different PDF text extraction methods on FinanceBench using
Dense BGE-M3 retrieval. Extraction is the only variable under study.

Extraction Methods:
  PyMuPDF     – Standard text extraction (Baseline)
  pdfplumber  – Better for tables/grid layouts
  marker-pdf  – Markdown-based extraction (vision-based / OCR)
  PyPDF2      – Traditional library (often noisier)
  pypdf       – Modern PyPDF2 successor

Settings:
  Embedding: BGE-M3 (Dense)
  Chunking: 1024 tokens, 128 overlap (RecursiveCharacterTextSplitter)
  Retrieval: Global (all 150 PDFs)
"""

import os
import sys
import json
import logging
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import numpy as np
from tqdm import tqdm

# Project root on sys.path
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Use project-standard dependencies
from src.core.rag_dependencies import (
    Document, 
    RecursiveCharacterTextSplitter, 
    HuggingFaceEmbeddings, 
    FAISS
)

# Setup logger
logger = logging.getLogger("ExtractionStudy")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMBED_MODEL = "BAAI/bge-m3"
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 128
PDF_DIR = _ROOT / "pdfs"
OUTPUT_DIR = _ROOT / "outputs" / "extraction_impact"
QUESTIONS_PATH = _ROOT / "data" / "financebench_open_source.jsonl"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def load_financebench_questions(path: str) -> List[Dict[str, Any]]:
    """Loads questions from FinanceBench JSONL file."""
    questions = []
    with open(path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            # Normalize fields
            q = {
                "question_id": data.get("financebench_id"),
                "question": data.get("question"),
                "pdf_name": data.get("doc_name") + ".pdf" if not data.get("doc_name").endswith(".pdf") else data.get("doc_name"),
                "gold_page": None
            }
            # Evidence structure check
            evidence = data.get("evidence", [])
            if evidence and isinstance(evidence, list):
                q["gold_page"] = evidence[0].get("evidence_page_num")
            
            if q["question"] and q["pdf_name"] and q["gold_page"] is not None:
                questions.append(q)
    return questions

# ---------------------------------------------------------------------------
# Extraction Functions
# ---------------------------------------------------------------------------

def extract_pymupdf(pdf_path: str) -> List[Document]:
    """Standard PyMuPDF extraction."""
    try:
        from langchain_community.document_loaders import PyMuPDFLoader
        loader = PyMuPDFLoader(pdf_path)
        return loader.load()
    except Exception:
        import fitz
        doc = fitz.open(pdf_path)
        return [Document(page_content=page.get_text(), metadata={"source": pdf_path, "page": i}) 
                for i, page in enumerate(doc)]

def extract_pdfplumber(pdf_path: str) -> List[Document]:
    """pdfplumber extraction (often better for tables)."""
    import pdfplumber
    docs = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                docs.append(Document(
                    page_content=text,
                    metadata={"source": pdf_path, "page": i}
                ))
    except Exception as e:
        logger.error(f"pdfplumber failed for {pdf_path}: {e}")
    return docs

def extract_pypdf2(pdf_path: str) -> List[Document]:
    """PyPDF2 extraction."""
    import PyPDF2
    docs = []
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                docs.append(Document(
                    page_content=text,
                    metadata={"source": pdf_path, "page": i}
                ))
    except Exception as e:
        logger.error(f"PyPDF2 failed for {pdf_path}: {e}")
    return docs

def extract_pypdf(pdf_path: str) -> List[Document]:
    """pypdf extraction (modern PyPDF2 successor)."""
    try:
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(pdf_path)
        return loader.load()
    except Exception as e:
        logger.warning(f"pypdf loader failed for {pdf_path}: {e}")
        return []

def extract_marker(pdf_path: str) -> List[Document]:
    """Marker-PDF (Markdown) extraction."""
    # Try cache first
    stem = Path(pdf_path).stem
    cache_path = _ROOT / "modality_fusion_rag" / "cache" / "marker" / f"{stem}.json"
    if cache_path.exists():
        with open(cache_path, 'r') as f:
            data = json.load(f)
            docs = []
            for page in data:
                # Some caches are flattened, others structured
                content = ""
                if "text_chunks" in page:
                    content = "\n\n".join(page["text_chunks"])
                elif "text" in page:
                    content = page["text"]
                
                docs.append(Document(
                    page_content=content,
                    metadata={"source": pdf_path, "page": page.get("page_num", 0)}
                ))
            return docs
    
    # Fallback to direct extraction
    try:
        from marker.converters.pdf import PdfConverter
        converter = PdfConverter(artifact_path=None)
        rendered = converter(pdf_path)
        markdown = rendered.markdown
        pages_md = re.split(r"\n\n---\n\n|\f", markdown)
        docs = []
        for i, page_md in enumerate(pages_md):
            docs.append(Document(
                page_content=page_md,
                metadata={"source": pdf_path, "page": i}
            ))
        return docs
    except Exception as e:
        logger.warning(f"Marker failed for {pdf_path}: {e}")
        return extract_pymupdf(pdf_path)

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_experiment(method_name: str, extraction_fn, questions: List[Dict]):
    logger.info(f"--- Starting Experiment: {method_name} ---")
    
    unique_pdfs = list(set([q['pdf_name'] for q in questions]))
    
    all_documents = []
    logger.info(f"Extracting {len(unique_pdfs)} PDFs using {method_name}...")
    for pdf_name in tqdm(unique_pdfs):
        pdf_path = PDF_DIR / pdf_name
        if not pdf_path.exists():
            logger.warning(f"PDF not found: {pdf_path}")
            continue
        
        try:
            docs = extraction_fn(str(pdf_path))
            all_documents.extend(docs)
        except Exception as e:
            logger.error(f"Failed to extract {pdf_name} with {method_name}: {e}")

    if not all_documents:
        logger.error(f"No documents extracted for {method_name}. Skipping.")
        return None

    # 2. Chunk
    logger.info(f"Chunking {len(all_documents)} pages...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    token_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = token_splitter.split_documents(all_documents)
    logger.info(f"Created {len(chunks)} chunks.")

    # 3. Embed and Index
    logger.info("Building FAISS index...")
    model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs=model_kwargs)
    
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # 4. Evaluate Retrieval
    logger.info("Evaluating retrieval performance...")
    results = []
    metrics = {f"recall@{k}": 0 for k in [1, 3, 5, 10]}
    
    for q in tqdm(questions):
        query = q['question']
        gold_page = q['gold_page']
        gold_pdf = q['pdf_name']
        
        # Search
        docs = vector_store.similarity_search(query, k=10)
        
        # Check hits
        found_at = -1
        for i, doc in enumerate(docs):
            doc_source = Path(doc.metadata['source']).name
            # Ensure it matches .pdf extension if gold_pdf has it
            if not doc_source.endswith(".pdf"):
                doc_source += ".pdf"
            
            doc_page = int(doc.metadata['page'])  # 0-indexed, matches evidence_page_num

            if doc_source == gold_pdf and doc_page == int(gold_page):
                found_at = i + 1
                break
        
        for k in [1, 3, 5, 10]:
            if found_at > 0 and found_at <= k:
                metrics[f"recall@{k}"] += 1
                
        results.append({
            "question_id": q.get("question_id"),
            "question": query,
            "gold_pdf": gold_pdf,
            "gold_page": gold_page,
            "found_at": found_at,
            "top_hits": [{"source": Path(d.metadata['source']).name, "page": int(d.metadata['page'])} for d in docs]
        })

    # Finalize Metrics
    for k in [1, 3, 5, 10]:
        metrics[f"recall@{k}"] /= len(questions)
        logger.info(f"Recall@{k}: {metrics[f'recall@{k}']:.4f}")

    # Save Results
    method_dir = OUTPUT_DIR / method_name
    os.makedirs(method_dir, exist_ok=True)
    
    with open(method_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    with open(method_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    logger.info(f"Results saved to {method_dir}")
    return metrics

if __name__ == "__main__":
    # 1. Load Data once
    questions = load_financebench_questions(str(QUESTIONS_PATH))
    logger.info(f"Loaded {len(questions)} valid FinanceBench questions.")

    methods = [
        ("pymupdf", extract_pymupdf),
        ("pdfplumber", extract_pdfplumber),
        ("pypdf2", extract_pypdf2),
        ("pypdf", extract_pypdf),
        ("marker", extract_marker)
    ]
    
    all_metrics = {}
    for name, fn in methods:
        try:
            metrics = run_experiment(name, fn, questions)
            if metrics:
                all_metrics[name] = metrics
        except Exception as e:
            logger.error(f"Experiment {name} failed: {e}", exc_info=True)

    # Summary
    logger.info("\n--- FINAL SUMMARY ---")
    for name, m in all_metrics.items():
        logger.info(f"{name:<12} | Recall@5: {m['recall@5']:.4f} | Recall@10: {m['recall@10']:.4f}")
    
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
