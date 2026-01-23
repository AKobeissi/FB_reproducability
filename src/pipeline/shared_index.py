"""
Shared index ingestion utilities.

This logic used to be duplicated across multiple experiments (e.g. HyDE shared,
unified pipeline). Centralizing it makes the pipeline easier to reason about:

  dataset -> required docs -> ensure shared Chroma store is populated.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from src.ingestion.pdf_utils import load_pdf_with_fallback
from src.retrieval.vectorstore import (
    build_chroma_store,
    populate_chroma_store,
    save_store_config,
    get_chroma_db_path,
)

logger = logging.getLogger(__name__)


def _iter_all_pdf_stems(pdf_local_dir: Path) -> Iterable[str]:
    if not pdf_local_dir:
        return []
    p = str(pdf_local_dir)
    if not os.path.exists(p):
        return []
    for f in glob.glob(os.path.join(p, "**", "*.pdf"), recursive=True):
        yield os.path.splitext(os.path.basename(f))[0]


def collect_required_docs(
    data: List[Dict[str, Any]],
    *,
    pdf_local_dir: Optional[Path],
    use_all_pdfs: bool,
) -> Dict[str, str]:
    """
    Returns mapping of doc_name -> doc_link.

    If use_all_pdfs=True, include every PDF in pdf_local_dir (doc_link empty).
    Always includes at least doc_name/doc_link present in dataset samples.
    """
    unique_docs: Dict[str, str] = {}

    if use_all_pdfs and pdf_local_dir:
        for stem in _iter_all_pdf_stems(pdf_local_dir):
            unique_docs.setdefault(stem, "")

    for sample in data or []:
        doc_name = sample.get("doc_name", "unknown")
        doc_link = sample.get("doc_link", "")
        unique_docs.setdefault(doc_name, doc_link)

    return unique_docs


@dataclass(frozen=True)
class SharedIndexState:
    vectordb: Any
    db_path: str
    available_docs: Set[str]
    pdf_source_map: Dict[str, str]


def _load_shared_meta(meta_path: str) -> Tuple[Set[str], Dict[str, str]]:
    available_docs: Set[str] = set()
    pdf_source_map: Dict[str, str] = {}

    if not os.path.exists(meta_path):
        return available_docs, pdf_source_map

    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        available_docs = set(meta.get("available_docs", []) or [])
        pdf_source_map = dict(meta.get("pdf_source_map", {}) or {})
    except Exception:
        # Non-fatal: treat as empty and rebuild as we ingest.
        return set(), {}

    return available_docs, pdf_source_map


def ensure_shared_chroma_index(
    experiment: Any,
    data: List[Dict[str, Any]],
    *,
    use_all_pdfs: Optional[bool] = None,
) -> SharedIndexState:
    """
    Ensure a shared Chroma index ("all") exists and is populated with all docs
    needed for this run.

    Returns a SharedIndexState with vectordb + metadata tracking.
    """
    # Resolve policy: default to experiment.use_all_pdfs if present.
    if use_all_pdfs is None:
        use_all_pdfs = bool(getattr(experiment, "use_all_pdfs", False))

    pdf_local_dir = getattr(experiment, "pdf_local_dir", None)
    pdf_local_dir = Path(pdf_local_dir) if pdf_local_dir else None

    required_docs = collect_required_docs(
        data, pdf_local_dir=pdf_local_dir, use_all_pdfs=bool(use_all_pdfs)
    )

    # Build / load shared store.
    _, vectordb, is_new = build_chroma_store(experiment, "all", lazy_load=True)
    _, db_path = get_chroma_db_path(experiment, "all")

    meta_path = os.path.join(db_path, "shared_meta.json")
    available_docs, pdf_source_map = _load_shared_meta(meta_path) if not is_new else (set(), {})

    docs_to_process = {k: v for k, v in required_docs.items() if k not in available_docs}

    if docs_to_process:
        logger.info("Shared index missing %s docs; ingesting...", len(docs_to_process))
        for doc_name, doc_link in docs_to_process.items():
            pdf_docs, src = load_pdf_with_fallback(doc_name, doc_link, pdf_local_dir)
            pdf_source_map[doc_name] = src
            if not pdf_docs:
                continue
            chunks = experiment._chunk_text_langchain(pdf_docs, metadata={"doc_name": doc_name})
            if not chunks:
                continue
            populate_chroma_store(experiment, vectordb, chunks, "all")
            available_docs.add(doc_name)

        # Persist store config + meta
        save_store_config(experiment, db_path)
        try:
            with open(meta_path, "w") as f:
                json.dump(
                    {
                        "available_docs": sorted(available_docs),
                        "pdf_source_map": pdf_source_map,
                    },
                    f,
                )
        except Exception as e:
            logger.warning("Failed writing shared meta (%s): %s", meta_path, e)

    return SharedIndexState(
        vectordb=vectordb,
        db_path=db_path,
        available_docs=available_docs,
        pdf_source_map=pdf_source_map,
    )

