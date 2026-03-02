"""
Slot-Coverage Reranking (Set-Cover + Diversity)

Retrieves a candidate pool, extracts slot hints from the query, then selects
an evidence set that maximizes slot coverage while minimizing redundancy.
FinanceBench-focused, retrieval-only (no generation).
"""
from __future__ import annotations

import logging
import os
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.ingestion.pdf_utils import load_pdf_with_fallback
from src.experiments.rag_shared_vector import (
    _get_doc_text,
    _log_pdf_sources,
)
from src.retrieval.vectorstore import (
    build_chroma_store,
    populate_chroma_store,
    save_store_config,
    get_chroma_db_path,
)

logger = logging.getLogger(__name__)

DEFAULT_CANDIDATE_K = 100
DEFAULT_SLOT_COVERAGE_LAMBDA = 0.25
DEFAULT_MAX_SLOTS = 8

FINANCE_KEYWORDS = [
    "revenue", "net income", "operating income", "operating profit",
    "gross profit", "gross margin", "operating margin", "ebitda",
    "earnings", "eps", "diluted eps", "free cash flow", "cash flow",
    "capex", "dividends", "share repurchase", "buyback", "assets",
    "liabilities", "equity", "debt", "interest expense", "operating expense",
    "rd", "r&d", "sg&a", "sales", "guidance",
]

STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "to", "in", "for", "by", "on", "with",
    "from", "at", "as", "is", "are", "was", "were", "be", "this", "that",
    "these", "those", "how", "what", "which", "who", "whom", "when", "where",
    "why", "does", "do", "did", "will", "would", "should", "can", "could",
    "company", "fiscal", "year", "quarter", "q1", "q2", "q3", "q4",
}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _extract_years(question: str) -> List[str]:
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", question)
    return sorted(set(years))


def _extract_quarters(question: str) -> List[str]:
    quarters = re.findall(r"\bQ[1-4]\b", question.upper())
    quarters += re.findall(r"\b[1-4]Q\b", question.upper())
    return sorted(set(quarters))


def _extract_keywords(question: str, max_slots: int) -> List[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9&\-]+", question.lower())
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    counts = Counter(tokens)
    keywords = [w for w, _ in counts.most_common(max_slots)]
    return keywords


def _extract_finance_terms(question: str) -> List[str]:
    q = _normalize_text(question)
    hits = []
    for kw in FINANCE_KEYWORDS:
        if kw in q:
            hits.append(kw)
    return hits


def extract_slots(question: str, doc_name: Optional[str], max_slots: int) -> List[str]:
    slots: List[str] = []
    slots.extend(_extract_years(question))
    slots.extend(_extract_quarters(question))
    slots.extend(_extract_finance_terms(question))

    if doc_name:
        normalized_doc = _normalize_text(doc_name)
        if normalized_doc and normalized_doc not in slots:
            slots.append(normalized_doc)

    keywords = _extract_keywords(question, max_slots=max_slots)
    for kw in keywords:
        if kw not in slots:
            slots.append(kw)

    return slots[:max_slots]


def _chunk_tokens(text: str) -> Set[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9&\-]+", text.lower())
    return {t for t in tokens if t not in STOPWORDS and len(t) > 2}


def _slot_hits(slots: List[str], chunk_text: str) -> Set[str]:
    chunk_lower = _normalize_text(chunk_text)
    hits = set()
    for s in slots:
        s_norm = _normalize_text(s)
        if s_norm and s_norm in chunk_lower:
            hits.add(s)
    return hits


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = a.intersection(b)
    union = a.union(b)
    return len(inter) / len(union) if union else 0.0


def greedy_set_cover(
    candidates: List[Any],
    slots: List[str],
    top_k: int,
    lambda_penalty: float,
) -> List[Any]:
    if not candidates:
        return []
    if not slots:
        return candidates[:top_k]

    candidate_data = []
    for idx, doc in enumerate(candidates):
        text = _get_doc_text(doc)
        tokens = _chunk_tokens(text)
        hits = _slot_hits(slots, text)
        candidate_data.append({
            "doc": doc,
            "idx": idx,
            "text": text,
            "tokens": tokens,
            "hits": hits,
        })

    selected = []
    covered: Set[str] = set()
    selected_tokens: List[Set[str]] = []

    for _ in range(min(top_k, len(candidate_data))):
        best = None
        best_score = None

        for item in candidate_data:
            if item["doc"] in selected:
                continue

            new_slots = item["hits"] - covered
            gain = len(new_slots)

            redundancy = 0.0
            if selected_tokens:
                redundancy = max(_jaccard(item["tokens"], t) for t in selected_tokens)

            score = gain - (lambda_penalty * redundancy)
            if best is None or score > best_score:
                best = item
                best_score = score

        if best is None:
            break

        selected.append(best["doc"])
        covered.update(best["hits"])
        selected_tokens.append(best["tokens"])

        if covered == set(slots):
            break

    return selected


def _extract_numbers(text: str) -> List[str]:
    if not text:
        return []
    clean = re.sub(r"[,$]", "", text)
    return re.findall(r"-?\d*\.?\d+", clean)


def guess_numeric_answer(chunks: List[Dict[str, Any]]) -> str:
    numbers: List[str] = []
    for chunk in chunks:
        numbers.extend(_extract_numbers(chunk.get("text", "")))

    if not numbers:
        return ""

    counts = Counter(numbers)
    best_num, _ = counts.most_common(1)[0]
    return best_num


def _docs_to_chunks(documents: List[Any]) -> List[Dict[str, Any]]:
    chunks = []
    for rank, doc in enumerate(documents):
        text = _get_doc_text(doc)
        chunks.append({
            "rank": rank + 1,
            "text": text,
            "score": doc.metadata.get("slot_coverage_score"),
            "metadata": doc.metadata,
        })
    return chunks


def run_slot_coverage_reranking(experiment, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING SLOT-COVERAGE RERANKING (SET-COVER)")
    logger.info("=" * 80)

    candidate_k = getattr(experiment, "k_cand", None) or DEFAULT_CANDIDATE_K
    lambda_penalty = getattr(experiment, "slot_coverage_lambda", DEFAULT_SLOT_COVERAGE_LAMBDA)
    max_slots = getattr(experiment, "slot_coverage_max_slots", DEFAULT_MAX_SLOTS)

    # Force retrieval-only evaluation for this experiment (no generation)
    experiment.eval_type = "retrieval"

    # --- PART 0: AUTO-DETECT PDF DIRECTORY ---
    current_pdf_dir = getattr(experiment, "pdf_local_dir", None)
    if not current_pdf_dir or not os.path.exists(current_pdf_dir) or not os.listdir(current_pdf_dir):
        logger.info(f"Default PDF dir '{current_pdf_dir}' seems invalid. Searching for 'pdfs' folder...")
        potential_paths = [
            os.path.join(os.getcwd(), "pdfs"),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../../pdfs")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../pdfs")),
        ]
        for p in potential_paths:
            if os.path.exists(p) and os.path.isdir(p) and os.listdir(p):
                logger.info(f"✓ Auto-detected PDF directory at: {p}")
                experiment.pdf_local_dir = Path(p)
                break
        else:
            logger.warning("Could not auto-locate 'pdfs' directory. Ingestion may fail.")

    # --- PART 1: ENSURE VECTOR STORE IS POPULATED ---
    unique_docs = {}
    for sample in data:
        doc_name = sample.get("doc_name", "unknown")
        if doc_name not in unique_docs:
            unique_docs[doc_name] = sample.get("doc_link", "")

    if getattr(experiment, "use_all_pdfs", False) and getattr(experiment, "pdf_local_dir", None):
        pdf_dir = str(experiment.pdf_local_dir)
        if os.path.exists(pdf_dir):
            import glob
            pdf_files = glob.glob(os.path.join(pdf_dir, "**", "*.pdf"), recursive=True)
            pdf_files += glob.glob(os.path.join(pdf_dir, "**", "*.PDF"), recursive=True)
            for pdf_path in pdf_files:
                filename = os.path.basename(pdf_path)
                doc_name = os.path.splitext(filename)[0]
                unique_docs[doc_name] = ""

    logger.info(f"Collected {len(unique_docs)} unique documents")

    try:
        base_retriever, vectordb, is_new = build_chroma_store(
            experiment,
            "all",
            lazy_load=True,
        )
    except Exception as e:
        logger.error("Chroma build failed: %s", e)
        return []

    db_name, db_path = get_chroma_db_path(experiment, "all")
    meta_path = os.path.join(db_path, "shared_meta.json")
    available_docs = set()
    pdf_source_map = {}

    if not is_new and os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
                available_docs = set(meta.get("available_docs", []))
                pdf_source_map = meta.get("pdf_source_map", {})
        except Exception:
            pass

    docs_to_process = {k: v for k, v in unique_docs.items() if k not in available_docs}
    if docs_to_process:
        logger.info(f"Ingesting {len(docs_to_process)} missing documents...")
        for doc_name, doc_link in docs_to_process.items():
            pdf_docs, pdf_source = load_pdf_with_fallback(
                doc_name=doc_name,
                doc_link=doc_link,
                local_dir=getattr(experiment, "pdf_local_dir", None),
            )
            pdf_source_map[doc_name] = pdf_source

            if not pdf_docs:
                logger.warning(f"No PDF pages for '{doc_name}'. Skipping.")
                continue

            chunks = experiment._chunk_text_langchain(
                pdf_docs,
                metadata={"doc_name": doc_name, "source": "pdf", "doc_link": doc_link},
            )

            if chunks:
                populate_chroma_store(experiment, vectordb, chunks, db_name)
                available_docs.add(doc_name)

        save_store_config(experiment, db_path)
        with open(meta_path, "w") as f:
            json.dump({"available_docs": list(available_docs), "pdf_source_map": pdf_source_map}, f)

    _log_pdf_sources(pdf_source_map)

    # --- PART 2: RUN EXPERIMENT LOOP ---
    base_retriever.search_kwargs["k"] = candidate_k

    results: List[Dict[str, Any]] = []

    for i, sample in enumerate(data):
        logger.info(f"\n--- Sample {i+1}/{len(data)} ---")

        doc_name = sample.get("doc_name", "unknown")
        question = sample.get("question", "")

        try:
            initial_docs = base_retriever.invoke(question)
        except Exception as e:
            logger.warning(f"Retrieval failed: {e}")
            initial_docs = []

        slots = extract_slots(question, doc_name, max_slots=max_slots)
        logger.info("Slots (%s): %s", len(slots), slots)

        selected_docs = greedy_set_cover(
            initial_docs,
            slots,
            top_k=experiment.top_k,
            lambda_penalty=lambda_penalty,
        )

        for doc in selected_docs:
            doc.metadata["slot_coverage_score"] = float(len(_slot_hits(slots, _get_doc_text(doc))))

        retrieved_chunks = _docs_to_chunks(selected_docs)
        numeric_guess = guess_numeric_answer(retrieved_chunks)

        gold_segments, gold_str = experiment._prepare_gold_evidence(sample.get("evidence", ""))

        result = {
            "sample_id": i,
            "doc_name": doc_name,
            "question": question,
            "reference_answer": sample.get("answer", ""),
            "question_type": sample.get("question_type", ""),
            "question_reasoning": sample.get("question_reasoning", ""),
            "gold_evidence": gold_str,
            "gold_evidence_segments": gold_segments,
            "retrieved_chunks": retrieved_chunks,
            "num_retrieved": len(retrieved_chunks),
            "generated_answer": numeric_guess,
            "experiment_type": "slot_coverage_reranking",
            "final_prompt": "",
            "context_length": sum(len(c.get("text", "")) for c in retrieved_chunks),
            "generation_length": len(numeric_guess),
        }

        results.append(result)
        logger.info("Completed sample %s", i + 1)

    return results
