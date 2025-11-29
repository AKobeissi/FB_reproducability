from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import Any, Dict, List, Sequence

from ...data import load_pdf_with_fallback

logger = logging.getLogger(__name__)


def run_random_single_store(experiment, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run single-store experiment that samples random chunks instead of using a retriever."""
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING RANDOM SINGLE STORE EXPERIMENT (Random chunk baseline)")
    logger.info("=" * 80)

    results: List[Dict[str, Any]] = []
    doc_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for sample in data:
        doc_name = sample.get("doc_name", "unknown")
        doc_groups[doc_name].append(sample)

    logger.info(f"Processing {len(doc_groups)} unique documents")

    rng = _initialize_rng(experiment)
    _ensure_component_overrides(experiment)

    for doc_name, samples in doc_groups.items():
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Document: {doc_name} ({len(samples)} samples)")
        logger.info(f"{'=' * 80}")

        doc_link = samples[0].get("doc_link", "")

        pdf_docs, pdf_source = load_pdf_with_fallback(
            doc_name=doc_name,
            doc_link=doc_link,
            local_dir=getattr(experiment, "pdf_local_dir", None),
        )

        if not pdf_docs:
            logger.warning(f"No PDF pages for '{doc_name}'. Skipping {len(samples)} samples.")
            skipped = experiment._create_skipped_results(  # type: ignore[attr-defined]
                samples,
                doc_name,
                doc_link,
                pdf_source,
                experiment.RANDOM_SINGLE,
                len(results),
                experiment=experiment,
            )
            results.extend(skipped)
            if hasattr(experiment, "notify_sample_complete"):
                experiment.notify_sample_complete(count=len(skipped), note=f"{doc_name} skipped (no pdf)")
            continue

        logger.info(f"PDF source: {pdf_source}")

        documents = experiment._chunk_text_langchain(  # type: ignore[attr-defined]
            pdf_docs,
            metadata={
                "doc_name": doc_name,
                "source": "pdf",
                "doc_link": doc_link,
                "pdf_source": pdf_source,
            },
        )

        if not documents:
            logger.warning(f"No chunks created for '{doc_name}'. Skipping samples.")
            skipped = experiment._create_skipped_results(  # type: ignore[attr-defined]
                samples,
                doc_name,
                doc_link,
                pdf_source,
                experiment.RANDOM_SINGLE,
                len(results),
                experiment=experiment,
            )
            results.extend(skipped)
            if hasattr(experiment, "notify_sample_complete"):
                experiment.notify_sample_complete(count=len(skipped), note=f"{doc_name} skipped (no chunks)")
            continue

        for idx, sample in enumerate(samples):
            logger.info(f"\n--- Sample {idx + 1}/{len(samples)} for {doc_name} ---")
            result = _process_sample_random_selection(
                experiment=experiment,
                sample=sample,
                doc_name=doc_name,
                doc_link=doc_link,
                pdf_source=pdf_source,
                documents=documents,
                rng=rng,
                sample_id=len(results),
            )
            results.append(result)
            logger.info(f"Completed sample {len(results)}")
            if hasattr(experiment, "notify_sample_complete"):
                experiment.notify_sample_complete(note=f"{doc_name}")

    return results


def _process_sample_random_selection(
    experiment,
    sample: Dict[str, Any],
    doc_name: str,
    doc_link: str,
    pdf_source: str,
    documents,
    rng: random.Random,
    sample_id: int,
) -> Dict[str, Any]:
    question = sample.get("question", "")
    reference_answer = sample.get("answer", "")
    gold_segments, gold_evidence_str = experiment._prepare_gold_evidence(sample.get("evidence", ""))  # type: ignore[attr-defined]

    sampled_docs = _select_random_documents(documents, getattr(experiment, "top_k", 5), rng)
    retrieved_chunks = _documents_to_chunks(sampled_docs)
    context = "\n\n".join(chunk["text"] for chunk in retrieved_chunks)

    generated_answer, prompt_snapshot = experiment._generate_answer(  # type: ignore[attr-defined]
        question,
        context,
        mode=experiment.experiment_type,
        return_prompt=True,
    )

    random_seed = getattr(experiment, "random_selection_seed", None)

    return {
        "sample_id": sample_id,
        "doc_name": doc_name,
        "doc_link": doc_link,
        "question": question,
        "reference_answer": reference_answer,
        "gold_evidence": gold_evidence_str,
        "gold_evidence_segments": gold_segments,
        "retrieved_chunks": retrieved_chunks,
        "num_retrieved": len(retrieved_chunks),
        "context_length": len(context),
        "generated_answer": generated_answer,
        "generation_length": len(generated_answer),
        "experiment_type": experiment.RANDOM_SINGLE,
        "vector_store_type": "RandomSelectionBaseline",
        "retrieval_strategy": "uniform_random_chunks",
        "pdf_source": pdf_source,
        "final_prompt": prompt_snapshot,
        "random_selection_seed": random_seed,
    }


def _ensure_component_overrides(experiment) -> None:
    if getattr(experiment, "_random_single_components_set", False):
        return

    experiment.register_component_usage(  # type: ignore[attr-defined]
        "vector_store",
        "None (random selection baseline)",
        {"reason": "Chunks sampled without embeddings"},
    )
    experiment.register_component_usage(  # type: ignore[attr-defined]
        "retriever",
        "RandomChunkSelector",
        {"strategy": "uniform_without_replacement", "top_k": getattr(experiment, "top_k", 5)},
    )
    setattr(experiment, "_random_single_components_set", True)


def _initialize_rng(experiment) -> random.Random:
    seed = getattr(experiment, "random_selection_seed", None)
    if seed is None:
        return random.Random()
    logger.info(f"Using reproducible random selection seed: {seed}")
    return random.Random(seed)


def _select_random_documents(documents: Sequence[Any], top_k: int, rng: random.Random) -> List[Any]:
    if not documents:
        return []

    k = max(1, top_k or 1)
    population = documents if isinstance(documents, list) else list(documents)

    if k >= len(population):
        shuffled = list(population)
        rng.shuffle(shuffled)
        return shuffled

    return rng.sample(population, k)


def _documents_to_chunks(documents: Sequence[Any]) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for rank, doc in enumerate(documents, start=1):
        text = getattr(doc, "page_content", None) or getattr(doc, "content", None) or str(doc)
        if isinstance(text, (bytes, bytearray)):
            text = text.decode("utf-8", errors="replace")
        metadata = getattr(doc, "metadata", {}) or {}
        chunks.append(
            {
                "rank": rank,
                "text": text,
                "score": None,
                "length": len(text),
                "metadata": metadata,
            }
        )
    return chunks
