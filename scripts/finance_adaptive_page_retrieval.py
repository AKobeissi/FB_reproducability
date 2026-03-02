#!/usr/bin/env python3
import argparse
import json
import logging
import pickle
import re
import hashlib
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.ingestion.page_processor import extract_pages_from_pdf
from src.evaluation.retrieval_evaluator import RetrievalEvaluator
from src.evaluation.generative_evaluator import GenerativeEvaluator


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("finance_adaptive_page")


SECTION_PATTERNS = {
    "balance_sheet": ["balance sheet", "statement of financial position", "assets", "liabilities", "equity"],
    "income_statement": ["income statement", "statement of operations", "statement of income", "net income", "revenue"],
    "cash_flow": ["cash flow", "statement of cash flows", "operating activities", "investing activities", "financing activities"],
    "notes": ["note ", "notes to", "footnote", "supplemental"],
    "mda": ["management discussion", "md&a", "results of operations", "liquidity and capital resources"],
    "debt_securities": ["notes due", "debt securities", "registered", "trading symbol"],
}

QUERY_SECTION_HINTS = {
    "balance_sheet": ["balance sheet", "financial position", "assets", "liabilities", "equity", "working capital", "quick ratio"],
    "income_statement": ["income statement", "statement of operations", "operating margin", "gross margin", "revenue", "net income"],
    "cash_flow": ["cash flow", "operating cash", "capex", "dividends", "free cash flow", "financing activities"],
    "notes": ["note", "footnote", "disclose", "exhibit"],
    "mda": ["what drove", "drivers", "why", "explain", "trend"],
    "debt_securities": ["debt securities", "registered", "trading symbol", "exchange"],
}


def normalize_text(text: str, max_chars: int = 8000) -> str:
    if not text:
        return " "
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text[:max_chars] if text else " "


def tokenize_finance(text: str) -> List[str]:
    if not text:
        return []
    text = text.lower()
    return re.findall(r"\$?\d+(?:[.,]\d+)?%?|[a-zA-Z][a-zA-Z0-9\-_/]*", text)


def zscore(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    std = float(arr.std())
    if std < 1e-8:
        return np.zeros_like(arr)
    return (arr - float(arr.mean())) / std


def cosine_scores(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    q_norm = np.linalg.norm(query_vec) + 1e-8
    d_norm = np.linalg.norm(doc_vecs, axis=1) + 1e-8
    return (doc_vecs @ query_vec) / (q_norm * d_norm)


def extract_gold_pages(evidence: Any) -> List[int]:
    if evidence is None:
        return []
    if isinstance(evidence, str):
        try:
            evidence = json.loads(evidence)
        except Exception:
            return []
    if isinstance(evidence, dict):
        evidence = [evidence]
    if not isinstance(evidence, list):
        return []
    pages = []
    for ev in evidence:
        if not isinstance(ev, dict):
            continue
        p = ev.get("evidence_page_num")
        if p is None:
            p = ev.get("page")
        if p is None:
            p = ev.get("page_ix")
        if p is None:
            continue
        try:
            pages.append(int(p))
        except Exception:
            continue
    return sorted(set(pages))


def query_sections(query: str) -> List[str]:
    q = query.lower()
    hits = []
    for section, patterns in QUERY_SECTION_HINTS.items():
        if any(p in q for p in patterns):
            hits.append(section)
    return hits


def page_section_score(page_text: str, targets: List[str]) -> float:
    if not targets:
        return 0.0
    txt = page_text.lower()
    score = 0.0
    for section in targets:
        patterns = SECTION_PATTERNS.get(section, [])
        if any(p in txt for p in patterns):
            score += 1.0
    return score


def extract_year_tokens(text: str) -> List[str]:
    return re.findall(r"\b(?:19|20)\d{2}\b", text)


def number_alignment_score(query: str, page_text: str) -> float:
    q_years = set(extract_year_tokens(query))
    if not q_years:
        return 0.0
    p_years = set(extract_year_tokens(page_text))
    if not p_years:
        return 0.0
    overlap = len(q_years & p_years)
    return overlap / max(1, len(q_years))


def adaptive_window(question: str, question_type: str, num_pages: int, base_window: int) -> int:
    window = base_window
    q = question.lower()
    qt = (question_type or "").lower()
    if qt == "metrics-generated" or any(k in q for k in ["ratio", "average", "change", "between", "fy", "%", "calculate"]):
        window += 1
    if num_pages >= 200:
        window += 1
    return min(window, 4)


@dataclass
class SweepConfig:
    page_k: int
    seed_k: int
    dense_w: float
    bm25_w: float
    section_w: float
    number_w: float
    base_window: int
    distance_penalty: float


class DocIndex:
    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def _rebuild_helpers(self):
        # Per-candidate metadata aligned with page_nums/page_texts/embeddings rows
        self.page_doc_names = [str(p.get("doc_name", self.doc_name)) for p in getattr(self, "pages", [])]

        # Global lookup: (doc_name, page_num) -> row index in the global candidate arrays
        # (If duplicates exist, keep the first occurrence deterministically.)
        self.doc_page_to_idx = {}
        for i, (d, pg) in enumerate(zip(getattr(self, "page_doc_names", []), getattr(self, "page_nums", []))):
            key = (str(d), int(pg))
            if key not in self.doc_page_to_idx:
                self.doc_page_to_idx[key] = i

    def __setstate__(self, state):
        self.__dict__.update(state)

        # Backward compatibility for older cached pickles that may not have `pages`
        if not hasattr(self, "pages"):
            page_nums = list(getattr(self, "page_nums", []))
            page_texts = list(getattr(self, "page_texts", []))
            fallback_doc = getattr(self, "doc_name", "")
            self.pages = [
                {"page": int(pn), "text": txt, "doc_name": fallback_doc}
                for pn, txt in zip(page_nums, page_texts)
            ]

        self._rebuild_helpers()

    def __init__(self, doc_name: str, pages: List[Dict[str, Any]], model: SentenceTransformer):
        self.doc_name = doc_name
        self.pages = pages  # Store the original page dicts for later reference
        self.page_nums = [int(p["page"]) for p in pages]
        self.page_texts = [normalize_text(p.get("text", "")) for p in pages]
        self.tokens = [tokenize_finance(t) for t in self.page_texts]
        self.bm25 = BM25Okapi(self.tokens)
        self.embeddings = model.encode(self.page_texts, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        self._rebuild_helpers()


def select_pages_from_features(sample_features: Dict[str, Any], cfg: SweepConfig) -> List[Dict[str, Any]]:
    """
    Select pages from the GLOBAL candidate pool in a doc-aware way.

    Returns a list of dicts:
      {"global_idx": int, "doc_name": str, "page": int, "score": float}

    This avoids collisions where page numbers repeat across documents (e.g., page 12 in many PDFs).
    """
    dense = sample_features["dense"]
    bm25 = sample_features["bm25"]
    section_prior = sample_features["section_prior"]
    number_prior = sample_features["number_prior"]
    page_nums = sample_features["page_nums"]
    doc_names = sample_features["doc_names"]
    doc_page_to_idx = sample_features["doc_page_to_idx"]
    query = sample_features["question"]
    q_type = sample_features["question_type"]

    fused = (
        cfg.dense_w * zscore(dense)
        + cfg.bm25_w * zscore(bm25)
        + cfg.section_w * zscore(section_prior)
        + cfg.number_w * zscore(number_prior)
    )

    seed_order = np.argsort(-fused)[:cfg.seed_k]
    win = adaptive_window(query, q_type, len(page_nums), cfg.base_window)

    # Key by GLOBAL row index, not local page number
    selected: Dict[int, float] = {}
    for idx in seed_order:
        idx = int(idx)
        seed_doc = str(doc_names[idx])
        seed_page = int(page_nums[idx])
        seed_score = float(fused[idx])

        for delta in range(-win, win + 1):
            cand_page = seed_page + delta
            if cand_page < 0:
                continue

            cand_idx = doc_page_to_idx.get((seed_doc, int(cand_page)))
            if cand_idx is None:
                continue

            cand_score = seed_score - cfg.distance_penalty * abs(delta)
            if cand_idx not in selected or cand_score > selected[cand_idx]:
                selected[cand_idx] = cand_score

    if len(selected) < cfg.page_k:
        for idx in np.argsort(-fused):
            idx = int(idx)
            if idx not in selected:
                selected[idx] = float(fused[idx])
            if len(selected) >= cfg.page_k:
                break

    best = sorted(selected.items(), key=lambda x: x[1], reverse=True)[:cfg.page_k]
    return [
        {
            "global_idx": int(i),
            "doc_name": str(doc_names[int(i)]),
            "page": int(page_nums[int(i)]),
            "score": float(score),
        }
        for i, score in best
    ]


def build_sample_features(dataset: List[Dict[str, Any]], global_index: DocIndex, model: SentenceTransformer) -> List[Dict[str, Any]]:
    out = []
    shared_doc_names = np.array(global_index.page_doc_names, dtype=object)
    shared_doc_page_to_idx = global_index.doc_page_to_idx
    for sample in tqdm(dataset, desc="Precomputing sample features (global)"):
        gold = extract_gold_pages(sample.get("evidence"))
        if not gold:
            continue
        sample_doc_name = sample.get("doc_name", "")
        gold_page_keys = [(str(sample_doc_name), int(p)) for p in gold]

        query = sample.get("question", "")
        q_vec = model.encode([query], convert_to_numpy=True, show_progress_bar=False)[0].astype(np.float32)
        dense = cosine_scores(q_vec, global_index.embeddings).astype(np.float32)
        bm25 = np.array(global_index.bm25.get_scores(tokenize_finance(query)), dtype=np.float32)
        targets = query_sections(query)
        section_prior = np.array([page_section_score(t, targets) for t in global_index.page_texts], dtype=np.float32)
        number_prior = np.array([number_alignment_score(query, t) for t in global_index.page_texts], dtype=np.float32)
        out.append({
            "doc_name": sample_doc_name,
            "question": query,
            "question_type": sample.get("question_type", ""),
            "gold_pages": gold,  # kept for backward compatibility / debugging
            "gold_page_keys": gold_page_keys,
            "page_nums": np.array(global_index.page_nums, dtype=np.int32),
            "dense": dense,
            "bm25": bm25,
            "section_prior": section_prior,
            "number_prior": number_prior,
            "doc_names": shared_doc_names,
            "doc_page_to_idx": shared_doc_page_to_idx,
        })
    return out


def build_inference_features(dataset: List[Dict[str, Any]], global_index: DocIndex, model: SentenceTransformer) -> List[Dict[str, Any]]:
    out = []
    shared_doc_names = np.array(global_index.page_doc_names, dtype=object)
    shared_doc_page_to_idx = global_index.doc_page_to_idx
    for sample in tqdm(dataset, desc="Precomputing inference features (global)"):
        query = sample.get("question", "")
        q_vec = model.encode([query], convert_to_numpy=True, show_progress_bar=False)[0].astype(np.float32)
        dense = cosine_scores(q_vec, global_index.embeddings).astype(np.float32)
        bm25 = np.array(global_index.bm25.get_scores(tokenize_finance(query)), dtype=np.float32)
        targets = query_sections(query)
        section_prior = np.array([page_section_score(t, targets) for t in global_index.page_texts], dtype=np.float32)
        number_prior = np.array([number_alignment_score(query, t) for t in global_index.page_texts], dtype=np.float32)
        out.append({
            "sample": sample,
            "doc_name": sample.get("doc_name", ""),
            "question": query,
            "question_type": sample.get("question_type", ""),
            "page_nums": np.array(global_index.page_nums, dtype=np.int32),
            "dense": dense,
            "bm25": bm25,
            "section_prior": section_prior,
            "number_prior": number_prior,
            "doc_names": shared_doc_names,
            "doc_page_to_idx": shared_doc_page_to_idx,
        })
    return out


def coerce_evidence_segments(evidence: Any, fallback_doc_name: str) -> List[Dict[str, Any]]:
    if evidence is None:
        return []

    raw = evidence
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return []
        try:
            raw = json.loads(stripped)
        except Exception:
            return [{
                "text": stripped,
                "doc_name": fallback_doc_name,
                "page": None,
                "page_text": None,
                "raw": stripped,
            }]

    if isinstance(raw, dict):
        entries = [raw]
    elif isinstance(raw, list):
        entries = raw
    else:
        entries = [raw]

    segments: List[Dict[str, Any]] = []
    for entry in entries:
        if isinstance(entry, dict):
            txt = (
                entry.get("evidence_text")
                or entry.get("text")
                or entry.get("evidence_text_full_page")
                or entry.get("excerpt")
                or ""
            )
            segments.append({
                "text": str(txt),
                "doc_name": entry.get("doc_name") or entry.get("document") or fallback_doc_name,
                "page": entry.get("evidence_page_num") or entry.get("page") or entry.get("page_num"),
                "page_text": entry.get("evidence_text_full_page"),
                "raw": entry,
            })
        else:
            segments.append({
                "text": str(entry),
                "doc_name": fallback_doc_name,
                "page": None,
                "page_text": None,
                "raw": entry,
            })

    return [s for s in segments if s.get("text")]


def aggregate_result_stats(results: List[Dict[str, Any]]) -> Dict[str, float]:
    if not results:
        return {
            "num_samples": 0,
            "avg_num_retrieved": 0.0,
            "avg_context_length": 0.0,
            "avg_generation_length": 0.0,
        }

    num_retrieved = [r.get("num_retrieved", 0) for r in results]
    context_lengths = [r.get("context_length", 0) for r in results]
    generation_lengths = [r.get("generation_length", 0) for r in results]

    return {
        "num_samples": len(results),
        "avg_num_retrieved": float(np.mean(num_retrieved)),
        "avg_context_length": float(np.mean(context_lengths)),
        "avg_generation_length": float(np.mean(generation_lengths)),
    }


def _build_generator(args) -> Optional[Any]:
    if args.skip_generation:
        logger.info("Skipping generation as requested.")
        return None

    from src.core.rag_experiments import RAGExperiment

    experiment = RAGExperiment(
        experiment_type=RAGExperiment.UNIFIED,
        llm_model=args.gen_model,
        embedding_model=args.gen_embedding_model,
        output_dir=args.gen_output_dir,
        vector_store_dir=args.gen_vector_store_dir,
        pdf_local_dir=args.pdf_dir,
        use_api=args.gen_use_api,
        api_base_url=args.gen_api_base_url,
        api_key_env=args.gen_api_key_env,
        eval_type="none",
    )
    return experiment


def _evaluate_outputs(results_payload: Dict[str, Any], page_k: int, eval_mode: str) -> Dict[str, Any]:
    samples = results_payload.get("results", [])

    ret_evaluator = RetrievalEvaluator()
    k_values = sorted(set([1, 3, 5, 10, page_k]))
    retrieval_metrics = ret_evaluator.compute_metrics(samples, k_values=k_values)

    use_semantic = eval_mode == "semantic"
    gen_evaluator = GenerativeEvaluator(
        use_bertscore=True,
        use_llm_judge=use_semantic,
        use_ragas=use_semantic,
        judge_pipeline=None,
    )

    metric_keys = set()
    for sample in samples:
        metrics = gen_evaluator.evaluate_sample(sample, langchain_llm=None)
        sample["generative_metrics"] = metrics
        metric_keys.update(metrics.keys())

    agg_gen = {}
    for key in metric_keys:
        vals = [s["generative_metrics"].get(key) for s in samples if s.get("generative_metrics") and s["generative_metrics"].get(key) is not None]
        if not vals:
            continue
        if all(isinstance(v, bool) for v in vals):
            agg_gen[f"avg_{key}"] = float(sum(vals) / len(vals))
        elif all(isinstance(v, (int, float)) for v in vals):
            agg_gen[f"avg_{key}"] = float(sum(vals) / len(vals))

    if "evaluation_summary" not in results_payload:
        results_payload["evaluation_summary"] = {}
    results_payload["evaluation_summary"]["retrieval"] = retrieval_metrics
    results_payload["evaluation_summary"]["generative"] = agg_gen
    return results_payload


def _resolve_inference_config(args, default_cfg: SweepConfig) -> SweepConfig:
    if args.best_from_leakage_safe:
        with Path(args.best_from_leakage_safe).open() as f:
            leak = json.load(f)

        groupings = leak.get("groupings") or {}
        if groupings:
            holdout_cfg = None
            kfold_cfg = None
            for group_key in ["doc_name", "sample"]:
                grouping_payload = groupings.get(group_key) or {}
                holdout_cfg = holdout_cfg or ((grouping_payload.get("holdout") or {}).get("best_config_from_tune"))
                kfold_cfg = kfold_cfg or ((grouping_payload.get("kfold") or {}).get("recommended_config"))
        else:
            holdout_cfg = ((leak.get("holdout") or {}).get("best_config_from_tune"))
            kfold_cfg = ((leak.get("kfold") or {}).get("recommended_config"))

        cfg_dict = holdout_cfg or kfold_cfg
        if not cfg_dict:
            raise ValueError(f"No usable config found in {args.best_from_leakage_safe}")
        return SweepConfig(**cfg_dict)

    if args.best_from_sweep:
        with Path(args.best_from_sweep).open() as f:
            sweep = json.load(f)
        best = sweep.get("best") or {}
        cfg_dict = best.get("config") or {}
        if not cfg_dict:
            raise ValueError(f"No best config found in {args.best_from_sweep}")
        return SweepConfig(**cfg_dict)

    return default_cfg


def run_inference(args, model: SentenceTransformer, doc_indices: Dict[str, DocIndex], dataset: List[Dict[str, Any]], default_cfg: SweepConfig):
    cfg = _resolve_inference_config(args, default_cfg)
    logger.info(f"Running inference with config: {cfg}")

    global_index = doc_indices["ALL_DOCS"]
    inference_features = build_inference_features(dataset, global_index, model)
    logger.info(f"Prepared inference features for {len(inference_features)} questions")

    generator = _build_generator(args)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.inference_output or f"results/finance_adaptive_sweeps/finance_adaptive_inference_{timestamp}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for idx, sf in enumerate(tqdm(inference_features, desc="Inference over questions")):
        sample = sf["sample"]
        query_doc_name = sf["doc_name"]

        retrieved_items = select_pages_from_features(sf, cfg)
        retrieved_chunks = []

        for item in retrieved_items:
            gidx = int(item["global_idx"])
            if gidx < 0 or gidx >= len(global_index.pages):
                continue

            page_record = global_index.pages[gidx]
            text = normalize_text(page_record.get("text", ""))
            if not text:
                continue

            cand_doc = str(item["doc_name"])
            cand_page = int(item["page"])
            retrieved_chunks.append({
                "text": text,
                "metadata": {
                    "doc_name": cand_doc,
                    "page": cand_page,
                    "global_idx": gidx,
                },
                "score": float(item.get("score", 0.0)),
            })

        context = "\n\n".join(c["text"] for c in retrieved_chunks)
        generated_answer = ""
        final_prompt = ""
        if generator is not None:
            generated_answer, final_prompt = generator._generate_answer(
                sample.get("question", ""),
                context,
                mode="unified",
                return_prompt=True,
            )

        gold_segments = coerce_evidence_segments(sample.get("evidence"), query_doc_name)
        gold_evidence = "\n\n".join(seg.get("text", "") for seg in gold_segments if seg.get("text"))

        result_item = {
            "sample_id": idx,
            "doc_name": query_doc_name,
            "doc_link": sample.get("doc_link", ""),
            "doc_type": sample.get("doc_type", ""),
            "question": sample.get("question", ""),
            "reference_answer": sample.get("answer", ""),
            "question_type": sample.get("question_type", ""),
            "question_reasoning": sample.get("question_reasoning", ""),
            "gold_evidence": gold_evidence,
            "gold_evidence_segments": gold_segments,
            "retrieved_chunks": retrieved_chunks,
            "num_retrieved": len(retrieved_chunks),
            "context_length": len(context),
            "generated_answer": generated_answer,
            "generation_length": len(generated_answer),
            "experiment_type": "finance_adaptive_page",
            "vector_store_type": "finance_adaptive_global",
            "pdf_source": str(args.pdf_dir),
            "final_prompt": final_prompt,
            "metadata_page_retrieval": {
                "retrieved_pages_global": [
                    {
                        "doc_name": str(it["doc_name"]),
                        "page": int(it["page"]),
                        "global_idx": int(it["global_idx"]),
                        "score": float(it.get("score", 0.0)),
                    }
                    for it in retrieved_items
                ],
                # Convenience field: pages retrieved in the query's ground-truth document only
                "retrieved_pages_in_query_doc": [
                    int(it["page"]) for it in retrieved_items if str(it["doc_name"]) == str(query_doc_name)
                ],
                "config": cfg.__dict__,
            },
        }
        results.append(result_item)

    payload = {
        "metadata": {
            "experiment_type": "finance_adaptive_page",
            "created_at": timestamp,
            "model_path": args.model_path,
            "inference_config": cfg.__dict__,
            "n_questions": len(results),
        },
        "num_samples": len(results),
        "framework": "Finance adaptive page retrieval + unified-style outputs",
        "aggregate_stats": aggregate_result_stats(results),
        "results": results,
    }

    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Saved inference outputs to: {out_path}")

    scored_payload = _evaluate_outputs(payload, page_k=cfg.page_k, eval_mode=args.eval_mode)
    scored_path = out_path.with_name(f"{out_path.stem}_scored{out_path.suffix}")
    with scored_path.open("w") as f:
        json.dump(scored_payload, f, indent=2)
    logger.info(f"Saved scored outputs to: {scored_path}")

    if generator is not None and hasattr(generator, "_unload_model"):
        generator._unload_model()


def evaluate(sample_features: List[Dict[str, Any]], cfg: SweepConfig) -> Dict[str, Any]:
    """
    Global, doc-aware page evaluation.

    A retrieved page counts only if BOTH the document and page number match a gold evidence page
    for the sample (i.e., compare on (doc_name, page), not page number alone).
    """
    recalls = []
    hits = 0
    total = 0

    for sf in sample_features:
        retrieved_items = select_pages_from_features(sf, cfg)

        if "gold_page_keys" in sf and sf["gold_page_keys"]:
            gold_set = {(str(d), int(p)) for d, p in sf["gold_page_keys"]}
        else:
            # Backward compatibility fallback (less strict); should not happen for newly built features
            sample_doc = str(sf.get("doc_name", ""))
            gold_set = {(sample_doc, int(p)) for p in sf.get("gold_pages", [])}

        ret_set = {(str(it["doc_name"]), int(it["page"])) for it in retrieved_items}
        inter = gold_set & ret_set

        total += 1
        if inter:
            hits += 1
        if gold_set:
            recalls.append(len(inter) / len(gold_set))
        else:
            recalls.append(0.0)

    return {
        "n_questions": total,
        "page_hit@k": (hits / total) if total else 0.0,
        "page_recall@k": float(np.mean(recalls)) if recalls else 0.0,
    }


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_global_doc_index(dataset, pdf_dir, model, cache_path=None):
    if cache_path and cache_path.exists():
        logger.info(f"Loading cached global doc index: {cache_path}")
        with cache_path.open("rb") as f:
            return pickle.load(f)
    docs = sorted(set(d["doc_name"] for d in dataset if d.get("doc_name")))
    all_pages = []
    for doc in tqdm(docs, desc="Building global page pool"):
        pdf_path = pdf_dir / f"{doc}.pdf"
        if not pdf_path.exists():
            logger.warning(f"Missing PDF: {pdf_path}")
            continue
        pages = extract_pages_from_pdf(pdf_path, doc)
        for p in pages:
            p = dict(p)
            p["doc_name"] = doc
            all_pages.append(p)
    global_index = DocIndex("ALL_DOCS", all_pages, model)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as f:
            pickle.dump(global_index, f)
    return global_index


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def build_runs(args) -> List[SweepConfig]:
    dense_weights = parse_float_list(args.dense_weights)
    bm25_weights = parse_float_list(args.bm25_weights)
    section_weights = parse_float_list(args.section_weights)
    number_weights = parse_float_list(args.number_weights)
    seed_ks = parse_int_list(args.seed_ks)
    base_windows = parse_int_list(args.base_windows)

    runs: List[SweepConfig] = []
    for seed_k in seed_ks:
        for dw in dense_weights:
            for bw in bm25_weights:
                for sw in section_weights:
                    for nw in number_weights:
                        for win in base_windows:
                            runs.append(SweepConfig(
                                page_k=args.page_k,
                                seed_k=seed_k,
                                dense_w=dw,
                                bm25_w=bw,
                                section_w=sw,
                                number_w=nw,
                                base_window=win,
                                distance_penalty=args.distance_penalty,
                            ))

    if args.max_runs > 0:
        runs = runs[:args.max_runs]
    return runs


def run_sweep(sample_features: List[Dict[str, Any]], runs: List[SweepConfig], desc: str = "Sweeping") -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    all_results = []
    for cfg in tqdm(runs, desc=desc):
        metrics = evaluate(sample_features, cfg)
        all_results.append({
            "config": cfg.__dict__,
            "metrics": metrics,
        })
    all_results.sort(key=lambda r: r["metrics"]["page_recall@k"], reverse=True)
    best = all_results[0] if all_results else None
    return all_results, best


def split_sample_features(
    sample_features: List[Dict[str, Any]],
    tune_frac: float,
    seed: int,
    group_by: str = "sample",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not sample_features:
        return [], []
    if tune_frac <= 0.0 or tune_frac >= 1.0:
        raise ValueError("--tune-frac must be in (0, 1)")

    rng = np.random.default_rng(seed)
    if group_by == "sample":
        indices = np.arange(len(sample_features))
        rng.shuffle(indices)

        split = int(round(len(indices) * tune_frac))
        split = max(1, min(split, len(indices) - 1))

        tune_idx = indices[:split]
        test_idx = indices[split:]
    elif group_by == "doc_name":
        doc_to_indices: Dict[str, List[int]] = {}
        for idx, sf in enumerate(sample_features):
            doc = sf.get("doc_name", "")
            doc_to_indices.setdefault(doc, []).append(idx)

        docs = np.array(sorted(doc_to_indices.keys()), dtype=object)
        if len(docs) < 2:
            raise ValueError("Need at least 2 unique doc_name values for doc_name holdout split")

        rng.shuffle(docs)
        split_docs = int(round(len(docs) * tune_frac))
        split_docs = max(1, min(split_docs, len(docs) - 1))

        tune_docs = set(docs[:split_docs].tolist())
        test_docs = set(docs[split_docs:].tolist())
        tune_idx = [i for i, sf in enumerate(sample_features) if sf.get("doc_name", "") in tune_docs]
        test_idx = [i for i, sf in enumerate(sample_features) if sf.get("doc_name", "") in test_docs]
    else:
        raise ValueError(f"Unsupported group_by: {group_by}")

    tune_features = [sample_features[i] for i in tune_idx]
    test_features = [sample_features[i] for i in test_idx]
    return tune_features, test_features


def _cfg_key(cfg: Dict[str, Any]) -> str:
    key_json = json.dumps(cfg, sort_keys=True)
    digest = hashlib.md5(key_json.encode("utf-8")).hexdigest()[:10]
    return f"cfg_{digest}"


def run_leakage_safe_holdout(
    sample_features: List[Dict[str, Any]],
    runs: List[SweepConfig],
    tune_frac: float,
    seed: int,
    group_by: str = "sample",
) -> Dict[str, Any]:
    tune_features, test_features = split_sample_features(
        sample_features,
        tune_frac=tune_frac,
        seed=seed,
        group_by=group_by,
    )
    tune_results, best_tune = run_sweep(tune_features, runs, desc="Sweep (holdout tune)")

    if not best_tune:
        return {
            "mode": "holdout",
            "error": "No configurations evaluated",
        }

    best_cfg = SweepConfig(**best_tune["config"])
    test_metrics = evaluate(test_features, best_cfg)
    tune_metrics = best_tune["metrics"]
    all_metrics = evaluate(sample_features, best_cfg)

    return {
        "mode": "holdout",
        "group_by": group_by,
        "seed": seed,
        "tune_frac": tune_frac,
        "n_tune": len(tune_features),
        "n_test": len(test_features),
        "best_config_from_tune": best_tune["config"],
        "tune_metrics_for_best": tune_metrics,
        "test_metrics_for_best": test_metrics,
        "all_metrics_for_best": all_metrics,
        "top_tune_results": tune_results[:10],
    }


def run_leakage_safe_kfold(
    sample_features: List[Dict[str, Any]],
    runs: List[SweepConfig],
    folds: int,
    seed: int,
    group_by: str = "sample",
) -> Dict[str, Any]:
    if folds < 2:
        raise ValueError("--cv-folds must be >= 2")
    n = len(sample_features)
    if n < folds:
        raise ValueError(f"Not enough samples ({n}) for {folds}-fold CV")

    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    if group_by == "sample":
        rng.shuffle(indices)
        fold_indices = np.array_split(indices, folds)
    elif group_by == "doc_name":
        doc_to_indices: Dict[str, List[int]] = {}
        for idx, sf in enumerate(sample_features):
            doc = sf.get("doc_name", "")
            doc_to_indices.setdefault(doc, []).append(idx)

        docs = np.array(sorted(doc_to_indices.keys()), dtype=object)
        if len(docs) < folds:
            raise ValueError(f"Not enough unique doc_name values ({len(docs)}) for {folds}-fold CV")

        rng.shuffle(docs)
        doc_folds = np.array_split(docs, folds)
        fold_indices = []
        for fold_docs in doc_folds:
            val_list: List[int] = []
            for d in fold_docs.tolist():
                val_list.extend(doc_to_indices[d])
            fold_indices.append(np.array(val_list, dtype=np.int64))
    else:
        raise ValueError(f"Unsupported group_by: {group_by}")


    fold_results = []
    config_counts: Dict[str, Dict[str, Any]] = {}
    recall_vals = []
    hit_vals = []
    all_val_outputs = []
    all_val_samples = []
    import copy
    import os
    from pathlib import Path
    from datetime import datetime
    # Directory for kfold-unified outputs
    kfold_out_dir = Path("results/kfold_unified_outputs")
    kfold_out_dir.mkdir(parents=True, exist_ok=True)

    for fold_id, val_idx in enumerate(fold_indices):
        val_set = set(val_idx.tolist())
        train_idx = [i for i in indices if i not in val_set]

        train_features = [sample_features[i] for i in train_idx]
        val_features = [sample_features[i] for i in val_idx.tolist()]

        train_results, best_train = run_sweep(train_features, runs, desc=f"Sweep (CV fold {fold_id+1}/{folds})")
        if not best_train:
            continue

        best_cfg = SweepConfig(**best_train["config"])
        val_metrics = evaluate(val_features, best_cfg)

        recall_vals.append(val_metrics.get("page_recall@k", 0.0))
        hit_vals.append(val_metrics.get("page_hit@k", 0.0))

        cfg_key = _cfg_key(best_train["config"])
        if cfg_key not in config_counts:
            config_counts[cfg_key] = {"config": best_train["config"], "count": 0}
        config_counts[cfg_key]["count"] += 1

        # --- KFold Unified-Style Inference ---
        # Use the best config for this fold, run inference on val_features only
        # Assume access to global_index and model from outer scope (or pass as args if needed)
        # Save outputs for each fold
        # NOTE: This code assumes global_index and model are available in the global scope.
        # If not, refactor to pass them as arguments.
        try:
            import numpy as np
            import json
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fold_out_path = kfold_out_dir / f"fold_{fold_id+1}_unified_style.json"
            fold_scored_path = kfold_out_dir / f"fold_{fold_id+1}_unified_style_scored.json"
            # Build features for val set
            val_samples = [sf.get("sample", None) or {k: sf[k] for k in ("doc_name","question","question_type","gold_pages","page_nums")} for sf in val_features]
            # Use the global_index and model from the outer scope
            # Rebuild features for val set
            from copy import deepcopy
            val_features_full = []
            for sample in val_samples:
                query = sample.get("question", "")
                q_vec = model.encode([query], convert_to_numpy=True, show_progress_bar=False)[0].astype(np.float32)
                dense = cosine_scores(q_vec, global_index.embeddings).astype(np.float32)
                bm25 = np.array(global_index.bm25.get_scores(tokenize_finance(query)), dtype=np.float32)
                targets = query_sections(query)
                section_prior = np.array([page_section_score(t, targets) for t in global_index.page_texts], dtype=np.float32)
                number_prior = np.array([number_alignment_score(query, t) for t in global_index.page_texts], dtype=np.float32)
                val_features_full.append({
                    "sample": sample,
                    "doc_name": sample.get("doc_name", ""),
                    "question": query,
                    "question_type": sample.get("question_type", ""),
                    "page_nums": np.array(global_index.page_nums, dtype=np.int32),
                    "dense": dense,
                    "bm25": bm25,
                    "section_prior": section_prior,
                    "number_prior": number_prior,
                    "doc_names": np.array(global_index.page_doc_names, dtype=object),
                })
            # Run inference for each val sample
            results = []
            for idx, sf in enumerate(val_features_full):
                sample = sf["sample"]
                doc_name = sf["doc_name"]
                # Use select_pages_from_features with best_cfg
                retrieved_pages = select_pages_from_features(sf, best_cfg)
                retrieved_chunks = []
                page_to_text = {p: t for p, t in zip(global_index.page_nums, global_index.page_texts)}
                for page in [r["page"] if isinstance(r, dict) else r for r in retrieved_pages]:
                    text = page_to_text.get(page, "")
                    if not text:
                        continue
                    retrieved_chunks.append({
                        "text": text,
                        "metadata": {
                            "doc_name": doc_name,
                            "page": int(page),
                        },
                        "score": None,
                    })
                context = "\n\n".join(c["text"] for c in retrieved_chunks)
                generated_answer = ""
                final_prompt = ""
                # Optionally, add generation here if desired (skip for now or add if generator is available)
                result_item = {
                    "sample_id": idx,
                    "doc_name": doc_name,
                    "question": sample.get("question", ""),
                    "reference_answer": sample.get("answer", ""),
                    "question_type": sample.get("question_type", ""),
                    "gold_evidence": sample.get("gold_evidence", ""),
                    "retrieved_chunks": retrieved_chunks,
                    "num_retrieved": len(retrieved_chunks),
                    "context_length": len(context),
                    "generated_answer": generated_answer,
                    "generation_length": len(generated_answer),
                    "experiment_type": "finance_adaptive_page_kfold",
                    "vector_store_type": "finance_adaptive_doc_local",
                    "pdf_source": "",
                    "final_prompt": final_prompt,
                    "metadata_page_retrieval": {
                        "retrieved_pages": [int(r["page"]) if isinstance(r, dict) else int(r) for r in retrieved_pages],
                        "config": best_cfg.__dict__,
                    },
                }
                results.append(result_item)
            payload = {
                "metadata": {
                    "experiment_type": "finance_adaptive_page_kfold",
                    "created_at": timestamp,
                    "model_path": "",
                    "inference_config": best_cfg.__dict__,
                    "n_questions": len(results),
                },
                "num_samples": len(results),
                "framework": "Finance adaptive page retrieval + unified-style outputs (kfold)",
                "aggregate_stats": aggregate_result_stats(results),
                "results": results,
            }
            with open(fold_out_path, "w") as f:
                json.dump(payload, f, indent=2)
            # Score outputs (retrieval metrics at k=1,3,5,10,20)
            scored_payload = _evaluate_outputs(payload, page_k=20, eval_mode="static")
            with open(fold_scored_path, "w") as f:
                json.dump(scored_payload, f, indent=2)
            all_val_outputs.extend(results)
            all_val_samples.extend(val_samples)
        except Exception as e:
            print(f"[KFold Unified Inference] Fold {fold_id+1} failed: {e}")

        fold_results.append({
            "fold": fold_id + 1,
            "n_train": len(train_features),
            "n_val": len(val_features),
            "best_config_from_train": best_train["config"],
            "train_metrics_for_best": best_train["metrics"],
            "val_metrics_for_best": val_metrics,
            "top_train_results": train_results[:5],
            "unified_style_output": str(fold_out_path),
            "unified_style_scored": str(fold_scored_path),
        })

    # Merge all validation outputs and score as a single file
    merged_out_path = kfold_out_dir / "kfold_merged_unified_style.json"
    merged_scored_path = kfold_out_dir / "kfold_merged_unified_style_scored.json"
    merged_payload = {
        "metadata": {
            "experiment_type": "finance_adaptive_page_kfold_merged",
            "created_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "model_path": "",
            "n_questions": len(all_val_outputs),
        },
        "num_samples": len(all_val_outputs),
        "framework": "Finance adaptive page retrieval + unified-style outputs (kfold merged)",
        "aggregate_stats": aggregate_result_stats(all_val_outputs),
        "results": all_val_outputs,
    }
    with open(merged_out_path, "w") as f:
        json.dump(merged_payload, f, indent=2)
    merged_scored_payload = _evaluate_outputs(merged_payload, page_k=20, eval_mode="static")
    with open(merged_scored_path, "w") as f:
        json.dump(merged_scored_payload, f, indent=2)

    config_ranking = sorted(config_counts.values(), key=lambda x: x["count"], reverse=True)
    recommended_config = config_ranking[0]["config"] if config_ranking else None

    return {
        "mode": "kfold",
        "group_by": group_by,
        "seed": seed,
        "folds": folds,
        "n_samples": n,
        "fold_results": fold_results,
        "cv_mean_page_recall@k": float(np.mean(recall_vals)) if recall_vals else 0.0,
        "cv_std_page_recall@k": float(np.std(recall_vals)) if recall_vals else 0.0,
        "cv_mean_page_hit@k": float(np.mean(hit_vals)) if hit_vals else 0.0,
        "cv_std_page_hit@k": float(np.std(hit_vals)) if hit_vals else 0.0,
        "config_frequency": config_ranking,
        "recommended_config": recommended_config,
    }


def run_leakage_safe_eval(args, sample_features: List[Dict[str, Any]], runs: List[SweepConfig], base_output_path: Path) -> Optional[Path]:
    mode = args.leakage_safe_mode
    if mode == "none":
        return None

    payload: Dict[str, Any] = {
        "metadata": {
            "created_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "mode": mode,
            "leakage_group_by": args.leakage_group_by,
            "n_questions_with_gold": len(sample_features),
            "n_configs": len(runs),
            "split_seed": args.split_seed,
            "tune_frac": args.tune_frac,
            "cv_folds": args.cv_folds,
        }
    }

    groupings = [args.leakage_group_by] if args.leakage_group_by != "both" else ["sample", "doc_name"]

    def _run_group(group_by: str) -> Dict[str, Any]:
        group_payload: Dict[str, Any] = {
            "group_by": group_by,
        }
        if mode in {"holdout", "both"}:
            group_payload["holdout"] = run_leakage_safe_holdout(
                sample_features=sample_features,
                runs=runs,
                tune_frac=args.tune_frac,
                seed=args.split_seed,
                group_by=group_by,
            )
        if mode in {"kfold", "both"}:
            group_payload["kfold"] = run_leakage_safe_kfold(
                sample_features=sample_features,
                runs=runs,
                folds=args.cv_folds,
                seed=args.split_seed,
                group_by=group_by,
            )
        return group_payload

    if len(groupings) == 1:
        single_payload = _run_group(groupings[0])
        payload.update(single_payload)
    else:
        payload["groupings"] = {g: _run_group(g) for g in groupings}

    leakage_output = Path(args.leakage_output) if args.leakage_output else base_output_path.with_name(f"{base_output_path.stem}_leakage_safe.json")
    leakage_output.parent.mkdir(parents=True, exist_ok=True)
    with leakage_output.open("w") as f:
        json.dump(payload, f, indent=2)

    logger.info(f"Saved leakage-safe evaluation to: {leakage_output}")
    return leakage_output


def main():
    parser = argparse.ArgumentParser(description="Finance-adaptive page retrieval sweep")
    parser.add_argument("--data", type=str, default="data/financebench_open_source.jsonl")
    parser.add_argument("--pdf-dir", type=str, default="pdfs")
    parser.add_argument("--model-path", type=str, default="models/finetuned_page_scorer_v2")
    parser.add_argument("--page-k", type=int, default=20)
    parser.add_argument("--seed-ks", type=str, default="6,8,10")
    parser.add_argument("--dense-weights", type=str, default="0.55,0.65,0.75")
    parser.add_argument("--bm25-weights", type=str, default="0.20,0.30,0.40")
    parser.add_argument("--section-weights", type=str, default="0.05,0.10,0.15")
    parser.add_argument("--number-weights", type=str, default="0.00,0.05")
    parser.add_argument("--base-windows", type=str, default="1,2")
    parser.add_argument("--distance-penalty", type=float, default=0.08)
    parser.add_argument("--max-runs", type=int, default=0, help="0 means all combinations")
    parser.add_argument("--output", type=str, default="results/finance_adaptive_page_sweep.json")
    parser.add_argument("--cache", type=str, default="outputs/cache/finance_adaptive_global_index.pkl")
    parser.add_argument("--run-inference", action="store_true", help="Run full per-question inference and output unified-style eval artifacts")
    parser.add_argument("--best-from-sweep", type=str, default="", help="Path to sweep JSON; use its best config for inference")
    parser.add_argument("--best-from-leakage-safe", type=str, default="", help="Path to leakage-safe JSON; prefer holdout tune config or CV recommended config")
    parser.add_argument("--inference-output", type=str, default="", help="Output JSON path for inference results")
    parser.add_argument("--skip-generation", action="store_true", help="Skip answer generation and evaluate retrieval only")
    parser.add_argument("--gen-model", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Generation model for inference mode")
    parser.add_argument("--gen-embedding-model", type=str, default="sentence-transformers/all-mpnet-base-v2", help="Embedding model for generator experiment init")
    parser.add_argument("--gen-output-dir", type=str, default="outputs/finance_adaptive_generation", help="Output dir used by generation experiment object")
    parser.add_argument("--gen-vector-store-dir", type=str, default="vector_stores", help="Vector store dir used by generation experiment object")
    parser.add_argument("--gen-use-api", action="store_true", help="Use API-based generation mode")
    parser.add_argument("--gen-api-base-url", type=str, default="https://api.openai.com/v1", help="Base URL for API generation")
    parser.add_argument("--gen-api-key-env", type=str, default="OPENAI_API_KEY", help="Env var for API key used in generation")
    parser.add_argument("--eval-mode", type=str, default="static", choices=["static", "semantic"], help="Evaluation mode for generative metrics")
    parser.add_argument("--leakage-safe-mode", type=str, default="none", choices=["none", "holdout", "kfold", "both"], help="Leakage-safe evaluation mode")
    parser.add_argument("--leakage-group-by", type=str, default="sample", choices=["sample", "doc_name", "both"], help="Split unit for leakage-safe eval")
    parser.add_argument("--tune-frac", type=float, default=0.8, help="Tune split fraction for holdout leakage-safe mode")
    parser.add_argument("--split-seed", type=int, default=42, help="Random seed for leakage-safe splitting")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of folds for k-fold leakage-safe mode")
    parser.add_argument("--leakage-output", type=str, default="", help="Output path for leakage-safe evaluation JSON")
    args = parser.parse_args()

    dataset = load_dataset(Path(args.data))
    logger.info(f"Loaded dataset rows: {len(dataset)}")

    model = SentenceTransformer(args.model_path)
    global_index = build_global_doc_index(dataset, Path(args.pdf_dir), model, Path(args.cache))
    logger.info(f"Built global doc index: {len(global_index.page_nums)} pages")
    sample_features = build_sample_features(dataset, global_index, model)
    logger.info(f"Prepared features for {len(sample_features)} questions")
    runs = build_runs(args)
    logger.info(f"Running sweep with {len(runs)} configs")
    all_results, best = run_sweep(sample_features, runs, desc="Sweeping")
    output = {
        "n_configs": len(all_results),
        "best": best,
        "results": all_results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(output, f, indent=2)
    if best:
        logger.info("=" * 80)
        logger.info(f"BEST page_recall@{args.page_k}: {best['metrics']['page_recall@k']:.4f}")
        logger.info(f"BEST page_hit@{args.page_k}: {best['metrics']['page_hit@k']:.4f}")
        logger.info(f"BEST config: {best['config']}")
        logger.info("=" * 80)
    logger.info(f"Saved sweep results to: {out_path}")
    run_leakage_safe_eval(args, sample_features=sample_features, runs=runs, base_output_path=out_path)
    if args.run_inference:
        default_cfg = SweepConfig(**best["config"]) if best else runs[0]
        run_inference(args, model=model, doc_indices={"ALL_DOCS": global_index}, dataset=dataset, default_cfg=default_cfg)

if __name__ == "__main__":
    main()
