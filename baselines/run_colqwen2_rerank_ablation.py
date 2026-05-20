"""
run_colqwen2_rerank_ablation.py
================================
Ablation: ColQwen2 visual page retrieval → cross-encoder reranking,
with NO doc-level aggregation (the bottleneck in colpali_page).

For each query:
  1. Score all pages globally with ColQwen2 (MaxSim late-interaction).
  2. Take global top-N pages (no doc filtering).
  3. Extract plain text from those pages via PyMuPDF.
  4. Rerank with CrossEncoder → keep top-5.

Four variants run from a single ColQwen2 embedding pass:
  A. top-20  + BGE-reranker-v2-m3
  B. top-100 + BGE-reranker-v2-m3
  C. top-20  + FT cross-encoder  (checkpoints/ft_cross_encoder)
  D. top-100 + FT cross-encoder

Outputs (per variant):
  results/colqwen2_rerank_ablation/<variant>/<variant>_<timestamp>.json
  results/colqwen2_rerank_ablation/<variant>/<variant>_<timestamp>_scored.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import fitz
    if not hasattr(fitz, "open"):
        raise ImportError
except Exception:
    import pymupdf as fitz

try:
    from PIL import Image
except ImportError:
    Image = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("colqwen2_rerank_ablation")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_FILE    = PROJECT_ROOT / "data" / "financebench_open_source.jsonl"
PDF_DIR      = PROJECT_ROOT / "pdfs"
FT_CKPT      = PROJECT_ROOT / "checkpoints" / "ft_cross_encoder"
BGE_RERANKER = "BAAI/bge-reranker-v2-m3"
COLQWEN2     = "vidore/colqwen2-v1.0-hf"
RESULTS_ROOT = PROJECT_ROOT / "results" / "colqwen2_rerank_ablation"

TOP_K_FINAL  = 5      # pages passed to evaluator
RERANK_BATCH = 64     # CrossEncoder batch size
DPI          = 150
PAGE_BATCH   = 4      # pages per ColQwen2 forward pass
QUERY_BATCH  = 1


# ── Dataset helpers ────────────────────────────────────────────────────────────
def load_dataset(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_gold_segments(evidence: Any, doc_name: str) -> List[Dict[str, Any]]:
    if not evidence:
        return []
    if isinstance(evidence, str):
        try:
            evidence = json.loads(evidence)
        except Exception:
            return []
    segs = []
    for ev in evidence:
        page = ev.get("evidence_page_num")
        segs.append({
            "text":     ev.get("evidence_text", ""),
            "doc_name": ev.get("doc_name", doc_name),
            "page":     page,
        })
    return segs


# ── PDF helpers ────────────────────────────────────────────────────────────────
def find_pdf(doc_name: str, pdf_dir: Path) -> Optional[Path]:
    for p in pdf_dir.glob("**/*.pdf"):
        if p.stem.lower() == doc_name.lower():
            return p
    return None


def render_pages(pdf_path: Path, doc_name: str, dpi: int) -> List[Dict[str, Any]]:
    if Image is None:
        raise RuntimeError("Pillow required.")
    pages = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.warning("Cannot open %s: %s", pdf_path, e)
        return pages
    try:
        for idx in range(len(doc)):
            try:
                page = doc[idx]
                pix  = page.get_pixmap(dpi=dpi)
                mode = "RGBA" if pix.alpha else "RGB"
                img  = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                if mode == "RGBA":
                    img = img.convert("RGB")
                pages.append({"image": img, "doc_name": doc_name,
                               "page": idx, "pdf_path": str(pdf_path)})
            except Exception as e:
                logger.warning("Page %d of %s: %s", idx, pdf_path, e)
    finally:
        doc.close()
    return pages


def load_page_text(pdf_path: str, page_idx: int) -> str:
    try:
        doc  = fitz.open(pdf_path)
        text = doc[page_idx].get_text("text") if page_idx < len(doc) else ""
        doc.close()
        return text or ""
    except Exception:
        return ""


# ── ColQwen2 helpers ───────────────────────────────────────────────────────────
def load_colqwen2():
    from transformers import ColQwen2ForRetrieval, ColQwen2Processor
    from transformers.utils.import_utils import is_flash_attn_2_available
    return ColQwen2ForRetrieval, ColQwen2Processor, is_flash_attn_2_available


def trim_zero_rows(t: torch.Tensor) -> torch.Tensor:
    if t.ndim != 2:
        return t
    keep = torch.any(t != 0, dim=-1)
    return t[keep] if torch.any(keep) else t


def score_query_vs_pages(
    processor: Any,
    q_emb: torch.Tensor,
    page_embs: List[torch.Tensor],
    device: torch.device,
    batch: int,
) -> np.ndarray:
    scores = []
    q_batch = [trim_zero_rows(q_emb).to(device)]
    with torch.inference_mode():
        for start in range(0, len(page_embs), batch):
            docs = [trim_zero_rows(e).to(device) for e in page_embs[start:start + batch]]
            s = processor.score_retrieval(q_batch, docs)
            if isinstance(s, torch.Tensor):
                s = s.detach().float().cpu().numpy().reshape(-1)
            else:
                s = np.asarray(s).reshape(-1)
            scores.append(s)
    return np.concatenate(scores) if scores else np.zeros(0, dtype=np.float32)


# ── Reranker helpers ───────────────────────────────────────────────────────────
def load_cross_encoder(model_path: str):
    from sentence_transformers.cross_encoder import CrossEncoder
    logger.info("Loading CrossEncoder: %s", model_path)
    return CrossEncoder(str(model_path), max_length=512)


def rerank(model: Any, query: str, pages: List[Dict[str, Any]], batch: int) -> List[Dict[str, Any]]:
    pairs  = [(query, p["text"]) for p in pages]
    scores_out: List[float] = []
    for i in range(0, len(pairs), batch):
        s = model.predict(pairs[i:i + batch])
        scores_out.extend(s.tolist() if hasattr(s, "tolist") else list(s))
    for page, sc in zip(pages, scores_out):
        page["reranker_score"] = float(sc)
    return sorted(pages, key=lambda x: x["reranker_score"], reverse=True)


# ── Evaluation ─────────────────────────────────────────────────────────────────
def evaluate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    from src.evaluation.retrieval_evaluator import RetrievalEvaluator
    ev = RetrievalEvaluator()
    return ev.compute_metrics(results, k_values=[1, 3, 5])


# ── Result formatting ──────────────────────────────────────────────────────────
def build_result(
    idx: int,
    sample: Dict[str, Any],
    top_pages: List[Dict[str, Any]],
    variant: str,
) -> Dict[str, Any]:
    gold_segs = parse_gold_segments(sample.get("evidence"), sample.get("doc_name", ""))
    retrieved_chunks = []
    for rank, p in enumerate(top_pages, 1):
        retrieved_chunks.append({
            "text": p["text"],
            "metadata": {
                "doc_name":       p["doc_name"],
                "page":           p["page"],
                "pdf_path":       p["pdf_path"],
                "rank":           rank,
                "score":          p.get("reranker_score", p.get("visual_score", 0.0)),
                "visual_score":   p.get("visual_score", 0.0),
                "reranker_score": p.get("reranker_score", 0.0),
                "retrieval":      variant,
            },
        })
    return {
        "sample_id":             idx,
        "doc_name":              sample.get("doc_name"),
        "question":              sample.get("question", ""),
        "reference_answer":      sample.get("answer"),
        "question_type":         sample.get("question_type"),
        "question_reasoning":    sample.get("question_reasoning"),
        "gold_evidence":         " ".join(s.get("text", "") for s in gold_segs),
        "gold_evidence_segments": gold_segs,
        "retrieved_chunks":      retrieved_chunks,
        "num_retrieved":         len(retrieved_chunks),
        "generated_answer":      "",
        "experiment_type":       variant,
    }


# ── Save helpers ───────────────────────────────────────────────────────────────
def save(results: List[Dict[str, Any]], summary: Dict[str, Any],
         variant: str, ts: str, metadata: Dict[str, Any]) -> Path:
    out_dir = RESULTS_ROOT / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata":           metadata,
        "experiment_name":    variant,
        "num_samples":        len(results),
        "evaluation_summary": {"retrieval": summary},
        "results":            results,
    }
    path = out_dir / f"{variant}_{ts}_scored.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved → %s", path)
    return path


# ── Main ───────────────────────────────────────────────────────────────────────
def main(args: argparse.Namespace) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load dataset
    data = load_dataset(DATA_FILE)
    logger.info("Loaded %d samples", len(data))

    # Find all PDFs
    all_pdfs = list(PDF_DIR.glob("**/*.pdf"))
    logger.info("Found %d PDFs in %s", len(all_pdfs), PDF_DIR)

    # Load ColQwen2
    ColQwen2ForRetrieval, ColQwen2Processor, is_flash_attn_2_available = load_colqwen2()

    if torch.cuda.is_available():
        device     = torch.device("cuda")
        dtype      = torch.bfloat16
        attn_impl  = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
        device_map = "auto"
    else:
        device     = torch.device("cpu")
        dtype      = torch.float32
        attn_impl  = "sdpa"
        device_map = None

    logger.info("Loading ColQwen2 (%s) …", COLQWEN2)
    mkw = {"attn_implementation": attn_impl}
    if device_map:
        mkw["device_map"] = device_map
    try:
        model = ColQwen2ForRetrieval.from_pretrained(COLQWEN2, dtype=dtype, **mkw).eval()
    except TypeError:
        model = ColQwen2ForRetrieval.from_pretrained(COLQWEN2, torch_dtype=dtype, **mkw).eval()
    processor = ColQwen2Processor.from_pretrained(COLQWEN2)

    # Render + embed all pages
    logger.info("Rendering and embedding %d PDFs at %d DPI …", len(all_pdfs), DPI)
    page_records: List[Dict[str, Any]] = []
    page_embs:    List[torch.Tensor]   = []

    with torch.inference_mode():
        for pdf_path in all_pdfs:
            doc_name = pdf_path.stem
            pages    = render_pages(pdf_path, doc_name, DPI)
            if not pages:
                continue
            for start in range(0, len(pages), PAGE_BATCH):
                batch  = pages[start:start + PAGE_BATCH]
                inputs = processor(images=[p["image"] for p in batch], return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                embs   = model(**inputs).embeddings.detach().cpu()
                for i, emb in enumerate(torch.unbind(embs, dim=0)):
                    page_embs.append(trim_zero_rows(emb))
                    page_records.append({
                        "doc_name": batch[i]["doc_name"],
                        "page":     batch[i]["page"],
                        "pdf_path": batch[i]["pdf_path"],
                    })

    logger.info("Embedded %d pages total", len(page_records))

    # Embed all queries
    logger.info("Embedding %d queries …", len(data))
    query_embs: List[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(data), QUERY_BATCH):
            batch   = [d.get("question", "") for d in data[start:start + QUERY_BATCH]]
            inputs  = processor(text=batch, return_tensors="pt")
            inputs  = {k: v.to(device) for k, v in inputs.items()}
            embs    = model(**inputs).embeddings.detach().cpu()
            for emb in torch.unbind(embs, dim=0):
                query_embs.append(trim_zero_rows(emb))

    # Score all queries vs all pages  →  (n_queries, n_pages)
    logger.info("Scoring %d queries × %d pages …", len(query_embs), len(page_records))
    all_scores = np.zeros((len(query_embs), len(page_records)), dtype=np.float32)
    for qi, q_emb in enumerate(query_embs):
        all_scores[qi] = score_query_vs_pages(
            processor, q_emb, page_embs, device, batch=64
        )

    # Free ColQwen2 VRAM before loading rerankers
    del model, query_embs, page_embs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Pre-cache page texts for all pages that will be needed
    # (we'll load on demand; cache to avoid re-opening PDFs)
    page_text_cache: Dict[Tuple[str, int], str] = {}

    def get_text(pdf_path: str, page_idx: int) -> str:
        key = (pdf_path, page_idx)
        if key not in page_text_cache:
            page_text_cache[key] = load_page_text(pdf_path, page_idx)
        return page_text_cache[key]

    # Build page lookup: doc_name → set of global indices
    doc_to_page_ids: Dict[str, List[int]] = defaultdict(list)
    for gi, pr in enumerate(page_records):
        doc_to_page_ids[pr["doc_name"]].append(gi)

    # Define the 4 variants
    variants = []
    if not args.skip_bge:
        variants += [
            ("colqwen2_top20_bge_rerank",  20,  BGE_RERANKER,    False),
            ("colqwen2_top100_bge_rerank", 100, BGE_RERANKER,    False),
        ]
    if not args.skip_ft:
        variants += [
            ("colqwen2_top20_ft_rerank",   20,  str(FT_CKPT),    True),
            ("colqwen2_top100_ft_rerank",  100, str(FT_CKPT),    True),
        ]

    for variant_name, top_n, reranker_path, is_ft in variants:
        logger.info("\n" + "=" * 70)
        logger.info("VARIANT: %s  (top-%d → %s)", variant_name, top_n, reranker_path)
        logger.info("=" * 70)

        ce_model = load_cross_encoder(reranker_path)

        results: List[Dict[str, Any]] = []
        for idx, sample in enumerate(data):
            scores = all_scores[idx]
            n_candidates = min(top_n, len(scores))

            # Global top-N (no doc aggregation)
            top_indices = np.argsort(scores)[::-1][:n_candidates]

            # Build page dicts with text
            candidates = []
            for gi in top_indices:
                pr   = page_records[int(gi)]
                text = get_text(pr["pdf_path"], pr["page"])
                candidates.append({
                    "doc_name":     pr["doc_name"],
                    "page":         pr["page"],
                    "pdf_path":     pr["pdf_path"],
                    "text":         text,
                    "visual_score": float(scores[int(gi)]),
                })

            # Cross-encoder rerank → top-5
            ranked    = rerank(ce_model, sample.get("question", ""), candidates, RERANK_BATCH)
            final_top = ranked[:TOP_K_FINAL]

            results.append(build_result(idx, sample, final_top, variant_name))

            if (idx + 1) % 25 == 0:
                logger.info("  %d/%d done", idx + 1, len(data))

        # Evaluate
        logger.info("Evaluating %s …", variant_name)
        summary = evaluate_results(results)
        logger.info("Results for %s:", variant_name)
        for k, v in sorted(summary.items()):
            logger.info("  %s: %.4f", k, v)

        metadata = {
            "experiment_type":  variant_name,
            "colqwen2_model":   COLQWEN2,
            "reranker_model":   reranker_path,
            "ft_reranker":      is_ft,
            "top_n_candidates": top_n,
            "top_k_final":      TOP_K_FINAL,
            "dpi":              DPI,
            "timestamp":        ts,
            "num_samples":      len(results),
        }
        save(results, summary, variant_name, ts, metadata)

        # Free reranker between variants
        del ce_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("\nAll variants complete. Results in: %s", RESULTS_ROOT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ColQwen2 → CrossEncoder ablation (no doc-level aggregation)")
    parser.add_argument("--skip-bge", action="store_true", help="Skip BGE reranker variants")
    parser.add_argument("--skip-ft",  action="store_true", help="Skip fine-tuned reranker variants")
    main(parser.parse_args())
