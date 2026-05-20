"""
Extract page-level text from FinanceBench PDFs.
Aligns evidence text to specific pages using fuzzy matching.
"""
import argparse
import pathlib
import json
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

try:
    import pymupdf as fitz
except ImportError:
    import fitz
from rapidfuzz import fuzz
from tqdm import tqdm
from src.utils.io import load_jsonl, save_jsonl
from src.utils.text import normalize_text
from src.utils.logging import get_logger

logger = get_logger("build_financebench_pages")


def parse_doc_metadata(doc_name: str, doc_info_map: dict) -> dict:
    info = doc_info_map.get(doc_name, {})
    company = info.get("company", doc_name.split("_")[0])
    form_type = info.get("doc_type", "").upper()
    period = str(info.get("doc_period", ""))
    year = re.findall(r"\d{4}", period)
    fiscal_year = year[0] if year else period
    return {
        "ticker": company,
        "company": company,
        "form_type": form_type,
        "fiscal_year": fiscal_year,
        "doc_period": period,
        "doc_name": doc_name,
        "filing_date": "",
        "section": "",
    }


def align_evidence_to_page(
    evidence_text: str,
    pages: list[dict],
    threshold: float = 0.35,
) -> tuple[int | None, float]:
    """Return (0-indexed page_number, score) for best matching page."""
    ev_norm = normalize_text(evidence_text[:500])  # use first 500 chars for speed
    best_page = None
    best_score = 0.0
    for p in pages:
        page_text = normalize_text(p["text"][:2000])
        if not page_text:
            continue
        score = fuzz.partial_ratio(ev_norm, page_text) / 100.0
        if score > best_score:
            best_score = score
            best_page = p["page_number"]
    if best_score >= threshold:
        return best_page, best_score
    return None, best_score


def build_financebench_pages(
    fb_path: str,
    docinfo_path: str,
    pdf_dir: str,
    out_dir: str,
    align_threshold: float = 0.35,
    min_chars: int = 50,
) -> None:
    fb_records = load_jsonl(fb_path)
    docinfo_records = load_jsonl(docinfo_path)
    doc_info_map = {r["doc_name"]: r for r in docinfo_records}
    logger.info(f"Loaded {len(fb_records)} FinanceBench records, {len(doc_info_map)} doc infos")

    # Build evidence index: doc_name -> list of evidence texts + qid
    evidence_index: dict[str, list[dict]] = {}
    for rec in fb_records:
        doc_name = rec["doc_name"]
        for ev in rec.get("evidence", []):
            ev_text = ev.get("evidence_text", "")
            if ev_text:
                evidence_index.setdefault(doc_name, []).append({
                    "qid": rec["financebench_id"],
                    "evidence_text": ev_text,
                })

    # Find PDFs
    pdf_dir = pathlib.Path(pdf_dir)
    pdf_map: dict[str, pathlib.Path] = {}
    for f in pdf_dir.iterdir():
        if f.suffix.lower() == ".pdf":
            pdf_map[f.stem] = f

    all_doc_names = set(r["doc_name"] for r in fb_records)
    found = all_doc_names & set(pdf_map.keys())
    missing = all_doc_names - set(pdf_map.keys())
    logger.info(f"FB PDFs found: {len(found)}, missing: {len(missing)}")

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_pages = []
    alignment_log = []
    skipped = 0
    aligned = 0
    failed = 0

    for doc_name in tqdm(sorted(found), desc="Extracting FB pages"):
        pdf_path = pdf_map[doc_name]
        meta = parse_doc_metadata(doc_name, doc_info_map)

        try:
            doc = fitz.open(str(pdf_path))
            raw_pages = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")
                raw_pages.append({"page_number": page_num, "text": text, "char_count": len(text)})
            doc.close()
        except Exception as e:
            logger.warning(f"Failed {doc_name}: {e}")
            continue

        # Find gold pages by aligning evidence
        gold_pages: set[int] = set()
        doc_evidences = evidence_index.get(doc_name, [])
        for ev_item in doc_evidences:
            page_num, score = align_evidence_to_page(
                ev_item["evidence_text"], raw_pages, align_threshold
            )
            alignment_log.append({
                "qid": ev_item["qid"],
                "doc_name": doc_name,
                "page_num": page_num,
                "score": score,
                "aligned": page_num is not None,
            })
            if page_num is not None:
                gold_pages.add(page_num)
                aligned += 1
            else:
                failed += 1

        for p in raw_pages:
            if p["char_count"] < min_chars:
                skipped += 1
                continue
            candidate_id = f"{doc_name}_page_{p['page_number']}"
            record = {
                "candidate_id": candidate_id,
                "doc_id": doc_name,
                "page_number": p["page_number"],
                "page_number_1indexed": p["page_number"] + 1,
                "text": p["text"],
                "char_count": p["char_count"],
                "label": 1 if p["page_number"] in gold_pages else 0,
                "source_dataset": "financebench",
                "metadata": {**meta, "page_number": p["page_number"]},
            }
            all_pages.append(record)

    logger.info(f"FB total pages: {len(all_pages)}, positives: {sum(p['label'] for p in all_pages)}")
    logger.info(f"Evidence alignments: {aligned} success, {failed} failed")

    save_jsonl(all_pages, out_dir / "financebench_pages.jsonl")
    save_jsonl(alignment_log, out_dir / "financebench_alignment_log.jsonl")

    summary = {
        "total_pages": len(all_pages),
        "positive_pages": int(sum(p["label"] for p in all_pages)),
        "unique_docs": len(found),
        "aligned_evidences": aligned,
        "failed_alignments": failed,
        "align_threshold": align_threshold,
    }
    with open(out_dir / "financebench_pages_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary: {summary}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fb_path", default="../data/financebench_open_source.jsonl")
    parser.add_argument("--docinfo_path", default="../data/financebench_document_information.jsonl")
    parser.add_argument("--pdf_dir", default="../pdfs")
    parser.add_argument("--out_dir", default="data/processed/pages")
    parser.add_argument("--align_threshold", type=float, default=0.35)
    parser.add_argument("--min_chars", type=int, default=50)
    args = parser.parse_args()
    build_financebench_pages(
        args.fb_path, args.docinfo_path, args.pdf_dir,
        args.out_dir, args.align_threshold, args.min_chars
    )
