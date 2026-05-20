"""
Extract page-level text from FinQA PDFs and build a pages JSONL.
Each record = one page from one document.
Gold evidence pages are flagged with label=1.
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
    import fitz  # PyMuPDF < 1.24
from tqdm import tqdm
from src.utils.io import load_jsonl, save_jsonl
from src.utils.logging import get_logger

logger = get_logger("build_finqa_pages")


def parse_doc_name(qid: str) -> str | None:
    """Extract doc_name like AAL_2014_10K from qid or evidence."""
    return None


def extract_pages(pdf_path: pathlib.Path) -> list[dict]:
    pages = []
    try:
        doc = fitz.open(str(pdf_path))
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            pages.append({
                "page_number": page_num,
                "page_number_1indexed": page_num + 1,
                "text": text,
                "char_count": len(text),
            })
        doc.close()
    except Exception as e:
        logger.warning(f"Failed to extract {pdf_path}: {e}")
    return pages


def parse_doc_metadata(doc_name: str) -> dict:
    """Parse ticker, year, form from doc_name like AAL_2014_10K or AAL_2023Q2_10Q."""
    parts = doc_name.split("_")
    ticker = parts[0] if parts else ""
    form_type = parts[-1] if len(parts) > 1 else ""
    # year / period
    period = ""
    if len(parts) >= 3:
        period = parts[1]
    year = re.findall(r"\d{4}", period)
    fiscal_year = year[0] if year else ""
    return {
        "ticker": ticker,
        "company": ticker,
        "form_type": form_type,
        "fiscal_year": fiscal_year,
        "doc_period": period,
    }


def build_finqa_pages(
    gold_pages_path: str,
    pdf_dirs: list[str],
    out_dir: str,
    min_chars: int = 50,
) -> None:
    gold_records = load_jsonl(gold_pages_path)
    logger.info(f"Loaded {len(gold_records)} FinQA gold-page records")

    # Build gold page index: doc_name -> set of 0-indexed page numbers
    gold_index: dict[str, set[int]] = {}
    for rec in gold_records:
        for ev in rec.get("evidences_updated", rec.get("evidences", [])):
            dn = ev["doc_name"]
            pg = ev["page_num"]  # 0-indexed in FinQA
            gold_index.setdefault(dn, set()).add(pg)

    all_docs = set(gold_index.keys())
    logger.info(f"Unique FinQA docs with gold pages: {len(all_docs)}")

    # Find PDFs
    pdf_map: dict[str, pathlib.Path] = {}
    for pdf_dir in pdf_dirs:
        p = pathlib.Path(pdf_dir)
        if not p.exists():
            continue
        for f in p.iterdir():
            if f.suffix.lower() == ".pdf":
                pdf_map[f.stem] = f

    found = set(all_docs) & set(pdf_map.keys())
    missing = set(all_docs) - set(pdf_map.keys())
    logger.info(f"PDFs found: {len(found)}, missing: {len(missing)}")
    if missing:
        logger.warning(f"Missing PDFs: {sorted(missing)[:10]}...")

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_pages = []
    skipped = 0

    for doc_name in tqdm(sorted(pdf_map.keys()), desc="Extracting FinQA pages"):
        pdf_path = pdf_map[doc_name]
        meta = parse_doc_metadata(doc_name)
        pages = extract_pages(pdf_path)

        doc_gold_pages = gold_index.get(doc_name, set())

        for p in pages:
            if p["char_count"] < min_chars:
                skipped += 1
                continue
            candidate_id = f"{doc_name}_page_{p['page_number']}"
            record = {
                "candidate_id": candidate_id,
                "doc_id": doc_name,
                "page_number": p["page_number"],
                "page_number_1indexed": p["page_number_1indexed"],
                "text": p["text"],
                "char_count": p["char_count"],
                "label": 1 if p["page_number"] in doc_gold_pages else 0,
                "source_dataset": "finqa",
                "metadata": {
                    **meta,
                    "doc_name": doc_name,
                    "page_number": p["page_number"],
                    "filing_date": "",
                    "section": "",
                },
            }
            all_pages.append(record)

    logger.info(f"Total pages: {len(all_pages)}, skipped (too short): {skipped}")
    positives = sum(1 for p in all_pages if p["label"] == 1)
    logger.info(f"Positive (gold) pages: {positives}")

    out_path = out_dir / "finqa_pages.jsonl"
    save_jsonl(all_pages, out_path)
    logger.info(f"Saved to {out_path}")

    # Summary
    summary = {
        "total_pages": len(all_pages),
        "positive_pages": positives,
        "negative_pages": len(all_pages) - positives,
        "unique_docs": len(set(p["doc_id"] for p in all_pages)),
        "skipped_short": skipped,
        "pdf_dirs": pdf_dirs,
    }
    with open(out_dir / "finqa_pages_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary: {summary}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_pages", default="../data/finqa_test_gold_pages.jsonl")
    parser.add_argument("--pdf_dirs", nargs="+", default=["../pdfs-extended-v4", "../pdfs"])
    parser.add_argument("--out_dir", default="data/processed/pages")
    parser.add_argument("--min_chars", type=int, default=50)
    args = parser.parse_args()
    build_finqa_pages(args.gold_pages, args.pdf_dirs, args.out_dir, args.min_chars)
