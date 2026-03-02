"""Compute per-page character length stats for all PDFs."""

from pathlib import Path
import statistics
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.ingestion.page_processor import extract_pages_from_pdf


def main() -> None:
    pdf_dir = Path("pdfs")
    lengths = []
    failures = []

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    print(f"Scanning {len(pdf_files)} PDFs in {pdf_dir}...")

    for idx, pdf_path in enumerate(pdf_files, start=1):
        print(f"[{idx}/{len(pdf_files)}] Reading {pdf_path.name}")
        try:
            pages = extract_pages_from_pdf(pdf_path, pdf_path.stem)
        except Exception as exc:
            failures.append((pdf_path.name, str(exc)))
            print(f"  ! Failed to read {pdf_path.name}: {exc}")
            continue

        print(f"  -> {len(pages)} pages extracted")

        for page in pages:
            text = page.get("text") or ""
            lengths.append(len(text))

    if not lengths:
        print("No pages found.")
        if failures:
            print("Failures:")
            for name, err in failures:
                print(f"  {name}: {err}")
        return

    print(f"Pages: {len(lengths)}")
    print(f"Mean chars: {statistics.mean(lengths):.2f}")
    print(f"Median chars: {statistics.median(lengths):.2f}")
    print(f"Min chars: {min(lengths)}")
    print(f"Max chars: {max(lengths)}")
    print(f"Std chars: {statistics.pstdev(lengths):.2f}")

    if failures:
        print("\nFailures:")
        for name, err in failures:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
