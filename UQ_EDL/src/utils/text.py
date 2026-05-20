import re
import unicodedata


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def token_overlap(a: str, b: str) -> float:
    ta = set(normalize_text(a).split())
    tb = set(normalize_text(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def truncate_text(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def build_model_input(question: str, candidate_text: str, metadata: dict | None = None) -> str:
    parts = [f"Question: {question}"]
    if metadata:
        meta_lines = ["", "Candidate metadata:"]
        if metadata.get("company"):
            meta_lines.append(f"Company: {metadata['company']}")
        if metadata.get("ticker"):
            meta_lines.append(f"Ticker: {metadata['ticker']}")
        if metadata.get("form_type"):
            meta_lines.append(f"Form: {metadata['form_type']}")
        if metadata.get("fiscal_year"):
            meta_lines.append(f"Fiscal year: {metadata['fiscal_year']}")
        if metadata.get("filing_date"):
            meta_lines.append(f"Filing date: {metadata['filing_date']}")
        if metadata.get("page_number") is not None:
            meta_lines.append(f"Page: {metadata['page_number']}")
        if metadata.get("section"):
            meta_lines.append(f"Section: {metadata['section']}")
        parts.append("\n".join(meta_lines))
    parts.append(f"\nCandidate page:\n{candidate_text}")
    return "\n".join(parts)
