from pathlib import Path

import pymupdf 


def load_pdf(file_path: str | Path) -> list[dict]:
    """
    Extract text from a PDF, keeping the page number for citations.

    Returns:
        A list of dictionaries, one per page:
        [{"source": "file.pdf", "page": 1, "text": "..."}]
    """
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path.suffix}")

    pages = []

    with pymupdf.open(path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if text:
                pages.append({
                    "source": path.name,
                    "page": page_number,
                    "text": text,
                })

    return pages