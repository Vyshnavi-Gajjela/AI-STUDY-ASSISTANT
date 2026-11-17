import re


def clean_text(text: str) -> str:
    """Normalize whitespace in extracted document text."""
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()