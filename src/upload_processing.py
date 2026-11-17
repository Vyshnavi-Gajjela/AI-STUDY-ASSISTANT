import tempfile
from pathlib import Path

from src.chunking import chunk_documents
from src.config import CHUNK_OVERLAP, CHUNK_SIZE
from src.document_loader import load_pdf
from src.embeddings import embed_texts
from src.ppt_loader import load_powerpoint
from src.vector_store import build_vector_store


def process_uploaded_document(file_name: str, file_bytes: bytes):
    """Extract, chunk, and index an uploaded PDF or PPTX."""
    suffix = Path(file_name).suffix.lower()
    if suffix not in {".pdf", ".pptx"}:
        raise ValueError("Upload a PDF or PowerPoint (.pptx) document.")

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_path = Path(temporary_directory) / f"upload{suffix}"
        temporary_path.write_bytes(file_bytes)

        if suffix == ".pdf":
            documents = load_pdf(temporary_path)
        else:
            documents = load_powerpoint(temporary_path)

    for item in documents:
        item["source"] = Path(file_name).name

    if not documents:
        raise ValueError("No readable text was found in the uploaded document.")

    chunks = chunk_documents(
        documents,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )
    if not chunks:
        raise ValueError("No text chunks could be created from this document.")

    embeddings = embed_texts([chunk["text"] for chunk in chunks])
    return build_vector_store(chunks, embeddings)
