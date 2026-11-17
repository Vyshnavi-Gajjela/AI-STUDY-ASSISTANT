from src.ppt_loader import load_powerpoint
from src.chunking import chunk_documents


def test_pptx_text_can_be_chunked():
    slides = load_powerpoint("data/documents/test_pptx.pptx")
    chunks = chunk_documents(slides, chunk_size=500, overlap=100)

    assert isinstance(chunks, list)
    for chunk in chunks:
        assert chunk["source"] == "test_pptx.pptx"
        assert "slide" in chunk
        assert chunk["text"].strip()