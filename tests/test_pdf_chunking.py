from src.chunking import chunk_documents


def test_chunk_documents_splits_text_and_preserves_source():
    document = {
        "source": "test_pdf.pdf",
        "page": 1,
        "text": " ".join(f"word{i}" for i in range(1, 1001)),
    }

    chunks = chunk_documents([document], chunk_size=500, overlap=100)

    assert len(chunks) == 3
    assert len(chunks[0]["text"].split()) == 500
    assert len(chunks[1]["text"].split()) == 500
    assert len(chunks[2]["text"].split()) == 200

    assert chunks[0]["text"].split()[-100:] == chunks[1]["text"].split()[:100]
    assert chunks[0]["source"] == "test_pdf.pdf"
    assert chunks[0]["page"] == 1


def test_chunk_documents_skips_empty_text():
    chunks = chunk_documents([{"source": "empty.pdf", "page": 1, "text": "  "}])

    assert chunks == []