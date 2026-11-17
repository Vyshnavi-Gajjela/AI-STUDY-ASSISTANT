import io

import numpy as np
import pymupdf
from pptx import Presentation

import src.upload_processing as upload_processing


def _pdf_bytes():
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "This uploaded PDF contains test text.")
    return pdf.tobytes()


def _pptx_bytes():
    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    textbox = slide.shapes.add_textbox(100, 100, 500, 100)
    textbox.text = "This uploaded presentation contains test text."
    output = io.BytesIO()
    presentation.save(output)
    return output.getvalue()


def test_process_uploaded_pdf(monkeypatch):
    monkeypatch.setattr(
        upload_processing,
        "embed_texts",
        lambda texts: np.ones((len(texts), 2), dtype="float32"),
    )

    index, chunks = upload_processing.process_uploaded_document(
        "sample.pdf",
        _pdf_bytes(),
    )

    assert index.ntotal == 1
    assert chunks[0]["source"] == "sample.pdf"
    assert "test text" in chunks[0]["text"]


def test_process_uploaded_pptx(monkeypatch):
    monkeypatch.setattr(
        upload_processing,
        "embed_texts",
        lambda texts: np.ones((len(texts), 2), dtype="float32"),
    )

    index, chunks = upload_processing.process_uploaded_document(
        "sample.pptx",
        _pptx_bytes(),
    )

    assert index.ntotal == 1
    assert chunks[0]["source"] == "sample.pptx"
    assert "test text" in chunks[0]["text"]
