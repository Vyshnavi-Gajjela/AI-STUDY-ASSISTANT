import pymupdf
import pytest

from src.document_loader import load_pdf


def create_pdf(path):
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "This is a test PDF.")
    document.save(path)
    document.close()


def test_load_pdf_extracts_page_text(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    create_pdf(pdf_path)

    pages = load_pdf(pdf_path)

    assert len(pages) == 1
    assert pages[0]["source"] == "sample.pdf"
    assert pages[0]["page"] == 1
    assert "This is a test PDF." in pages[0]["text"]


def test_load_pdf_missing_file_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pdf(tmp_path / "missing.pdf")


def test_load_pdf_wrong_extension_raises_error(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Not a PDF", encoding="utf-8")

    with pytest.raises(ValueError):
        load_pdf(file_path)