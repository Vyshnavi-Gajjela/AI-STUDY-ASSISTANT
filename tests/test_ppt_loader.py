from pptx import Presentation
import pytest

from src.ppt_loader import load_powerpoint


def create_pptx(path):
    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])

    textbox = slide.shapes.add_textbox(100, 100, 500, 100)
    textbox.text = "This is a test slide."

    presentation.save(path)


def test_load_powerpoint_extracts_slide_text(tmp_path):
    pptx_path = tmp_path / "sample.pptx"
    create_pptx(pptx_path)

    slides = load_powerpoint(pptx_path)

    assert len(slides) == 1
    assert slides[0]["source"] == "sample.pptx"
    assert slides[0]["slide"] == 1
    assert "This is a test slide." in slides[0]["text"]


def test_load_powerpoint_missing_file_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_powerpoint(tmp_path / "missing.pptx")


def test_load_powerpoint_wrong_extension_raises_error(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Not a PowerPoint", encoding="utf-8")

    with pytest.raises(ValueError):
        load_powerpoint(file_path)