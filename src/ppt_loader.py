from pathlib import Path

from pptx import Presentation


def load_powerpoint(file_path: str | Path) -> list[dict]:
    """
    Extract text from a PPT or PPTX file, keeping the slide number.

    Returns:
        A list of dictionaries, one per slide:
        [{"source": "file.pptx", "slide": 1, "text": "..."}]
    """
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"PowerPoint file not found: {path}")

    if path.suffix.lower() not in {".ppt", ".pptx"}:
        raise ValueError(f"Expected a PPT or PPTX file, got: {path.suffix}")

    presentation = Presentation(str(path))
    slides = []

    for slide_number, slide in enumerate(presentation.slides, start=1):
        text_parts = []

        for shape in slide.shapes:
            if shape.has_text_frame:
                text = shape.text.strip()
                if text:
                    text_parts.append(text)

            if shape.has_table:
                for row in shape.table.rows:
                    row_text = [
                        cell.text.strip()
                        for cell in row.cells
                        if cell.text.strip()
                    ]
                    if row_text:
                        text_parts.append(" | ".join(row_text))

        slide_text = "\n".join(text_parts).strip()

        if slide_text:
            slides.append({
                "source": path.name,
                "slide": slide_number,
                "text": slide_text,
            })

    return slides