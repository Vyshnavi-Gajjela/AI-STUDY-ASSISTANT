from src.llm import generate_answer
from src.prompts import build_summary_prompt


SUMMARY_BATCH_CHARACTERS = 8000


def summarize_text(text: str) -> str:
    """Summarize study material using the configured local language model."""
    if not text or not text.strip():
        raise ValueError("Text to summarize cannot be empty.")

    prompt = build_summary_prompt(text)
    return generate_answer(prompt)


def summarize_chunks(chunks: list[dict]) -> str:
    """Summarize every chunk hierarchically to keep each model request bounded."""
    texts = [chunk.get("text", "").strip() for chunk in chunks]
    texts = [text for text in texts if text]
    if not texts:
        raise ValueError("Text to summarize cannot be empty.")

    batches = []
    current = []
    current_length = 0
    for text in texts:
        if current and current_length + len(text) > SUMMARY_BATCH_CHARACTERS:
            batches.append("\n\n".join(current))
            current = []
            current_length = 0
        current.append(text)
        current_length += len(text)
    if current:
        batches.append("\n\n".join(current))

    summaries = [summarize_text(batch) for batch in batches]
    while len(summaries) > 1:
        batches = []
        current = []
        current_length = 0
        for summary in summaries:
            if current and current_length + len(summary) > SUMMARY_BATCH_CHARACTERS:
                batches.append("\n\n".join(current))
                current = []
                current_length = 0
            current.append(summary)
            current_length += len(summary)
        if current:
            batches.append("\n\n".join(current))
        if len(batches) == len(summaries):
            batches = [
                "\n\n".join(summaries[:2]),
                *summaries[2:],
            ]
        summaries = [summarize_text(batch) for batch in batches]
    return summaries[0]