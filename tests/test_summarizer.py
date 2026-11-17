import pytest

import src.summarizer as summarizer


def test_summarize_text_uses_local_model(monkeypatch):
    captured = {}

    def fake_generate_answer(prompt):
        captured["prompt"] = prompt
        return "A short summary."

    monkeypatch.setattr(summarizer, "generate_answer", fake_generate_answer)

    result = summarizer.summarize_text("Photosynthesis helps plants make food.")

    assert result == "A short summary."
    assert "Photosynthesis" in captured["prompt"]
    assert "Summary:" in captured["prompt"]


def test_summarize_text_rejects_empty_text():
    with pytest.raises(ValueError):
        summarizer.summarize_text("   ")


def test_summarize_chunks_summarizes_all_batches(monkeypatch):
    calls = []

    def fake_generate_answer(prompt):
        calls.append(prompt)
        return "Brief summary."

    monkeypatch.setattr(summarizer, "generate_answer", fake_generate_answer)
    monkeypatch.setattr(summarizer, "SUMMARY_BATCH_CHARACTERS", 5)

    result = summarizer.summarize_chunks([
        {"text": "first passage"},
        {"text": "second passage"},
    ])

    assert result == "Brief summary."
    assert len(calls) == 3