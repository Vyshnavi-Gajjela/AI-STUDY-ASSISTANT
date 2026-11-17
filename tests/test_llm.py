import json
from urllib.error import URLError

import pytest

import src.llm as llm


class FakeResponse:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return json.dumps(self.value).encode("utf-8")


def test_generate_answer_returns_ollama_text(monkeypatch):
    monkeypatch.setattr(
        llm,
        "urlopen",
        lambda request, timeout: FakeResponse({"response": "Document answer"}),
    )

    assert llm.generate_answer("Question") == "Document answer"


def test_generate_answer_reports_unreachable_hosted_endpoint(monkeypatch):
    monkeypatch.setattr(
        llm,
        "urlopen",
        lambda request, timeout: (_ for _ in ()).throw(URLError("offline")),
    )

    with pytest.raises(RuntimeError, match="For a hosted app"):
        llm.generate_answer("Question")


def test_generate_answer_reports_timeout(monkeypatch):
    monkeypatch.setattr(
        llm,
        "urlopen",
        lambda request, timeout: (_ for _ in ()).throw(TimeoutError()),
    )

    with pytest.raises(RuntimeError, match="too long to respond"):
        llm.generate_answer("Question")


def test_generate_answer_rejects_empty_response(monkeypatch):
    monkeypatch.setattr(
        llm,
        "urlopen",
        lambda request, timeout: FakeResponse({"response": "  "}),
    )

    with pytest.raises(RuntimeError, match="empty answer"):
        llm.generate_answer("Question")


def test_generate_answer_uses_gemini_provider(monkeypatch):
    captured = {}
    monkeypatch.setattr(llm, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(llm, "GEMINI_API_KEY", "test-key")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        return FakeResponse({
            "candidates": [{
                "content": {"parts": [{"text": "Gemini answer"}]},
            }],
        })

    monkeypatch.setattr(llm, "urlopen", fake_urlopen)

    assert llm.generate_answer("Question") == "Gemini answer"
    assert "test-key" in captured["url"]
    assert captured["timeout"] == 180


def test_generate_answer_requires_gemini_api_key(monkeypatch):
    monkeypatch.setattr(llm, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(llm, "GEMINI_API_KEY", "")

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        llm.generate_answer("Question")


def test_generate_answer_rejects_unknown_provider(monkeypatch):
    monkeypatch.setattr(llm, "LLM_PROVIDER", "unknown")

    with pytest.raises(RuntimeError, match="Unsupported LLM_PROVIDER"):
        llm.generate_answer("Question")
