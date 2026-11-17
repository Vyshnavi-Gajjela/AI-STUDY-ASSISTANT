import pytest

from src.embeddings import embed_texts
from src.vector_store import build_vector_store
from src.rag import build_rag_prompt, answer_question

def test_answer_question_uses_local_model():
    chunks = [
        {"text": "The capital of France is Paris."},
        {"text": "Plants use sunlight for photosynthesis."},
    ]

    embeddings = embed_texts([chunk["text"] for chunk in chunks])
    index, stored_chunks = build_vector_store(chunks, embeddings)

    answer = answer_question(
        question="What is the capital of France?",
        index=index,
        chunks=stored_chunks,
        top_k=1,
    )

    print("\nGenerated answer:", answer)
    assert answer.strip()


def test_answer_question_does_not_generate_when_search_has_no_support(monkeypatch):
    import src.rag as rag

    monkeypatch.setattr(
        rag,
        "retrieve_chunks",
        lambda **kwargs: [{
            "chunk": {"text": "Unrelated material."},
            "semantic_score": 0.1,
            "keyword_score": 0.0,
        }],
    )
    monkeypatch.setattr(
        rag,
        "generate_answer",
        lambda prompt: pytest.fail("Should not generate an unsupported answer."),
    )

    answer = answer_question(
        question="What is the document's account number?",
        index=object(),
        chunks=[{"text": "Unrelated material."}],
    )

    assert answer == "The document does not contain information about that topic."