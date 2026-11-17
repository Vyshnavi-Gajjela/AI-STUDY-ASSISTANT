from src.embeddings import embed_texts
from src.vector_store import build_vector_store
from src.retriever import retrieve_chunks


def test_retrieve_chunks_returns_relevant_chunk():
    chunks = [
        {"text": "The capital of France is Paris."},
        {"text": "Plants use sunlight for photosynthesis."},
    ]

    embeddings = embed_texts([chunk["text"] for chunk in chunks])
    index, stored_chunks = build_vector_store(chunks, embeddings)

    results = retrieve_chunks(
        question="What is the capital of France?",
        index=index,
        chunks=stored_chunks,
        top_k=1,
    )

    assert len(results) == 1
    assert results[0]["chunk"]["text"] == "The capital of France is Paris."