import numpy as np
import pytest

from src.vector_store import build_vector_store, search_vector_store
def test_search_vector_store_returns_closest_chunk():
    chunks = [
        {"text": "First chunk"},
        {"text": "Second chunk"},
    ]
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype="float32",
    )

    index, stored_chunks = build_vector_store(chunks, embeddings)

    results = search_vector_store(
        query_embedding=np.array([0.9, 0.1], dtype="float32"),
        index=index,
        chunks=stored_chunks,
        top_k=1,
    )

    assert len(results) == 1
    assert results[0]["chunk"]["text"] == "First chunk"
    assert results[0]["score"] > 0


def test_search_vector_store_rejects_wrong_dimension():
    chunks = [{"text": "First chunk"}]
    embeddings = np.array([[1.0, 0.0]], dtype="float32")
    index, stored_chunks = build_vector_store(chunks, embeddings)

    with pytest.raises(ValueError):
        search_vector_store(
            query_embedding=np.array([1.0, 0.0, 0.0], dtype="float32"),
            index=index,
            chunks=stored_chunks,
        )