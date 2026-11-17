import faiss
import numpy as np


def build_vector_store(chunks: list[dict], embeddings):
    """Build a FAISS index and keep its matching text chunks."""
    if len(chunks) != len(embeddings):
        raise ValueError("The number of chunks must match the number of embeddings.")

    if len(chunks) == 0:
        raise ValueError("Cannot build a vector store with no chunks.")

    vectors = np.asarray(embeddings, dtype="float32")

    if vectors.ndim != 2:
        raise ValueError("Embeddings must be a 2D array.")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    return index, chunks


def search_vector_store(query_embedding, index, chunks: list[dict], top_k: int = 5):
    """Return the most similar chunks and their similarity scores."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if index.ntotal != len(chunks):
        raise ValueError("The index and chunks must contain the same number of items.")

    if index.ntotal == 0:
        return []

    query_vector = np.asarray(query_embedding, dtype="float32")

    # FAISS expects a 2D array: one row for the query.
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)

    if query_vector.ndim != 2 or query_vector.shape[0] != 1:
        raise ValueError("Provide one query embedding.")

    if query_vector.shape[1] != index.d:
        raise ValueError("The query embedding dimension does not match the index.")

    scores, indices = index.search(query_vector, min(top_k, index.ntotal))

    results = []
    for score, chunk_index in zip(scores[0], indices[0]):
        if chunk_index == -1:
            continue

        results.append({
            "chunk": chunks[int(chunk_index)],
            "score": float(score),
        })

    return results