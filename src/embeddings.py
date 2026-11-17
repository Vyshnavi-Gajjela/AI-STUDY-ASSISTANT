from functools import lru_cache

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _embedding_model():
    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: list[str]):
    """Convert a list of text strings into normalized embedding vectors."""
    if not texts:
        return []

    return _embedding_model().encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )