from src.embeddings import embed_texts


def test_embed_texts_returns_one_vector_per_text():
    vectors = embed_texts(["A test sentence.", "Another sentence."])

    assert vectors.shape == (2, 384)