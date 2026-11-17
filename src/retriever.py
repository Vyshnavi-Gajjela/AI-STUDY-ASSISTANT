import math
import re
from collections import Counter

from src.config import RETRIEVAL_CANDIDATES
from src.embeddings import embed_texts
from src.vector_store import search_vector_store


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "how", "i", "in", "into", "is", "it", "me", "of", "on",
    "or", "please", "tell", "that", "the", "their", "this", "to", "was",
    "what", "when", "where", "which", "who", "why", "with", "would",
}


def _tokenize(text: str) -> list[str]:
    """Lowercase text and split it into word tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _exact_scores(query: str, chunks: list[dict]) -> list[float]:
    """Score literal phrase and term matches separately from semantic search."""
    query_tokens = [token for token in _tokenize(query) if token not in STOP_WORDS]
    if not query_tokens:
        return [0.0] * len(chunks)

    normalized_query = " ".join(query_tokens)
    scores = []
    for chunk in chunks:
        chunk_tokens = _tokenize(chunk.get("text", ""))
        token_set = set(chunk_tokens)
        coverage = sum(token in token_set for token in query_tokens) / len(query_tokens)
        normalized_text = " ".join(chunk_tokens)
        phrase_match = normalized_query in normalized_text
        scores.append(coverage + (1.0 if phrase_match else 0.0))
    return scores


def _bm25_scores(query: str, chunks: list[dict]) -> list[float]:
    """Calculate BM25 keyword relevance scores for all chunks."""
    query_tokens = [
        token for token in _tokenize(query)
        if token not in STOP_WORDS
    ]

    if not query_tokens or not chunks:
        return [0.0] * len(chunks)

    tokenized_chunks = [
        _tokenize(chunk.get("text", ""))
        for chunk in chunks
    ]

    document_count = len(tokenized_chunks)
    lengths = [len(tokens) for tokens in tokenized_chunks]
    average_length = sum(lengths) / max(document_count, 1)

    document_frequency = Counter()
    for tokens in tokenized_chunks:
        document_frequency.update(set(tokens))

    k1 = 1.5
    b = 0.75
    scores = []

    for tokens, length in zip(tokenized_chunks, lengths):
        term_counts = Counter(tokens)
        score = 0.0

        for term in set(query_tokens):
            frequency = term_counts.get(term, 0)
            if frequency == 0:
                continue

            df = document_frequency[term]
            idf = math.log(
                1 + (document_count - df + 0.5) / (df + 0.5)
            )

            denominator = frequency + k1 * (
                1 - b + b * length / max(average_length, 1)
            )

            score += idf * frequency * (k1 + 1) / denominator

        scores.append(score)

    return scores


def retrieve_chunks(
    question: str,
    index,
    chunks: list[dict],
    top_k: int = 10,
    candidate_k: int = RETRIEVAL_CANDIDATES,
) -> list[dict]:
    """
    Retrieve and rerank chunks using semantic search, BM25, and RRF.

    Each result includes:
      chunk, score, semantic_rank, keyword_rank
    """
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if candidate_k < 1:
        raise ValueError("candidate_k must be at least 1.")

    if not chunks:
        return []

    if index.ntotal != len(chunks):
        raise ValueError(
            "The index and chunks must contain the same number of items."
        )

    candidate_k = min(candidate_k, len(chunks))

    # Semantic retrieval.
    query_embedding = embed_texts([question])
    semantic_results = search_vector_store(
        query_embedding=query_embedding,
        index=index,
        chunks=chunks,
        top_k=candidate_k,
    )

    # Use list positions as stable IDs for this retrieval call.
    semantic_rank = {
        id(result["chunk"]): rank
        for rank, result in enumerate(semantic_results, start=1)
    }
    semantic_scores = {
        id(result["chunk"]): result["score"]
        for result in semantic_results
    }

    # Keyword retrieval.
    keyword_scores = _bm25_scores(question, chunks)
    keyword_order = sorted(
        range(len(chunks)),
        key=lambda i: keyword_scores[i],
        reverse=True,
    )

    keyword_rank = {
        i: rank
        for rank, i in enumerate(keyword_order, start=1)
        if keyword_scores[i] > 0
    }

    # Literal phrase/term search complements semantic and BM25 retrieval.
    exact_scores = _exact_scores(question, chunks)
    exact_order = sorted(
        range(len(chunks)),
        key=lambda i: exact_scores[i],
        reverse=True,
    )
    exact_rank = {
        i: rank
        for rank, i in enumerate(exact_order, start=1)
        if exact_scores[i] > 0
    }

    # Reciprocal Rank Fusion combines rankings with different score scales.
    rrf_constant = 60
    results = []

    for position, chunk in enumerate(chunks):
        sem_rank = semantic_rank.get(id(chunk))
        key_rank = keyword_rank.get(position)
        literal_rank = exact_rank.get(position)

        fused_score = 0.0
        if sem_rank is not None:
            fused_score += 1 / (rrf_constant + sem_rank)
        if key_rank is not None:
            fused_score += 1 / (rrf_constant + key_rank)
        if literal_rank is not None:
            fused_score += 1 / (rrf_constant + literal_rank)

        results.append({
            "chunk": chunk,
            "score": fused_score,
            "semantic_rank": sem_rank,
            "keyword_rank": key_rank,
            "semantic_score": semantic_scores.get(id(chunk), 0.0),
            "keyword_score": keyword_scores[position],
            "exact_score": exact_scores[position],
        })

    results.sort(key=lambda item: item["score"], reverse=True)

    return results[:min(top_k, len(results))]