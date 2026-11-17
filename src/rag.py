import re

from src.retriever import retrieve_chunks
from src.llm import generate_answer


OUT_OF_SCOPE_ANSWER = "The document does not contain information about that topic."
MIN_SEMANTIC_SCORE = 0.25


def has_document_support(results: list[dict]) -> bool:
    """Avoid generating an answer when neither search method found support."""
    return any(
        result["semantic_score"] >= MIN_SEMANTIC_SCORE
        or result["keyword_score"] > 0
        for result in results
    )


def _format_location(chunk: dict) -> str:
    source = chunk.get("source", "unknown source")
    locations = []

    if "page" in chunk:
        locations.append(f"page {chunk['page']}")

    if "slide" in chunk:
        locations.append(f"slide {chunk['slide']}")

    if locations:
        return f"{source}, {', '.join(locations)}"

    return source


def _requested_section(question: str) -> str | None:
    """
    Detect a section name mentioned in a question, for example:
    'Summarize the conclusion section' -> 'conclusion'.
    """
    match = re.search(
        r"\b(?:the\s+)?(.{1,80}?)\s+section\b",
        question.lower(),
    )

    if not match:
        return None

    section = match.group(1).strip(" .?!,:;\"'")
    section = re.sub(r"^(?:summarize|summary of|explain|describe)\s+", "", section)
    section = re.sub(r"^(?:the\s+)", "", section)

    return section or None


def _matches_section(chunk: dict, section: str) -> bool:
    """Check whether a chunk contains a section heading or its phrase."""
    text = chunk.get("text", "").lower()

    # Match common heading forms, such as "V. CONCLUSION".
    if section in text:
        return True

    return False


def build_rag_prompt(
    question: str,
    index,
    chunks: list[dict],
    top_k: int = 10,
    retrieved_results: list[dict] | None = None,
) -> str:
    results = retrieved_results
    if results is None:
        results = retrieve_chunks(
            question=question,
            index=index,
            chunks=chunks,
            top_k=top_k,
        )

    section = _requested_section(question)

    # If the user names a section, use matching section chunks when found.
    # Otherwise, retain the normal hybrid-retrieval results.
    if section:
        section_results = [
            result
            for result in results
            if _matches_section(result["chunk"], section)
        ]

        if section_results:
            results = section_results

    if not results:
        context = "No relevant passages were retrieved."
    else:
        context_parts = []

        for number, result in enumerate(results, start=1):
            chunk = result["chunk"]
            location = _format_location(chunk)
            text = chunk.get("text", "").strip()

            context_parts.append(
                f"[Passage {number} | Source: {location}]\n{text}"
            )

        context = "\n\n".join(context_parts)

    section_instruction = ""
    if section:
        section_instruction = (
            f"- Focus on the section named '{section}'. "
            "Do not summarize unrelated sections.\n"
        )

    return f"""You are an AI study assistant.

Answer the user's question using only the study-material passages below.

Instructions:
- Answer directly and clearly.
- Include the important details supported by the passages.
- Do not invent facts, names, findings, or conclusions.
- If the passages do not contain enough information, say so.
- Do not add unrelated information.
{section_instruction}
Study-material passages:
{context}

User question:
{question}

Answer:"""


def answer_question(
    question: str,
    index,
    chunks: list[dict],
    top_k: int = 10,
    retrieved_results: list[dict] | None = None,
) -> str:
    results = retrieved_results
    if results is None:
        results = retrieve_chunks(
            question=question,
            index=index,
            chunks=chunks,
            top_k=top_k,
        )
    if not has_document_support(results):
        return OUT_OF_SCOPE_ANSWER

    prompt = build_rag_prompt(
        question=question,
        index=index,
        chunks=chunks,
        top_k=top_k,
        retrieved_results=results,
    )

    return generate_answer(prompt)