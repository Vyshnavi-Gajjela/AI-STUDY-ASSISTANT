def build_summary_prompt(text: str) -> str:
    """Create a prompt asking the local model to summarize study material."""
    return f"""Summarize the study material below.

Use clear, simple language.
Keep the important ideas, facts, and terms.
Do not add information that is not in the material.

Study material:
{text}

Summary:"""


def build_question_prompt(question: str, context: str) -> str:
    """Create a prompt that answers a question using only supplied context."""
    return f"""Answer the question using only the study-material context below.
If the context does not contain the answer, say that the materials do not provide enough information.

Study-material context:
{context}

Question:
{question}

Answer:"""