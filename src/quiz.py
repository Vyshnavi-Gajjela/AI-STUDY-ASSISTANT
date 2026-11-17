import json
import re

from src.llm import generate_answer


def build_quiz_prompt(chunks: list[dict], question_count: int) -> str:
    if not 5 <= question_count <= 15:
        raise ValueError("The quiz must contain between 5 and 15 questions.")
    if not chunks:
        raise ValueError("Cannot create a quiz from an empty document.")

    source_text = "\n\n".join(
        chunk.get("text", "").strip()
        for chunk in chunks
        if chunk.get("text", "").strip()
    )
    if not source_text:
        raise ValueError("Cannot create a quiz from an empty document.")

    return f"""Create exactly {question_count} multiple-choice questions using only the document text below.
Do not use outside knowledge. Every correct answer and explanation must be supported by this text.
Each question must have exactly four plausible options labeled A, B, C, and D, with exactly one correct answer.
Avoid ambiguous questions and do not repeat questions.
Return only valid JSON with this shape:
{{"questions":[{{"question":"...","options":{{"A":"...","B":"...","C":"...","D":"..."}},"correct_answer":"A","explanation":"..."}}]}}

Document text:
{source_text}
"""


def _parse_quiz(response: str, expected_count: int) -> list[dict]:
    cleaned = response.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL)
    if fence:
        cleaned = fence.group(1)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as error:
        raise ValueError("The model returned invalid quiz JSON. Please try again.") from error

    questions = parsed.get("questions") if isinstance(parsed, dict) else None
    if not isinstance(questions, list) or len(questions) != expected_count:
        raise ValueError(
            f"The model must return exactly {expected_count} quiz questions."
        )

    validated = []
    for question in questions:
        if not isinstance(question, dict):
            raise ValueError("The model returned a malformed quiz question.")
        prompt = question.get("question")
        options = question.get("options")
        answer = question.get("correct_answer")
        explanation = question.get("explanation")
        if (
            not isinstance(prompt, str)
            or not prompt.strip()
            or not isinstance(options, dict)
            or set(options) != {"A", "B", "C", "D"}
            or not all(isinstance(value, str) and value.strip() for value in options.values())
            or len({value.strip().casefold() for value in options.values()}) != 4
            or not isinstance(answer, str)
            or answer not in options
            or not isinstance(explanation, str)
            or not explanation.strip()
        ):
            raise ValueError("The model returned a malformed quiz question.")
        validated.append({
            "question": prompt.strip(),
            "options": {key: options[key].strip() for key in ("A", "B", "C", "D")},
            "correct_answer": answer,
            "explanation": explanation.strip(),
        })
    return validated


def generate_quiz(chunks: list[dict], question_count: int = 5) -> list[dict]:
    prompt = build_quiz_prompt(chunks, question_count)
    response = generate_answer(prompt)
    return _parse_quiz(response, question_count)
