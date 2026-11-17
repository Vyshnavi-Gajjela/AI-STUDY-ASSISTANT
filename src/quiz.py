import json
import re

from src.llm import generate_answer


QUIZ_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "questions": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "question": {"type": "STRING"},
                    "options": {
                        "type": "OBJECT",
                        "properties": {
                            "A": {"type": "STRING"},
                            "B": {"type": "STRING"},
                            "C": {"type": "STRING"},
                            "D": {"type": "STRING"},
                        },
                        "required": ["A", "B", "C", "D"],
                    },
                    "correct_answer": {
                        "type": "STRING",
                        "enum": ["A", "B", "C", "D"],
                    },
                    "explanation": {"type": "STRING"},
                },
                "required": [
                    "question",
                    "options",
                    "correct_answer",
                    "explanation",
                ],
            },
        },
    },
    "required": ["questions"],
}


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
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        fence = re.search(
            r"```(?:json)?\s*(.*?)\s*```",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if fence:
            cleaned = fence.group(1).strip()
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                parsed = None
        else:
            parsed = None

        if parsed is None:
            decoder = json.JSONDecoder()
            parsed = None
            for position, character in enumerate(cleaned):
                if character != "{":
                    continue
                try:
                    parsed, _ = decoder.raw_decode(cleaned[position:])
                    break
                except json.JSONDecodeError:
                    continue
        if parsed is None:
            raise ValueError(
                "The AI returned quiz text in an unreadable format. Please retry."
            )

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


def _generate_quiz_response(prompt: str) -> str:
    return generate_answer(prompt, response_schema=QUIZ_RESPONSE_SCHEMA)


def generate_quiz(chunks: list[dict], question_count: int = 5) -> list[dict]:
    prompt = build_quiz_prompt(chunks, question_count)
    response = _generate_quiz_response(prompt)

    try:
        return _parse_quiz(response, question_count)
    except ValueError as first_error:
        repair_prompt = f"""Convert the quiz response below into valid JSON matching this exact structure:
{{"questions":[{{"question":"...","options":{{"A":"...","B":"...","C":"...","D":"..."}},"correct_answer":"A","explanation":"..."}}]}}

Return exactly {question_count} questions, with four options and a correct_answer key of A, B, C, or D. Preserve the document-supported facts and do not add outside information.

Quiz response to repair:
{response}
"""
        repaired = _generate_quiz_response(repair_prompt)
        try:
            return _parse_quiz(repaired, question_count)
        except ValueError as repair_error:
            raise ValueError(
                "The AI could not produce a valid quiz after retrying. Please "
                "generate the quiz again."
            ) from repair_error
