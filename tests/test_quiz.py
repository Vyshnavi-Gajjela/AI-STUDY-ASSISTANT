import json

import pytest

import src.quiz as quiz


def _quiz_response(count=5):
    question = {
        "question": "Which process is described?",
        "options": {
            "A": "Photosynthesis",
            "B": "Respiration",
            "C": "Fermentation",
            "D": "Transpiration",
        },
        "correct_answer": "A",
        "explanation": "The document says plants use photosynthesis.",
    }
    return json.dumps({"questions": [question] * count})


def test_generate_quiz_validates_and_returns_questions(monkeypatch):
    monkeypatch.setattr(
        quiz,
        "_generate_quiz_response",
        lambda prompt: _quiz_response(),
    )

    questions = quiz.generate_quiz(
        [{"text": "Plants use photosynthesis."}],
        question_count=5,
    )

    assert len(questions) == 5
    assert list(questions[0]["options"]) == ["A", "B", "C", "D"]
    assert questions[0]["correct_answer"] == "A"


@pytest.mark.parametrize("count", [4, 16])
def test_build_quiz_prompt_rejects_out_of_range_question_count(count):
    with pytest.raises(ValueError):
        quiz.build_quiz_prompt([{"text": "Document text"}], count)


def test_generate_quiz_rejects_invalid_model_response(monkeypatch):
    responses = iter(["not JSON", "still not JSON"])
    monkeypatch.setattr(
        quiz,
        "_generate_quiz_response",
        lambda prompt: next(responses),
    )

    with pytest.raises(ValueError, match="after retrying"):
        quiz.generate_quiz([{"text": "Document text"}], question_count=5)


def test_parse_quiz_accepts_json_fence_and_surrounding_text():
    response = f"Generated quiz:\n```JSON\n{_quiz_response()}\n```\nDone."

    questions = quiz._parse_quiz(response, expected_count=5)

    assert len(questions) == 5


def test_generate_quiz_retries_invalid_json(monkeypatch):
    responses = iter(["not JSON", _quiz_response()])
    calls = []

    def fake_generate(prompt):
        calls.append(prompt)
        return next(responses)

    monkeypatch.setattr(quiz, "_generate_quiz_response", fake_generate)

    questions = quiz.generate_quiz(
        [{"text": "Plants use photosynthesis."}],
        question_count=5,
    )

    assert len(questions) == 5
    assert len(calls) == 2
    assert "response to repair" in calls[1]
