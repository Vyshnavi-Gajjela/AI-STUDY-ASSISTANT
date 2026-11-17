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
    monkeypatch.setattr(quiz, "generate_answer", lambda prompt: _quiz_response())

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
    monkeypatch.setattr(quiz, "generate_answer", lambda prompt: "not JSON")

    with pytest.raises(ValueError, match="invalid quiz JSON"):
        quiz.generate_quiz([{"text": "Document text"}], question_count=5)
