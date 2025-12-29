from pydantic import BaseModel


class Answer(BaseModel):
    """Answer model to store reward and answers."""

    reward: float
    correct_answer: str
    model_answer: str
    is_correct: bool
    is_truncated: bool


class AnswerWithConfidence(BaseModel):
    """Answer model with a confidence score."""

    answer: Answer
    confidence: float


def get_answers_with_confidence(
    answers: list[Answer],
    confidences: list[float],
) -> list[AnswerWithConfidence]:
    """Combine answers with their corresponding confidence scores."""
    answers_with_confidence = [
        AnswerWithConfidence(answer=answer, confidence=confidence)
        for answer, confidence in zip(answers, confidences, strict=True)
    ]
    return answers_with_confidence
