from pydantic import BaseModel


class Answer(BaseModel):
    """Configuration for the program."""

    reward: float
    correct_answer: str
    model_answer: str
