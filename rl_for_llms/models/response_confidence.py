import statistics

from pydantic import BaseModel, computed_field


class TokenStep(BaseModel):
    """Data for a single generated token step."""

    token_id: int
    token_text: str
    confidence_logit: float
    confidence_sigmoid: float
    confidence_logprob: float


class SampleResult(BaseModel):
    """Result for a single sample (question + generated answer)."""

    question: str
    full_response: str
    correct_answer: str
    model_answer: str
    is_correct: bool
    is_truncated: bool
    contains_confidence_token: bool
    steps: list[TokenStep]
    last_token_confidence_logprob: float

    @computed_field  # type: ignore[prop-decorator]
    @property
    def mean_confidence_sigmoid(self) -> float:
        """Return the mean confidence sigmoid value across all steps."""
        if not self.steps:
            return 0.0
        return statistics.mean(step.confidence_sigmoid for step in self.steps)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def mean_confidence_logit(self) -> float:
        """Return the mean confidence logit value across all steps."""
        if not self.steps:
            return 0.0
        return statistics.mean(step.confidence_logit for step in self.steps)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def std_confidence_logit(self) -> float:
        """Return the standard deviation of confidence logit values across all steps."""
        if len(self.steps) < 2:  # noqa: PLR2004
            return 0.0
        return statistics.stdev(step.confidence_logit for step in self.steps)


class ResponseConfidenceResult(BaseModel):
    """Aggregated result for all sampled answers."""

    samples: list[SampleResult]
    overall_mean_confidence_logit: float
    overall_std_confidence_logit: float
    overall_mean_confidence_logprob: float
