import itertools
import random
import statistics
from collections import defaultdict

from rl_for_llms.models.answer import AnswerWithConfidence
from rl_for_llms.utils.config_utils import get_config
from rl_for_llms.utils.constant_utils import get_default_confidence_score
from rl_for_llms.utils.dataset_utils import load_training_data_from_disk, trim_dataset
from rl_for_llms.utils.llm_utils import (
    get_llm_output_with_step_data,
    get_token_to_id_mapping,
    get_tokenizer,
)


def get_mean_and_std_of_confidence_token_logit(
    sample_size: int = 16,
) -> tuple[float, float]:
    """Return the mean and standard deviation of the confidence token logit."""
    config = get_config()
    confidence_token_id = get_token_to_id_mapping(config.hf_model_id)[
        config.confidence_token
    ]
    tokenizer = get_tokenizer(config.hf_model_id)
    dataset = trim_dataset(
        load_training_data_from_disk(),
        config.dataset_use_row_percentage,
        tokenizer,
        config.max_prompt_length,
    )
    messages = [
        row["prompt"][-1]["content"] for row in dataset.select(range(sample_size))
    ]
    step_data = [
        get_llm_output_with_step_data(
            message, (confidence_token_id,), config.hf_model_id
        )[1]
        for message in messages
    ]
    logit_values = [
        [x[confidence_token_id]["logit"] for x in message_logits]
        for message_logits in step_data
    ]
    flattened_logit_values = list(itertools.chain.from_iterable(logit_values))
    mean_value = statistics.mean(flattened_logit_values)
    std_value = statistics.stdev(flattened_logit_values)
    return mean_value, std_value


def pick_best_answer(
    weights: dict[str, float],
    answers_with_confidence: list[AnswerWithConfidence],
) -> AnswerWithConfidence:
    """Pick the best answer based on the given weights."""
    max_weight = max(weights.values())
    top_answers = [
        a
        for a in answers_with_confidence
        if weights[a.answer.model_answer] == max_weight
    ]
    return random.choice(top_answers)  # noqa: S311


def compute_answer_metrics(
    answers_with_confidence: list[AnswerWithConfidence],
    temperature: float,
) -> dict[tuple[str, ...], float]:
    """Compute binary classification metrics."""
    metrics: dict[tuple[str, ...], float] = {}
    sample_amount = len(answers_with_confidence)
    if sample_amount == 0:
        return metrics
    truncation_percentage = statistics.mean(
        [float(x.answer.is_truncated) for x in answers_with_confidence]
    )
    metrics[(f"truncation_percentage_t={temperature}",)] = truncation_percentage
    flags_for_correctness = [
        float(x.answer.is_correct) for x in answers_with_confidence
    ]
    pass_at_1_accuracy = statistics.mean(flags_for_correctness)
    metrics[
        (
            "accuracy",
            f"pass@1_t={temperature}",
        )
    ] = pass_at_1_accuracy
    pass_at_k_accuracy = max(flags_for_correctness)
    metrics[
        (
            "accuracy",
            f"pass@{sample_amount}_t={temperature}",
        )
    ] = pass_at_k_accuracy
    answer_weights: dict[str, float] = defaultdict(float)
    weighted_answer_weights: dict[str, float] = defaultdict(float)
    for answer_with_confidence in answers_with_confidence:
        answer_weights[answer_with_confidence.answer.model_answer] += (
            get_default_confidence_score()
        )
        weighted_answer_weights[answer_with_confidence.answer.model_answer] += (
            answer_with_confidence.confidence
        )
    metrics[
        (
            "accuracy",
            f"majority_voting_t={temperature}",
        )
    ] = float(
        pick_best_answer(answer_weights, answers_with_confidence).answer.is_correct
    )
    metrics[
        (
            "accuracy",
            f"confidence_weighted_majority_voting_t={temperature}",
        )
    ] = float(
        pick_best_answer(
            weighted_answer_weights, answers_with_confidence
        ).answer.is_correct
    )
    max_confidence = max(x.confidence for x in answers_with_confidence)
    max_confidence_answers = [
        x for x in answers_with_confidence if x.confidence == max_confidence
    ]
    metrics[
        (
            "accuracy",
            f"highest_confidence_t={temperature}",
        )
    ] = float(random.choice(max_confidence_answers).answer.is_correct)  # noqa: S311
    return metrics
