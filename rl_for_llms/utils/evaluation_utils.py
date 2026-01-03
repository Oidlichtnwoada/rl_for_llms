import itertools
import json
import random
import statistics
from collections import defaultdict

import pandas as pd
from scipy.stats import skew

from rl_for_llms.models.answer import AnswerWithConfidence
from rl_for_llms.utils.config_utils import get_config
from rl_for_llms.utils.constant_utils import (
    get_default_confidence_score,
    get_default_metric_separator,
)
from rl_for_llms.utils.dataset_utils import load_training_data_from_disk, trim_dataset
from rl_for_llms.utils.llm_utils import (
    get_llm_output_with_step_data,
    get_token_to_id_mapping,
    get_tokenizer,
)
from rl_for_llms.utils.path_utils import get_evaluation_metric_dir


def get_mean_and_std_of_confidence_token_logit(
    sample_size: int = 16,
) -> tuple[float, float, float]:
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
    file_path = get_evaluation_metric_dir() / "confidence_logit_values.json"
    with file_path.open("w") as file:
        file.write(json.dumps(flattened_logit_values))
    mean_value = statistics.mean(flattened_logit_values)
    std_value = statistics.stdev(flattened_logit_values)
    skewness = skew(flattened_logit_values, bias=False)
    return mean_value, std_value, skewness


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


def aggregate_metrics(
    metrics_list: list[dict[tuple[str, ...], float]],
) -> dict[tuple[str, ...], float]:
    """Aggregate metrics by computing mean and standard deviation."""
    aggregated_metrics: dict[tuple[str, ...], float] = {}
    if not metrics_list:
        return aggregated_metrics
    metric_keys = set().union(*metrics_list)
    for key in metric_keys:
        values = [metrics[key] for metrics in metrics_list if key in metrics]
        if len(values) == 0:
            raise ValueError
        aggregated_metrics[(*key, "mean")] = statistics.mean(values)
        stddev = statistics.stdev(values) if len(values) > 1 else 0.0
        aggregated_metrics[(*key, "std")] = stddev
    return aggregated_metrics


def change_metric_keys(
    metrics: dict[tuple[str, ...], float],
    prefix: tuple[str, ...] = (),
    postfix: tuple[str, ...] = (),
) -> dict[tuple[str, ...], float]:
    """Change metric keys by adding prefix and postfix."""
    changed_metrics: dict[tuple[str, ...], float] = {}
    for key, value in metrics.items():
        new_key = prefix + key + postfix
        changed_metrics[new_key] = value
    return changed_metrics


def get_df_from_metrics(
    metrics: dict[tuple[str, ...], float],
    sep: str = get_default_metric_separator(),
) -> pd.DataFrame:
    """Convert metrics dictionary to a pandas DataFrame."""
    data = {sep.join(key): value for key, value in metrics.items()}
    df = pd.DataFrame([data])
    return df


def get_eval_metrics_df_name(
    metric_key_prefix: str, *, is_aggregated: bool = True, is_bc: bool = True
) -> str:
    """Get the evaluation metrics DataFrame name."""
    scope = "agg" if is_aggregated else "concat"
    metric_type = "bc" if is_bc else "answer"
    return f"{scope}_{metric_key_prefix}_{metric_type}_metrics"


def store_eval_df(
    file_name: str,
    df: pd.DataFrame,
) -> None:
    """Store evaluation DataFrame to a CSV file."""
    config = get_config()
    shorthand = config.get_config_shorthand()
    df.to_csv(get_evaluation_metric_dir() / f"{file_name}_{shorthand}.csv", index=False)
