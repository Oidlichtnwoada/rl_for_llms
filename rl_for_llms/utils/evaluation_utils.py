import itertools
import statistics

from rl_for_llms.models.answer import AnswerWithConfidence
from rl_for_llms.utils.config_utils import get_config
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


def compute_answer_metrics(
    answers_with_confidence: list[AnswerWithConfidence],  # noqa: ARG001
) -> dict[tuple[str, ...], float]:
    """Compute binary classification metrics."""
    return {}
