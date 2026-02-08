import typing
from functools import cache

from math_verify import (
    ExprExtractionConfig,
    LatexExtractionConfig,
    StringExtractionConfig,
    parse,
    verify,
)
from transformers import TrainerState

from rl_for_llms.models.answer import Answer
from rl_for_llms.models.config import Config
from rl_for_llms.utils.latex_utils import (
    get_boxed_expression,
    get_last_boxed_expression,
)
from rl_for_llms.utils.llm_utils import check_model_output_for_completion


@cache
def get_default_extraction_config() -> list[typing.Any]:
    """Return the default extraction configuration for math verification."""
    return [LatexExtractionConfig(), ExprExtractionConfig(), StringExtractionConfig()]


def parse_answer(answer: str) -> list[typing.Any]:
    """Parse the model answer to a standard format."""
    parsed_answer = list(
        parse(
            answer,
            extraction_config=get_default_extraction_config(),
            fallback_mode="first_match",
            extraction_mode="any_match",
            parsing_timeout=60,
            raise_on_error=False,
        )
    )
    return parsed_answer


def verify_answer(
    parsed_correct_answer: list[typing.Any],
    parsed_model_answer: list[typing.Any],
) -> bool:
    """Verify the model answer against the correct answer."""
    verification_result = bool(
        verify(
            parsed_correct_answer,
            parsed_model_answer,
            float_rounding=6,
            numeric_precision=15,
            strict=True,
            allow_set_relation_comp=False,
            timeout_seconds=60,
            raise_on_error=False,
        )
    )
    return verification_result


def get_math_verification_answer(
    correct_answer: str,
    model_answer: str,
    *,
    is_truncated: bool,
    contains_confidence_token: bool,
) -> Answer:
    """Return the math verification answer."""
    boxed_correct_answer = get_boxed_expression(correct_answer)
    parsed_boxed_correct_answer = parse_answer(boxed_correct_answer)
    boxed_model_answer = get_last_boxed_expression(model_answer)
    parsed_boxed_model_answer = parse_answer(boxed_model_answer)
    verification_result = verify_answer(
        parsed_boxed_correct_answer, parsed_boxed_model_answer
    )
    verification_reward = float(verification_result)
    answer = Answer(
        reward=verification_reward,
        correct_answer=boxed_correct_answer,
        model_answer=boxed_model_answer,
        is_correct=verification_result,
        is_truncated=is_truncated,
        contains_confidence_token=contains_confidence_token,
    )
    return answer


def default_reward_function(
    prompt: str,
    completion: str,
    completion_ids: list[int],
    prompt_id: str,
    answer: str,
    trainer_state: TrainerState,  # noqa: ARG001
    config: Config,
) -> Answer:
    """Return a reward based on the math verification of the model answer."""
    is_completed, contains_confidence_token = check_model_output_for_completion(
        completion_ids, prompt, prompt_id, config
    )
    is_truncated = not is_completed
    verification_answer = get_math_verification_answer(
        answer,
        completion,
        is_truncated=is_truncated,
        contains_confidence_token=contains_confidence_token,
    )
    return verification_answer


def default_batch_reward_function(
    prompts: list[typing.Any],
    completions: list[typing.Any],
    **kwargs: typing.Any,  # noqa: ANN401
) -> list[float]:
    """Return a list of reward values for a batch of prompts and completions."""
    batch_size = len(prompts)
    answers = [
        default_reward_function(
            prompt[-1]["content"],
            completion[-1]["content"],
            completion_ids,
            prompt_id,
            answer,
            trainer_state,
            config,
        )
        for prompt, completion, completion_ids, prompt_id, answer, trainer_state, config in zip(
            prompts,
            completions,
            kwargs["completion_ids"],
            kwargs["id"],
            kwargs["answer"],
            [kwargs["trainer_state"]] * batch_size,
            [kwargs["trainer"].config] * batch_size,
            strict=True,
        )
    ]
    kwargs["trainer"].answers = answers
    rewards = [answer.reward for answer in answers]
    return rewards


def get_class_weights_for_single_group(
    rewards: list[float], correct_class_reward_value: float = 1.0
) -> tuple[float, float]:
    """Get class weights for imbalanced rewards within a single group."""
    total_samples = len(rewards)
    correct_samples = len([x for x in rewards if x == correct_class_reward_value])
    incorrect_samples = total_samples - correct_samples
    incorrect_sample_weight = (
        total_samples / (2 * incorrect_samples) if incorrect_samples > 0 else 0.0
    )
    correct_sample_weight = (
        total_samples / (2 * correct_samples) if correct_samples > 0 else 0.0
    )
    return incorrect_sample_weight, correct_sample_weight


def get_class_weights_for_rewards(
    rewards: list[float],
    num_generations: int,
    correct_class_reward_value: float = 1.0,
) -> list[tuple[float, float]]:
    """Get class weights for imbalanced rewards, computed per-group."""
    num_groups = len(rewards) // num_generations
    weights_per_group = []
    for group_idx in range(num_groups):
        start_idx = group_idx * num_generations
        end_idx = start_idx + num_generations
        group_rewards = rewards[start_idx:end_idx]
        weights = get_class_weights_for_single_group(
            group_rewards, correct_class_reward_value
        )
        weights_per_group.append(weights)
    return weights_per_group
