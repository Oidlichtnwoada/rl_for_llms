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


def get_math_verification_reward(
    correct_answer: str,
    model_answer: str,
) -> float:
    """Return a reward based on the math verification of the model answer."""
    boxed_correct_answer = get_boxed_expression(correct_answer)
    parsed_boxed_correct_answer = parse_answer(boxed_correct_answer)
    boxed_model_answer = get_last_boxed_expression(model_answer)
    parsed_boxed_model_answer = parse_answer(boxed_model_answer)
    verification_result = verify_answer(
        parsed_boxed_correct_answer, parsed_boxed_model_answer
    )
    verification_reward = float(verification_result)
    return verification_reward


def default_reward_function(
    prompt: str,
    completion: str,
    completion_ids: list[int],
    prompt_id: str,
    answer: str,
    trainer_state: TrainerState,  # noqa: ARG001
) -> float:
    """Return a reward value that rewards completions with more unique letters."""
    check_model_output_for_completion(completion_ids, prompt, prompt_id)
    reward = get_math_verification_reward(answer, completion)
    return reward


def default_batch_reward_function(
    prompts: list[typing.Any],
    completions: list[typing.Any],
    **kwargs: dict[str, typing.Any],
) -> list[float]:
    """Return a list of reward values for a batch of prompts and completions."""
    batch_size = len(prompts)
    rewards = [
        default_reward_function(
            prompt[-1]["content"],
            completion[-1]["content"],
            completion_ids,
            prompt_id,
            answer,
            trainer_state,
        )
        for prompt, completion, completion_ids, prompt_id, answer, trainer_state in zip(
            prompts,
            completions,
            kwargs["completion_ids"],
            kwargs["id"],
            kwargs["answer"],
            [kwargs["trainer_state"]] * batch_size,
            strict=True,
        )
    ]
    return rewards
