import typing

from transformers import TrainerState


def default_reward_function(
    prompt: str,  # noqa: ARG001
    completion: str,
    completion_ids: list[int],  # noqa: ARG001
    prompt_id: str,  # noqa: ARG001
    answer: str,  # noqa: ARG001
    trainer_state: TrainerState,  # noqa: ARG001
) -> float:
    """Return a reward value that rewards completions with more unique letters."""
    return float(len(set(completion)))


def default_batch_reward_function(
    prompts: list[typing.Any],
    completions: list[typing.Any],
    **kwargs: dict[str, typing.Any],
) -> list[float]:
    """Return a list of reward values for a batch of prompts and completions."""
    batch_size = len(prompts)
    rewards = [
        default_reward_function(
            prompt[0]["content"],
            completion[0]["content"],
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
