import typing


def default_reward_function(
    prompts: list[typing.Any],  # noqa: ARG001
    completions: list[typing.Any],
    **kwargs: dict[str, typing.Any],  # noqa: ARG001
) -> list[float]:
    """Return a reward value that rewards completions with more unique letters."""
    completion_contents = [completion[0]["content"] for completion in completions]
    return [float(len(set(content))) for content in completion_contents]
