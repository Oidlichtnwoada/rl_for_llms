from functools import cache

import torch
from transformers import Pipeline, pipeline

from rl_for_llms.utils.constant_utils import get_default_hf_model_id
from rl_for_llms.utils.torch_utils import get_device, is_bf16_supported


@cache
def get_pipeline(model_id: str) -> Pipeline:
    """Return the pipline for the given model ID, loading it if not cached."""
    dtype = torch.bfloat16 if is_bf16_supported() else "auto"
    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=dtype,
        device=get_device(),
    )
    return pipe


def get_user_message(content: str) -> dict[str, str]:
    """Return the user message for the given content."""
    return {"role": "user", "content": content.strip()}


def get_assistant_message(content: str) -> dict[str, str]:
    """Return the assistant message for the given content."""
    return {"role": "assistant", "content": content.strip()}


def get_llm_output(
    message: str,
    model_id: str = get_default_hf_model_id(),
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_output_tokens: int = 512,
    *,
    do_sampling: bool = True,
) -> str:
    """Return the LLM output for the given message and model ID."""
    pipe = get_pipeline(model_id)
    messages = [get_user_message(message)]
    outputs = pipe(
        messages,
        max_new_tokens=max_output_tokens,
        do_sample=do_sampling,
        temperature=temperature,
        top_p=top_p,
        return_full_text=False,
    )
    output_message = str(outputs[0]["generated_text"]).strip()
    return output_message
