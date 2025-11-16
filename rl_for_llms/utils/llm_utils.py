import typing
from functools import cache

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Pipeline,
    PreTrainedTokenizerBase,
    pipeline,
)

from rl_for_llms.utils.constant_utils import get_default_hf_model_id
from rl_for_llms.utils.torch_utils import get_device, is_bf16_supported


@cache
def get_pipeline(model_id: str) -> Pipeline:
    """Return the pipline for the given model ID, loading it if not cached."""
    dtype = torch.bfloat16 if is_bf16_supported() else "auto"
    pipe = pipeline(
        "text-generation",
        model=model_id,
        dtype=dtype,
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
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_output_tokens: int = 512,
    top_k: int | None = None,
    *,
    do_sampling: bool = False,
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
        top_k=top_k,
        return_full_text=False,
    )
    output_message = str(outputs[0]["generated_text"]).strip()
    return output_message


def get_model_representation(model_id: str) -> str:
    """Return a string representation of the model for the given model ID."""
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model_representation = str(model)
    return model_representation


def tokenize_text(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
) -> list[int]:
    """Return the tokenized representation of the given text."""
    token_list = list(map(int, tokenizer.encode(text)))
    return token_list


@cache
def get_tokenizer(
    model_id: str, truncation_side: str = "left"
) -> PreTrainedTokenizerBase:
    """Return the tokenizer for the specified model ID."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, truncation_side=truncation_side, trust_remote_code=True
    )  # type: ignore[no-untyped-call]
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return typing.cast("PreTrainedTokenizerBase", tokenizer)
