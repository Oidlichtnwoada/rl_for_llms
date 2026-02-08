import logging
import typing
from functools import cache

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    Pipeline,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    pipeline,
)

from rl_for_llms.models.config import Config
from rl_for_llms.utils.logging_utils import log_msg
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


def get_system_message(content: str) -> dict[str, str]:
    """Return the system message for the given content."""
    return {"role": "system", "content": content.strip()}


def get_user_message(content: str) -> dict[str, str]:
    """Return the user message for the given content."""
    return {"role": "user", "content": content.strip()}


def get_assistant_message(content: str) -> dict[str, str]:
    """Return the assistant message for the given content."""
    return {"role": "assistant", "content": content.strip()}


def get_llm_output_with_step_data(
    message: str,
    model_id: str,
    target_token_ids: tuple[int, ...] = (),
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_output_tokens: int = 8192,
    top_k: int | None = None,
    *,
    do_sampling: bool = False,
) -> tuple[str, list[dict[int, dict[str, float]]]]:
    """Return the LLM output along with step data for the specified target token IDs."""
    pipe = get_pipeline(model_id)
    model = pipe.model
    tokenizer = typing.cast("PreTrainedTokenizer", pipe.tokenizer)
    messages = [get_user_message(message)]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(  # type: ignore[operator]
        **inputs,
        max_new_tokens=max_output_tokens,
        do_sample=do_sampling,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        output_logits=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated_ids = outputs.sequences[0][len(inputs["input_ids"][0]) :]
    output_message = str(
        tokenizer.decode(generated_ids, skip_special_tokens=True)
    ).strip()
    step_data = []
    for step_tensor in outputs.logits:
        step_probs = torch.softmax(step_tensor / temperature, dim=-1)
        step_vals = {}
        for tid in target_token_ids:
            logit_val = step_tensor[0, tid].item()
            prob_val = step_probs[0, tid].item()
            step_vals[tid] = {"logit": logit_val, "prob": prob_val}
        step_data.append(step_vals)
    return output_message, step_data


def get_model_representation(model_id: str) -> str:
    """Return a string representation of the model for the given model ID."""
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model_representation = str(model)
    return model_representation


def get_id_to_token_mapping(model_id: str) -> dict[int, str]:
    """Generate a mapping from token IDs to token texts for the specified model ID."""
    tokenizer = get_tokenizer(model_id)
    vocab_map = tokenizer.get_vocab()
    id_to_token = {token_id: token_text for token_text, token_id in vocab_map.items()}
    sorted_vocab = dict(sorted(id_to_token.items()))
    return sorted_vocab


def get_token_to_id_mapping(model_id: str) -> dict[str, int]:
    """Generate a mapping from token texts to token IDs for the specified model ID."""
    tokenizer = get_tokenizer(model_id)
    vocab_map = tokenizer.get_vocab()
    sorted_vocab = dict(sorted(vocab_map.items()))
    return sorted_vocab


def tokenize_messages(
    messages: list[dict[str, typing.Any]],
    tokenizer: PreTrainedTokenizerBase,
) -> list[int]:
    """Return the tokenized representation of the given text."""
    batch_encoding = typing.cast(
        "BatchEncoding", tokenizer.apply_chat_template(messages)
    )
    token_list = list(map(int, batch_encoding.data["input_ids"]))
    return token_list


@cache
def get_tokenizer(
    model_id: str,
    truncation_side: str = "left",
    padding_side: str = "left",
) -> PreTrainedTokenizerBase:
    """Return the tokenizer for the specified model ID."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        truncation_side=truncation_side,
        padding_side=padding_side,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return typing.cast("PreTrainedTokenizerBase", tokenizer)


def check_model_output_for_completion(
    completion_ids: list[int],
    prompt: str,
    prompt_id: str,
    config: Config,
) -> bool:
    """Check if the model could finish its answer for the given prompt."""
    completion_length = len(completion_ids)
    last_completion_token = completion_ids[-1]
    eos_token_id = get_tokenizer(config.hf_model_id).eos_token_id
    if (
        completion_length >= config.max_completion_length
        and last_completion_token != eos_token_id
    ):
        log_msg(
            f'model could not finish its answer for prompt ("{prompt}") with ID ("{prompt_id}")',
            level=logging.WARNING,
        )
        return False
    return True
