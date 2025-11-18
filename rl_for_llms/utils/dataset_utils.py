import typing

from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from rl_for_llms.utils.config_utils import get_config
from rl_for_llms.utils.hash_utils import generate_deterministic_id
from rl_for_llms.utils.llm_utils import (
    get_user_message,
    tokenize_messages,
)


def trim_dataset(
    dataset: Dataset,
    target_percentage: float,
    tokenizer: PreTrainedTokenizerBase,
    max_prompt_tokens: int,
) -> Dataset:
    """Trim the dataset to the target percentage by removing random rows."""
    total_rows = len(dataset)
    target_rows = int(total_rows * target_percentage)
    filtered_dataset = dataset.filter(
        lambda x: len(tokenize_messages(x["prompt"], tokenizer)) <= max_prompt_tokens
    )
    target_rows = min(target_rows, len(filtered_dataset))
    trimmed_dataset = filtered_dataset.shuffle().select(range(target_rows))
    return trimmed_dataset


def clean_dataset_value(value: dict[str, typing.Any]) -> dict[str, typing.Any]:
    """Clean a single dataset value by removing whitespaces."""
    preliminary_dict = {
        "prompt": [get_user_message(value["prompt"][0]["content"])],
        "answer": value["answer"].strip(),
        "id": value["id"].strip(),
    }
    new_id = generate_deterministic_id(preliminary_dict)
    final_dict = {**preliminary_dict, "id": new_id}
    return final_dict


def filter_function(content: str, seen_content: set[str]) -> bool:
    """Filter function to check for uniqueness of content."""
    result = content not in seen_content
    seen_content.add(content)
    return result


def add_system_message(sample: dict[str, typing.Any]) -> dict[str, typing.Any]:
    """Add the system message to the sample's prompt."""
    config = get_config()
    sample["prompt"][0]["content"] = (
        f"{config.system_message}\n{sample['prompt'][0]['content']}"
    )
    return sample


def clean_dataset(dataset: Dataset) -> Dataset:
    """Clean the dataset by removing duplicates and adding system messages."""
    cleaned_dataset = dataset.map(clean_dataset_value)
    if len(cleaned_dataset) != len(set(cleaned_dataset["id"])):
        raise ValueError
    seen_prompts: set[str] = set()
    cleaned_dataset = cleaned_dataset.filter(
        lambda x: filter_function(x["prompt"][0]["content"], seen_prompts)
    )
    if len(cleaned_dataset) != len(
        {x[0]["content"] for x in cleaned_dataset["prompt"]}
    ):
        raise ValueError
    cleaned_dataset = cleaned_dataset.map(add_system_message)
    return cleaned_dataset
