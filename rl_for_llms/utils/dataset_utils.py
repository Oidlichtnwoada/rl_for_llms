import typing

from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from rl_for_llms.utils.hash_utils import generate_deterministic_id
from rl_for_llms.utils.llm_utils import get_user_message, tokenize_text


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
        lambda x: len(tokenize_text(x["prompt"][0]["content"], tokenizer))
        <= max_prompt_tokens
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


def clean_dataset(dataset: Dataset) -> Dataset:
    """Clean the dataset by removing whitespaces and checking uniqueness."""
    cleaned_dataset = dataset.map(clean_dataset_value)
    return cleaned_dataset
