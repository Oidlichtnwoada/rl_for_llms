import typing
from copy import deepcopy
from functools import partial

from datasets import Dataset, Value, concatenate_datasets, load_dataset, load_from_disk
from transformers import PreTrainedTokenizerBase

from rl_for_llms.models.config import Config
from rl_for_llms.utils.constant_utils import (
    get_default_evaluation_file_names,
    get_train_split,
)
from rl_for_llms.utils.hash_utils import generate_deterministic_id
from rl_for_llms.utils.llm_utils import (
    get_system_message,
    get_user_message,
    tokenize_messages,
)
from rl_for_llms.utils.path_utils import get_evaluation_data_dir, get_training_data_dir


def trim_dataset(
    dataset: Dataset,
    target_percentage: float,
    tokenizer: PreTrainedTokenizerBase,
    max_prompt_tokens: int,
    seed: int = 0,
) -> Dataset:
    """Trim the dataset deterministically to the target percentage by removing random rows."""
    total_rows = len(dataset)
    target_rows = int(total_rows * target_percentage)
    filtered_dataset = dataset.filter(
        lambda x: len(tokenize_messages(x["prompt"], tokenizer)) <= max_prompt_tokens
    )
    target_rows = min(target_rows, len(filtered_dataset))
    trimmed_dataset = filtered_dataset.shuffle(seed=seed).select(range(target_rows))
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


def add_system_message(
    sample: dict[str, typing.Any], config: Config
) -> dict[str, typing.Any]:
    """Add the system message to the sample's prompt."""
    sample["prompt"] = [get_system_message(config.system_message)] + sample["prompt"]
    return sample


def clean_dataset(dataset: Dataset, config: Config) -> Dataset:
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
    cleaned_dataset = cleaned_dataset.map(
        partial(add_system_message, config=config), load_from_cache_file=False
    )
    return cleaned_dataset


def load_training_data_from_disk(config: Config) -> Dataset:
    """Load training data previously saved to disk."""
    training_data_dir = get_training_data_dir()
    dataset = load_from_disk(training_data_dir)
    dataset = dataset.remove_columns(["ability", "data_source"])
    ids = [str(x["index"]) for x in list(dataset["extra_info"])]
    dataset = dataset.add_column("id", ids)
    dataset = dataset.remove_columns(["extra_info"])
    answers = [str(x["ground_truth"]) for x in list(dataset["reward_model"])]
    dataset = dataset.add_column("answer", answers)
    dataset = dataset.remove_columns(["reward_model"])
    cleaned_dataset = clean_dataset(dataset, config)
    return cleaned_dataset


def load_evaluation_data(config: Config) -> Dataset:
    """Load evaluation data."""
    file_paths = [
        get_evaluation_data_dir() / file_name
        for file_name in get_default_evaluation_file_names()
    ]
    dataset_dicts = [
        load_dataset("json", data_files=[str(file_path.resolve())])
        for file_path in file_paths
    ]
    datasets = [dataset_dict[str(get_train_split())] for dataset_dict in dataset_dicts]
    target_columns = {"problem", "answer", "id"}
    for index, dataset in enumerate(datasets):
        if "unique_id" in dataset.column_names:
            new_dataset = dataset.rename_columns({"unique_id": "id"})
        else:
            new_dataset = deepcopy(dataset)
        columns_to_drop = set(new_dataset.column_names) - target_columns
        new_dataset = new_dataset.remove_columns(list(columns_to_drop))
        for column in target_columns:
            new_dataset = new_dataset.cast_column(column, Value("string"))
        datasets[index] = new_dataset
    merged_dataset = concatenate_datasets(datasets)
    prompts = [[get_user_message(x)] for x in list(merged_dataset["problem"])]
    merged_dataset = merged_dataset.add_column("prompt", prompts)
    merged_dataset = merged_dataset.remove_columns(["problem"])
    cleaned_dataset = clean_dataset(merged_dataset, config)
    return cleaned_dataset
