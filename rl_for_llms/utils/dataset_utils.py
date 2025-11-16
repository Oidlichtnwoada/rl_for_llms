from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from rl_for_llms.utils.llm_utils import tokenize_text


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
