from datasets import Dataset


def trim_dataset(dataset: Dataset, target_percentage: float) -> Dataset:
    """Trim the dataset to the target percentage by removing random rows."""
    total_rows = len(dataset)
    target_rows = int(total_rows * target_percentage)
    trimmed_dataset = dataset.shuffle().select(range(target_rows))
    return trimmed_dataset
