from datasets import Split


def get_hf_training_ds_path() -> str:
    """Return the Hugging Face dataset path for training data."""
    return "Keven16/LaSeR_training_data"


def get_relative_training_file_path() -> str:
    """Return the relative file path for training data within the dataset."""
    return "RL_data/DeepMath-103K/train.parquet"


def get_train_split() -> Split:
    """Return the default training split."""
    return Split.TRAIN


def get_gitignore_file_name() -> str:
    """Return the .gitignore file name."""
    return ".gitignore"


def get_default_evaluation_file_names() -> tuple[str, ...]:
    """Return the default evaluation file name."""
    return (
        "aime-2024-test.jsonl",
        "aime-2025-test.jsonl",
        "amc-2023-test.jsonl",
        "math-500-test.jsonl",
        "olympiad-bench-test.jsonl",
    )


def get_python_debug_modules() -> list[str]:
    """Return the list of Python modules used for debugging."""
    return ["debugpy", "pydevd"]


def get_boolean_classification_threshold() -> float:
    """Return the threshold for boolean classification tasks."""
    return 0.5


def get_default_confidence_score() -> float:
    """Return the default confidence score."""
    return 1.0
