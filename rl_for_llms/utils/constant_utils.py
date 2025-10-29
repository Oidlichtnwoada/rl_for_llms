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
