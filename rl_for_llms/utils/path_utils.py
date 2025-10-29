import pathlib


def get_repo_root() -> pathlib.Path:
    """Return the root directory of the repository."""
    return pathlib.Path(__file__).parent.parent.parent.resolve()


def get_data_dir() -> pathlib.Path:
    """Return the data directory path."""
    return get_repo_root() / "data"


def get_evaluation_data_dir() -> pathlib.Path:
    """Return the evaluation data directory path."""
    return get_data_dir() / "evaluation"


def get_training_data_dir() -> pathlib.Path:
    """Return the training data directory path."""
    return get_data_dir() / "training"
