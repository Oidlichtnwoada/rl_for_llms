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


def is_folder_empty(
    folder_path: pathlib.Path, ignore_file_names: tuple[str, ...] = ()
) -> bool:
    """Check if a folder is empty."""
    if not folder_path.is_dir():
        raise ValueError
    relevant_folder_contents = [
        x
        for x in folder_path.rglob("*")
        if not x.is_file() or x.name not in ignore_file_names
    ]
    relevant_content_amount = len(relevant_folder_contents)
    return relevant_content_amount == 0
