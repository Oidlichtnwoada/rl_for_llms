import pathlib


def get_repo_root() -> pathlib.Path:
    """Return the root directory of the repository."""
    return pathlib.Path(__file__).parent.parent.parent.resolve()


def get_data_dir() -> pathlib.Path:
    """Return the data directory path."""
    return get_repo_root() / "data"


def get_thesis_dir() -> pathlib.Path:
    """Return the thesis directory path."""
    return get_repo_root() / "thesis"


def get_charts_dir() -> pathlib.Path:
    """Return the charts directory path."""
    return get_thesis_dir() / "charts"


def get_checkpoints_data_dir() -> pathlib.Path:
    """Return the checkpoints data directory path."""
    return get_data_dir() / "checkpoints"


def get_evaluation_data_dir() -> pathlib.Path:
    """Return the evaluation data directory path."""
    return get_data_dir() / "evaluation"


def get_training_data_dir() -> pathlib.Path:
    """Return the training data directory path."""
    return get_data_dir() / "training"


def get_evaluation_metric_dir() -> pathlib.Path:
    """Return the evaluation metrics data directory path."""
    return get_evaluation_data_dir() / "metrics"


def get_evaluation_final_dir() -> pathlib.Path:
    """Return the final evaluation data directory path."""
    return get_evaluation_data_dir() / "final"


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


def get_checkpoint_folder_for_model_id(
    model_id: str, *, create: bool = True
) -> pathlib.Path:
    """Return the checkpoint folder path for a given model ID."""
    first_part, second_part = model_id.split("/")
    full_path = get_checkpoints_data_dir() / first_part / second_part
    if create:
        full_path.mkdir(exist_ok=True, parents=True)
    return full_path
