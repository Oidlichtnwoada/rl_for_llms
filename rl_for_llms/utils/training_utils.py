from datasets import Dataset, load_dataset, load_from_disk

from rl_for_llms.utils.constant_utils import (
    get_gitignore_file_name,
    get_hf_training_ds_path,
    get_relative_training_file_path,
    get_train_split,
)
from rl_for_llms.utils.path_utils import get_training_data_dir, is_folder_empty


def download_training_data() -> None:
    """Download training data from a remote source."""
    training_data_dir = get_training_data_dir()
    if not is_folder_empty(
        training_data_dir, ignore_file_names=(get_gitignore_file_name(),)
    ):
        return
    dataset = load_dataset(
        path=get_hf_training_ds_path(),
        data_files=get_relative_training_file_path(),
        split=get_train_split(),
    )
    dataset.save_to_disk(training_data_dir)


def load_training_data_from_disk() -> Dataset:
    """Load training data previously saved to disk."""
    training_data_dir = get_training_data_dir()
    dataset = load_from_disk(training_data_dir)
    return dataset
