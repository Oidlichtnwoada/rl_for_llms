import time

from datasets import Split

from rl_for_llms.models.dataset import Dataset


class _ProcessStartTime:
    _value: float = time.time()

    @staticmethod
    def get() -> float:
        """Return the time at which the Python process started (captured at first module import)."""
        return _ProcessStartTime._value


def get_process_start_time() -> float:
    """Return the time at which the Python process started (captured at first module import)."""
    return _ProcessStartTime.get()


def get_hf_training_ds_path(dataset: Dataset) -> str:
    """Return the Hugging Face dataset path for training data."""
    match dataset:
        case Dataset.DEEPMATH_103K:
            return "Keven16/LaSeR_training_data"
        case Dataset.GSM8K:
            return "openai/gsm8k"


def get_relative_training_file_path(dataset: Dataset) -> str | None:
    """Return the relative file path for training data within the dataset."""
    match dataset:
        case Dataset.DEEPMATH_103K:
            return "RL_data/DeepMath-103K/train.parquet"
        case Dataset.GSM8K:
            return None


def get_hf_training_ds_subset(dataset: Dataset) -> str | None:
    """Return the subset name for training data within the dataset."""
    match dataset:
        case Dataset.DEEPMATH_103K:
            return None
        case Dataset.GSM8K:
            return "main"


def get_default_evaluation_file_names(dataset: Dataset) -> tuple[str, ...]:
    """Return the default evaluation file name."""
    match dataset:
        case Dataset.DEEPMATH_103K:
            return (
                "aime-2024-test.jsonl",
                "aime-2025-test.jsonl",
                "amc-2023-test.jsonl",
                "math-500-test.jsonl",
                "olympiad-bench-test.jsonl",
            )
        case Dataset.GSM8K:
            return ()


def get_train_split() -> Split:
    """Return the default training split."""
    return Split.TRAIN


def get_test_split() -> Split:
    """Return the default test split."""
    return Split.TEST


def get_gitignore_file_name() -> str:
    """Return the .gitignore file name."""
    return ".gitignore"


def get_python_debug_modules() -> list[str]:
    """Return the list of Python modules used for debugging."""
    return ["debugpy", "pydevd"]


def get_boolean_classification_threshold() -> float:
    """Return the threshold for boolean classification tasks."""
    return 0.5


def get_default_confidence_score() -> float:
    """Return the default confidence score."""
    return 1.0


def get_confidence_namespace() -> str:
    """Return the namespace for confidence-related metrics."""
    return "confidence"


def get_total_namespace() -> str:
    """Return the namespace for total-related metrics."""
    return "total"


def get_grpo_namespace() -> str:
    """Return the namespace for GRPO-related metrics."""
    return "grpo"


def get_answer_namespace() -> str:
    """Return the namespace for answer-related metrics."""
    return "answer"


def get_loss_name() -> str:
    """Return the name for loss metrics."""
    return "loss"


def get_default_metric_separator() -> str:
    """Return the default separator for metric names."""
    return "/"


def get_default_eps() -> float:
    """Return the default epsilon value."""
    return 1e-4


def get_no_name() -> str:
    """Return a string 'no'."""
    return "no"


def get_confidence_loss_name() -> str:
    """Return the name for confidence loss."""
    return "confloss"


def get_confidence_reward_name() -> str:
    """Return the name for confidence reward."""
    return "confrew"


def get_eval_before_train_prefix() -> str:
    """Return the prefix for evaluation before training metrics."""
    return "eval_before_train"


def get_eval_after_train_prefix() -> str:
    """Return the prefix for evaluation after training metrics."""
    return "eval_after_train"


def get_mean_name() -> str:
    """Return the name for mean statistics."""
    return "mean"


def get_std_name() -> str:
    """Return the name for standard deviation statistics."""
    return "std"


def get_tempmod_name() -> str:
    """Return the suffix name for temperature modulation evaluation files."""
    return "tempmod"


def get_inv_prefix() -> str:
    """Return the prefix for inverted evaluation variants."""
    return "inv"


def get_filter_name() -> str:
    """Return the suffix name for SMC particle-filtering evaluation files."""
    return "filter"
