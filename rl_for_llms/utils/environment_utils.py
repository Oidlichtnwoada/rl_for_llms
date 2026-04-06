import os
import sys

from rl_for_llms.models.method import Method
from rl_for_llms.utils.constant_utils import (
    get_eval_before_train_prefix,
    get_python_debug_modules,
)


def is_local_mode_enforced() -> bool:
    """Check if local mode is enforced via environment variable."""
    return os.getenv("MODE", "LOCAL") == "LOCAL"


def use_confidence_loss() -> bool:
    """Check if confidence loss should be used via environment variable."""
    return os.getenv("USE_CONFIDENCE_LOSS", "1") == "1"


def use_confidence_reward() -> bool:
    """Check if confidence reward should be used via environment variable."""
    return os.getenv("USE_CONFIDENCE_REWARD", "1") == "1"


def get_method() -> Method:
    """Return the confidence method via environment variable."""
    if os.getenv("METHOD", "DENSE") == "DENSE":
        return Method.DENSE
    return Method.LASER


def get_confidence_loss_factor() -> float:
    """Return the confidence loss factor according to used method."""
    if get_method() == Method.DENSE:
        return 0.01
    return 0.1


def get_per_device_rollouts_per_batch() -> int:
    """Return the number of rollouts per batch per device from environment variable."""
    return int(os.getenv("PER_DEVICE_ROLLOUTS_PER_BATCH", "1"))


def get_base_num_generations() -> int:
    """Return the base number of generations from environment variable."""
    return int(os.getenv("BASE_NUM_GENERATIONS", "4"))


def get_base_num_train_epochs() -> int:
    """Return the base number of training epochs from environment variable."""
    return int(os.getenv("BASE_NUM_TRAIN_EPOCHS", "1"))


def get_skip_eval_before_train() -> bool:
    """Check if evaluation should be skipped before training via environment variable."""
    return os.getenv(f"SKIP_{get_eval_before_train_prefix().upper()}", "0") == "1"


def get_lora_train_confidence_token_embedding() -> bool:
    """Check if the confidence token embedding should be trained during LoRA fine-tuning via environment variable."""
    if use_confidence_loss():
        return os.getenv("LORA_TRAIN_CONFIDENCE_TOKEN_EMBEDDING", "1") == "1"
    return False


def get_wandb_api_key() -> str | None:
    """Return the Weights & Biases API key from environment variable."""
    return os.getenv("WANDB_API_KEY")


def is_wandb_enabled() -> bool:
    """Check if Weights & Biases is enabled via API key."""
    return get_wandb_api_key() is not None


def is_debug_mode() -> bool:
    """Check if the code is running in debug mode."""
    first_check = sys.gettrace() is not None
    second_check = any(x in sys.modules for x in get_python_debug_modules())
    return first_check or second_check


def get_seed() -> int:
    """Return the seed from environment variable."""
    return int(os.getenv("SEED", "42"))


def get_hf_model_id() -> str:
    """Return the HF model id from environment variable."""
    return os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
