import os
import sys

import torch

from rl_for_llms.utils.constant_utils import get_python_debug_modules
from rl_for_llms.utils.torch_utils import is_mps_device_used


def is_local_mode_enforced() -> bool:
    """Check if local mode is enforced via environment variable."""
    return os.getenv("MODE") == "LOCAL"


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


def setup_environment() -> None:
    """Set up the environment for training."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if is_mps_device_used():
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    if is_debug_mode():
        os.environ["TORCHDYNAMO_DISABLE"] = "1"
        torch.autograd.set_detect_anomaly(True)
