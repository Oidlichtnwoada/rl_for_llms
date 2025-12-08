import logging

import torch

from rl_for_llms.utils.environment_utils import is_local_mode_enforced, is_wandb_enabled
from rl_for_llms.utils.logging_utils import log_msg


def get_device() -> torch.device:
    """Return the best available device for PyTorch computations."""
    if torch.cuda.is_available():
        device_string = "cuda"
    elif torch.mps.is_available():
        device_string = "mps"
    else:
        device_string = "cpu"
    device = torch.device(device_string)
    return device


def is_cuda_device_used() -> bool:
    """Check if the current device is a CUDA device."""
    device_string = str(get_device())
    is_cuda_device = device_string == "cuda"
    return is_cuda_device


def is_mps_device_used() -> bool:
    """Check if the current device is an MPS device."""
    device_string = str(get_device())
    is_mps_device = device_string == "mps"
    return is_mps_device


def is_cpu_device_used() -> bool:
    """Check if the current device is a CPU device."""
    device_string = str(get_device())
    is_cpu_device = device_string == "cpu"
    return is_cpu_device


def get_cuda_default_value[T](value_if_cuda: T, value_if_not_cuda: T) -> T:
    """Return the appropriate value based on whether a CUDA device is used."""
    if is_cuda_device_used() and not is_local_mode_enforced():
        return value_if_cuda
    return value_if_not_cuda


def get_cuda_device_count() -> int:
    """Return the appropriate value based on whether a CUDA device is used."""
    if is_cuda_device_used():
        return int(torch.cuda.device_count())
    return 0


def is_bf16_supported() -> bool:
    """Check if the current device supports bfloat16 precision."""
    if is_cuda_device_used():
        return bool(torch.cuda.is_bf16_supported())
    return False


def get_logging_integrations() -> list[str]:
    """Get the logging integrations."""
    integrations = ["tensorboard"]
    if is_wandb_enabled():
        integrations.append("wandb")
    else:
        log_msg(
            text="no wandb api key was provided, so wandb logging is disabled",
            level=logging.WARNING,
        )
    return integrations


def get_mode(model: torch.nn.Module) -> str:
    """Return the current mode of the model."""
    mode = "train" if model.training else "eval"
    return mode
