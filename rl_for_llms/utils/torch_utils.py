import torch


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


def get_cuda_default_value[T](value_if_cuda: T, value_if_not_cuda: T) -> T:
    """Return the appropriate value based on whether a CUDA device is used."""
    if is_cuda_device_used():
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
