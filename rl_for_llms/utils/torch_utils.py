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


def is_bf16_supported() -> bool:
    """Check if the current device supports bfloat16 precision."""
    device_string = str(get_device())
    if device_string == "cuda":
        return bool(torch.cuda.is_bf16_supported())
    return False
