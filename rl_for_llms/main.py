from rl_for_llms.utils.logging_utils import log_msg
from rl_for_llms.utils.torch_utils import get_cuda_device_count
from rl_for_llms.utils.training_utils import (
    start_training,
)


def main() -> None:
    """Execute the main function of the module."""
    log_msg(
        f"program is starting execution of the main function, found {get_cuda_device_count()} CUDA devices"
    )
    start_training()
    log_msg(
        f"program has finishing executing the main function, found {get_cuda_device_count()} CUDA devices"
    )


if __name__ == "__main__":
    main()
