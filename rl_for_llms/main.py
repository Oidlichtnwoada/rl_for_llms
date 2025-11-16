from rl_for_llms.utils.logging_utils import log_msg
from rl_for_llms.utils.training_utils import (
    start_training,
)


def main() -> None:
    """Execute the main function of the module."""
    log_msg("program is starting execution of the main function")
    start_training()
    log_msg("program has finishing executing the main function")


if __name__ == "__main__":
    main()
