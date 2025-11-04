from rl_for_llms.utils.logging_utils import log_msg
from rl_for_llms.utils.training_utils import (
    start_training,
)


def main() -> None:
    """Execute the main function of the module."""
    log_msg("rl_for_llms is starting with the main function.")
    start_training()
    log_msg("rl_for_llms is finishing with the main function.")


if __name__ == "__main__":
    main()
