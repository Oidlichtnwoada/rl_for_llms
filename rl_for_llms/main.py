from rl_for_llms.logging.logging import log_msg
from rl_for_llms.utils.training_utils import (
    download_training_data,
)


def main() -> None:
    """Execute the main function of the module."""
    log_msg("rl_for_llms is starting with the main function.")
    download_training_data()
    log_msg("rl_for_llms is finishing with the main function.")


if __name__ == "__main__":
    main()
