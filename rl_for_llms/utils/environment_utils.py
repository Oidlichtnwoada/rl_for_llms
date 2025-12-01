import os


def is_local_mode_enforced() -> bool:
    """Check if local mode is enforced via environment variable."""
    return os.getenv("MODE") == "LOCAL"


def get_wandb_api_key() -> str | None:
    """Return the Weights & Biases API key from environment variable."""
    return os.getenv("WANDB_API_KEY")


def is_wandb_enabled() -> bool:
    """Check if Weights & Biases is enabled via API key."""
    return get_wandb_api_key() is not None

def set_environment_variables() -> None:
    """Set important environment variables for optimal performance."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
