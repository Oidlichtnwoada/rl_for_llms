import os


def is_local_mode_enforced() -> bool:
    """Check if local mode is enforced via environment variable."""
    return os.getenv("MODE") == "LOCAL"
