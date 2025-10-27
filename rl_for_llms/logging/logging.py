import logging
from functools import cache


@cache
def get_logger() -> logging.Logger:
    """Get a logger."""
    logger = logging.getLogger("rl_for_llms")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s/%(name)s/%(levelname)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def log_msg(text: str, level: int = logging.INFO) -> None:
    """Log a message."""
    get_logger().log(msg=text, level=level)
