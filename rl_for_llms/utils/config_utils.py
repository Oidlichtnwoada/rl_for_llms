from functools import cache

from rl_for_llms.models.config import Config


@cache
def get_config() -> Config:
    """Return the configuration."""
    return Config()
