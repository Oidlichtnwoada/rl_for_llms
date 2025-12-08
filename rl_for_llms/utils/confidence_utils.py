import math

import torch

from rl_for_llms.utils.config_utils import get_config


def get_confidence_token_logit_sigmoid(
    confidence_token_logit: torch.Tensor,
) -> torch.Tensor:
    """Return the sigmoid of the confidence token logit."""
    config = get_config()
    scale = math.pi / (config.confidence_token_logit_std * math.sqrt(3))
    return torch.sigmoid(
        scale * (confidence_token_logit - config.confidence_token_logit_mean)
    )