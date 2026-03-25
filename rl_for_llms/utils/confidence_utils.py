import math

import torch

from rl_for_llms.models.config import Config


def get_confidence_token_logit_sigmoid(
    confidence_token_logit: torch.Tensor,
    config: Config,
) -> torch.Tensor:
    """Return the sigmoid of the confidence token logit."""
    scale = math.pi / (config.confidence_token_logit_std * math.sqrt(3))
    return torch.sigmoid(
        scale * (confidence_token_logit - config.confidence_token_logit_mean)
    )


def apply_laser_score_transformation(
    confidence_log_prob: torch.Tensor,
    config: Config,
) -> torch.Tensor:
    """Apply LaSeR scale and offset to a pre-computed confidence token log-probability."""
    return config.confidence_token_logprob_kl_scale * (
        confidence_log_prob - config.confidence_token_logprob_mean
    )


def compute_laser_self_rewarding_score(
    logits: torch.Tensor,
    confidence_token_id: int,
    config: Config,
) -> torch.Tensor:
    """Return the LaSeR self-rewarding score from full vocabulary logits."""
    log_probs = torch.log_softmax(logits / config.temperature, dim=-1)
    confidence_log_prob = log_probs[..., confidence_token_id]
    return apply_laser_score_transformation(confidence_log_prob, config)
