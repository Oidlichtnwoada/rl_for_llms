import torch
from torch import Tensor


def get_mean_aggregation_weights(mask: Tensor) -> Tensor:
    """Return uniform per-token weights (1 / sequence_length) that sum to 1, zero at padding."""
    sequence_lengths = mask.float().sum(dim=-1, keepdim=True).clamp(min=1)
    return mask.float() / sequence_lengths


def get_exponentially_increasing_aggregation_weights(
    mask: Tensor, base: float = 1.01
) -> Tensor:
    """Return weights proportional to exp(base * t / (T-1)) normalised to sum to 1, zero at padding."""
    return _get_exponential_aggregation_weights(mask, increasing=True, base=base)


def get_exponentially_decreasing_aggregation_weights(
    mask: Tensor, base: float = 1.01
) -> Tensor:
    """Return weights proportional to exp(-base * t / (T-1)) normalised to sum to 1, zero at padding."""
    return _get_exponential_aggregation_weights(mask, increasing=False, base=base)


def _get_exponential_aggregation_weights(
    mask: Tensor, *, increasing: bool, base: float = 1.01
) -> Tensor:
    """Return normalised exponential weights over valid positions, with relative positions in [0, 1]."""
    batch_size, seq_len = mask.shape
    sequence_lengths = mask.float().sum(dim=-1, keepdim=True)
    max_pos = (sequence_lengths - 1).clamp(min=1)
    positions = (
        torch.arange(seq_len, device=mask.device, dtype=torch.float32)
        .unsqueeze(0)
        .expand(batch_size, seq_len)
    )
    rel_positions = positions / max_pos
    scale = base if increasing else -base
    raw_weights = torch.exp(scale * rel_positions) * mask.float()
    weight_sums = raw_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return raw_weights / weight_sums
