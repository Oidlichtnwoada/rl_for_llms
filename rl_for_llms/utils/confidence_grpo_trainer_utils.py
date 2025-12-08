import typing

import torch
from trl.trainer.grpo_trainer import GRPOTrainer

from rl_for_llms.models.config import Config
from rl_for_llms.utils.llm_utils import get_token_to_id_mapping


class ConfidenceGRPOTrainer(GRPOTrainer):
    """GRPO trainer with auxiliary confidence loss."""

    def __init__(self, config: Config, **kwargs: typing.Any) -> None:  # noqa: ANN401
        """Initialize the class with confidence loss parameters."""
        super().__init__(**kwargs)
        self.confidence_token_id = get_token_to_id_mapping(config.hf_model_id)[
            config.confidence_token
        ]
        self.confidence_loss_factor = config.confidence_loss_factor

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,  # noqa: FBT001, FBT002
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the combined GRPO loss and confidence loss."""
        grpo_loss = super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )
        confidence_loss = 0
        total_loss = typing.cast(
            "torch.Tensor", grpo_loss + self.confidence_loss_factor * confidence_loss
        )
        return total_loss
