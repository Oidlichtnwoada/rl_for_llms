import typing
from collections import Counter

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.hooks import RemovableHandle
from trl.trainer.grpo_trainer import GRPOTrainer

from rl_for_llms.models.config import Config
from rl_for_llms.utils.confidence_utils import get_confidence_token_logit_sigmoid
from rl_for_llms.utils.llm_utils import get_token_to_id_mapping
from rl_for_llms.utils.torch_utils import get_mode


class ConfidenceGRPOTrainer(GRPOTrainer):
    """GRPO trainer with auxiliary confidence loss."""

    def __init__(self, config: Config, **kwargs: typing.Any) -> None:  # noqa: ANN401
        """Initialize the class with confidence loss parameters."""
        super().__init__(**kwargs)
        self.confidence_token_id = get_token_to_id_mapping(config.hf_model_id)[
            config.confidence_token
        ]
        self.confidence_loss_factor = config.confidence_loss_factor
        self.lm_head_attribute_name = config.lm_head_attribute_name
        self.hook_handle: RemovableHandle | None = None
        self.confidence_logits: Tensor | None = None
        self.loss_criterion = torch.nn.BCELoss()

    def get_reward_function_name(self) -> str:
        """Return the name of the reward function."""
        if len(self.reward_func_names) != 1:
            raise ValueError
        reward_func_name = self.reward_func_names[0]
        return reward_func_name

    def get_last_rewards(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> list[float]:
        """Return the last computed rewards."""
        last_rewards = list(
            dict(self._logs["rewards"])[self.get_reward_function_name()]
        )
        advantages = inputs["advantages"].detach().cpu()
        inputs_length = advantages.shape[0]
        if len(last_rewards) != inputs_length:
            raise ValueError
        sorted_last_rewards = sorted(last_rewards)
        advantages_np = advantages.numpy()
        indices = np.argsort(advantages_np)
        rewards_for_advantages_np = np.array(sorted_last_rewards)[indices]
        rewards_for_advantages_list = list(rewards_for_advantages_np.tolist())
        rewards_counter = Counter(rewards_for_advantages_np)
        reward_sorted_counts = [
            rewards_counter[key] for key in sorted(rewards_counter.keys())
        ]
        advantages_counter = Counter(advantages_np)
        advantage_sorted_counts = [
            advantages_counter[key] for key in sorted(advantages_counter.keys())
        ]
        if reward_sorted_counts != advantage_sorted_counts:
            raise ValueError
        return rewards_for_advantages_list

    def register_hook(self) -> None:
        """Register a hook to capture confidence logits."""
        lm_head = getattr(self.model, self.lm_head_attribute_name, None)
        if lm_head is None:
            raise ValueError
        self.hook_handle = lm_head.register_forward_hook(self.logits_hook)

    def remove_hook(self) -> None:
        """Remove the registered hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def logits_hook(
        self,
        module: Module,  # noqa: ARG002
        inputs: tuple[Tensor, ...],  # noqa: ARG002
        output: torch.Tensor,
    ) -> None:
        """Capture the logits for the confidence token."""
        logits = output
        if logits.dim() != 3:  # noqa: PLR2004
            raise ValueError
        confidence_logits = logits[:, :-1, self.confidence_token_id]
        self.confidence_logits = confidence_logits

    def get_confidence_loss(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute the confidence loss based on the last rewards and confidence logits."""
        if self.confidence_logits is None:
            raise ValueError
        estimated_rewards = get_confidence_token_logit_sigmoid(
            self.confidence_logits
        ).float()
        sequence_length = estimated_rewards.shape[-1]
        real_rewards = (
            torch.tensor(self.get_last_rewards(inputs))
            .unsqueeze(-1)
            .expand(-1, sequence_length)
            .to(estimated_rewards.device)
            .float()
        )
        loss = typing.cast(
            "torch.Tensor", self.loss_criterion(estimated_rewards, real_rewards)
        )
        return loss

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,  # noqa: FBT001, FBT002
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the combined GRPO loss and confidence loss."""
        self.register_hook()
        grpo_loss = typing.cast(
            "torch.Tensor",
            super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            ),
        )
        self.remove_hook()
        confidence_loss = self.confidence_loss_factor * self.get_confidence_loss(inputs)
        self.confidence_logits = None
        total_loss = grpo_loss + confidence_loss
        mode = get_mode(model)
        self._metrics[mode]["grpo_loss"].append(grpo_loss.item())
        self._metrics[mode]["confidence_loss"].append(confidence_loss.item())
        self._metrics[mode]["total_loss"].append(total_loss.item())
        return total_loss
