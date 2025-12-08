import typing
from collections import Counter

import numpy as np
import torch
from trl.trainer.grpo_trainer import GRPOTrainer

from rl_for_llms.models.config import Config
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

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,  # noqa: FBT001, FBT002
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the combined GRPO loss and confidence loss."""
        _ = self.get_last_rewards(inputs)
        grpo_loss = super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )
        confidence_loss = self.confidence_loss_factor * torch.tensor(0.0)
        total_loss = typing.cast("torch.Tensor", grpo_loss + confidence_loss)
        mode = get_mode(model)
        self._metrics[mode]["grpo_loss"].append(grpo_loss.item())
        self._metrics[mode]["confidence_loss"].append(confidence_loss.item())
        self._metrics[mode]["total_loss"].append(total_loss.item())
        return total_loss
