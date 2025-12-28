import typing
from collections import Counter
from functools import partial, update_wrapper

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module, functional
from torch.utils.hooks import RemovableHandle
from trl.trainer.grpo_trainer import GRPOTrainer

from rl_for_llms.models.answer import Answer
from rl_for_llms.models.config import Config
from rl_for_llms.utils.classification_utils import compute_binary_classification_metrics
from rl_for_llms.utils.confidence_utils import get_confidence_token_logit_sigmoid
from rl_for_llms.utils.llm_utils import get_token_to_id_mapping
from rl_for_llms.utils.reward_utils import get_class_weights_for_rewards
from rl_for_llms.utils.torch_utils import get_mode


class ConfidenceGRPOTrainer(GRPOTrainer):
    """GRPO trainer with auxiliary confidence loss."""

    def __init__(self, config: Config, **kwargs: typing.Any) -> None:  # noqa: ANN401
        """Initialize the class with confidence loss parameters."""
        if len(kwargs["reward_funcs"]) != 1:
            raise ValueError
        reward_func = kwargs["reward_funcs"][0]
        wrapped_reward_func = partial(reward_func, trainer=self)
        update_wrapper(wrapped_reward_func, reward_func)
        kwargs["reward_funcs"] = [wrapped_reward_func]
        super().__init__(**kwargs)
        self.confidence_token_id = get_token_to_id_mapping(config.hf_model_id)[
            config.confidence_token
        ]
        self.confidence_loss_factor = (
            config.confidence_loss_factor if config.use_confidence_loss else 0.0
        )
        self.answers: list[Answer] = []
        self.lm_head_attribute_name = config.lm_head_attribute_name
        self.hook_handle: RemovableHandle | None = None
        self.confidence_logits: Tensor | None = None

    def get_last_rewards(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> list[float]:
        """Return the last computed rewards."""
        last_rewards = [answer.reward for answer in self.answers]
        advantages = inputs["advantages"].detach().cpu()
        inputs_length = advantages.shape[0]
        if len(last_rewards) != inputs_length:
            raise ValueError
        sorted_last_rewards = sorted(last_rewards)
        advantages_np = advantages.numpy()
        indices = np.argsort(np.argsort(advantages_np))
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
        mask = inputs["completion_mask"].bool()
        sequence_lengths = mask.sum(dim=-1)
        masked_estimated_rewards = estimated_rewards * mask
        sum_estimated_rewards = masked_estimated_rewards.sum(dim=-1)
        mean_estimated_rewards = sum_estimated_rewards / sequence_lengths
        maximum_sequence_length = estimated_rewards.shape[-1]
        last_rewards = self.get_last_rewards(inputs)
        real_rewards = (
            torch.tensor(last_rewards)
            .unsqueeze(-1)
            .expand(-1, maximum_sequence_length)
            .to(estimated_rewards.device)
            .float()
        )
        if mask.sum().item() == 0:
            raise ValueError
        per_sample_loss_masked = (
            functional.binary_cross_entropy(
                estimated_rewards, real_rewards, reduction="none"
            )
            * mask
        )
        incorrect_sample_weight, correct_sample_weight = get_class_weights_for_rewards(
            last_rewards
        )
        sample_weights = torch.tensor(
            [
                correct_sample_weight if reward == 1.0 else incorrect_sample_weight
                for reward in last_rewards
            ],
            device=estimated_rewards.device,
        )
        per_rollout_loss_weighted = (
            per_sample_loss_masked.sum(dim=-1) / sequence_lengths
        ) * sample_weights
        mean_rollout_loss = per_rollout_loss_weighted.mean()
        binary_classification_metrics = compute_binary_classification_metrics(
            last_rewards, mean_estimated_rewards.detach().cpu().tolist()
        )
        self.add_metrics("confidence", binary_classification_metrics)
        return mean_rollout_loss

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
        self.add_metrics("grpo", {("loss",): grpo_loss.item()})
        self.add_metrics("confidence", {("loss",): confidence_loss.item()})
        self.add_metrics("total", {("loss",): total_loss.item()})
        return total_loss

    def add_metrics(
        self, namespace: str, metrics: dict[tuple[str, ...], float], sep: str = "/"
    ) -> None:
        """Add metrics with the appropriate mode and namespace prefix."""
        mode = get_mode(typing.cast("torch.nn.Module", self.model))
        for key, value in metrics.items():
            self._metrics[mode][sep.join((namespace, *key))].append(value)
