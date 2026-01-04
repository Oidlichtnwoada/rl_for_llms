import pathlib
import shutil
import statistics
import typing
from functools import partial, update_wrapper
from itertools import chain

import numpy as np
import torch
from datasets import Dataset
from peft import PeftModel
from torch import Tensor
from torch.nn import Module, functional
from torch.utils.hooks import RemovableHandle
from transformers import PreTrainedModel
from trl.trainer.grpo_trainer import GRPOTrainer

from rl_for_llms.models.answer import (
    Answer,
    AnswerWithConfidence,
    get_answers_with_confidence,
)
from rl_for_llms.models.config import Config
from rl_for_llms.utils.classification_utils import compute_binary_classification_metrics
from rl_for_llms.utils.confidence_utils import get_confidence_token_logit_sigmoid
from rl_for_llms.utils.config_utils import get_config
from rl_for_llms.utils.constant_utils import (
    get_answer_namespace,
    get_confidence_namespace,
    get_default_confidence_score,
    get_default_eps,
    get_default_metric_separator,
    get_grpo_namespace,
    get_loss_name,
    get_total_namespace,
)
from rl_for_llms.utils.evaluation_utils import (
    aggregate_metrics,
    change_metric_keys,
    compute_answer_metrics,
    get_df_from_metrics,
    get_eval_metrics_df_name,
    store_eval_df,
)
from rl_for_llms.utils.llm_utils import get_token_to_id_mapping
from rl_for_llms.utils.path_utils import get_evaluation_metric_dir
from rl_for_llms.utils.reward_utils import get_class_weights_for_rewards
from rl_for_llms.utils.torch_utils import convert_tensor_to_list, get_mode


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
        self.config = config
        self.confidence_token_id = get_token_to_id_mapping(self.config.hf_model_id)[
            self.config.confidence_token
        ]
        self.confidence_loss_factor = (
            self.config.confidence_loss_factor
            if self.config.use_confidence_loss
            else 0.0
        )
        self.answers: list[Answer] = []
        self.lm_head_attribute_name = self.config.lm_head_attribute_name
        self.hook_handle: RemovableHandle | None = None
        self.all_confidence_logits: Tensor | None = None
        self.all_confidence_logits_excluding_last: Tensor | None = None
        self.eval_mode: bool = False
        self.eval_binary_classification_metrics_inputs: list[
            tuple[list[float], list[float]]
        ] = []
        self.eval_binary_classification_metrics_outputs: list[
            dict[tuple[str, ...], float]
        ] = []
        self.eval_answer_metrics_inputs: list[
            tuple[list[AnswerWithConfidence], float, int]
        ] = []
        self.eval_answer_metrics_outputs: list[dict[tuple[str, ...], float]] = []

    def get_lm_head(self, *, unwrap_model: bool) -> Module:
        """Get the language model head."""
        if unwrap_model:
            wrapped_model = getattr(self, "model_wrapped", self.model)
            model = self.accelerator.unwrap_model(wrapped_model)
        else:
            model = self.model
        lm_head = getattr(model, self.lm_head_attribute_name, None)
        if lm_head is None:
            raise ValueError
        return typing.cast("Module", lm_head)

    def register_hook(self, *, unwrap_model: bool) -> None:
        """Register a forward hook to capture confidence logits."""
        lm_head = self.get_lm_head(unwrap_model=unwrap_model)
        self.hook_handle = lm_head.register_forward_hook(self.logits_hook)

    def clear_confidence_logits(self) -> None:
        """Clear confidence logits."""
        self.all_confidence_logits = None
        self.all_confidence_logits_excluding_last = None

    def remove_hook(
        self,
        *,
        clear_confidence_logits: bool = True,
    ) -> None:
        """Remove the registered hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
        if clear_confidence_logits:
            self.clear_confidence_logits()

    def _generate_and_score_completions(
        self, inputs: list[dict[str, torch.Tensor | typing.Any]]
    ) -> dict[str, torch.Tensor | typing.Any]:
        """Generate completions and capture mean_estimated_rewards for advantage blending."""
        self.register_hook(unwrap_model=True)
        output = super()._generate_and_score_completions(inputs)
        mask = output["completion_mask"].bool()
        device = output["advantages"].device
        output["mean_estimated_rewards"] = torch.tensor(
            self.get_mean_estimated_rewards(mask), device=device
        )
        rewards = torch.tensor(
            [answer.reward for answer in self.answers],
            device=device,
        )
        output["rewards"] = rewards
        output["indices"] = torch.tensor(list(range(len(self.answers))), device=device)
        self.remove_hook()
        return output

    def logits_hook(
        self,
        module: Module,  # noqa: ARG002
        inputs: tuple[Tensor, ...],  # noqa: ARG002
        output: torch.Tensor,
    ) -> None:
        """Capture confidence logits during the forward pass."""
        logits = output
        if logits.dim() != 3:  # noqa: PLR2004
            raise ValueError
        confidence_logits = logits[:, :, self.confidence_token_id]
        if self.all_confidence_logits is None:
            self.all_confidence_logits = confidence_logits
        else:
            self.all_confidence_logits = torch.cat(
                (self.all_confidence_logits, confidence_logits), dim=1
            )
        self.all_confidence_logits_excluding_last = self.all_confidence_logits[:, :-1]

    def get_mean_estimated_rewards(self, completion_mask: torch.Tensor) -> list[float]:
        """Get mean estimated rewards for each sample in the batch."""
        if (
            self.all_confidence_logits is None
            or self.all_confidence_logits_excluding_last is None
        ):
            raise ValueError
        mask = completion_mask.bool()
        sequence_lengths = mask.sum(dim=-1)
        if self.all_confidence_logits.shape[1] == mask.shape[1]:
            confidence_logits = self.all_confidence_logits
        elif self.all_confidence_logits_excluding_last.shape[1] == mask.shape[1]:
            confidence_logits = self.all_confidence_logits_excluding_last
        else:
            raise ValueError
        estimated_rewards = get_confidence_token_logit_sigmoid(
            confidence_logits
        ).float()
        masked_estimated_rewards = estimated_rewards * mask
        sum_estimated_rewards = masked_estimated_rewards.sum(dim=-1)
        mean_estimated_rewards = sum_estimated_rewards / sequence_lengths
        mean_estimated_rewards_list = convert_tensor_to_list(mean_estimated_rewards)
        return mean_estimated_rewards_list

    def get_num_generations(self) -> int:
        """Return the number of generations per prompt."""
        return typing.cast("int", self.num_generations)

    def get_confidence_loss(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute the confidence loss based on the last rewards and confidence logits."""
        if self.all_confidence_logits_excluding_last is None:
            raise ValueError
        estimated_rewards = get_confidence_token_logit_sigmoid(
            self.all_confidence_logits_excluding_last
        ).float()
        mask = inputs["completion_mask"].bool()
        sequence_lengths = mask.sum(dim=-1)
        maximum_sequence_length = estimated_rewards.shape[-1]
        last_rewards = convert_tensor_to_list(inputs["rewards"])
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
        num_generations = self.get_num_generations()
        weights_per_group = get_class_weights_for_rewards(last_rewards, num_generations)
        sample_weights = []
        for group_idx, (incorrect_weight, correct_weight) in enumerate(
            weights_per_group
        ):
            start_idx = group_idx * num_generations
            end_idx = start_idx + num_generations
            group_rewards = last_rewards[start_idx:end_idx]
            for reward in group_rewards:
                sample_weights.extend(
                    [correct_weight if reward == 1.0 else incorrect_weight]
                )
        sample_weights_tensor = torch.tensor(
            sample_weights, device=estimated_rewards.device
        )
        per_rollout_loss_weighted = (
            per_sample_loss_masked.sum(dim=-1) / sequence_lengths
        ) * sample_weights_tensor
        mean_rollout_loss = per_rollout_loss_weighted.mean()
        mean_estimated_rewards_list = self.get_mean_estimated_rewards(
            completion_mask=mask
        )
        binary_classification_metrics = compute_binary_classification_metrics(
            last_rewards, mean_estimated_rewards_list
        )
        self.add_metrics(get_confidence_namespace(), binary_classification_metrics)
        answers_with_confidence = get_answers_with_confidence(
            self.answers,
            mean_estimated_rewards_list
            if self.is_confidence_trained()
            else [get_default_confidence_score()] * len(self.answers),
        )
        answer_metrics = compute_answer_metrics(
            answers_with_confidence, self.config.temperature, self.get_num_generations()
        )
        self.add_metrics(get_answer_namespace(), answer_metrics)
        if self.eval_mode:
            self.eval_binary_classification_metrics_inputs.append(
                (last_rewards, mean_estimated_rewards_list)
            )
            self.eval_binary_classification_metrics_outputs.append(
                binary_classification_metrics
            )
            self.eval_answer_metrics_inputs.append(
                (
                    answers_with_confidence,
                    self.config.temperature,
                    self.get_num_generations(),
                )
            )
            self.eval_answer_metrics_outputs.append(answer_metrics)
        return mean_rollout_loss

    @staticmethod
    def compute_advantages_from_rewards(
        rewards: list[float],
        num_generations: int,
        eps: float = get_default_eps(),
    ) -> list[float]:
        """Compute normalized advantages from rewards, per-group."""
        rewards_np = np.array(rewards)
        num_groups = len(rewards) // num_generations
        all_advantages = []
        for group_idx in range(num_groups):
            start_idx = group_idx * num_generations
            end_idx = start_idx + num_generations
            group_rewards = rewards_np[start_idx:end_idx]
            group_advantages = (group_rewards - np.mean(group_rewards)) / (
                float(np.std(group_rewards)) + eps
            )
            all_advantages.extend(group_advantages.tolist())
        return all_advantages

    def blend_advantages(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> None:
        """Blend real advantages with estimated reward advantages per-group if stddev > threshold."""
        if not (
            self.is_confidence_trained()
            and self.config.use_confidence_reward
            and (self.state.global_step >= self.config.confidence_loss_warmup_steps)
        ):
            return
        mean_estimated_rewards = convert_tensor_to_list(
            inputs["mean_estimated_rewards"]
        )
        num_generations = self.get_num_generations()
        num_groups = len(mean_estimated_rewards) // num_generations
        percentage = self.config.confidence_reward_percentage
        blended_advantages = inputs["advantages"].clone()
        estimated_advantages = self.compute_advantages_from_rewards(
            mean_estimated_rewards, num_generations
        )
        estimated_advantages_tensor = torch.tensor(
            estimated_advantages, device=inputs["advantages"].device
        )
        for group_idx in range(num_groups):
            start_idx = group_idx * num_generations
            end_idx = start_idx + num_generations
            group_estimated_rewards = mean_estimated_rewards[start_idx:end_idx]
            group_std = (
                statistics.stdev(group_estimated_rewards)
                if len(group_estimated_rewards) > 1
                else 0.0
            )
            if group_std > self.config.minimum_confidence_std:
                blended_advantages[start_idx:end_idx] = (1 - percentage) * inputs[
                    "advantages"
                ][start_idx:end_idx] + percentage * estimated_advantages_tensor[
                    start_idx:end_idx
                ]
        inputs["advantages"] = blended_advantages

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,  # noqa: FBT001, FBT002
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the combined GRPO loss and confidence loss."""
        indices = list(map(int, convert_tensor_to_list(inputs["indices"])))
        ordered_inputs = {
            key: value[indices] if value.ndim > 0 else value
            for key, value in inputs.items()
        }
        self.blend_advantages(ordered_inputs)
        self.register_hook(unwrap_model=False)
        grpo_loss = typing.cast(
            "torch.Tensor",
            super().compute_loss(
                model,
                ordered_inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            ),
        )
        confidence_loss = self.confidence_loss_factor * self.get_confidence_loss(
            ordered_inputs
        )
        self.remove_hook()
        total_loss = grpo_loss + confidence_loss
        self.add_metrics(get_grpo_namespace(), {(get_loss_name(),): grpo_loss.item()})
        self.add_metrics(
            get_confidence_namespace(), {(get_loss_name(),): confidence_loss.item()}
        )
        self.add_metrics(get_total_namespace(), {(get_loss_name(),): total_loss.item()})
        self.answers = []
        return total_loss

    def add_metrics(
        self,
        namespace: str,
        metrics: dict[tuple[str, ...], float],
        sep: str = get_default_metric_separator(),
    ) -> None:
        """Add metrics with the appropriate mode and namespace prefix."""
        mode = get_mode(typing.cast("torch.nn.Module", self.model))
        for key, value in metrics.items():
            self._metrics[mode][sep.join((namespace, *key))].append(value)

    def clear_eval_inputs_and_outputs(self) -> None:
        """Clear evaluation inputs and outputs."""
        self.eval_binary_classification_metrics_inputs.clear()
        self.eval_binary_classification_metrics_outputs.clear()
        self.eval_answer_metrics_inputs.clear()
        self.eval_answer_metrics_outputs.clear()

    def is_confidence_trained(self) -> bool:
        """Check if the confidence loss is being used for training."""
        return self.confidence_loss_factor > 0.0

    def evaluate(
        self,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        """Evaluate the model and return evaluation metrics."""
        self.save_model_to_eval_folder(metric_key_prefix)
        self.load_model_from_eval_folder(metric_key_prefix)
        self.eval_mode = True
        eval_output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        self.eval_mode = False
        self.merge_eval_metrics(metric_key_prefix)
        self.clear_eval_inputs_and_outputs()
        return eval_output

    @staticmethod
    def get_model_output_dir(model_identifier: str) -> pathlib.Path:
        """Get the model output directory for saving/loading."""
        config = get_config()
        shorthand = config.get_config_shorthand()
        standardized_hf_model_id = (
            config.hf_model_id.replace("/", "_")
            .replace("-", "_")
            .replace(".", "_")
            .lower()
        )
        model_output_dir = (
            get_evaluation_metric_dir()
            / f"{standardized_hf_model_id}_{model_identifier}_{shorthand}"
        )
        return model_output_dir

    def save_model_to_eval_folder(
        self,
        model_identifier: str,
    ) -> None:
        """Save the model to disk."""
        model_output_dir = self.get_model_output_dir(model_identifier)
        shutil.rmtree(model_output_dir, ignore_errors=True)
        model_output_dir.mkdir(parents=True, exist_ok=True)
        self.save_model(str(model_output_dir))

    def load_model_from_eval_folder(self, model_identifier: str) -> None:
        """Load the model from disk."""
        model_output_dir = self.get_model_output_dir(model_identifier)
        config = get_config()
        if config.enable_lora:
            peft_model = typing.cast("PeftModel", self.model)
            peft_model.load_adapter(
                str(model_output_dir), adapter_name=model_identifier
            )
            peft_model.set_adapter(model_identifier)
        else:
            self.model = typing.cast("PreTrainedModel", self.model).from_pretrained(
                model_output_dir
            )

    def merge_eval_metrics(self, metric_key_prefix: str) -> None:
        """Merge evaluation metrics from multiple evaluation runs."""
        concatenated_eval_binary_classification_metrics_inputs = tuple(
            list(chain.from_iterable(x))
            for x in zip(*self.eval_binary_classification_metrics_inputs, strict=True)
        )
        concatenated_eval_binary_classification_metrics_outputs = change_metric_keys(
            compute_binary_classification_metrics(
                *concatenated_eval_binary_classification_metrics_inputs  # type: ignore[arg-type]
            ),
            prefix=(metric_key_prefix, get_confidence_namespace()),
            postfix=(),
        )
        aggregated_eval_binary_classification_metrics_outputs = change_metric_keys(
            aggregate_metrics(self.eval_binary_classification_metrics_outputs),
            prefix=(metric_key_prefix, get_confidence_namespace()),
            postfix=(),
        )
        answers_with_confidence, temperatures, num_generations_list = zip(
            *self.eval_answer_metrics_inputs, strict=True
        )
        if len(set(temperatures)) != 1:
            raise ValueError
        if len(set(num_generations_list)) != 1:
            raise ValueError
        temperature = temperatures[0]
        num_generations = num_generations_list[0]
        concatenated_eval_answer_metrics_inputs = (
            list(chain.from_iterable(answers_with_confidence)),
            temperature,
            num_generations,
        )
        concatenated_eval_answer_metrics_outputs = change_metric_keys(
            compute_answer_metrics(*concatenated_eval_answer_metrics_inputs),
            prefix=(metric_key_prefix, get_answer_namespace()),
            postfix=(),
        )
        aggregated_eval_answer_metrics_outputs = change_metric_keys(
            aggregate_metrics(self.eval_answer_metrics_outputs),
            prefix=(metric_key_prefix, get_answer_namespace()),
            postfix=(),
        )
        concatenated_eval_binary_classification_metrics_df = get_df_from_metrics(
            concatenated_eval_binary_classification_metrics_outputs
        )
        store_eval_df(
            get_eval_metrics_df_name(
                metric_key_prefix, is_aggregated=False, is_bc=True
            ),
            concatenated_eval_binary_classification_metrics_df,
        )
        aggregated_eval_binary_classification_metrics_df = get_df_from_metrics(
            aggregated_eval_binary_classification_metrics_outputs
        )
        store_eval_df(
            get_eval_metrics_df_name(metric_key_prefix, is_aggregated=True, is_bc=True),
            aggregated_eval_binary_classification_metrics_df,
        )
        concatenated_eval_answer_metrics_df = get_df_from_metrics(
            concatenated_eval_answer_metrics_outputs
        )
        store_eval_df(
            get_eval_metrics_df_name(
                metric_key_prefix, is_aggregated=False, is_bc=False
            ),
            concatenated_eval_answer_metrics_df,
        )
        aggregated_eval_answer_metrics_df = get_df_from_metrics(
            aggregated_eval_answer_metrics_outputs
        )
        store_eval_df(
            get_eval_metrics_df_name(
                metric_key_prefix, is_aggregated=True, is_bc=False
            ),
            aggregated_eval_answer_metrics_df,
        )
