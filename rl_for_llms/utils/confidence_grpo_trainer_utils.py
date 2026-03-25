import pathlib
import shutil
import statistics
import typing
from functools import partial, update_wrapper
from itertools import chain

import numpy as np
import torch
from datasets import Dataset
from torch import Tensor
from torch.nn import Module, functional
from torch.utils.hooks import RemovableHandle
from trl.trainer.grpo_trainer import GRPOTrainer

from rl_for_llms.models.answer import (
    Answer,
    get_answers_with_confidence,
)
from rl_for_llms.models.config import Config
from rl_for_llms.utils.classification_utils import compute_binary_classification_metrics
from rl_for_llms.utils.confidence_utils import get_confidence_token_logit_sigmoid
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
from rl_for_llms.utils.group_utils import iter_groups
from rl_for_llms.utils.llm_utils import get_confidence_token_id, load_checkpoint_weights
from rl_for_llms.utils.path_utils import (
    delete_csv_files_in_evaluation_metric_dir,
    get_evaluation_metric_dir,
    standardize_model_id,
)
from rl_for_llms.utils.reward_utils import get_class_weights_for_single_group
from rl_for_llms.utils.torch_utils import convert_tensor_to_list, get_mode


class ConfidenceGRPOTrainer(GRPOTrainer):
    """GRPO trainer with auxiliary confidence loss."""

    def __init__(self, config: Config, **kwargs: typing.Any) -> None:  # noqa: ANN401
        """Initialize the ConfidenceGRPOTrainer."""
        if len(kwargs["reward_funcs"]) < 1:
            raise ValueError
        wrapped_reward_funcs = []
        for reward_func in kwargs["reward_funcs"]:
            wrapped_reward_func = partial(reward_func, trainer=self)
            update_wrapper(wrapped_reward_func, reward_func)
            wrapped_reward_funcs.append(wrapped_reward_func)
        kwargs["reward_funcs"] = wrapped_reward_funcs
        super().__init__(**kwargs)
        self.config = config
        self.confidence_token_id: int = get_confidence_token_id(self.config)
        self.confidence_loss_factor: float = (
            self.config.confidence_loss_factor
            if self.config.use_confidence_loss
            else 0.0
        )
        self.answers: list[Answer] = []
        self.lm_head_attribute_name: str = self.config.lm_head_attribute_name
        self.hook_handle: RemovableHandle | None = None
        self.all_confidence_logits: Tensor | None = None
        self.all_confidence_logits_excluding_last: Tensor | None = None
        self.eval_mode: bool = False
        self.eval_inputs: dict[str, list[typing.Any]] = {"bc": [], "answer": []}
        self.eval_outputs: dict[str, list[dict[tuple[str, ...], float]]] = {
            "bc": [],
            "answer": [],
        }

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

    def register_hook(
        self, *, unwrap_model: bool, clear_confidence_logits: bool = True
    ) -> None:
        """Register a forward hook to capture confidence logits."""
        self.remove_hook(clear_confidence_logits=clear_confidence_logits)
        lm_head = self.get_lm_head(unwrap_model=unwrap_model)
        self.hook_handle = lm_head.register_forward_hook(self.logits_hook)

    def clear_confidence_logits(self) -> None:
        """Clear confidence logits."""
        self.all_confidence_logits = None
        self.all_confidence_logits_excluding_last = None

    def remove_hook(self, *, clear_confidence_logits: bool = True) -> None:
        """Remove the registered hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
        if clear_confidence_logits:
            self.clear_confidence_logits()

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
        elif confidence_logits.shape[1] == 1:
            self.all_confidence_logits = torch.cat(
                (self.all_confidence_logits, confidence_logits), dim=1
            )
        self.all_confidence_logits_excluding_last = self.all_confidence_logits[:, :-1]

    def get_num_generations(self) -> int:
        """Return the number of generations per prompt."""
        return typing.cast("int", self.num_generations)

    def is_confidence_trained(self) -> bool:
        """Check if the confidence loss is being used."""
        return self.confidence_loss_factor > 0.0

    def _get_masked_estimated_rewards(
        self, completion_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            confidence_logits, self.config
        ).float()
        return estimated_rewards, mask, sequence_lengths

    def _compute_mean_estimated_rewards(
        self, completion_mask: torch.Tensor
    ) -> list[float]:
        estimated_rewards, mask, sequence_lengths = self._get_masked_estimated_rewards(
            completion_mask
        )
        sum_estimated_rewards = (estimated_rewards * mask).sum(dim=-1)
        mean_estimated_rewards = sum_estimated_rewards / sequence_lengths
        return convert_tensor_to_list(mean_estimated_rewards)

    def _compute_sample_weights(
        self, rewards: list[float], batch_size: int
    ) -> list[float]:
        num_generations = self.get_num_generations()
        sample_weights = [1.0] * batch_size
        for start, _, group_rewards in iter_groups(rewards, num_generations):
            if len(group_rewards) == num_generations:
                incorrect_weight, correct_weight = get_class_weights_for_single_group(
                    group_rewards
                )
                for i, reward in enumerate(group_rewards):
                    sample_weights[start + i] = (
                        correct_weight if reward == 1.0 else incorrect_weight
                    )
        return sample_weights

    def _log_ordered_metrics(
        self,
        rewards: list[float],
        mean_estimated_rewards: list[float],
        answers: list[Answer],
    ) -> None:
        confidences = (
            mean_estimated_rewards
            if self.is_confidence_trained()
            else [get_default_confidence_score()] * len(answers)
        )
        answers_with_confidence = get_answers_with_confidence(answers, confidences)

        bc_metrics = compute_binary_classification_metrics(
            rewards, mean_estimated_rewards
        )
        self.add_metrics(get_confidence_namespace(), bc_metrics)

        num_generations = self.get_num_generations()
        answer_metrics = compute_answer_metrics(
            answers_with_confidence, self.config.temperature, num_generations
        )
        self.add_metrics(get_answer_namespace(), answer_metrics)

        if self.eval_mode:
            self.eval_inputs["bc"].append((rewards, mean_estimated_rewards))
            self.eval_outputs["bc"].append(bc_metrics)
            self.eval_inputs["answer"].append(
                (answers_with_confidence, self.config.temperature, num_generations)
            )
            self.eval_outputs["answer"].append(answer_metrics)

    @staticmethod
    def _compute_advantages_from_rewards(
        rewards: list[float],
        num_generations: int,
        eps: float = get_default_eps(),
    ) -> list[float]:
        all_advantages: list[float] = []
        for _, _, group in iter_groups(rewards, num_generations):
            group_np = np.array(group)
            group_advantages = (group_np - np.mean(group_np)) / (
                float(np.std(group_np)) + eps
            )
            all_advantages.extend(group_advantages.tolist())
        return all_advantages

    def _get_total_samples_processed(self) -> int:
        """Return the total number of samples processed based on the global step and config."""
        return (
            self.state.global_step
            * self.config.gradient_accumulation_steps
            * self.config.per_device_rollouts_per_batch
        )

    def _is_reasoning_warmup_complete(self) -> bool:
        """Check if the reasoning warmup phase is complete."""
        return (
            self._get_total_samples_processed() >= self.config.reasoning_warmup_samples
        )

    def _is_confidence_warmup_complete(self) -> bool:
        """Check if the confidence warmup phase is complete."""
        return (
            self._get_total_samples_processed()
            >= self.config.reasoning_warmup_samples
            + self.config.confidence_warmup_samples
        )

    def _blend_advantages(
        self,
        advantages: torch.Tensor,
        mean_estimated_rewards: list[float],
    ) -> torch.Tensor:
        if not (
            self.is_confidence_trained()
            and self.config.use_confidence_reward
            and self._is_confidence_warmup_complete()
        ):
            return advantages
        num_generations = self.get_num_generations()
        percentage = self.config.confidence_reward_percentage
        blended = advantages.clone()
        for start, _, group_estimated in iter_groups(
            mean_estimated_rewards, num_generations
        ):
            group_std = (
                statistics.stdev(group_estimated) if len(group_estimated) > 1 else 0.0
            )
            if group_std > self.config.minimum_confidence_std:
                group_est_advantages = self._compute_advantages_from_rewards(
                    group_estimated, num_generations
                )
                for i in range(num_generations):
                    idx = start + i
                    blended[idx] = (1 - percentage) * advantages[
                        idx
                    ] + percentage * group_est_advantages[i]
        return blended

    def _generate_and_score_completions(
        self, inputs: list[dict[str, torch.Tensor | typing.Any]]
    ) -> dict[str, torch.Tensor | typing.Any]:
        self.answers = []
        self.register_hook(unwrap_model=True)
        output = super()._generate_and_score_completions(inputs)
        device = output["advantages"].device
        mask = output["completion_mask"].bool()

        mean_estimated_rewards = self._compute_mean_estimated_rewards(mask)
        rewards = [answer.reward for answer in self.answers]

        output["mean_estimated_rewards"] = torch.tensor(
            mean_estimated_rewards, device=device
        )
        output["rewards"] = torch.tensor(rewards, device=device)
        output["sample_weights"] = torch.tensor(
            self._compute_sample_weights(rewards, len(self.answers)), device=device
        )

        output["advantages"] = self._blend_advantages(
            output["advantages"], mean_estimated_rewards
        )

        self._log_ordered_metrics(rewards, mean_estimated_rewards, self.answers)
        self.remove_hook()
        return output

    def _compute_confidence_loss(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.all_confidence_logits_excluding_last is None:
            raise ValueError
        estimated_rewards = get_confidence_token_logit_sigmoid(
            self.all_confidence_logits_excluding_last, self.config
        ).float()
        mask = inputs["completion_mask"].bool()
        sequence_lengths = mask.sum(dim=-1)
        real_rewards = (
            inputs["rewards"]
            .unsqueeze(-1)
            .expand(-1, estimated_rewards.shape[-1])
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
        per_rollout_loss_weighted = (
            per_sample_loss_masked.sum(dim=-1) / sequence_lengths
        ) * inputs["sample_weights"]
        return per_rollout_loss_weighted.mean()

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,  # noqa: FBT001, FBT002
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        """Compute the combined GRPO loss and confidence loss."""
        self.register_hook(unwrap_model=False)
        grpo_loss = typing.cast(
            "torch.Tensor",
            super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            ),
        )
        confidence_loss = (
            self.confidence_loss_factor if self._is_reasoning_warmup_complete() else 0.0
        ) * self._compute_confidence_loss(inputs)
        self.remove_hook()
        total_loss = grpo_loss + confidence_loss
        self.add_metrics(get_grpo_namespace(), {(get_loss_name(),): grpo_loss.item()})
        self.add_metrics(
            get_confidence_namespace(), {(get_loss_name(),): confidence_loss.item()}
        )
        self.add_metrics(get_total_namespace(), {(get_loss_name(),): total_loss.item()})
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
        for key in self.eval_inputs:
            self.eval_inputs[key].clear()
            self.eval_outputs[key].clear()

    def evaluate(
        self,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        """Evaluate the model and return evaluation metrics."""
        delete_csv_files_in_evaluation_metric_dir(self.config.started_at)
        self._save_checkpoint_to_disk(metric_key_prefix)
        self.eval_mode = True
        eval_output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        self.eval_mode = False
        self._merge_eval_metrics(metric_key_prefix)
        self.clear_eval_inputs_and_outputs()
        return eval_output

    def get_config_shorthand(self) -> str:
        """Get a shorthand representation of the config."""
        if self.state.global_step == 0:
            return "base"
        return self.config.get_config_shorthand()

    def _get_model_output_dir(self, model_identifier: str) -> pathlib.Path:
        shorthand = self.get_config_shorthand()
        standardized_hf_model_id = standardize_model_id(self.config.hf_model_id)
        return (
            get_evaluation_metric_dir()
            / f"{standardized_hf_model_id}_{model_identifier}_{shorthand}"
        )

    def _save_checkpoint_to_disk(self, model_identifier: str) -> None:
        model_output_dir = self._get_model_output_dir(model_identifier)
        shutil.rmtree(model_output_dir, ignore_errors=True)
        model_output_dir.mkdir(parents=True, exist_ok=True)
        self.save_model(str(model_output_dir))

    def load_checkpoint_from_disk(self, model_identifier: str) -> None:
        """Load previously saved weights into the model (adapter or full)."""
        model_output_dir = self._get_model_output_dir(model_identifier)
        if not model_output_dir.exists():
            raise FileNotFoundError(model_output_dir)
        model = self.accelerator.unwrap_model(
            getattr(self, "model_wrapped", self.model)
        )
        load_checkpoint_weights(model, model_output_dir)

    def _merge_eval_metrics(self, metric_key_prefix: str) -> None:
        bc_concat, bc_agg = self._compute_eval_bc_metrics(metric_key_prefix)
        answer_concat, answer_agg = self._compute_eval_answer_metrics(metric_key_prefix)
        shorthand = self.get_config_shorthand()
        metric_bundles: list[tuple[bool, bool, dict[tuple[str, ...], float]]] = [
            (False, True, bc_concat),
            (True, True, bc_agg),
            (False, False, answer_concat),
            (True, False, answer_agg),
        ]
        for is_aggregated, is_bc, data in metric_bundles:
            store_eval_df(
                get_eval_metrics_df_name(
                    metric_key_prefix, is_aggregated=is_aggregated, is_bc=is_bc
                ),
                get_df_from_metrics(data),
                shorthand,
            )

    def _compute_eval_metric_pair(
        self,
        prefix: str,
        namespace: str,
        recomputed: dict[tuple[str, ...], float],
        outputs_key: str,
    ) -> tuple[dict[tuple[str, ...], float], dict[tuple[str, ...], float]]:
        ns = (prefix, namespace)
        concatenated = change_metric_keys(recomputed, prefix=ns)
        aggregated = change_metric_keys(
            aggregate_metrics(self.eval_outputs[outputs_key]), prefix=ns
        )
        return concatenated, aggregated

    def _compute_eval_bc_metrics(
        self, prefix: str
    ) -> tuple[dict[tuple[str, ...], float], dict[tuple[str, ...], float]]:
        concatenated_inputs = tuple(
            list(chain.from_iterable(x))
            for x in zip(*self.eval_inputs["bc"], strict=True)
        )
        recomputed = compute_binary_classification_metrics(
            *concatenated_inputs  # type: ignore[arg-type]
        )
        return self._compute_eval_metric_pair(
            prefix, get_confidence_namespace(), recomputed, "bc"
        )

    def _compute_eval_answer_metrics(
        self, prefix: str
    ) -> tuple[dict[tuple[str, ...], float], dict[tuple[str, ...], float]]:
        answers_with_confidence, temperatures, num_generations_list = zip(
            *self.eval_inputs["answer"], strict=True
        )
        if len(set(temperatures)) != 1:
            raise ValueError
        if len(set(num_generations_list)) != 1:
            raise ValueError
        recomputed = compute_answer_metrics(
            list(chain.from_iterable(answers_with_confidence)),
            temperatures[0],
            num_generations_list[0],
        )
        return self._compute_eval_metric_pair(
            prefix, get_answer_namespace(), recomputed, "answer"
        )
