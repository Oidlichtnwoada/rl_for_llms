import pathlib
import random
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
from transformers import LogitsProcessor
from transformers.cache_utils import CacheLayerMixin, DynamicCache
from transformers.generation.logits_process import LogitsProcessorList
from trl.trainer.grpo_trainer import GRPOTrainer

from rl_for_llms.models.aggregation_strategy import AggregationStrategy
from rl_for_llms.models.answer import (
    Answer,
    get_answers_with_confidence,
)
from rl_for_llms.models.config import Config
from rl_for_llms.models.method import Method
from rl_for_llms.models.variant import Variant
from rl_for_llms.utils.aggregation_utils import (
    get_exponentially_decreasing_aggregation_weights,
    get_exponentially_increasing_aggregation_weights,
    get_mean_aggregation_weights,
)
from rl_for_llms.utils.chart_utils import get_variant_method_shorthand
from rl_for_llms.utils.classification_utils import compute_binary_classification_metrics
from rl_for_llms.utils.confidence_utils import (
    compute_laser_self_rewarding_score,
    get_confidence_token_logit_sigmoid,
)
from rl_for_llms.utils.constant_utils import (
    get_answer_namespace,
    get_confidence_namespace,
    get_default_confidence_score,
    get_default_eps,
    get_default_metric_separator,
    get_filter_name,
    get_grpo_namespace,
    get_loss_name,
    get_no_name,
    get_tempmod_name,
    get_total_namespace,
)
from rl_for_llms.utils.evaluation_utils import (
    aggregate_metrics,
    change_metric_keys,
    compute_answer_metrics,
    compute_mean_std_metrics,
    get_df_from_metrics,
    get_eval_metrics_df_name,
    get_variant_checkpoint_dir,
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
        self._laser_full_logits: Tensor | None = None
        self.eval_mode: bool = False
        self.use_tempmod: bool = False
        self.use_filtering: bool = False
        self._loaded_variant: Variant | None = None
        self._loaded_method: Method | None = None
        self._smc_current_cache: DynamicCache | None = None
        self._smc_cache_hook_handle: RemovableHandle | None = None
        self.eval_inputs: dict[str, list[typing.Any]] = {"bc": [], "answer": []}
        self.eval_outputs: dict[str, list[dict[tuple[str, ...], float]]] = {
            "bc": [],
            "answer": [],
        }
        self._eval_run_results: list[
            tuple[dict[tuple[str, ...], float], dict[tuple[str, ...], float]]
        ] = []

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
        if self.use_filtering and self.eval_mode and unwrap_model:
            wrapped = getattr(self, "model_wrapped", self.model)
            model = self.accelerator.unwrap_model(wrapped)
            self._smc_cache_hook_handle = model.register_forward_pre_hook(
                self._capture_cache_pre_hook, with_kwargs=True
            )

    def clear_confidence_logits(self) -> None:
        """Clear confidence logits."""
        self.all_confidence_logits = None
        self.all_confidence_logits_excluding_last = None
        self._laser_full_logits = None

    def remove_hook(self, *, clear_confidence_logits: bool = True) -> None:
        """Remove the registered hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
        if self._smc_cache_hook_handle is not None:
            self._smc_cache_hook_handle.remove()
            self._smc_cache_hook_handle = None
        if clear_confidence_logits:
            self.clear_confidence_logits()
            self._smc_current_cache = None

    def _capture_cache_pre_hook(
        self,
        module: Module,  # noqa: ARG002
        args: tuple[typing.Any, ...],  # noqa: ARG002
        kwargs: dict[str, typing.Any],
    ) -> tuple[tuple[typing.Any, ...], dict[str, typing.Any]] | None:
        """Capture a reference to the DynamicCache before each decode step for SMC filtering."""
        cache = kwargs.get("past_key_values")
        if isinstance(cache, DynamicCache):
            self._smc_current_cache = cache
        return None

    def logits_hook(
        self,
        module: Module,  # noqa: ARG002
        inputs: tuple[Tensor, ...],  # noqa: ARG002
        output: torch.Tensor,
    ) -> Tensor | None:
        """Capture confidence logits during the forward pass."""
        logits = output
        if logits.dim() != 3:  # noqa: PLR2004
            raise ValueError
        match self.config.method:
            case Method.DENSE:
                return self._dense_logits_hook(logits)
            case Method.LASER:
                return self._laser_logits_hook(logits)

    def _apply_temperature_modulation(
        self,
        logits: Tensor,
        confidence_logits: Tensor,
        min_temperature: float = 1e-6,
    ) -> Tensor:
        """Return logits scaled by per-sequence confidence-derived temperature, pre-compensating for any base config.temperature applied downstream by transformers."""
        conf = get_confidence_token_logit_sigmoid(
            confidence_logits[:, -1], self.config
        ).float()
        t_min = self.config.temperature_modulation_min_temperature
        t_max = self.config.temperature_modulation_max_temperature
        weight = (
            conf if self.config.temperature_modulation_invert_mapping else (1.0 - conf)
        )
        temps = t_min + weight * (t_max - t_min)
        temps = temps / self.config.temperature
        temps = temps.clamp(min=min_temperature)
        return logits / temps.view(-1, 1, 1)

    def _dense_logits_hook(self, logits: Tensor) -> Tensor | None:
        """Capture raw confidence logit at every position for Dense."""
        confidence_logits = logits[:, :, self.confidence_token_id]
        self._accumulate_confidence_values(confidence_logits)
        if self.use_tempmod and self.eval_mode and logits.size(1) == 1:
            return self._apply_temperature_modulation(logits, confidence_logits)
        return None

    def _smc_pick_replacement(self, alive: list[int], confidence: Tensor) -> int:
        """Stochastically pick a replacement alive index, weighted proportionally to confidence scores."""
        if not alive:
            raise ValueError
        if len(alive) == 1:
            return alive[0]
        weights = [confidence[i].item() for i in alive]
        total_weight = sum(weights)
        if total_weight <= 0.0:
            chosen: list[int] = random.choices(alive, k=1)  # noqa: S311
            return chosen[0]
        chosen = random.choices(alive, weights=weights, k=1)  # noqa: S311
        return chosen[0]

    def _smc_copy_sequence(
        self, dead_i: int, alive_i: int, scores: Tensor, input_ids: Tensor
    ) -> None:
        """Copy state from alive_i to dead_i in-place."""
        scores[dead_i].copy_(scores[alive_i])
        input_ids[dead_i].copy_(input_ids[alive_i])

        if self.all_confidence_logits is not None:
            self.all_confidence_logits[dead_i].copy_(
                self.all_confidence_logits[alive_i]
            )

        if self._smc_current_cache is not None:
            for layer in self._smc_current_cache.layers:
                if (
                    isinstance(layer, CacheLayerMixin)
                    and layer.keys is not None
                    and layer.values is not None
                ):
                    layer.keys[dead_i].copy_(layer.keys[alive_i])
                    layer.values[dead_i].copy_(layer.values[alive_i])

    def _smc_resample_group(
        self,
        group_indices: list[int],
        confidence: Tensor,
        scores: Tensor,
        input_ids: Tensor,
        finished: list[bool],
    ) -> None:
        """Resample dead sequences within a group using confidence-weighted stochastic replacement."""
        threshold = self.config.filtering_threshold
        dead = [
            i
            for i in group_indices
            if not finished[i] and confidence[i].item() < threshold
        ]
        alive = [
            i
            for i in group_indices
            if not finished[i] and confidence[i].item() >= threshold
        ]

        if not dead:
            return
        if not alive:
            unfinished = [i for i in group_indices if not finished[i]]
            if not unfinished:
                return
            best = max(unfinished, key=lambda i: confidence[i].item())
            alive = [best]
            dead = [i for i in unfinished if i != best]

        for dead_i in dead:
            alive_i = self._smc_pick_replacement(alive, confidence)
            self._smc_copy_sequence(dead_i, alive_i, scores, input_ids)

    def apply_smc_filtering(
        self, scores: Tensor, input_ids: Tensor, finished_mask: Tensor
    ) -> None:
        """Resample low-confidence sequences within each group using SMC particle filtering."""
        if self.all_confidence_logits is None:
            return

        confidence = (
            get_confidence_token_logit_sigmoid(
                self.all_confidence_logits[:, -1], self.config
            )
            .float()
            .detach()
        )
        num_gen = self.get_num_generations()
        batch_size = scores.size(0)
        finished = finished_mask.tolist()

        for _, _, group in iter_groups(list(range(batch_size)), num_gen):
            self._smc_resample_group(group, confidence, scores, input_ids, finished)

    def _laser_logits_hook(self, logits: Tensor) -> Tensor | None:
        """Capture full logits for LaSeR (consumed by ``_get_laser_post_response_scores``)."""
        self._laser_full_logits = logits
        if self.use_tempmod and self.eval_mode and logits.size(1) == 1:
            raise ValueError
        return None

    def _accumulate_confidence_values(self, confidence_values: Tensor) -> None:
        """Accumulate per-position confidence values across decoding steps."""
        if self.all_confidence_logits is None:
            self.all_confidence_logits = confidence_values
        elif confidence_values.shape[1] == 1:
            self.all_confidence_logits = torch.cat(
                (self.all_confidence_logits, confidence_values), dim=1
            )
        self.all_confidence_logits_excluding_last = self.all_confidence_logits[:, :-1]

    def get_num_generations(self) -> int:
        """Return the number of generations per prompt."""
        return typing.cast("int", self.num_generations)

    def is_confidence_trained(self) -> bool:
        """Check if the confidence loss is being used."""
        return self.confidence_loss_factor > 0.0

    def _get_aligned_confidence_values(self, mask: torch.Tensor) -> torch.Tensor:
        """Return stored confidence values aligned with the given mask shape."""
        if (
            self.all_confidence_logits is None
            or self.all_confidence_logits_excluding_last is None
        ):
            raise ValueError
        if self.all_confidence_logits.shape[1] == mask.shape[1]:
            return self.all_confidence_logits
        if self.all_confidence_logits_excluding_last.shape[1] == mask.shape[1]:
            return self.all_confidence_logits_excluding_last
        raise ValueError

    @staticmethod
    def _extract_post_response_values(
        values: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Extract values at the first position after the last non-padding token per sequence."""
        post_response_positions = values.shape[1] - mask.shape[1] + mask.sum(dim=-1) - 1
        batch_indices = torch.arange(values.shape[0], device=values.device)
        return values[batch_indices, post_response_positions]

    def _get_masked_estimated_rewards(
        self, completion_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = completion_mask.bool()
        sequence_lengths = mask.sum(dim=-1)
        confidence_logits = self._get_aligned_confidence_values(mask)
        estimated_rewards = get_confidence_token_logit_sigmoid(
            confidence_logits, self.config
        ).float()
        return estimated_rewards, mask, sequence_lengths

    def _compute_mean_estimated_rewards(
        self, completion_mask: torch.Tensor
    ) -> list[float]:
        match self.config.method:
            case Method.DENSE:
                return self._compute_dense_aggregated_rewards(completion_mask)
            case Method.LASER:
                return self._compute_laser_mean_scores(completion_mask)

    def _compute_dense_aggregated_rewards(
        self, completion_mask: torch.Tensor
    ) -> list[float]:
        """Compute per-sequence aggregated confidence for Dense (sigmoid over all positions)."""
        estimated_rewards, mask, _ = self._get_masked_estimated_rewards(completion_mask)
        aggregation_weights = self._get_confidence_aggregation_weights(mask)
        aggregated_rewards = (estimated_rewards * aggregation_weights).sum(dim=-1)
        return convert_tensor_to_list(aggregated_rewards)

    def _get_laser_post_response_scores(self, mask: torch.Tensor) -> torch.Tensor:
        """Return per-sequence LaSeR self-rewarding scores at the post-EOS position."""
        if self._laser_full_logits is None:
            raise ValueError
        post_response_logits = self._extract_post_response_values(
            self._laser_full_logits, mask
        )
        return compute_laser_self_rewarding_score(
            post_response_logits, self.confidence_token_id, self.config
        )

    def _compute_laser_mean_scores(self, completion_mask: torch.Tensor) -> list[float]:
        """Compute per-sequence self-rewarding score for LaSeR (last position only)."""
        mask = completion_mask.bool()
        scores = self._get_laser_post_response_scores(mask)
        return convert_tensor_to_list(scores)

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

    def _run_post_generation_forward_for_laser(
        self, output: dict[str, torch.Tensor | typing.Any]
    ) -> None:
        """Run an extra teacher-forced forward pass to capture post-EOS logits for LaSeR."""
        prompt_ids = output["prompt_ids"]
        prompt_mask = output["prompt_mask"]
        completion_ids = output["completion_ids"]
        completion_mask = output["completion_mask"]

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        self.register_hook(unwrap_model=True, clear_confidence_logits=True)
        with torch.no_grad():
            self._get_per_token_logps_and_entropies(
                self.model,
                input_ids,
                attention_mask,
                logits_to_keep,
                batch_size=input_ids.size(0),
            )

    def _generate_single_turn(
        self,
        prompt_ids: list[list[int]],
        images: list[typing.Any] | None,
        multimodal_fields: dict[str, typing.Any],
    ) -> tuple[list[list[int]], list[list[float]] | None]:
        """Override to dynamically inject SMC LogitsProcessor during generation."""
        unwrapped_model = self.accelerator.unwrap_model(
            getattr(self, "model_wrapped", self.model)
        )
        original_generate = unwrapped_model.generate

        def patched_generate(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:  # noqa: ANN401
            if self.use_filtering and self.eval_mode:
                proc_list = kwargs.get("logits_processor", LogitsProcessorList())
                if not any(isinstance(p, SMCLogitsProcessor) for p in proc_list):
                    input_ids_kwarg = kwargs.get("input_ids")
                    prompt_len = (
                        input_ids_kwarg.size(1)
                        if isinstance(input_ids_kwarg, torch.Tensor)
                        else len(prompt_ids[0])
                    )
                    proc_list.append(SMCLogitsProcessor(self, prompt_len))
                kwargs["logits_processor"] = proc_list
            return original_generate(*args, **kwargs)

        try:
            unwrapped_model.generate = patched_generate
            result = super()._generate_single_turn(
                prompt_ids, images, multimodal_fields
            )  # type: ignore[no-untyped-call]
            return typing.cast(
                "tuple[list[list[int]], list[list[float]] | None]", result
            )
        finally:
            unwrapped_model.generate = original_generate

    def _generate_and_score_completions(
        self, inputs: list[dict[str, torch.Tensor | typing.Any]]
    ) -> dict[str, torch.Tensor | typing.Any]:
        self.answers = []
        self.register_hook(unwrap_model=True)
        output = super()._generate_and_score_completions(inputs)
        device = output["advantages"].device
        mask = output["completion_mask"].bool()

        if self.config.method == Method.LASER:
            self._run_post_generation_forward_for_laser(output)

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
        match self.config.method:
            case Method.DENSE:
                return self._compute_dense_confidence_loss(inputs)
            case Method.LASER:
                return self._compute_laser_confidence_loss(inputs)

    def _get_aggregation_weights(self, mask: torch.Tensor) -> torch.Tensor:
        """Return per-token aggregation weights based on the configured strategy."""
        base = self.config.exponential_weight_base
        match self.config.loss_aggregation_strategy:
            case AggregationStrategy.MEAN:
                return get_mean_aggregation_weights(mask)
            case AggregationStrategy.EXPONENTIALLY_INCREASING:
                return get_exponentially_increasing_aggregation_weights(mask, base)
            case AggregationStrategy.EXPONENTIALLY_DECREASING:
                return get_exponentially_decreasing_aggregation_weights(mask, base)

    def _get_confidence_aggregation_weights(self, mask: torch.Tensor) -> torch.Tensor:
        """Return per-token aggregation weights based on the confidence aggregation strategy."""
        base = self.config.exponential_weight_base
        match self.config.confidence_aggregation_strategy:
            case AggregationStrategy.MEAN:
                return get_mean_aggregation_weights(mask)
            case AggregationStrategy.EXPONENTIALLY_INCREASING:
                return get_exponentially_increasing_aggregation_weights(mask, base)
            case AggregationStrategy.EXPONENTIALLY_DECREASING:
                return get_exponentially_decreasing_aggregation_weights(mask, base)

    def _compute_dense_confidence_loss(
        self, inputs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute Dense BCE loss over all token positions."""
        if self.all_confidence_logits_excluding_last is None:
            raise ValueError
        estimated_rewards = get_confidence_token_logit_sigmoid(
            self.all_confidence_logits_excluding_last, self.config
        ).float()
        mask = inputs["completion_mask"].bool()
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
        aggregation_weights = self._get_aggregation_weights(mask)
        per_rollout_loss_weighted = (
            (per_sample_loss_masked * aggregation_weights).sum(dim=-1)
        ) * inputs["sample_weights"]
        return per_rollout_loss_weighted.mean()

    def _compute_laser_confidence_loss(
        self, inputs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute LaSeR MSE loss at the post-EOS position."""
        mask = inputs["completion_mask"].bool()
        scores = self._get_laser_post_response_scores(mask).float()
        rewards = inputs["rewards"].float()
        per_sample_loss = functional.mse_loss(scores, rewards, reduction="none")
        weighted_loss = per_sample_loss * inputs["sample_weights"]
        return weighted_loss.mean()

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

    def _validate_filtering_configuration(self) -> None:
        """Validate that inference-time filtering is only used on supported generation paths."""
        if not self.use_filtering:
            return
        if self.config.method != Method.DENSE:
            raise ValueError
        if self.use_vllm:
            raise ValueError
        if getattr(self, "use_transformers_paged", False):
            raise ValueError

    def evaluate(
        self,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
        *,
        use_tempmod: bool = False,
        use_filtering: bool = False,
    ) -> dict[str, float]:
        """Evaluate the model and return evaluation metrics."""
        self.use_tempmod = use_tempmod
        self.use_filtering = use_filtering
        self._validate_filtering_configuration()
        try:
            delete_csv_files_in_evaluation_metric_dir(self.config.started_at)
            self._save_checkpoint_to_disk(metric_key_prefix)
            for _ in range(self.config.num_eval_repetitions):
                self.eval_mode = True
                super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
                self.eval_mode = False
                bc_concat, _ = self._compute_eval_bc_metrics(metric_key_prefix)
                _, answer_agg = self._compute_eval_answer_metrics(metric_key_prefix)
                self._eval_run_results.append((bc_concat, answer_agg))
                self.clear_eval_inputs_and_outputs()
            self._merge_eval_metrics(metric_key_prefix)
            self._eval_run_results.clear()
        finally:
            self.eval_mode = False
            self.use_tempmod = False
            self.use_filtering = False
        return {}

    def get_config_shorthand(self) -> str:
        """Get a shorthand representation of the config."""
        if self._loaded_variant is not None and self._loaded_method is not None:
            return get_variant_method_shorthand(
                self._loaded_variant, self._loaded_method
            )
        if self.state.global_step == 0:
            return "base"
        return self.config.get_config_shorthand()

    def get_eval_shorthand(self) -> str:
        """Get a shorthand representation of the config for evaluation CSV files."""
        base = self.get_config_shorthand()
        tempmod_suffix = (
            get_tempmod_name()
            if self.use_tempmod
            else f"{get_no_name()}{get_tempmod_name()}"
        )
        filter_suffix = (
            get_filter_name()
            if self.use_filtering
            else f"{get_no_name()}{get_filter_name()}"
        )
        return f"{base}_{tempmod_suffix}_{filter_suffix}"

    def _load_weights_from_dir(self, checkpoint_dir: str | pathlib.Path) -> None:
        """Unwrap the model and load weights from the given checkpoint directory."""
        model = self.accelerator.unwrap_model(
            getattr(self, "model_wrapped", self.model)
        )
        load_checkpoint_weights(model, checkpoint_dir)

    def load_final_checkpoint(self, variant: Variant, method: Method) -> None:
        """Load model weights from the final evaluation directory for a given variant and method."""
        self._load_weights_from_dir(
            get_variant_checkpoint_dir(variant, self.config.hf_model_id, method)
        )
        self._loaded_variant = variant
        self._loaded_method = method

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
        self._load_weights_from_dir(model_output_dir)

    def _merge_eval_metrics(self, metric_key_prefix: str) -> None:
        bc_concat_runs = [r[0] for r in self._eval_run_results]
        answer_agg_runs = [r[1] for r in self._eval_run_results]
        shorthand = self.get_eval_shorthand()
        metric_bundles: list[tuple[bool, bool, list[dict[tuple[str, ...], float]]]] = [
            (False, True, bc_concat_runs),
            (True, False, answer_agg_runs),
        ]
        for is_aggregated, is_bc, runs in metric_bundles:
            store_eval_df(
                get_eval_metrics_df_name(
                    metric_key_prefix, is_aggregated=is_aggregated, is_bc=is_bc
                ),
                get_df_from_metrics(compute_mean_std_metrics(runs)),
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


class SMCLogitsProcessor(LogitsProcessor):
    """Logits processor that coordinates SMC filtering securely inside the generation loop."""

    def __init__(self, trainer: ConfidenceGRPOTrainer, prompt_len: int) -> None:
        """Initialize with trainer reference and initial condition constraints."""
        self.trainer = trainer
        self.prompt_len = prompt_len
        self.finished_mask: torch.Tensor | None = None

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Execute particle filtering over probabilities iteratively."""
        finished_mask = self.finished_mask
        if finished_mask is None:
            finished_mask = torch.zeros(
                input_ids.size(0), dtype=torch.bool, device=input_ids.device
            )
            self.finished_mask = finished_mask

        if input_ids.size(1) > self.prompt_len:
            last_tokens = input_ids[:, -1]
            eos_id = self.trainer._tokenizer.eos_token_id  # noqa: SLF001
            pad_id = self.trainer._tokenizer.pad_token_id  # noqa: SLF001

            if isinstance(eos_id, int):
                finished_mask |= last_tokens == eos_id
            elif isinstance(eos_id, list | tuple):
                for e_id in eos_id:
                    finished_mask |= last_tokens == e_id

            if isinstance(pad_id, int):
                finished_mask |= last_tokens == pad_id

        self.trainer.apply_smc_filtering(scores, input_ids, finished_mask)
        return scores
