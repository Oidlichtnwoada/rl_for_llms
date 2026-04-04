import time
from typing import Literal

from peft import TaskType
from pydantic import BaseModel, Field

from rl_for_llms.models.dataset import Dataset
from rl_for_llms.models.method import Method
from rl_for_llms.models.model_specifics import get_model_specifics
from rl_for_llms.models.variant import Variant
from rl_for_llms.utils.constant_utils import (
    get_confidence_loss_name,
    get_confidence_reward_name,
    get_no_name,
)
from rl_for_llms.utils.environment_utils import (
    get_base_num_generations,
    get_base_num_train_epochs,
    get_confidence_loss_factor,
    get_lora_train_confidence_token_embedding,
    get_method,
    get_per_device_rollouts_per_batch,
    get_seed,
    get_skip_eval_before_train,
    use_confidence_loss,
    use_confidence_reward,
)
from rl_for_llms.utils.torch_utils import (
    get_cuda_default_value,
    get_logging_integrations,
)


class Config(BaseModel):
    """Configuration for the program."""

    seed: int = Field(default_factory=get_seed)
    started_at: float = Field(default_factory=time.time)
    hf_model_id: str = Field(default=get_model_specifics().hf_model_id)
    lm_head_attribute_name: str = Field(
        default=get_model_specifics().lm_head_attribute_name
    )
    confidence_token: str = Field(default=get_model_specifics().confidence_token)
    method: Method = Field(default=get_method())
    confidence_token_logprob_kl_scale: float = Field(default=0.1)
    confidence_token_logprob_mean: float = Field(
        default=get_model_specifics().confidence_token_logprob_mean
    )
    confidence_token_logit_mean: float = Field(
        default=get_model_specifics().confidence_token_logit_mean
    )
    confidence_token_logit_std: float = Field(
        default=get_model_specifics().confidence_token_logit_std
    )
    confidence_loss_factor: float = Field(default=get_confidence_loss_factor())
    use_confidence_loss: bool = Field(default=use_confidence_loss())
    use_confidence_reward: bool = Field(default=use_confidence_reward())
    reasoning_warmup_samples: int = Field(default=2048)
    confidence_warmup_samples: int = Field(default=2048)
    minimum_confidence_std: float = Field(default=0.1)
    confidence_reward_percentage: float = Field(default=0.1)
    learning_rate: float = Field(default=2e-5)
    num_train_epochs: int = Field(
        default=get_cuda_default_value(
            2 * get_base_num_train_epochs(), get_base_num_train_epochs()
        )
    )
    use_vllm: bool = Field(default=False)
    vllm_gpu_memory_utilization: float = Field(default=0.4)
    vllm_split_model_across_gpus: bool = Field(default=False)
    vllm_mode: str = Field(default="colocate")
    vllm_enable_sleep_mode: bool = Field(
        default=get_cuda_default_value(value_if_cuda=True, value_if_not_cuda=False)
    )
    vllm_importance_sampling_correction: bool = Field(default=False)
    report_to: list[str] = Field(default_factory=lambda: get_logging_integrations())
    dataset: Dataset = Field(default=Dataset.GSM8K)
    train_dataset_rows: int = Field(default=get_cuda_default_value(-1, 8192))
    eval_dataset_rows: int = Field(default=get_cuda_default_value(-1, 512))
    max_prompt_length: int = Field(default=get_cuda_default_value(2048, 128), le=8192)
    max_completion_length: int = Field(
        default=get_cuda_default_value(8192, 1024), le=8192
    )
    skip_eval_before_train: bool = Field(default=get_skip_eval_before_train())
    num_generations: int = Field(
        default=get_cuda_default_value(
            2 * get_base_num_generations(), get_base_num_generations()
        ),
        le=16,
    )
    per_device_rollouts_per_batch: int = Field(
        default=get_per_device_rollouts_per_batch()
    )
    gradient_accumulation_steps: int = Field(default=4)
    temperature: float = Field(default=1.0)
    top_p: float = Field(default=1.0)
    save_steps: int = Field(default=1024)
    eval_strategy: str = Field(default="no")
    eval_steps: float = Field(default=0.35)
    gradient_checkpointing: bool = Field(default=True)
    log_completions: bool = Field(default=True)
    beta: float = Field(default=0.0)
    enable_lora: bool = Field(default=True)
    lora_target_modules: str | list[str] = Field(default="all-linear")
    lora_rank: int = Field(default=16)
    lora_alpha_factor_for_rank: int = Field(default=2)
    lora_dropout: float = Field(default=0.0)
    lora_bias: Literal["none", "all", "lora_only"] = Field(default="none")
    lora_task_type: str = Field(default=TaskType.CAUSAL_LM)
    lora_train_confidence_token_embedding: bool = Field(
        default=get_lora_train_confidence_token_embedding()
    )
    system_message: str = Field(
        default="Reason step by step, then provide the final answer within the last \\boxed{} command of your output."
    )
    dataloader_pin_memory: bool = Field(
        default=get_cuda_default_value(value_if_cuda=True, value_if_not_cuda=False)
    )
    evaluation_variants: tuple[Variant, ...] = Field(
        default=(
            Variant.BASE,
            Variant.GRPO,
            Variant.ONLY_CONFLOSS,
            Variant.WITH_CONFREW,
        )
    )
    evaluation_methods: tuple[Method, ...] = Field(default=(Method.DENSE, Method.LASER))

    def get_config_shorthand(self) -> str:
        """Return a shorthand representation of the config."""
        confidence_loss_spec = (
            get_confidence_loss_name()
            if self.use_confidence_loss
            else f"{get_no_name()}{get_confidence_loss_name()}"
        )
        confidence_reward_spec = (
            get_confidence_reward_name()
            if self.use_confidence_reward and self.use_confidence_loss
            else f"{get_no_name()}{get_confidence_reward_name()}"
        )
        parts = [confidence_loss_spec, confidence_reward_spec]
        if self.use_confidence_loss:
            parts.insert(0, self.method.value.lower())
        return "_".join(parts)
