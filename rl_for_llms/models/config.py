from typing import Literal

from peft import TaskType
from pydantic import BaseModel, Field

from rl_for_llms.utils.environment_utils import (
    get_base_num_generations,
    get_per_device_rollouts_per_batch,
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

    hf_model_id: str = Field(
        default=get_cuda_default_value(
            "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"
        )
    )
    lm_head_attribute_name: str = Field(default="lm_head")
    confidence_token: str = Field(default="<|vision_pad|>")
    confidence_token_logit_mean: float = Field(default=-4.0)
    confidence_token_logit_std: float = Field(default=1.5)
    confidence_loss_factor: float = Field(default=0.05)
    use_confidence_loss: bool = Field(default=use_confidence_loss())
    use_confidence_reward: bool = Field(default=use_confidence_reward())
    confidence_loss_warmup_steps: int = Field(default=1024)
    minimum_confidence_std: float = Field(default=0.1)
    confidence_reward_percentage: float = Field(default=0.1)
    learning_rate: float = Field(default=1e-6)
    num_train_epochs: int = Field(default=get_cuda_default_value(2, 1))
    use_vllm: bool = Field(default=False)
    vllm_gpu_memory_utilization: float = Field(default=0.4)
    vllm_split_model_across_gpus: bool = Field(default=False)
    vllm_mode: str = Field(default="colocate")
    vllm_enable_sleep_mode: bool = Field(
        default=get_cuda_default_value(value_if_cuda=True, value_if_not_cuda=False)
    )
    vllm_importance_sampling_correction: bool = Field(default=False)
    report_to: list[str] = Field(default_factory=lambda: get_logging_integrations())
    dataset_use_row_percentage: float = Field(default=get_cuda_default_value(1.0, 0.1))
    max_prompt_length: int = Field(default=get_cuda_default_value(2048, 64), le=8192)
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
    gradient_accumulation_steps: int = Field(default=1)
    temperature: float = Field(default=1.0)
    top_p: float = Field(default=1.0)
    save_steps: int = Field(default=1024)
    eval_strategy: str = Field(default="steps")
    eval_steps: float = Field(default=0.2)
    gradient_checkpointing: bool = Field(default=True)
    log_completions: bool = Field(default=True)
    beta: float = Field(default=1e-2)
    enable_lora: bool = Field(default=True)
    lora_target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    lora_rank: int = Field(default=16)
    lora_alpha_factor_for_rank: int = Field(default=2)
    lora_dropout: float = Field(default=0.1)
    lora_bias: Literal["none", "all", "lora_only"] = Field(default="none")
    lora_task_type: str = Field(default=TaskType.CAUSAL_LM)
    system_message: str = Field(
        default="Reason step by step, then provide the final answer within the last \\boxed{} command of your output."
    )
    dataloader_pin_memory: bool = Field(
        default=get_cuda_default_value(value_if_cuda=True, value_if_not_cuda=False)
    )

    def get_config_shorthand(self) -> str:
        """Return a shorthand representation of the config."""
        confidence_loss_spec = "confloss" if self.use_confidence_loss else "noconfloss"
        confidence_reward_spec = (
            "confrew"
            if self.use_confidence_reward and self.use_confidence_loss
            else "noconfrew"
        )
        shorthand = f"{confidence_loss_spec}_{confidence_reward_spec}"
        return shorthand
