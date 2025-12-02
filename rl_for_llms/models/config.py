from typing import Literal

from peft import TaskType
from pydantic import BaseModel, Field

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
    learning_rate: float = Field(default=1e-6)
    num_train_epochs: int = Field(default=get_cuda_default_value(3, 1))
    use_vllm: bool = Field(default=True)
    vllm_gpu_memory_utilization: float = Field(default=0.3)
    vllm_split_model_across_gpus: bool = Field(default=False)
    vllm_mode: str = Field(default="colocate")
    vllm_enable_sleep_mode: bool = Field(
        default=get_cuda_default_value(value_if_cuda=True, value_if_not_cuda=False)
    )
    enable_llm_weight_reloading: bool = Field(
        default=get_cuda_default_value(value_if_cuda=False, value_if_not_cuda=True)
    )
    report_to: list[str] = Field(default_factory=lambda: get_logging_integrations())
    dataset_use_row_percentage: float = Field(default=get_cuda_default_value(1.0, 0.1))
    max_prompt_length: int = Field(default=get_cuda_default_value(2048, 64), le=8192)
    max_completion_length: int = Field(
        default=get_cuda_default_value(8192, 1024), le=8192
    )
    num_generations: int = Field(default=get_cuda_default_value(8, 4), le=16)
    per_device_rollouts_per_batch: int = Field(default=get_cuda_default_value(2, 1))
    gradient_accumulation_steps: int = Field(default=1)
    temperature: float = Field(default=1.0)
    top_p: float = Field(default=1.0)
    save_steps: int = Field(default=512)
    eval_steps: int = Field(default=4096)
    gradient_checkpointing: bool = Field(default=False)
    log_completions: bool = Field(default=True)
    beta: float = Field(default=0.0)
    enable_lora: bool = Field(default=True)
    lora_target_modules: list[str] = Field(
        default_factory=lambda: ["gate_proj", "up_proj", "down_proj"]
    )
    lora_rank: int = Field(default=16)
    lora_alpha_factor_for_rank: int = Field(default=2)
    lora_dropout: float = Field(default=0.05)
    lora_bias: Literal["none", "all", "lora_only"] = Field(default="none")
    lora_task_type: str = Field(default=TaskType.CAUSAL_LM)
    system_message: str = Field(
        default="Reason step by step, then provide the final answer within the last \\boxed{} command of your output."
    )
    dataloader_pin_memory: bool = Field(
        default=get_cuda_default_value(value_if_cuda=True, value_if_not_cuda=False)
    )
