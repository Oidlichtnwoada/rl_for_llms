from typing import Literal

from peft import TaskType
from pydantic import BaseModel, Field

from rl_for_llms.utils.constant_utils import get_default_hf_model_id


class Config(BaseModel):
    """Configuration for the program."""

    hf_model_id: str = Field(default=get_default_hf_model_id())
    learning_rate: float = Field(default=1e-5)
    num_train_epochs: int = Field(default=1)
    use_vllm: bool = Field(default=False)
    report_to: list[str] = Field(default_factory=lambda: ["tensorboard"])
    lora_rank: int = Field(default=16)
    target_modules: list[str] = Field(
        default_factory=lambda: ["gate_proj", "up_proj", "down_proj"]
    )
    dataset_use_row_percentage: float = Field(default=0.1)
    max_prompt_length: int = Field(default=512, le=2048)
    max_completion_length: int = Field(default=256, le=8192)
    num_generations: int = Field(default=4, le=16)
    temperature: float = Field(default=1.0)
    top_p: float = Field(default=1.0)
    vllm_gpu_memory_utilization: float = Field(default=0.6)
    vllm_tensor_parallel_size: int = Field(default=1)
    vllm_mode: str = Field(default="colocate")
    save_steps: int = Field(default=1000)
    eval_steps: int = Field(default=1000)
    per_device_rollouts_per_batch: int = Field(default=1)
    gradient_accumulation_steps: int = Field(default=1)
    gradient_checkpointing: bool = Field(default=True)
    log_completions: bool = Field(default=True)
    beta: float = Field(default=0.0)
    lora_alpha_factor_for_rank: int = Field(default=2)
    lora_dropout: float = Field(default=0.05)
    bias: Literal["none", "all", "lora_only"] = Field(default="none")
    task_type: str = Field(default=TaskType.CAUSAL_LM)
