from pydantic import BaseModel, Field

from rl_for_llms.utils.constant_utils import get_default_hf_model_id


class Config(BaseModel):
    """Configuration for the program."""

    hf_model_id: str = Field(default=get_default_hf_model_id())
    learning_rate: float = Field(default=1e-6)
    num_train_epochs: int = Field(default=1)
    use_vllm: bool = Field(default=False)
    report_to: list[str] = Field(default_factory=lambda: ["tensorboard"])
    lora_rank: int = Field(default=16)
    target_modules: list[str] = Field(
        default_factory=lambda: ["gate_proj", "up_proj", "down_proj"]
    )
    dataset_use_row_percentage: float = Field(default=0.1)
