import functools

from pydantic import BaseModel, ConfigDict

from rl_for_llms.utils.environment_utils import get_hf_model_id


class ModelSpecifics(BaseModel):
    """Model-specific information for training and evaluation."""

    model_config = ConfigDict(frozen=True)

    hf_model_id: str
    lm_head_attribute_name: str
    confidence_token: str
    confidence_token_logprob_mean: float
    confidence_token_logit_mean: float
    confidence_token_logit_std: float


@functools.cache
def get_model_specifics() -> ModelSpecifics:
    """Return model-specific information for the configured HF model."""
    hf_model_id = get_hf_model_id()
    match hf_model_id:
        case "Qwen/Qwen2.5-0.5B-Instruct":
            return ModelSpecifics(
                hf_model_id=hf_model_id,
                lm_head_attribute_name="lm_head",
                confidence_token="<|vision_pad|>",
                confidence_token_logprob_mean=-28.5,
                confidence_token_logit_mean=-3.9,
                confidence_token_logit_std=1.5,
            )
        case "Qwen/Qwen3.5-0.8B":
            return ModelSpecifics(
                hf_model_id=hf_model_id,
                lm_head_attribute_name="lm_head",
                confidence_token="<|vision_pad|>",
                confidence_token_logprob_mean=-23.0,
                confidence_token_logit_mean=-5.0,
                confidence_token_logit_std=1.7,
            )
        case _:
            raise ValueError
