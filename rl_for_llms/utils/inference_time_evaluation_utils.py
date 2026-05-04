from rl_for_llms.models.method import Method
from rl_for_llms.models.variant import Variant
from rl_for_llms.utils.config_utils import get_config
from rl_for_llms.utils.constant_utils import get_eval_after_train_prefix
from rl_for_llms.utils.training_utils import get_confidence_grpo_trainer


def inference_time_evaluate(
    variant: Variant,
    method: Method,
    *,
    use_tempmod: bool = True,
    use_filtering: bool = True,
) -> None:
    """Load weights for the given variant and run post-training evaluation, optionally with confidence-based temperature modulation and/or SMC particle filtering."""
    if use_filtering and method != Method.DENSE:
        raise ValueError
    config = get_config()
    trainer = get_confidence_grpo_trainer(config)
    if variant.is_trained():
        trainer.load_final_checkpoint(variant, method)
    trainer.evaluate(
        metric_key_prefix=get_eval_after_train_prefix(),
        use_tempmod=use_tempmod,
        use_filtering=use_filtering,
    )
