from enum import StrEnum

from rl_for_llms.utils.constant_utils import (
    get_confidence_loss_name,
    get_confidence_reward_name,
    get_no_name,
)


class Variant(StrEnum):
    """Model variant types."""

    BASE = "base"
    GRPO = f"{get_no_name()}{get_confidence_loss_name()}_{get_no_name()}{get_confidence_reward_name()}"
    ONLY_CONFLOSS = (
        f"{get_confidence_loss_name()}_{get_no_name()}{get_confidence_reward_name()}"
    )
    WITH_CONFREW = f"{get_confidence_loss_name()}_{get_confidence_reward_name()}"

    def get_shorthand(self) -> str:
        """Return a shorthand representation of the variant."""
        match self:
            case Variant.BASE:
                return "base"
            case Variant.GRPO:
                return "-l-r"
            case Variant.ONLY_CONFLOSS:
                return "+l-r"
            case Variant.WITH_CONFREW:
                return "+l+r"
