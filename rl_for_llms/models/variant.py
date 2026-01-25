from enum import StrEnum


class Variant(StrEnum):
    """Model variant types."""

    BASE = "_base"
    GRPO = "_noconfloss_noconfrew"
    ONLY_CONFLOSS = "_confloss_noconfrew"
    WITH_CONFREW = "_confloss_confrew"

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
