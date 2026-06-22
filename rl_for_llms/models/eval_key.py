from pydantic import BaseModel, ConfigDict

from rl_for_llms.models.inference_time_variant_enums import (
    FilteringVariant,
    TemperatureModulationVariant,
)
from rl_for_llms.models.method import Method
from rl_for_llms.models.variant import Variant
from rl_for_llms.utils.constant_utils import get_base_name


class EvalKey(BaseModel):
    """Unified key identifying a single evaluation series for charting."""

    model_config = ConfigDict(frozen=True)

    variant: Variant
    method: Method | None = None
    tempmod: TemperatureModulationVariant | None = None
    filtering: FilteringVariant | None = None

    def _variant_method_shorthand(self) -> str:
        """Return the file shorthand for the underlying variant and method."""
        if self.variant.has_trained_confidence() and self.method is not None:
            return f"{self.method.value.lower()}_{self.variant.value}"
        return self.variant.value

    def shorthand(self) -> str:
        """Return the CSV filename shorthand for this key."""
        base = self._variant_method_shorthand()
        if self.tempmod is None and self.filtering is None:
            return base
        # Mirror trainer's get_eval_shorthand: omit suffix when both axes are base
        if (
            self.tempmod is not None
            and self.filtering is not None
            and self.tempmod.is_base()
            and self.filtering.is_base()
        ):
            return base
        return f"{base}_{self.tempmod}_{self.filtering}"

    def label(self) -> str:
        """Return the display label for this key."""
        if self.tempmod is not None and self.filtering is not None:
            if self.tempmod.is_base() and self.filtering.is_base():
                return get_base_name()
            return f"{self.tempmod.get_shorthand()}{self.filtering.get_shorthand()}"
        shorthand = self.variant.get_shorthand()
        if self.method is not None:
            return f"{shorthand} ({self.method.get_display_name()})"
        return shorthand

    def has_confidence_metrics(self) -> bool:
        """Return True if this key produces confidence-related metrics."""
        return self.variant.has_trained_confidence()
