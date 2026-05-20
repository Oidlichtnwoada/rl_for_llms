from enum import StrEnum

from rl_for_llms.utils.constant_utils import (
    get_filter_name,
    get_inv_prefix,
    get_no_name,
    get_tempmod_name,
)


class TemperatureModulationVariant(StrEnum):
    """Model variant types for temperature modulation."""

    NO_TEMP_MOD = f"{get_no_name()}{get_tempmod_name()}"
    TEMP_MOD = f"{get_tempmod_name()}"
    INV_TEMP_MOD = f"{get_inv_prefix()}{get_tempmod_name()}"

    def get_shorthand(self) -> str:
        """Return a shorthand representation of the variant."""
        match self:
            case TemperatureModulationVariant.NO_TEMP_MOD:
                return "-t"
            case TemperatureModulationVariant.TEMP_MOD:
                return "+t"
            case TemperatureModulationVariant.INV_TEMP_MOD:
                return "+it"

    def is_base(self) -> bool:
        """Check if the variant is the base variant with no temperature modulation."""
        return self == TemperatureModulationVariant.NO_TEMP_MOD


class FilteringVariant(StrEnum):
    """Model variant types for filtering."""

    NO_FILTER = f"{get_no_name()}{get_filter_name()}"
    FILTER = f"{get_filter_name()}"

    def get_shorthand(self) -> str:
        """Return a shorthand representation of the variant."""
        match self:
            case FilteringVariant.NO_FILTER:
                return "-f"
            case FilteringVariant.FILTER:
                return "+f"

    def is_base(self) -> bool:
        """Check if the variant is the base variant with no filtering."""
        return self == FilteringVariant.NO_FILTER
