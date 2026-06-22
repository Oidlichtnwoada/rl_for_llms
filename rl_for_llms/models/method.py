from enum import StrEnum, auto


class Method(StrEnum):
    """Method types for training and evaluation."""

    DENSE = auto()
    LASER = auto()

    def get_display_name(self) -> str:
        """Return the display name with the canonical casing (Dense, LaSeR)."""
        match self:
            case Method.DENSE:
                return "Dense"
            case Method.LASER:
                return "LaSeR"
