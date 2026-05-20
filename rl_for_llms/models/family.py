from enum import StrEnum, auto


class Family(StrEnum):
    """Method family types."""

    CONFIDENCE = auto()
    INFERENCE_TIME = auto()
