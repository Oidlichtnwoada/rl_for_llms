from enum import StrEnum, auto


class Method(StrEnum):
    """Method types for training and evaluation."""

    DENSE = auto()
    LASER = auto()
