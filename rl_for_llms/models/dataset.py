from enum import StrEnum, auto


class Dataset(StrEnum):
    """Dataset types for training and evaluation."""

    DEEPMATH_103K = auto()
    GSM8K = auto()
