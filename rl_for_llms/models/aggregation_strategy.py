from enum import StrEnum, auto


class AggregationStrategy(StrEnum):
    """Aggregation strategies for combining multiple values into one value."""

    MEAN = auto()
    EXPONENTIALLY_INCREASING = auto()
    EXPONENTIALLY_DECREASING = auto()
