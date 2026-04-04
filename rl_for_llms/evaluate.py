from rl_for_llms.models.variant import Variant
from rl_for_llms.utils.chart_utils import (
    create_answer_accuracy_chart,
    create_confidence_chart,
)
from rl_for_llms.utils.evaluation_utils import (
    get_response_and_confidence_tokens_for_answers,
)


def main() -> None:
    """Execute the main function of the module."""
    create_answer_accuracy_chart()
    create_confidence_chart()
    _ = get_response_and_confidence_tokens_for_answers(Variant.BASE, sample_size=64)


if __name__ == "__main__":
    main()
