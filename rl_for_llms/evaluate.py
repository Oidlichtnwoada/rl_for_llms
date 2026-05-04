from rl_for_llms.models.method import Method
from rl_for_llms.models.variant import Variant
from rl_for_llms.utils.chart_utils import (
    create_answer_accuracy_chart,
    create_confidence_chart,
)
from rl_for_llms.utils.evaluation_utils import (
    get_response_and_confidence_tokens_for_answers,
)
from rl_for_llms.utils.inference_time_evaluation_utils import inference_time_evaluate


def main() -> None:
    """Execute the main function of the module."""
    create_answer_accuracy_chart()
    create_confidence_chart()
    for use_tempmod, use_filtering in (
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ):
        inference_time_evaluate(
            Variant.WITH_CONFREW,
            Method.DENSE,
            use_tempmod=use_tempmod,
            use_filtering=use_filtering,
        )
    _ = get_response_and_confidence_tokens_for_answers(Variant.BASE, sample_size=64)


if __name__ == "__main__":
    main()
