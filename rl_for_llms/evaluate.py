from rl_for_llms.models.family import Family
from rl_for_llms.models.method import Method
from rl_for_llms.models.variant import Variant
from rl_for_llms.utils.chart_utils import (
    create_answer_accuracy_chart,
    create_confidence_chart,
)
from rl_for_llms.utils.environment_utils import (
    get_inference_time_evaluation_use_filtering,
    get_inference_time_evaluation_use_tempmod,
)
from rl_for_llms.utils.evaluation_utils import (
    get_response_and_confidence_tokens_for_answers,
)
from rl_for_llms.utils.inference_time_evaluation_utils import inference_time_evaluate


def main() -> None:
    """Execute the main function of the module."""
    create_answer_accuracy_chart(Family.CONFIDENCE)
    create_answer_accuracy_chart(Family.INFERENCE_TIME)
    create_confidence_chart(Family.CONFIDENCE)
    create_confidence_chart(Family.INFERENCE_TIME)
    _ = get_response_and_confidence_tokens_for_answers(
        Variant.WITH_CONFREW,
        sample_size=64,
    )
    inference_time_evaluate(
        Variant.WITH_CONFREW,
        Method.DENSE,
        use_tempmod=get_inference_time_evaluation_use_tempmod(),
        use_filtering=get_inference_time_evaluation_use_filtering(),
    )


if __name__ == "__main__":
    main()
