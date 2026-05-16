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
from rl_for_llms.utils.inference_time_evaluation_utils import inference_time_evaluate


def main() -> None:
    """Execute the main function of the module."""
    create_answer_accuracy_chart()
    create_confidence_chart()
    inference_time_evaluate(
        Variant.WITH_CONFREW,
        Method.DENSE,
        use_tempmod=get_inference_time_evaluation_use_tempmod(),
        use_filtering=get_inference_time_evaluation_use_filtering(),
    )


if __name__ == "__main__":
    main()
