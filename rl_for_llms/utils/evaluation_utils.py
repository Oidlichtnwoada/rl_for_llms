import random
import statistics
from collections import defaultdict

import pandas as pd
import torch
from accelerate.utils import is_peft_model
from peft import PeftModel, set_peft_model_state_dict
from peft.utils import load_peft_weights

from rl_for_llms.models.answer import AnswerWithConfidence
from rl_for_llms.models.response_confidence import (
    ResponseConfidenceResult,
    SampleResult,
    TokenStep,
)
from rl_for_llms.models.variant import Variant
from rl_for_llms.utils.chart_utils import (
    create_confidence_evolution_chart,
    get_eval_prefix,
)
from rl_for_llms.utils.confidence_utils import get_confidence_token_logit_sigmoid
from rl_for_llms.utils.config_utils import get_config
from rl_for_llms.utils.constant_utils import (
    get_default_confidence_score,
    get_default_metric_separator,
)
from rl_for_llms.utils.dataset_utils import load_training_data_from_disk, trim_dataset
from rl_for_llms.utils.group_utils import iter_groups
from rl_for_llms.utils.llm_utils import (
    check_contains_confidence_token,
    get_llm_output_with_step_data,
    get_pipeline,
    get_token_to_id_mapping,
    get_tokenizer,
)
from rl_for_llms.utils.path_utils import (
    get_evaluation_final_dir,
    get_evaluation_metric_dir,
    standardize_model_id,
)
from rl_for_llms.utils.reward_utils import get_math_verification_answer


def _get_variant_checkpoint_dir(variant: Variant, config_hf_model_id: str) -> str:
    """Return the checkpoint directory path for a variant in data/evaluation/final."""
    standardized_hf_model_id = standardize_model_id(config_hf_model_id)
    prefix = get_eval_prefix(variant)
    return str(
        get_evaluation_final_dir()
        / f"{standardized_hf_model_id}_{prefix}_{variant.value}"
    )


def get_response_and_confidence_tokens_for_answers(
    variant: Variant,
    sample_size: int = 1,
    *,
    generate_chart: bool = True,
) -> ResponseConfidenceResult:
    """Return response and confidence data for sampled answers using a specific variant."""
    config = get_config()
    confidence_token_id = get_token_to_id_mapping(config.hf_model_id)[
        config.confidence_token
    ]
    tokenizer = get_tokenizer(config.hf_model_id)
    dataset = trim_dataset(
        load_training_data_from_disk(config),
        config.train_dataset_rows,
        tokenizer,
        config.max_prompt_length,
        config.seed,
    )

    pipe = get_pipeline(config.hf_model_id)

    checkpoint_dir = _get_variant_checkpoint_dir(variant, config.hf_model_id)

    model = pipe.model
    if is_peft_model(model):
        adapter_weights = load_peft_weights(checkpoint_dir)
        set_peft_model_state_dict(model, adapter_weights, adapter_name="default")
    else:
        model = PeftModel.from_pretrained(model, checkpoint_dir)  # type: ignore[assignment]
        pipe.model = model

    rows = dataset.select(range(sample_size))
    messages = [row["prompt"][-1]["content"] for row in rows]
    answers = [row["answer"] for row in rows]

    samples: list[SampleResult] = []
    all_logit_values: list[float] = []

    for message, correct_answer in zip(messages, answers, strict=True):
        output_message, step_data, token_ids, token_texts = (
            get_llm_output_with_step_data(
                message,
                config.hf_model_id,
                (confidence_token_id,),
            )
        )

        verification = get_math_verification_answer(
            correct_answer,
            output_message,
            is_truncated=False,
            contains_confidence_token=check_contains_confidence_token(
                token_ids, config
            ),
        )

        steps: list[TokenStep] = []
        for i, step_vals in enumerate(step_data):
            logit_val = step_vals[confidence_token_id]["logit"]
            sigmoid_val = get_confidence_token_logit_sigmoid(
                torch.tensor(logit_val), config
            ).item()
            steps.append(
                TokenStep(
                    token_id=token_ids[i],
                    token_text=token_texts[i],
                    confidence_logit=logit_val,
                    confidence_sigmoid=sigmoid_val,
                )
            )
            all_logit_values.append(logit_val)

        samples.append(
            SampleResult(
                question=message,
                full_response=output_message,
                correct_answer=verification.correct_answer,
                model_answer=verification.model_answer,
                is_correct=verification.is_correct,
                is_truncated=verification.is_truncated,
                contains_confidence_token=verification.contains_confidence_token,
                steps=steps,
            )
        )

    overall_mean_logit = statistics.mean(all_logit_values) if all_logit_values else 0.0
    overall_std_logit = (
        statistics.stdev(all_logit_values) if len(all_logit_values) > 1 else 0.0
    )

    result = ResponseConfidenceResult(
        samples=samples,
        overall_mean_confidence_logit=overall_mean_logit,
        overall_std_confidence_logit=overall_std_logit,
    )

    if generate_chart:
        create_confidence_evolution_chart(result, variant)

    return result


def pick_best_answer(
    weights: dict[str, float],
    answers_with_confidence: list[AnswerWithConfidence],
) -> AnswerWithConfidence:
    """Pick the best answer based on the given weights."""
    max_weight = max(weights.values())
    top_answers = [
        a
        for a in answers_with_confidence
        if weights[a.answer.model_answer] == max_weight
    ]
    return random.choice(top_answers)  # noqa: S311


def get_correctness_flags(
    answers_with_confidence: list[AnswerWithConfidence],
) -> list[float]:
    """Extract correctness flags from answers."""
    return [float(x.answer.is_correct) for x in answers_with_confidence]


def compute_answer_metrics_for_group(
    answers_with_confidence: list[AnswerWithConfidence],
) -> dict[str, float]:
    """Compute answer metrics for a single group (same prompt)."""
    group_metrics: dict[str, float] = {}
    if not answers_with_confidence:
        return group_metrics
    flags_for_correctness = get_correctness_flags(answers_with_confidence)
    group_metrics["pass_at_1"] = statistics.mean(flags_for_correctness)
    group_metrics["pass_at_k"] = max(flags_for_correctness)
    answer_weights: dict[str, float] = defaultdict(float)
    weighted_answer_weights: dict[str, float] = defaultdict(float)
    for answer_with_confidence in answers_with_confidence:
        answer_weights[answer_with_confidence.answer.model_answer] += (
            get_default_confidence_score()
        )
        weighted_answer_weights[answer_with_confidence.answer.model_answer] += (
            answer_with_confidence.confidence
        )
    group_metrics["majority_voting"] = float(
        pick_best_answer(answer_weights, answers_with_confidence).answer.is_correct
    )
    group_metrics["confidence_weighted_majority_voting"] = float(
        pick_best_answer(
            weighted_answer_weights, answers_with_confidence
        ).answer.is_correct
    )
    max_confidence = max(x.confidence for x in answers_with_confidence)
    max_confidence_answers = [
        x for x in answers_with_confidence if x.confidence == max_confidence
    ]
    group_metrics["highest_confidence"] = float(
        random.choice(max_confidence_answers).answer.is_correct  # noqa: S311
    )
    return group_metrics


def compute_answer_metrics(
    answers_with_confidence: list[AnswerWithConfidence],
    temperature: float,
    num_generations: int | None = None,
) -> dict[tuple[str, ...], float]:
    """Compute answer metrics, supporting multiple groups."""
    metrics: dict[tuple[str, ...], float] = {}
    sample_amount = len(answers_with_confidence)
    if sample_amount == 0:
        return metrics
    truncation_percentage = statistics.mean(
        [float(x.answer.is_truncated) for x in answers_with_confidence]
    )
    metrics[(f"truncation_percentage_t={temperature}",)] = truncation_percentage
    confidence_token_inclusion_percentage = statistics.mean(
        [float(x.answer.contains_confidence_token) for x in answers_with_confidence]
    )
    metrics[(f"confidence_token_inclusion_percentage_t={temperature}",)] = (
        confidence_token_inclusion_percentage
    )
    pass_at_1_accuracy = statistics.mean(get_correctness_flags(answers_with_confidence))
    metrics[("accuracy", f"pass@1_t={temperature}")] = pass_at_1_accuracy
    if num_generations is None:
        num_generations = sample_amount
    group_metrics_list = [
        compute_answer_metrics_for_group(group)
        for _, _, group in iter_groups(answers_with_confidence, num_generations)
    ]
    if group_metrics_list:
        metrics[("accuracy", f"pass@{num_generations}_t={temperature}")] = (
            statistics.mean(m["pass_at_k"] for m in group_metrics_list)
        )
        metrics[("accuracy", f"majority_voting_t={temperature}")] = statistics.mean(
            m["majority_voting"] for m in group_metrics_list
        )
        metrics[
            ("accuracy", f"confidence_weighted_majority_voting_t={temperature}")
        ] = statistics.mean(
            m["confidence_weighted_majority_voting"] for m in group_metrics_list
        )
        metrics[("accuracy", f"highest_confidence_t={temperature}")] = statistics.mean(
            m["highest_confidence"] for m in group_metrics_list
        )
    return metrics


def aggregate_metrics(
    metrics_list: list[dict[tuple[str, ...], float]],
) -> dict[tuple[str, ...], float]:
    """Aggregate metrics by computing mean and standard deviation."""
    aggregated_metrics: dict[tuple[str, ...], float] = {}
    if not metrics_list:
        return aggregated_metrics
    metric_keys = set().union(*metrics_list)
    for key in metric_keys:
        values = [metrics[key] for metrics in metrics_list if key in metrics]
        if len(values) == 0:
            raise ValueError
        aggregated_metrics[(*key, "mean")] = statistics.mean(values)
        stddev = statistics.stdev(values) if len(values) > 1 else 0.0
        aggregated_metrics[(*key, "std")] = stddev
    return aggregated_metrics


def change_metric_keys(
    metrics: dict[tuple[str, ...], float],
    prefix: tuple[str, ...] = (),
    postfix: tuple[str, ...] = (),
) -> dict[tuple[str, ...], float]:
    """Change metric keys by adding prefix and postfix."""
    changed_metrics: dict[tuple[str, ...], float] = {}
    for key, value in metrics.items():
        new_key = prefix + key + postfix
        changed_metrics[new_key] = value
    return changed_metrics


def get_df_from_metrics(
    metrics: dict[tuple[str, ...], float],
    sep: str = get_default_metric_separator(),
) -> pd.DataFrame:
    """Convert metrics dictionary to a pandas DataFrame."""
    data = {sep.join(key): value for key, value in metrics.items()}
    df = pd.DataFrame([data])
    return df


def get_eval_metrics_df_name(
    metric_key_prefix: str, *, is_aggregated: bool = True, is_bc: bool = True
) -> str:
    """Get the evaluation metrics DataFrame name."""
    scope = "agg" if is_aggregated else "concat"
    metric_type = "bc" if is_bc else "answer"
    return f"{scope}_{metric_key_prefix}_{metric_type}_metrics"


def store_eval_df(
    file_name: str,
    df: pd.DataFrame,
    shorthand: str,
) -> None:
    """Store evaluation DataFrame to a CSV file."""
    df.to_csv(get_evaluation_metric_dir() / f"{file_name}_{shorthand}.csv", index=False)
