import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rl_for_llms.models.config import Config
from rl_for_llms.models.variant import Variant
from rl_for_llms.utils.config_utils import get_config
from rl_for_llms.utils.constant_utils import (
    get_eval_after_train_prefix,
    get_eval_before_train_prefix,
)
from rl_for_llms.utils.path_utils import get_charts_dir, get_evaluation_final_dir


def get_common_answer_metrics(config: Config) -> tuple[str, ...]:
    """Return the common answer metrics."""
    return (
        "accuracy/pass@1",
        f"accuracy/pass@{config.num_generations}",
        "accuracy/majority_voting",
        "truncation_percentage",
    )


def get_confidence_answer_metrics() -> tuple[str, ...]:
    """Return the confidence-based answer metrics."""
    return (
        "accuracy/highest_confidence",
        "accuracy/confidence_weighted_majority_voting",
    )


def configure_matplotlib_fonts() -> None:
    """Configure matplotlib fonts to match the LaTeX report."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Merriweather", "DejaVu Serif", "Times New Roman"],
            "font.sans-serif": ["Public Sans", "DejaVu Sans", "Helvetica"],
            "mathtext.fontset": "dejavuserif",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )


def format_metric_label(metric_name: str) -> str:
    """Convert metric name to display label (two lines, capitalized)."""
    name = metric_name.split("/")[-1].replace("_", " ").title()
    words = name.split()
    mid = (len(words) + 1) // 2
    return " ".join(words[:mid]) + "\n" + " ".join(words[mid:])


def get_csv_path_for_variant(variant: Variant) -> pathlib.Path:
    """Return the CSV path for the answer metrics of a variant."""
    final_dir = get_evaluation_final_dir()
    prefix = (
        get_eval_before_train_prefix()
        if not variant.is_trained()
        else get_eval_after_train_prefix()
    )
    return final_dir / f"agg_{prefix}_answer_metrics_{variant.value}.csv"


def load_metrics_for_variant(
    variant: Variant, config: Config
) -> dict[str, tuple[float, float]]:
    """Load metrics (mean, std) for a variant."""
    df = pd.read_csv(get_csv_path_for_variant(variant))
    prefix = (
        get_eval_before_train_prefix()
        if not variant.is_trained()
        else get_eval_after_train_prefix()
    )
    metrics = {}
    for metric in get_common_answer_metrics(config) + get_confidence_answer_metrics():
        mean_key = f"{prefix}/answer/{metric}_t=1.0/mean"
        std_key = f"{prefix}/answer/{metric}_t=1.0/std"
        if mean_key in df.columns and std_key in df.columns:
            metrics[metric] = (df[mean_key].iloc[0], df[std_key].iloc[0])
    return metrics


def compute_ylim(max_value: float) -> float:
    """Compute y-axis limit as second next multiple of 10."""
    return ((int(max_value) // 10) + 2) * 10


def create_answer_accuracy_chart(*, add_stddev_to_label: bool = False) -> None:
    """Create an answer accuracy chart comparing all variants."""
    configure_matplotlib_fonts()
    config = get_config()
    variants = config.evaluation_variants
    all_metrics = {v: load_metrics_for_variant(v, config) for v in variants}
    common_metrics = get_common_answer_metrics(config)
    confidence_metrics = get_confidence_answer_metrics()

    _, ax = plt.subplots(figsize=(14, 7))
    width = 0.18
    all_max_values = []

    x_common = np.arange(len(common_metrics))
    offsets = np.arange(len(variants)) - (len(variants) - 1) / 2
    for i, variant in enumerate(variants):
        metrics = all_metrics[variant]
        means = [metrics.get(m, (0, 0))[0] * 100 for m in common_metrics]
        stds = [metrics.get(m, (0, 0))[1] * 100 for m in common_metrics]
        all_max_values.extend(means)
        bars = ax.bar(
            x_common + offsets[i] * width, means, width, label=variant.get_shorthand()
        )
        for bar, mean, std in zip(bars, means, stds, strict=False):
            label_text = (
                f"{mean:.2f}%\n±{std:.2f}%" if add_stddev_to_label else f"{mean:.2f}%"
            )
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                label_text,
                ha="center",
                va="bottom",
                fontsize=6,
            )

    confidence_variants = [v for v in variants if v.has_trained_confidence()]
    x_conf_start = len(common_metrics)
    conf_offsets = (
        np.arange(len(confidence_variants)) - (len(confidence_variants) - 1) / 2
    )
    for j, metric in enumerate(confidence_metrics):
        x_pos = x_conf_start + j
        for i, variant in enumerate(confidence_variants):
            metrics = all_metrics[variant]
            if metric in metrics:
                mean, std = metrics[metric][0] * 100, metrics[metric][1] * 100
                all_max_values.append(mean)
                bar = ax.bar(
                    x_pos + conf_offsets[i] * width,
                    mean,
                    width,
                    color=f"C{variants.index(variant)}",
                )
                label_text = (
                    f"{mean:.2f}%\n±{std:.2f}%"
                    if add_stddev_to_label
                    else f"{mean:.2f}%"
                )
                ax.text(
                    bar[0].get_x() + bar[0].get_width() / 2,
                    bar[0].get_height() + 0.5,
                    label_text,
                    ha="center",
                    va="bottom",
                    fontsize=6,
                )

    x_all = np.arange(len(common_metrics) + len(confidence_metrics))
    labels = [format_metric_label(m) for m in common_metrics + confidence_metrics]
    ax.set_ylabel("Percentage [%]")
    ax.set_title("Answer Accuracy Metrics By Variant")
    ax.set_xticks(x_all)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(loc="upper right")
    ax.set_ylim(0, compute_ylim(max(all_max_values)))
    ax.grid(axis="y", alpha=0.3)

    charts_dir = get_charts_dir()
    charts_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(
        charts_dir / "answer_accuracy_chart.pdf", format="pdf", bbox_inches="tight"
    )
    plt.close()
