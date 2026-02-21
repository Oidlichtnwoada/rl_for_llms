import pathlib
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.container import BarContainer

from rl_for_llms.models.config import Config
from rl_for_llms.models.variant import Variant
from rl_for_llms.utils.config_utils import get_config
from rl_for_llms.utils.constant_utils import (
    get_eval_after_train_prefix,
    get_eval_before_train_prefix,
)
from rl_for_llms.utils.path_utils import get_charts_dir, get_evaluation_final_dir


def get_metric_units() -> tuple[tuple[str, str], ...]:
    """Return metric units as a tuple of (metric_name, unit) pairs."""
    return (
        ("accuracy", "%"),
        ("fpr", "%"),
        ("fnr", "%"),
        ("tpr", "%"),
        ("tnr", "%"),
        ("ppv", "%"),
        ("npv", "%"),
        ("roc_auc", "%"),
        ("pr_auc", "%"),
        ("f1_score", "%"),
        ("balanced_accuracy", "%"),
        ("harmonic_balanced_accuracy", "%"),
        ("mcc", ""),
    )


def get_unit_for_metric(metric_name: str) -> str:
    """Return the unit for a given metric name."""
    base_name = metric_name.rsplit("/", maxsplit=1)[-1]
    for name, unit in get_metric_units():
        if name == base_name:
            return unit
    return "%"


def is_percentage_metric(metric_name: str) -> bool:
    """Return True if the metric should be displayed as percentage."""
    return get_unit_for_metric(metric_name) == "%"


def get_common_answer_metrics(config: Config) -> tuple[str, ...]:
    """Return the common answer metrics."""
    return (
        "accuracy/pass@1",
        f"accuracy/pass@{config.num_generations}",
        "accuracy/majority_voting",
        "truncation_percentage",
        "confidence_token_inclusion_percentage",
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
    name = metric_name.rsplit("/", maxsplit=1)[-1].replace("_", " ").title()
    words = name.split()
    mid = (len(words) + 1) // 2
    return " ".join(words[:mid]) + "\n" + " ".join(words[mid:])


def get_eval_prefix(variant: Variant) -> str:
    """Return the evaluation prefix for a variant."""
    return (
        get_eval_before_train_prefix()
        if not variant.is_trained()
        else get_eval_after_train_prefix()
    )


def get_csv_path(
    variant: Variant, metric_type: str, aggregation: Literal["agg", "concat"]
) -> pathlib.Path:
    """Return the CSV path for metrics of a variant."""
    final_dir = get_evaluation_final_dir()
    prefix = get_eval_prefix(variant)
    return (
        final_dir / f"{aggregation}_{prefix}_{metric_type}_metrics_{variant.value}.csv"
    )


def load_agg_metrics_from_csv(
    variant: Variant,
    metric_type: str,
    metric_keys: tuple[str, ...],
) -> dict[str, tuple[float, float]]:
    """Load aggregated metrics (mean, std) from CSV for a variant, filtered to given keys."""
    all_metrics = load_all_agg_metrics_from_csv(variant, metric_type)
    return {k: v for k, v in all_metrics.items() if k in metric_keys}


def load_all_agg_metrics_from_csv(
    variant: Variant, metric_type: str
) -> dict[str, tuple[float, float]]:
    """Load all aggregated metrics (mean, std) from CSV for a variant."""
    df = pd.read_csv(get_csv_path(variant, metric_type, "agg"))
    prefix = get_eval_prefix(variant)
    metrics: dict[str, tuple[float, float]] = {}
    for col in df.columns:
        if col.endswith("/mean"):
            base_key = col[len(prefix) + 1 : -5]
            std_key = f"{prefix}/{base_key}/std"
            if std_key in df.columns:
                metrics[base_key] = (df[col].iloc[0], df[std_key].iloc[0])
    return metrics


def load_all_concat_metrics_from_csv(
    variant: Variant, metric_type: str
) -> dict[str, float]:
    """Load all concatenated metrics (single values) from CSV for a variant."""
    df = pd.read_csv(get_csv_path(variant, metric_type, "concat"))
    prefix = get_eval_prefix(variant)
    metrics: dict[str, float] = {}
    for col in df.columns:
        base_key = col[len(prefix) + 1 :]
        metrics[base_key] = df[col].iloc[0]
    return metrics


def compute_ylim(max_value: float) -> float:
    """Compute y-axis limit as second next multiple of 10."""
    return ((int(max_value) // 10) + 2) * 10


def add_bar_labels(
    ax: Axes,
    bars: BarContainer,
    means: list[float],
    stds: list[float],
    *,
    add_stddev: bool = False,
) -> None:
    """Add text labels on top of bars."""
    for bar, mean, std in zip(bars, means, stds, strict=False):
        if not np.isnan(mean):
            label_text = f"{mean:.2f}%\n±{std:.2f}%" if add_stddev else f"{mean:.2f}%"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                label_text,
                ha="center",
                va="bottom",
                fontsize=6,
            )


def finalize_chart(
    ax: Axes,
    labels: list[str],
    x_positions: np.ndarray,
    all_max_values: list[float],
    *,
    title: str,
    filename: str,
    default_ylim: float = 100,
) -> None:
    """Apply shared chart formatting and save."""
    ax.set_ylabel("Percentage [%]")
    ax.set_title(title)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(loc="upper right")
    ax.set_ylim(
        0, compute_ylim(max(all_max_values)) if all_max_values else default_ylim
    )
    ax.grid(axis="y", alpha=0.3)
    save_chart(filename)


def save_chart(filename: str) -> None:
    """Save chart to PDF in charts directory."""
    charts_dir = get_charts_dir()
    charts_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(charts_dir / filename, format="pdf", bbox_inches="tight")
    plt.close()


def create_answer_accuracy_chart(*, add_stddev_to_label: bool = False) -> None:
    """Create an answer accuracy chart comparing all variants."""
    configure_matplotlib_fonts()
    config = get_config()
    variants = config.evaluation_variants
    common_metrics = get_common_answer_metrics(config)
    confidence_metrics = get_confidence_answer_metrics()
    all_metric_keys = tuple(
        f"answer/{m}_t=1.0" for m in common_metrics + confidence_metrics
    )

    all_metrics = {
        v: load_agg_metrics_from_csv(v, "answer", all_metric_keys) for v in variants
    }

    _, ax = plt.subplots(figsize=(14, 7))
    width = 0.18
    all_max_values = []

    x_common = np.arange(len(common_metrics))
    offsets = np.arange(len(variants)) - (len(variants) - 1) / 2
    for i, variant in enumerate(variants):
        metrics = all_metrics[variant]
        means = [
            metrics.get(f"answer/{m}_t=1.0", (0, 0))[0] * 100 for m in common_metrics
        ]
        stds = [
            metrics.get(f"answer/{m}_t=1.0", (0, 0))[1] * 100 for m in common_metrics
        ]
        all_max_values.extend(means)
        bars = ax.bar(
            x_common + offsets[i] * width, means, width, label=variant.get_shorthand()
        )
        add_bar_labels(ax, bars, means, stds, add_stddev=add_stddev_to_label)

    confidence_variants = [v for v in variants if v.has_trained_confidence()]
    x_conf_start = len(common_metrics)
    conf_offsets = (
        np.arange(len(confidence_variants)) - (len(confidence_variants) - 1) / 2
    )
    for j, metric in enumerate(confidence_metrics):
        x_pos = x_conf_start + j
        for i, variant in enumerate(confidence_variants):
            metrics = all_metrics[variant]
            key = f"answer/{metric}_t=1.0"
            if key in metrics:
                mean, std = metrics[key][0] * 100, metrics[key][1] * 100
                all_max_values.append(mean)
                bar = ax.bar(
                    x_pos + conf_offsets[i] * width,
                    mean,
                    width,
                    color=f"C{variants.index(variant)}",
                )
                add_bar_labels(ax, bar, [mean], [std], add_stddev=add_stddev_to_label)

    x_all = np.arange(len(common_metrics) + len(confidence_metrics))
    labels = [format_metric_label(m) for m in common_metrics + confidence_metrics]
    finalize_chart(
        ax,
        labels,
        x_all,
        all_max_values,
        title="Answer Accuracy Metrics By Variant",
        filename="answer_accuracy_chart.pdf",
    )


def create_confidence_chart() -> None:
    """Create a confidence prediction metrics chart comparing trained variants."""
    configure_matplotlib_fonts()
    config = get_config()
    variants = [v for v in config.evaluation_variants if v.has_trained_confidence()]

    if not variants:
        return

    sample_metrics = load_all_concat_metrics_from_csv(variants[0], "bc")
    all_metric_keys = [k for k in sample_metrics if k.startswith("confidence/")]

    metric_order = [name for name, _ in get_metric_units()]
    percentage_keys = sorted(
        [k for k in all_metric_keys if is_percentage_metric(k)],
        key=lambda k: (
            metric_order.index(k.split("/")[-1])
            if k.split("/")[-1] in metric_order
            else len(metric_order)
        ),
    )
    mcc_key = next((k for k in all_metric_keys if not is_percentage_metric(k)), None)

    all_metrics = {v: load_all_concat_metrics_from_csv(v, "bc") for v in variants}

    _, ax = plt.subplots(figsize=(14, 7))
    width = 0.25
    x = np.arange(len(percentage_keys))
    offsets = np.arange(len(variants)) - (len(variants) - 1) / 2
    all_max_values = []

    for i, variant in enumerate(variants):
        metrics = all_metrics[variant]
        means = [metrics.get(k, 0) * 100 for k in percentage_keys]
        all_max_values.extend([m for m in means if not np.isnan(m)])

        legend_label = variant.get_shorthand()
        if mcc_key and mcc_key in metrics:
            mcc_value = metrics[mcc_key]
            legend_label = f"{legend_label} (MCC: {mcc_value:.2f})"

        bars = ax.bar(x + offsets[i] * width, means, width, label=legend_label)
        add_bar_labels(ax, bars, means, [0.0] * len(means))

    labels = [format_metric_label(k) for k in percentage_keys]
    finalize_chart(
        ax,
        labels,
        x,
        all_max_values,
        title="Confidence Prediction Metrics By Variant",
        filename="confidence_chart.pdf",
    )
