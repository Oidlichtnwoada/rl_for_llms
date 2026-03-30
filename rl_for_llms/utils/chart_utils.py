import pathlib
from typing import Literal

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.container import BarContainer
from matplotlib.figure import Figure

from rl_for_llms.models.config import Config
from rl_for_llms.models.method import Method
from rl_for_llms.models.response_confidence import ResponseConfidenceResult
from rl_for_llms.models.variant import Variant
from rl_for_llms.utils.config_utils import get_config
from rl_for_llms.utils.constant_utils import (
    get_eval_after_train_prefix,
    get_eval_before_train_prefix,
)
from rl_for_llms.utils.font_utils import get_chart_font_families
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


def get_variant_method_color_map() -> dict[tuple[Variant, Method | None], str]:
    """Return a mapping from (variant, method) to consistent color string."""
    return {
        (Variant.BASE, None): "#1f77b4",
        (Variant.GRPO, None): "#ff7f0e",
        (Variant.ONLY_CONFLOSS, Method.DENSE): "#2ca02c",
        (Variant.ONLY_CONFLOSS, Method.LASER): "#98df8a",
        (Variant.WITH_CONFREW, Method.DENSE): "#d62728",
        (Variant.WITH_CONFREW, Method.LASER): "#ff9896",
    }


def get_variant_method_shorthand(variant: Variant, method: Method | None = None) -> str:
    """Return the file shorthand for a variant and optional method."""
    if variant.has_trained_confidence() and method is not None:
        return f"{method.value.lower()}_{variant.value}"
    return variant.value


def get_variant_method_label(variant: Variant, method: Method | None = None) -> str:
    """Return the display label for a variant and optional method."""
    shorthand = variant.get_shorthand()
    if method is not None:
        return f"{shorthand} ({method.value.lower()})"
    return shorthand


def build_variant_method_keys(
    variants: tuple[Variant, ...],
    methods: tuple[Method, ...],
) -> list[tuple[Variant, Method | None]]:
    """Build a list of (variant, method) keys, expanding confidence variants by method."""
    keys: list[tuple[Variant, Method | None]] = []
    for variant in variants:
        if variant.has_trained_confidence():
            keys.extend((variant, method) for method in methods)
        else:
            keys.append((variant, None))
    return keys


def configure_matplotlib_fonts() -> None:
    """Configure matplotlib fonts to match the LaTeX report."""
    font_families = get_chart_font_families()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": font_families["serif"],
            "font.sans-serif": font_families["sans-serif"],
            "font.monospace": font_families["monospace"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
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
    variant: Variant,
    metric_type: str,
    aggregation: Literal["agg", "concat"],
    method: Method | None = None,
) -> pathlib.Path:
    """Return the CSV path for metrics of a variant."""
    final_dir = get_evaluation_final_dir()
    prefix = get_eval_prefix(variant)
    shorthand = get_variant_method_shorthand(variant, method)
    return final_dir / f"{aggregation}_{prefix}_{metric_type}_metrics_{shorthand}.csv"


def load_agg_metrics_from_csv(
    variant: Variant,
    metric_type: str,
    metric_keys: tuple[str, ...],
    method: Method | None = None,
) -> dict[str, tuple[float, float]]:
    """Load aggregated metrics (mean, std) from CSV for a variant, filtered to given keys."""
    all_metrics = load_all_agg_metrics_from_csv(variant, metric_type, method=method)
    return {k: v for k, v in all_metrics.items() if k in metric_keys}


def load_all_agg_metrics_from_csv(
    variant: Variant,
    metric_type: str,
    method: Method | None = None,
) -> dict[str, tuple[float, float]]:
    """Load all aggregated metrics (mean, std) from CSV for a variant."""
    df = pd.read_csv(get_csv_path(variant, metric_type, "agg", method=method))
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
    variant: Variant,
    metric_type: str,
    method: Method | None = None,
) -> dict[str, float]:
    """Load all concatenated metrics (single values) from CSV for a variant."""
    df = pd.read_csv(get_csv_path(variant, metric_type, "concat", method=method))
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
                fontsize=5,
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
    all_keys = build_variant_method_keys(
        config.evaluation_variants, config.evaluation_methods
    )
    common_metrics = get_common_answer_metrics(config)
    confidence_metrics = get_confidence_answer_metrics()
    all_metric_keys = tuple(
        f"answer/{m}_t=1.0" for m in common_metrics + confidence_metrics
    )

    all_metrics = {
        key: load_agg_metrics_from_csv(key[0], "answer", all_metric_keys, method=key[1])
        for key in all_keys
    }

    color_map = get_variant_method_color_map()
    _, ax = plt.subplots(figsize=(14, 7))
    n_keys = len(all_keys)
    width = 0.8 / max(n_keys, 1)
    all_max_values: list[float] = []

    x_common = np.arange(len(common_metrics))
    offsets = np.arange(n_keys) - (n_keys - 1) / 2
    for i, key in enumerate(all_keys):
        variant, method = key
        metrics = all_metrics[key]
        means = [
            metrics.get(f"answer/{m}_t=1.0", (0, 0))[0] * 100 for m in common_metrics
        ]
        stds = [
            metrics.get(f"answer/{m}_t=1.0", (0, 0))[1] * 100 for m in common_metrics
        ]
        all_max_values.extend(means)
        bars = ax.bar(
            x_common + offsets[i] * width,
            means,
            width,
            label=get_variant_method_label(variant, method),
            color=color_map[key],
        )
        add_bar_labels(ax, bars, means, stds, add_stddev=add_stddev_to_label)

    confidence_keys = [key for key in all_keys if key[0].has_trained_confidence()]
    x_conf_start = len(common_metrics)
    conf_offsets = np.arange(len(confidence_keys)) - (len(confidence_keys) - 1) / 2
    for j, metric in enumerate(confidence_metrics):
        x_pos = x_conf_start + j
        for i, key in enumerate(confidence_keys):
            variant, method = key
            metrics = all_metrics[key]
            m_key = f"answer/{metric}_t=1.0"
            if m_key in metrics:
                mean, std = metrics[m_key][0] * 100, metrics[m_key][1] * 100
                all_max_values.append(mean)
                bar = ax.bar(
                    x_pos + conf_offsets[i] * width,
                    mean,
                    width,
                    color=color_map[key],
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
    confidence_keys = build_variant_method_keys(
        tuple(v for v in config.evaluation_variants if v.has_trained_confidence()),
        config.evaluation_methods,
    )

    if not confidence_keys:
        return

    first_key = confidence_keys[0]
    sample_metrics = load_all_concat_metrics_from_csv(
        first_key[0], "bc", method=first_key[1]
    )
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

    all_metrics = {
        key: load_all_concat_metrics_from_csv(key[0], "bc", method=key[1])
        for key in confidence_keys
    }

    color_map = get_variant_method_color_map()
    _, ax = plt.subplots(figsize=(14, 7))
    n_keys = len(confidence_keys)
    width = 0.8 / max(n_keys, 1)
    x = np.arange(len(percentage_keys))
    offsets = np.arange(n_keys) - (n_keys - 1) / 2
    all_max_values: list[float] = []

    for i, key in enumerate(confidence_keys):
        variant, method = key
        metrics = all_metrics[key]
        means = [metrics.get(k, 0) * 100 for k in percentage_keys]
        all_max_values.extend([m for m in means if not np.isnan(m)])

        legend_label = get_variant_method_label(variant, method)
        if mcc_key and mcc_key in metrics:
            mcc_value = metrics[mcc_key]
            legend_label = f"{legend_label} [MCC: {mcc_value:.2f}]"

        bars = ax.bar(
            x + offsets[i] * width,
            means,
            width,
            label=legend_label,
            color=color_map[key],
        )
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


def get_confidence_colormap() -> mcolors.LinearSegmentedColormap:
    """Return a red-to-green colormap for confidence values in [0, 1]."""
    return mcolors.LinearSegmentedColormap.from_list(
        "confidence_rg", ["#d62728", "#f0e442", "#2ca02c"]
    )


def escape_latex(text: str) -> str:
    """Escape characters that matplotlib interprets as LaTeX markup."""
    for char in ("\\", "$", "_", "^", "{", "}", "%", "#", "&", "~"):
        text = text.replace(char, f"\\{char}")
    return text


def add_gradient_line(
    ax: Axes,
    x: list[float],
    y: list[float],
) -> None:
    """Add a line colored by confidence value using a red-to-green gradient."""
    cmap = get_confidence_colormap()
    points = np.column_stack([x, y])
    segments = np.array([[points[i], points[i + 1]] for i in range(len(points) - 1)])
    segment_colors = np.array([(y[i] + y[i + 1]) / 2 for i in range(len(y) - 1)])
    lc = LineCollection(segments.tolist(), cmap=cmap, norm=mcolors.Normalize(0, 1))
    lc.set_array(segment_colors)
    lc.set_linewidth(1.0)
    ax.add_collection(lc)
    scatter_colors = np.array(y)
    ax.scatter(
        x,
        y,
        c=scatter_colors,
        cmap=cmap,
        norm=mcolors.Normalize(0, 1),
        s=8,
        zorder=5,
        edgecolors="none",
    )


def _configure_token_row_axes(
    ax: Axes,
    row_tokens: list[str],
    row_confs: list[float],
    row_len: int,
    row_height: float,
    fig_height: float,
    full_width: float,
    margin_left: float,
    tokens_per_row: int,
    y_cursor: float,
) -> float:
    """Configure a single row of token axes and return the updated y_cursor."""
    h = row_height / fig_height
    row_width = full_width * (row_len / tokens_per_row)
    y_cursor -= h
    ax.set_position((margin_left, y_cursor, row_width, h * 0.85))

    x_positions = list(range(row_len))
    add_gradient_line(ax, [float(x) for x in x_positions], row_confs)

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.5, row_len - 0.5)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        row_tokens,
        fontsize=3,
        fontfamily="monospace",
        rotation=10,
        ha="right",
    )
    ax.tick_params(axis="x", length=2, pad=1)
    ax.set_ylabel("Confidence", fontsize=5, labelpad=2)
    ax.tick_params(axis="y", labelsize=4)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.3, alpha=0.5)
    ax.grid(axis="y", alpha=0.2, linewidth=0.3)
    return y_cursor


def _compute_fig_height_and_axes(
    result: ResponseConfidenceResult,
    tokens_per_row: int,
    row_height: float,
    title_height: float,
    sample_gap: float,
) -> tuple[float, Figure, list[list[Axes]]]:
    """Compute figure height, create figure and allocate axes per sample."""
    num_samples = len(result.samples)
    total_height = 0.0
    for sample in result.samples:
        n_tokens = max(len(sample.steps), 1)
        total_height += (
            title_height + ((n_tokens - 1) // tokens_per_row + 1) * row_height
        )
    total_height += max(num_samples - 1, 0) * sample_gap

    fig_height = total_height + 0.5
    fig = plt.figure(figsize=(14, fig_height))

    sample_axes: list[list[Axes]] = []
    for sample in result.samples:
        n_tokens = max(len(sample.steps), 1)
        n_rows = (n_tokens - 1) // tokens_per_row + 1
        axes_for_sample: list[Axes] = []
        for _ in range(n_rows):
            ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
            axes_for_sample.append(ax)
        sample_axes.append(axes_for_sample)

    return fig_height, fig, sample_axes


def create_confidence_evolution_chart(
    result: ResponseConfidenceResult,
    variant: Variant,
    tokens_per_row: int = 80,
) -> None:
    """Generate a PDF chart showing confidence evolution for each sample."""
    configure_matplotlib_fonts()

    if len(result.samples) == 0:
        return

    row_height = 0.9
    title_height = 0.25
    sample_gap = 0.4
    fig_height, fig, sample_axes = _compute_fig_height_and_axes(
        result,
        tokens_per_row,
        row_height,
        title_height,
        sample_gap,
    )

    margin_left = 0.06
    full_width = 0.98 - margin_left
    y_cursor = 1.0 - 0.3 / fig_height

    for idx, sample in enumerate(result.samples):
        if idx > 0:
            y_cursor -= sample_gap / fig_height

        if not sample.steps:
            for ax in sample_axes[idx]:
                ax.set_visible(False)
            continue

        n_tokens = len(sample.steps)
        n_rows = (n_tokens - 1) // tokens_per_row + 1

        correctness_label = "Correct" if sample.is_correct else "Incorrect"
        title_text = (
            f"Sample {idx + 1} [{correctness_label}]\n"
            f"Mean Confidence: {sample.mean_confidence_sigmoid:.3f}\n"
            f"Question: {escape_latex(sample.question)}"
        )
        title_h = title_height / fig_height
        y_cursor -= title_h
        fig.text(
            margin_left,
            y_cursor + title_h * 0.5,
            title_text,
            fontsize=5,
            va="center",
            ha="left",
            wrap=True,
        )

        token_texts = [
            step.token_text.replace("\n", "\\n").replace("\r", "\\r")
            for step in sample.steps
        ]
        confidence_values = [step.confidence_sigmoid for step in sample.steps]

        for row_idx in range(n_rows):
            start = row_idx * tokens_per_row
            end = min(start + tokens_per_row, n_tokens)
            y_cursor = _configure_token_row_axes(
                sample_axes[idx][row_idx],
                token_texts[start:end],
                confidence_values[start:end],
                len(token_texts[start:end]),
                row_height,
                fig_height,
                full_width,
                margin_left,
                tokens_per_row,
                y_cursor,
            )

    fig.text(
        0.5,
        1.0 - 0.1 / fig_height,
        f"Confidence Evolution [Variant: {variant.get_shorthand()}]",
        fontsize=10,
        ha="center",
        va="top",
    )
    save_chart("confidence_evolution_chart.pdf")
