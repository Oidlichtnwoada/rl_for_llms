import statistics

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from rl_for_llms.utils.constant_utils import get_boolean_classification_threshold
from rl_for_llms.utils.math_utils import compute_harmonic_mean


def compute_binary_classification_metrics(  # noqa: C901, PLR0912
    true_labels: list[float],
    predicted_labels: list[float],
    threshold: float = get_boolean_classification_threshold(),
) -> dict[tuple[str, ...], float]:
    """Compute binary classification metrics."""
    unique_true_labels = set(true_labels)
    if not unique_true_labels.issubset({0.0, 1.0}):
        raise ValueError
    true_labels_np = np.array(true_labels).astype(bool)
    predicted_labels_np = np.array(predicted_labels) >= threshold
    tn, fp, fn, tp = confusion_matrix(
        true_labels_np, predicted_labels_np, labels=[False, True]
    ).ravel()
    metrics: dict[tuple[str, ...], float] = {}
    if (denominator := (tp + fn)) > 0:
        tpr = tp / denominator
        metrics[("tpr",)] = tpr
    if (denominator := (tn + fp)) > 0:
        tnr = tn / denominator
        metrics[("tnr",)] = tnr
    if (denominator := (fp + tn)) > 0:
        fpr = fp / denominator
        metrics[("fpr",)] = fpr
    if (denominator := (fn + tp)) > 0:
        fnr = fn / denominator
        metrics[("fnr",)] = fnr
    if (denominator := (tp + fp)) > 0:
        ppv = tp / denominator
        metrics[("ppv",)] = ppv
    if (denominator := (tn + fn)) > 0:
        npv = tn / denominator
        metrics[("npv",)] = npv
    if (
        ("tpr",) in metrics
        and ("ppv",) in metrics
        and (metrics[("tpr",)] > 0 or metrics[("ppv",)] > 0)
    ):
        f1_score = compute_harmonic_mean(
            metrics[("tpr",)],
            metrics[("ppv",)],
        )
        metrics[("f1_score",)] = f1_score
    if (denominator := (tn + fp + fn + tp)) > 0:
        accuracy = (tp + tn) / denominator
        metrics[("accuracy",)] = accuracy
    if ("tpr",) in metrics and ("tnr",) in metrics:
        balanced_accuracy = statistics.mean((metrics[("tpr",)], metrics[("tnr",)]))
        metrics[("balanced_accuracy",)] = balanced_accuracy
        if metrics[("tpr",)] > 0 or metrics[("tnr",)] > 0:
            harmonic_balanced_accuracy = compute_harmonic_mean(
                metrics[("tpr",)], metrics[("tnr",)]
            )
            metrics[("harmonic_balanced_accuracy",)] = harmonic_balanced_accuracy
    if (denominator := ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5) > 0:
        mcc = (tp * tn - fp * fn) / denominator
        metrics[("mcc",)] = mcc
    if ("tpr",) in metrics and ("tnr",) in metrics:
        metrics[("roc_auc",)] = roc_auc_score(true_labels_np, predicted_labels)
    if ("tpr",) in metrics:
        metrics[("pr_auc",)] = average_precision_score(true_labels_np, predicted_labels)
    return metrics
