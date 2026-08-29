"""Tests for robustness probability-stability and metric-delta helpers."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluate_initial_robustness import metric_deltas, stability_metrics


def test_stability_metrics_measure_probability_and_label_changes() -> None:
    labels = np.asarray([0, 0, 1, 1])
    clean = np.asarray([0.1, 0.4, 0.6, 0.9])
    transformed = np.asarray([0.2, 0.7, 0.3, 0.8])

    result = stability_metrics(labels, clean, transformed, threshold=0.5)

    assert result["mean_absolute_probability_change"] == pytest.approx(0.2)
    assert result["prediction_flip_rate"] == 0.5
    assert result["real_mean_probability_shift"] == pytest.approx(0.2)
    assert result["ai_generated_mean_probability_shift"] == pytest.approx(-0.2)


def test_metric_deltas_are_transformed_minus_clean() -> None:
    keys = (
        "roc_auc",
        "average_precision",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "brier_score",
    )
    clean = {key: 0.8 for key in keys}
    transformed = {key: 0.7 for key in keys}

    deltas = metric_deltas(clean, transformed)

    assert all(value == pytest.approx(-0.1) for value in deltas.values())
