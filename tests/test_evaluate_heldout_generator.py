"""Tests for held-out-generator evaluation summaries."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluate_heldout_generator import (
    class_rates,
    domain_metric_deltas,
    probability_distribution_summary,
)
from src.metrics import binary_classification_metrics


def test_class_rates_name_real_and_ai_errors_explicitly() -> None:
    metrics = {"confusion_matrix": [[80, 20], [30, 70]]}

    rates = class_rates(metrics)

    assert rates == {
        "true_negative_rate_real_recall": 0.8,
        "false_positive_rate_real_called_ai": 0.2,
        "true_positive_rate_ai_recall": 0.7,
        "false_negative_rate_ai_called_real": 0.3,
    }


def test_probability_summary_separates_real_and_flux_scores() -> None:
    labels = np.asarray([0, 0, 1, 1])
    probabilities = np.asarray([0.1, 0.3, 0.7, 0.9])

    result = probability_distribution_summary(labels, probabilities)

    assert result["real_0"]["samples"] == 2
    assert result["flux_ai_generated_1"]["samples"] == 2
    assert result["real_0"]["mean"] == pytest.approx(0.2)
    assert result["flux_ai_generated_1"]["mean"] == pytest.approx(0.8)


def test_domain_deltas_are_heldout_minus_internal_clean() -> None:
    labels = np.asarray([0, 0, 1, 1])
    clean = binary_classification_metrics(labels, [0.1, 0.2, 0.8, 0.9], 0.5)
    heldout = binary_classification_metrics(labels, [0.1, 0.7, 0.3, 0.9], 0.5)

    deltas = domain_metric_deltas(clean, heldout)

    assert deltas["balanced_accuracy"] == -0.5
    assert deltas["true_negative_rate_real_recall"] == -0.5
    assert deltas["true_positive_rate_ai_recall"] == -0.5
    assert deltas["false_positive_rate_real_called_ai"] == 0.5
    assert deltas["false_negative_rate_ai_called_real"] == 0.5


def test_class_rates_rejects_missing_class() -> None:
    with pytest.raises(ValueError, match="Both classes"):
        class_rates({"confusion_matrix": [[4, 0], [0, 0]]})
