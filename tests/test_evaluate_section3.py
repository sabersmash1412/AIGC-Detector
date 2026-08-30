"""Tests for frozen Section 3 test aggregation and comparisons."""

from __future__ import annotations

import numpy as np

from src.evaluate_section3 import comparison_deltas, evaluate_model_conditions


def _probabilities(improved: bool) -> dict[str, np.ndarray]:
    if improved:
        transformed = np.asarray([0.1, 0.2, 0.8, 0.9])
    else:
        transformed = np.asarray([0.1, 0.7, 0.3, 0.9])
    return {
        "clean": np.asarray([0.1, 0.2, 0.8, 0.9]),
        "jpeg_q50": transformed,
        "gaussian_blur_sigma1": transformed,
    }


def test_evaluate_model_conditions_separates_clean_and_transformed_summary() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)

    result = evaluate_model_conditions(labels, _probabilities(False), threshold=0.5)

    assert result["clean"]["metrics"]["balanced_accuracy"] == 1.0
    assert result["transformed_summary"]["mean_balanced_accuracy"] == 0.5
    assert result["transformed_summary"]["mean_prediction_flip_rate"] == 0.5
    assert result["transformed_summary"]["worst_balanced_accuracy_condition"] in {
        "jpeg_q50",
        "gaussian_blur_sigma1",
    }


def test_comparison_deltas_are_candidate_minus_baseline() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    baseline = evaluate_model_conditions(labels, _probabilities(False), threshold=0.5)
    improved = evaluate_model_conditions(labels, _probabilities(True), threshold=0.5)

    deltas = comparison_deltas(baseline, improved)

    assert deltas["clean_balanced_accuracy"] == 0.0
    assert deltas["mean_transformed_balanced_accuracy"] == 0.5
    assert deltas["mean_prediction_flip_rate"] == -0.5
    assert deltas["by_condition"]["jpeg_q50"]["balanced_accuracy"] == 0.5
