"""Tests for deterministic stratified paired-bootstrap uncertainty."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src.bootstrap_uncertainty import (
    bootstrap_binary_metrics,
    confidence_interval,
    make_stratified_draws,
    validate_protocol,
)


PROTOCOL_PATH = Path("configs/section4c_bootstrap.json")


def test_checked_in_bootstrap_protocol_is_frozen() -> None:
    validate_protocol(json.loads(PROTOCOL_PATH.read_text(encoding="utf-8")))


@pytest.mark.parametrize(
    "rule",
    [
        "retraining_allowed",
        "threshold_changes_allowed",
        "model_reselection_allowed",
        "organiser_validation_subset_used",
        "bootstrap_results_allowed_to_change_models_or_thresholds",
    ],
)
def test_protocol_rejects_relaxed_guardrail(rule: str) -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol = copy.deepcopy(protocol)
    protocol["frozen_guardrails"][rule] = True

    with pytest.raises(ValueError, match="Frozen uncertainty rule changed"):
        validate_protocol(protocol)


def test_stratified_draws_are_deterministic_and_class_preserving() -> None:
    labels = np.asarray([0, 0, 0, 1, 1])
    first = make_stratified_draws(labels, replicates=20, seed=42)
    second = make_stratified_draws(labels, replicates=20, seed=42)

    assert np.array_equal(first.negative, second.negative)
    assert np.array_equal(first.positive, second.positive)
    assert first.negative.shape == (20, 3)
    assert first.positive.shape == (20, 2)
    assert np.all(labels[first.negative] == 0)
    assert np.all(labels[first.positive] == 1)


def test_perfect_classifier_has_degenerate_auc_and_accuracy_interval() -> None:
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    probabilities = np.asarray([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])
    draws = make_stratified_draws(labels, replicates=100, seed=7)

    metrics = bootstrap_binary_metrics(
        probabilities, threshold=0.5, draws=draws
    )

    assert np.all(metrics["roc_auc"] == 1.0)
    assert np.all(metrics["balanced_accuracy"] == 1.0)
    assert np.all(metrics["real_recall"] == 1.0)
    assert np.all(metrics["ai_recall"] == 1.0)


def test_flip_rate_uses_paired_source_indices() -> None:
    labels = np.asarray([0, 0, 1, 1])
    clean = np.asarray([0.1, 0.9, 0.1, 0.9])
    transformed = 1.0 - clean
    draws = make_stratified_draws(labels, replicates=50, seed=5)

    metrics = bootstrap_binary_metrics(
        transformed,
        threshold=0.5,
        draws=draws,
        clean_probabilities=clean,
    )

    assert np.all(metrics["prediction_flip_rate"] == 1.0)


def test_confidence_interval_uses_requested_percentiles() -> None:
    values = np.arange(100, dtype=np.float64)

    interval = confidence_interval(49.5, values, confidence_level=0.90)

    assert interval["lower"] == pytest.approx(np.quantile(values, 0.05))
    assert interval["upper"] == pytest.approx(np.quantile(values, 0.95))
    assert interval["replicates"] == 100
