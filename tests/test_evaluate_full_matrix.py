"""Tests for the frozen Section 4A matrix protocol and summaries."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src.evaluate_full_matrix import (
    add_predeclared_summaries,
    subset_summary,
    validate_section4_protocol,
)
from src.evaluate_section3 import evaluate_model_conditions


PROTOCOL_PATH = Path("configs/section4_evaluation.json")


def _protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _model_result() -> dict:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    protocol = _protocol()
    probabilities = {"clean": np.asarray([0.1, 0.2, 0.8, 0.9])}
    for index, condition in enumerate(protocol["full_matrix_conditions"][1:]):
        probabilities[condition] = (
            np.asarray([0.1, 0.2, 0.8, 0.9])
            if index % 2 == 0
            else np.asarray([0.1, 0.7, 0.3, 0.9])
        )
    return evaluate_model_conditions(labels, probabilities, threshold=0.5)


def test_checked_in_section4_protocol_is_frozen_and_complete() -> None:
    validate_section4_protocol(_protocol())


@pytest.mark.parametrize(
    "rule",
    [
        "retraining_allowed",
        "threshold_changes_allowed",
        "model_reselection_from_matrix_results_allowed",
    ],
)
def test_protocol_rejects_enabling_post_test_tuning(rule: str) -> None:
    protocol = copy.deepcopy(_protocol())
    protocol["frozen_evaluation_rules"][rule] = True

    with pytest.raises(ValueError, match="Frozen evaluation rule changed"):
        validate_section4_protocol(protocol)


def test_subset_summary_reports_worst_condition_and_flip_rate() -> None:
    result = _model_result()
    conditions = ("jpeg_q90", "jpeg_q70")

    summary = subset_summary(result, conditions)

    assert summary["mean_balanced_accuracy"] == 0.75
    assert summary["worst_balanced_accuracy"] == 0.5
    assert summary["worst_balanced_accuracy_condition"] == "jpeg_q70"
    assert summary["maximum_prediction_flip_rate"] == 0.5


def test_predeclared_summaries_keep_seen_new_and_families_separate() -> None:
    protocol = _protocol()
    summaries = add_predeclared_summaries(_model_result(), protocol)

    assert len(summaries["all_14_transformed"]["conditions"]) == 14
    assert len(summaries["six_representative_transforms"]["conditions"]) == 6
    assert len(summaries["eight_new_severities"]["conditions"]) == 8
    assert tuple(summaries["by_predeclared_family"]) == tuple(
        protocol["severity_families"]
    )
