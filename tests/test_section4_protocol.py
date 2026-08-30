"""Tests that freeze the complete Section 4 transformation matrix."""

from __future__ import annotations

import json
from pathlib import Path

from src.image_transforms import (
    DEFAULT_ROBUSTNESS_CONDITIONS,
    FULL_ROBUSTNESS_CONDITIONS,
)


PROTOCOL_PATH = Path("configs/section4_evaluation.json")


def _protocol() -> dict[str, object]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def test_full_matrix_matches_transform_registry_order() -> None:
    protocol = _protocol()

    assert tuple(protocol["full_matrix_conditions"]) == FULL_ROBUSTNESS_CONDITIONS
    assert len(protocol["full_matrix_conditions"]) == 15


def test_new_severities_are_exactly_full_minus_representative() -> None:
    protocol = _protocol()
    new_conditions = set(protocol["new_severity_conditions"])

    assert tuple(protocol["previously_evaluated_representative_conditions"]) == (
        DEFAULT_ROBUSTNESS_CONDITIONS
    )
    assert new_conditions == set(FULL_ROBUSTNESS_CONDITIONS).difference(
        DEFAULT_ROBUSTNESS_CONDITIONS
    )
    assert len(new_conditions) == 8


def test_full_matrix_cannot_change_frozen_models() -> None:
    protocol = _protocol()
    rules = protocol["frozen_evaluation_rules"]

    assert rules["retraining_allowed"] is False
    assert rules["threshold_changes_allowed"] is False
    assert rules["model_reselection_from_matrix_results_allowed"] is False
    assert protocol["preselected_primary_model"] == "E3"
    assert protocol["data"]["organiser_validation_subset_used"] is False
