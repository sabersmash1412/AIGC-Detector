"""Tests that prevent Section 3 protocol and implementation drift."""

from __future__ import annotations

import json
from pathlib import Path

from src.image_transforms import DEFAULT_ROBUSTNESS_CONDITIONS


PROTOCOL_PATH = Path("configs/section3_experiment.json")


def _protocol() -> dict[str, object]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def test_protocol_conditions_match_registered_representative_conditions() -> None:
    protocol = _protocol()

    assert tuple(protocol["representative_conditions"]) == DEFAULT_ROBUSTNESS_CONDITIONS


def test_e2_and_e3_differ_only_by_declared_consistency_term() -> None:
    protocol = _protocol()
    experiments = protocol["experiments"]
    paired_training = protocol["paired_training"]

    assert experiments["E3"]["consistency_weight"] == 1.0
    assert "adding only consistency loss" in experiments["E3"]["training"]
    assert paired_training["initialization"].startswith("Existing E1")


def test_protocol_keeps_selection_on_validation_data() -> None:
    protocol = _protocol()

    assert protocol["model_selection"]["data"] == "validation only"
    assert protocol["threshold_selection"]["data"] == "validation only"
    assert protocol["data"]["organiser_validation_subset_used"] is False


def test_threshold_grid_and_tie_break_are_frozen() -> None:
    selection = _protocol()["threshold_selection"]

    assert selection["candidate_minimum"] == 0.001
    assert selection["candidate_maximum"] == 0.999
    assert selection["candidate_step"] == 0.001
    assert selection["tie_break_1"] == "Threshold closest to 0.5."
    assert selection["tie_break_2"].startswith("Lower threshold")
