from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.evaluate_e4 import (
    assert_metrics_reproduced,
    heldout_comparison_deltas,
    validate_evaluation_protocol,
)
from src.metrics import binary_classification_metrics


PROTOCOL_PATH = Path("configs/e4_posthoc_evaluation.json")


def _protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def test_checked_in_e4_evaluation_protocol_is_frozen() -> None:
    protocol = _protocol()
    validate_evaluation_protocol(protocol)
    assert protocol["positioning"]["E4_results_are_design_independent_fresh_test"] is False
    assert protocol["positioning"]["fresh_external_test_still_required"] is True
    assert protocol["frozen_rules"]["organiser_validation_subset_used"] is False


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("positioning", "E3_remains_original_primary_model", False),
        ("positioning", "E4_results_are_design_independent_fresh_test", True),
        ("frozen_rules", "retraining_allowed", True),
        ("frozen_rules", "threshold_changes_allowed", True),
        ("frozen_rules", "overwrite_original_section4_artifacts_allowed", True),
        ("frozen_rules", "organiser_validation_subset_used", True),
    ],
)
def test_e4_evaluation_protocol_rejects_guardrail_drift(
    section: str, key: str, value: bool
) -> None:
    protocol = copy.deepcopy(_protocol())
    protocol[section][key] = value
    with pytest.raises(ValueError):
        validate_evaluation_protocol(protocol)


def test_e4_evaluation_protocol_rejects_original_report_overwrite() -> None:
    protocol = copy.deepcopy(_protocol())
    protocol["outputs"]["report"] = protocol["frozen_inputs"]["cifake_full_matrix"][
        "original_report"
    ]
    with pytest.raises(ValueError, match="overwrite"):
        validate_evaluation_protocol(protocol)


def test_heldout_deltas_include_real_fpr_and_ai_recall() -> None:
    labels = [0, 0, 1, 1]
    e3 = binary_classification_metrics(labels, [0.8, 0.7, 0.9, 0.8], 0.5)
    e4 = binary_classification_metrics(labels, [0.1, 0.2, 0.9, 0.4], 0.5)
    deltas = heldout_comparison_deltas(e3, e4)
    assert deltas["false_positive_rate_real_called_ai"] == pytest.approx(-1.0)
    assert deltas["true_positive_rate_ai_recall"] == pytest.approx(-0.5)
    assert deltas["balanced_accuracy"] == pytest.approx(0.25)


def test_metric_reproduction_accepts_identical_and_rejects_drift() -> None:
    metrics = binary_classification_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.5)
    assert_metrics_reproduced(metrics, copy.deepcopy(metrics), context="toy")
    changed = copy.deepcopy(metrics)
    changed["balanced_accuracy"] -= 0.01
    with pytest.raises(ValueError, match="did not reproduce"):
        assert_metrics_reproduced(metrics, changed, context="toy")
