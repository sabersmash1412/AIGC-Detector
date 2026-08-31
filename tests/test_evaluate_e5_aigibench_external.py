from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src.evaluate_e5_aigibench_external import (
    CLASS_NAMES,
    EXPECTED_PRIMARY,
    evaluate_frozen_gates,
    triage_class_metrics,
    validate_run_lock,
)


ROOT = Path(__file__).resolve().parents[1]
RUN_LOCK = ROOT / "configs/e5_aigibench_external_evaluation_run.json"


def test_committed_run_lock_retains_frozen_design() -> None:
    validate_run_lock(json.loads(RUN_LOCK.read_text(encoding="utf-8")))


def test_run_lock_rejects_threshold_drift() -> None:
    run = json.loads(RUN_LOCK.read_text(encoding="utf-8"))
    run["comparison_models"]["E5"]["ai_threshold"] = 0.8
    with pytest.raises(ValueError, match="selection changed"):
        validate_run_lock(run)


def test_triage_class_metrics_uses_all_class_examples_as_risk_denominator() -> None:
    probabilities = np.asarray([0.1, 0.2, 0.4, 0.7, 0.9], dtype=np.float64)
    metrics = triage_class_metrics(
        probabilities,
        true_label=0,
        real_threshold=0.2,
        ai_threshold=0.8,
        confidence_level=0.95,
    )
    assert metrics["called_real"] == 2
    assert metrics["uncertain"] == 2
    assert metrics["called_ai_generated"] == 1
    assert metrics["decisive_coverage"] == pytest.approx(0.6)
    assert metrics["decisive_accuracy"] == pytest.approx(2 / 3)
    assert metrics["confident_error_rate"] == pytest.approx(0.2)
    assert metrics["confident_error_wilson_upper"] > 0.2


def _passing_condition() -> dict[str, object]:
    return {
        "binary_metrics": {"roc_auc": 0.9},
        "triage_by_class": {
            CLASS_NAMES[0]: {
                "confident_error_wilson_upper": 0.04,
                "decisive_coverage": 0.7,
            },
            CLASS_NAMES[1]: {
                "confident_error_wilson_upper": 0.08,
                "decisive_coverage": 0.65,
            },
        },
    }


def test_frozen_gate_pass_and_failure_are_all_required() -> None:
    run = json.loads(RUN_LOCK.read_text(encoding="utf-8"))
    conditions = {condition: _passing_condition() for condition in EXPECTED_PRIMARY}
    passed = evaluate_frozen_gates(
        conditions,
        primary_conditions=EXPECTED_PRIMARY,
        criteria=run["frozen_success_criteria"],
    )
    assert passed["status"] == "PASS"
    assert all(passed["checks"].values())

    failed_conditions = copy.deepcopy(conditions)
    failed_conditions["resize_0_5x"]["triage_by_class"][CLASS_NAMES[0]][
        "confident_error_wilson_upper"
    ] = 0.051
    failed = evaluate_frozen_gates(
        failed_conditions,
        primary_conditions=EXPECTED_PRIMARY,
        criteria=run["frozen_success_criteria"],
    )
    assert failed["status"] == "FAIL"
    assert failed["checks"]["real_called_ai_risk"] is False

