from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.e5_protocol import (
    validate_e5_protocol,
    validate_frozen_e5_inputs,
    wilson_interval,
)


PROTOCOL_PATH = Path("configs/e5_source_matched_adaptation.json")


def _protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def test_checked_in_e5_protocol_and_existing_inputs_are_frozen() -> None:
    protocol = _protocol()
    validate_e5_protocol(protocol)
    validate_frozen_e5_inputs(protocol, Path.cwd())
    assert protocol["training"]["group_sampling_per_epoch"] == {
        "cifake_real": 0.25,
        "cifake_ai_generated": 0.25,
        "sid_set_train_real": 0.25,
        "sid_set_train_flux": 0.25,
    }


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("research_positioning", "E3_remains_original_primary_binary_baseline", False),
        ("research_positioning", "E4_remains_documented_failed_replacement", False),
        ("research_positioning", "universal_detector_claim_allowed", True),
        ("research_positioning", "fresh_external_test_required", False),
    ],
)
def test_e5_protocol_rejects_research_positioning_drift(
    section: str, key: str, value: bool
) -> None:
    protocol = copy.deepcopy(_protocol())
    protocol[section][key] = value
    with pytest.raises(ValueError):
        validate_e5_protocol(protocol)


@pytest.mark.parametrize(
    "key",
    [
        "allowed_for_training",
        "allowed_for_validation",
        "allowed_for_model_threshold_or_hyperparameter_selection",
    ],
)
def test_e5_protocol_rejects_prior_audit_leakage(key: str) -> None:
    protocol = copy.deepcopy(_protocol())
    protocol["forbidden_development_data"]["section4_sid_flux_audit"][key] = True
    with pytest.raises(ValueError, match="audit leakage"):
        validate_e5_protocol(protocol)


def test_e5_protocol_rejects_source_label_confounding() -> None:
    protocol = copy.deepcopy(_protocol())
    protocol["training"]["group_sampling_per_epoch"] = {
        "cifake_ai_generated": 0.5,
        "cifake_real": 0.25,
        "sid_set_train_real": 0.25,
    }
    with pytest.raises(ValueError, match="group balance"):
        validate_e5_protocol(protocol)


def test_e5_protocol_rejects_changed_anchor_search() -> None:
    protocol = copy.deepcopy(_protocol())
    protocol["training"]["anchor_weights"] = [0.0, 0.01]
    with pytest.raises(ValueError, match="anchor"):
        validate_e5_protocol(protocol)


def test_e5_protocol_rejects_relaxed_error_constraint() -> None:
    protocol = copy.deepcopy(_protocol())
    protocol["validation_and_decision_selection"]["confident_error_constraints"][
        "maximum_wilson_upper_real_called_ai"
    ] = 0.2
    with pytest.raises(ValueError, match="false-accusation"):
        validate_e5_protocol(protocol)


def test_e5_protocol_rejects_all_uncertain_loophole() -> None:
    protocol = copy.deepcopy(_protocol())
    protocol["validation_and_decision_selection"][
        "anti_trivial_abstention_constraints"
    ]["minimum_mean_source_condition_decisive_coverage"] = 0.0
    with pytest.raises(ValueError, match="abstention"):
        validate_e5_protocol(protocol)


def test_e5_protocol_requires_new_external_domains() -> None:
    protocol = copy.deepcopy(_protocol())
    protocol["fresh_external_evaluation"][
        "generator_family_must_be_absent_from_E5_development"
    ] = False
    with pytest.raises(ValueError, match="fresh-test"):
        validate_e5_protocol(protocol)


@pytest.mark.parametrize(
    ("successes", "trials", "expected"),
    [
        (0, 1000, (0.0, 0.0038267584855551234)),
        (50, 1000, (0.03813026239274882, 0.06531382024425081)),
        (1000, 1000, (0.9961732415144449, 1.0)),
    ],
)
def test_wilson_interval_known_values(
    successes: int, trials: int, expected: tuple[float, float]
) -> None:
    observed = wilson_interval(successes, trials)
    assert observed == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("successes,trials", [(-1, 10), (11, 10), (0, 0)])
def test_wilson_interval_rejects_invalid_counts(successes: int, trials: int) -> None:
    with pytest.raises(ValueError):
        wilson_interval(successes, trials)
