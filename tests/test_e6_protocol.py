from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path

import pytest

from src.e6_protocol import (
    assignment_sha256,
    build_pair_assignment,
    load_e6_protocol,
    pair_rows,
    validate_artifacts_absent,
    validate_e6_development_protocol,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/e6_development_protocol.json"


def _protocol() -> dict:
    return load_e6_protocol(PROTOCOL_PATH)


def test_checked_in_e6_protocol_preserves_the_pre_payload_contract() -> None:
    protocol = _protocol()
    validate_e6_development_protocol(protocol)

    assert protocol["metadata_access_preflight"]["image_payload_downloaded"] is False
    assert protocol["shortcut_gate"]["forensic_training_allowed_before_pass"] is False
    assert protocol["shortcut_gate"]["semantic_training_allowed_before_pass"] is False


def test_biggan_row_topology_and_assignment_are_deterministic_and_paired() -> None:
    protocol = _protocol()
    first = build_pair_assignment(protocol)
    second = build_pair_assignment(copy.deepcopy(protocol))

    assert first == second
    assert assignment_sha256(first) == assignment_sha256(second)
    assert len(first) == 1400
    assert len({item["pair_index"] for item in first}) == 1400
    assert len({item["pair_key"] for item in first}) == 1400
    assert pair_rows(protocol, 0) == (2, 3)
    assert pair_rows(protocol, 1999) == (27988, 27989)
    assert all(item["ai_row_index"] == item["real_row_index"] + 1 for item in first)

    role_counts = Counter(item["role"] for item in first)
    assert role_counts == {
        "development_train": 800,
        "development_model_selection": 200,
        "development_calibration": 200,
        "development_threshold_selection": 200,
    }


@pytest.mark.parametrize("cycle", [-1, 2000, True, 1.5])
def test_biggan_pair_rows_reject_invalid_cycles(cycle: object) -> None:
    with pytest.raises(ValueError, match="cycle"):
        pair_rows(_protocol(), cycle)  # type: ignore[arg-type]


def test_protocol_rejects_an_existing_or_test_generator_as_the_new_family() -> None:
    protocol = copy.deepcopy(_protocol())
    protocol["development_source"]["selected_generator"] = "Midjourney V6"

    with pytest.raises(ValueError, match="BigGAN"):
        validate_e6_development_protocol(protocol)


def test_protocol_rejects_sid_tampering_relabelled_as_full_synthesis() -> None:
    protocol = copy.deepcopy(_protocol())
    protocol["task_scope"][
        "sid_set_label_2_allowed_as_third_full_synthetic_generator"
    ] = True

    with pytest.raises(ValueError, match="sid_set_label_2"):
        validate_e6_development_protocol(protocol)


def test_protocol_rejects_unpinned_or_changed_source_revision() -> None:
    protocol = copy.deepcopy(_protocol())
    protocol["development_source"]["repository_revision"] = "main"

    with pytest.raises(ValueError, match="40-character revision"):
        validate_e6_development_protocol(protocol)


def test_protocol_rejects_model_selection_using_threshold_data() -> None:
    protocol = copy.deepcopy(_protocol())
    protocol["role_permissions"]["development_model_selection"][
        "select_decision_thresholds"
    ] = True

    with pytest.raises(ValueError, match="role conflation"):
        validate_e6_development_protocol(protocol)


def test_protocol_rejects_changed_or_unbalanced_role_counts() -> None:
    protocol = copy.deepcopy(_protocol())
    protocol["selection"]["pairs_per_role"]["development_train"] = 900

    with pytest.raises(ValueError, match="role pair counts"):
        validate_e6_development_protocol(protocol)


def test_protocol_rejects_consumed_test_reuse() -> None:
    protocol = copy.deepcopy(_protocol())
    protocol["isolation"]["consumed_test_ids_excluded"].remove(
        "aigibench_midjourney_v6"
    )

    with pytest.raises(ValueError, match="consumed-test exclusion"):
        validate_e6_development_protocol(protocol)


def test_protocol_rejects_relaxed_metadata_shortcut_gate() -> None:
    protocol = copy.deepcopy(_protocol())
    protocol["shortcut_gate"]["maximum_acceptable_auc"] = 0.95

    with pytest.raises(ValueError, match="shortcut acceptance gate"):
        validate_e6_development_protocol(protocol)


def test_protocol_rejects_training_before_shortcut_gate_passes() -> None:
    protocol = copy.deepcopy(_protocol())
    protocol["shortcut_gate"]["forensic_training_allowed_before_pass"] = True

    with pytest.raises(ValueError, match="forensic_training_allowed_before_pass"):
        validate_e6_development_protocol(protocol)


def test_protocol_detects_artifact_that_predates_the_lock(tmp_path: Path) -> None:
    protocol = _protocol()
    raw_root = tmp_path / protocol["outputs"]["raw_root"]
    raw_root.mkdir(parents=True)

    with pytest.raises(ValueError, match="existed before protocol lock"):
        validate_artifacts_absent(protocol, tmp_path)
