from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from src.e6_acquisition_protocol import (
    DEFAULT_E6_ACQUISITION_PROTOCOL,
    acquisition_payload_paths,
    load_e6_acquisition_protocol,
    sha256_file,
    validate_acquisition_payloads_absent,
    validate_e6_acquisition_protocol,
    validate_upstream_locks,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / DEFAULT_E6_ACQUISITION_PROTOCOL
LOCK_PATH = ROOT / "reports/e6_acquisition_protocol_lock.json"


def _protocol() -> dict:
    return load_e6_acquisition_protocol(PROTOCOL_PATH)


def _minimal_project(tmp_path: Path) -> Path:
    for relative in (
        Path("configs/e6_development_protocol.json"),
        Path("configs/e6_acquisition_protocol.json"),
        Path("reports/e6_development_protocol_lock.json"),
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    return tmp_path


def test_checked_in_acquisition_protocol_and_upstream_locks_pass_offline() -> None:
    protocol = _protocol()

    validate_e6_acquisition_protocol(protocol)
    development, lock = validate_upstream_locks(protocol, ROOT)

    assert development["development_source"]["selected_generator"] == "BigGAN"
    assert lock["status"] == "PASS"
    assert lock["image_payload_present_at_freeze"] is False
    assert protocol["trusted_asset_transport"]["redirects_allowed"] is False
    assert protocol["trusted_asset_transport"][
        "signed_url_or_query_persisted"
    ] is False


def test_checked_in_lock_receipt_binds_the_frozen_pre_payload_state() -> None:
    protocol = _protocol()
    observed = json.loads(LOCK_PATH.read_text(encoding="utf-8"))

    assert observed["status"] == "PASS"
    assert observed["acquisition_protocol"] == {
        "path": "configs/e6_acquisition_protocol.json",
        "sha256": sha256_file(PROTOCOL_PATH),
        "schema_version": 1,
    }
    assert observed["checked_absent_paths"] == [
        path.relative_to(ROOT).as_posix()
        for path in acquisition_payload_paths(protocol, ROOT)
    ]
    assert observed["payload_present_at_freeze"] is False
    assert observed["guardrails"]["training_or_feature_extraction_performed"] is False
    assert observed["guardrails"]["evaluation_performed"] is False


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("source_access", "rows_endpoint", "http://example.test/rows", "endpoint"),
        ("trusted_asset_transport", "redirects_allowed", True, "transport"),
        (
            "trusted_asset_transport",
            "signed_url_or_query_persisted",
            True,
            "transport",
        ),
        ("resource_caps", "maximum_asset_bytes", 2**40, "resource caps"),
        (
            "filesystem_safety",
            "symlink_in_any_output_component_allowed",
            True,
            "no-symlink",
        ),
    ],
)
def test_protocol_rejects_remote_or_filesystem_security_drift(
    section: str, key: str, value: object, message: str
) -> None:
    protocol = copy.deepcopy(_protocol())
    protocol[section][key] = value

    with pytest.raises(ValueError, match=message):
        validate_e6_acquisition_protocol(protocol)


def test_protocol_rejects_new_image_type() -> None:
    protocol = copy.deepcopy(_protocol())
    protocol["image_validation"]["allowed_formats"]["TIFF"] = {
        "media_types": ["image/tiff"],
        "extensions": [".tiff"],
    }

    with pytest.raises(ValueError, match="image types"):
        validate_e6_acquisition_protocol(protocol)


def test_protocol_rejects_decoded_hash_or_phash_threshold_drift() -> None:
    protocol = copy.deepcopy(_protocol())
    protocol["identity_algorithms"]["decoded_rgba_sha256"]["colour_mode"] = "RGB"
    with pytest.raises(ValueError, match="RGBA"):
        validate_e6_acquisition_protocol(protocol)

    protocol = copy.deepcopy(_protocol())
    protocol["identity_algorithms"]["perceptual_hash"][
        "near_duplicate_max_distance_inclusive"
    ] = 10
    with pytest.raises(ValueError, match="pHash"):
        validate_e6_acquisition_protocol(protocol)


def test_existing_overlap_and_conflicting_labels_can_never_use_replacements() -> None:
    protocol = copy.deepcopy(_protocol())
    policy = protocol["overlap_and_replacement"]
    assert policy["conflicting_label_duplicate_action"] == (
        "hard_abort_without_replacement"
    )
    assert policy["existing_dataset_exact_decoded_or_near_overlap_action"] == (
        "hard_abort_without_replacement"
    )

    policy["existing_dataset_exact_decoded_or_near_overlap_action"] = (
        "reject_whole_pair_and_replace"
    )
    with pytest.raises(ValueError, match="overlap precedence"):
        validate_e6_acquisition_protocol(protocol)


def test_only_corrupt_or_within_e6_same_label_duplicates_may_use_reserve() -> None:
    protocol = _protocol()
    policy = protocol["overlap_and_replacement"]

    assert policy["corrupt_remote_asset_action"] == "reject_whole_pair_and_replace"
    assert policy[
        "within_e6_same_label_same_role_or_cross_role_duplicate_action"
    ] == "reject_later_whole_pair_and_replace"
    assert policy["replacement_source"] == (
        "next unused pair in the frozen SHA-256 rank order"
    )
    assert policy["replacement_keeps_rejected_pair_role"] is True
    assert policy["replacement_is_whole_pair"] is True
    assert policy["manual_replacement_allowed"] is False


def test_upstream_lock_hash_binding_fails_closed(tmp_path: Path) -> None:
    project = _minimal_project(tmp_path)
    upstream_lock = project / "reports/e6_development_protocol_lock.json"
    upstream_lock.write_text(
        upstream_lock.read_text(encoding="utf-8") + " ", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="lock SHA-256 changed"):
        validate_upstream_locks(_protocol(), project)


def test_any_preexisting_payload_or_staging_path_blocks_the_lock(
    tmp_path: Path,
) -> None:
    project = _minimal_project(tmp_path)
    protocol = load_e6_acquisition_protocol(
        project / "configs/e6_acquisition_protocol.json"
    )
    staging = project / "data/raw/.e6_tiny_genimage_biggan_staging"
    staging.mkdir(parents=True)

    with pytest.raises(ValueError, match="existed before protocol lock"):
        validate_acquisition_payloads_absent(protocol, project)


def test_payload_inventory_has_no_absolute_or_parent_traversal_paths() -> None:
    paths = acquisition_payload_paths(_protocol(), ROOT)

    assert len(paths) == 8
    assert len(set(paths)) == len(paths)
    assert all(path.is_absolute() and path.is_relative_to(ROOT) for path in paths)
