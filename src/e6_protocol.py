"""Validation and deterministic split helpers for the frozen E6 protocol.

This module deliberately performs no network access, image download, feature
extraction or model training.  E6D freezes those decisions first so later
scripts cannot quietly change the data after seeing a result.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_E6_PROTOCOL = Path("configs/e6_development_protocol.json")
EXPECTED_REPOSITORY = "TheKernel01/Tiny-GenImage"
EXPECTED_REPOSITORY_REVISION = "89c4fe9efd0ebc7ce5c7641ef57d578ccd639c69"
EXPECTED_UPLOADER_REVISION = "ad5cacfb9321b0f3abb54941e224f2e8af357d9d"
EXPECTED_GENERATOR = "BigGAN"
EXPECTED_SEED = 606
EXPECTED_PAIR_COUNTS = {
    "development_train": 800,
    "development_model_selection": 200,
    "development_calibration": 200,
    "development_threshold_selection": 200,
}
EXPECTED_PERMISSIONS = {
    "development_train": (True, False, False, False),
    "development_model_selection": (False, True, False, False),
    "development_calibration": (False, False, True, False),
    "development_threshold_selection": (False, False, False, True),
}
EXPECTED_CONSUMED_TESTS = {
    "cifake_internal_test",
    "sid_set_validation_flux_audit",
    "aigibench_midjourney_v6",
}
EXPECTED_OUTPUTS = {
    "raw_root": "data/raw/e6_tiny_genimage_biggan",
    "manifest_root": "data/processed/e6_tiny_genimage_biggan",
    "provenance": "data/processed/e6_tiny_genimage_biggan_provenance.json",
    "existing_validation_role_root": "data/processed/e6_existing_validation_roles",
    "shortcut_audit": "reports/e6_tiny_genimage_shortcut_audit.json",
    "clip_feature_root": "data/features/e6_tiny_genimage_clip_vit_b32_quickgelu_openai",
    "forensic_feature_root": "data/features/e6_tiny_genimage_forensic_v1",
}


def load_e6_protocol(path: Path = DEFAULT_E6_PROTOCOL) -> dict[str, Any]:
    """Load an E6 development protocol from JSON."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("E6 protocol must be a JSON object")
    return payload


def _require_bool(container: dict[str, Any], key: str, expected: bool) -> None:
    if container.get(key) is not expected:
        raise ValueError(f"E6 protocol requires {key}={expected}")


def _require_revision(value: Any, description: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError(f"{description} must be a pinned 40-character revision")
    return value


def _safe_project_path(relative: Any, description: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{description} must be a project-relative path")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{description} must stay inside the project")
    return path


def pair_rows(protocol: dict[str, Any], cycle: int) -> tuple[int, int]:
    """Return the frozen real and BigGAN row indexes for one source cycle."""

    topology = protocol["row_topology"]
    available = int(topology["available_cycles"])
    if isinstance(cycle, bool) or not isinstance(cycle, int) or not 0 <= cycle < available:
        raise ValueError(f"BigGAN cycle must be in [0, {available})")
    stride = int(topology["rows_per_cycle"])
    return (
        cycle * stride + int(topology["biggan_real_offset"]),
        cycle * stride + int(topology["biggan_ai_offset"]),
    )


def _rank_digest(protocol: dict[str, Any], cycle: int) -> bytes:
    selection = protocol["selection"]
    revision = protocol["development_source"]["repository_revision"]
    key = f"e6d:{int(selection['seed'])}:{revision}:biggan:{cycle}"
    return hashlib.sha256(key.encode("utf-8")).digest()


def build_pair_assignment(protocol: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """Build the exact role assignment without reading any image or remote row."""

    validate_e6_development_protocol(protocol)
    candidate_pairs = int(protocol["selection"]["candidate_pairs"])
    ranked = sorted(
        range(candidate_pairs), key=lambda cycle: (_rank_digest(protocol, cycle), cycle)
    )
    assignments: list[dict[str, Any]] = []
    cursor = 0
    for role in protocol["selection"]["roles_in_assignment_order"]:
        count = int(protocol["selection"]["pairs_per_role"][role])
        for cycle in ranked[cursor : cursor + count]:
            real_row, ai_row = pair_rows(protocol, cycle)
            assignments.append(
                {
                    "role": role,
                    "pair_index": cycle,
                    "pair_key": f"tiny_genimage_train_biggan_cycle_{cycle}",
                    "real_row_index": real_row,
                    "ai_row_index": ai_row,
                }
            )
        cursor += count
    if cursor != int(protocol["selection"]["selected_pairs"]):
        raise ValueError("E6 selected-pair count does not match the role allocation")
    return tuple(assignments)


def assignment_sha256(assignments: tuple[dict[str, Any], ...]) -> str:
    """Return a stable identity for an already validated role assignment."""

    payload = json.dumps(
        assignments, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_e6_development_protocol(protocol: dict[str, Any]) -> None:
    """Reject leakage, role conflation, source drift and relaxed shortcut gates."""

    if protocol.get("schema_version") != 1:
        raise ValueError("Unsupported E6 protocol schema_version")
    if protocol.get("protocol_id") != "e6_tiny_genimage_biggan_development_v1":
        raise ValueError("Unexpected E6 protocol_id")
    if protocol.get("status") != (
        "frozen_before_e6_third_source_image_payload_or_feature_access"
    ):
        raise ValueError("E6 protocol must be frozen before image payload or feature access")
    if protocol.get("frozen_at_utc") != "2026-09-28T08:20:00Z":
        raise ValueError("E6 protocol freeze time changed")

    class_mapping = protocol["class_mapping"]
    if class_mapping != {"real": 0, "ai_generated": 1}:
        raise ValueError("E6 class mapping changed")

    scope = protocol["task_scope"]
    if scope.get("target") != "fully_synthetic_image_detection":
        raise ValueError("E6 must remain a fully synthetic image task")
    if scope.get("sid_set_released_full_synthetic_generator") != "FLUX":
        raise ValueError("SID-Set full-synthetic provenance changed")
    if scope.get("sid_set_label_2_meaning") != "locally_tampered_image":
        raise ValueError("SID-Set label 2 must remain identified as tampered")
    _require_bool(scope, "sid_set_label_2_allowed_as_third_full_synthetic_generator", False)
    _require_bool(scope, "organiser_validation_subset_allowed", False)

    source = protocol["development_source"]
    if source.get("dataset_repository") != EXPECTED_REPOSITORY:
        raise ValueError("E6 dataset repository changed")
    if _require_revision(source.get("repository_revision"), "dataset revision") != (
        EXPECTED_REPOSITORY_REVISION
    ):
        raise ValueError("E6 dataset revision changed")
    if source.get("repository_status") != "third_party_compact_mirror_of_genimage":
        raise ValueError("Tiny-GenImage mirror status must remain explicit")
    if source.get("source_split") != "train":
        raise ValueError("E6 may only use the Tiny-GenImage train split")
    if source.get("selected_generator") != EXPECTED_GENERATOR:
        raise ValueError("E6 third generator must remain BigGAN")
    if source.get("generator_architecture_family") != "GAN":
        raise ValueError("BigGAN architecture family changed")
    if source.get("upstream_real_source") != "ImageNet nature images from GenImage":
        raise ValueError("E6 real source changed")
    if source.get("declared_license") != (
        "CC-BY-NC-SA-4.0 with GenImage dataset terms"
    ):
        raise ValueError("E6 source licence record changed")
    if source.get("permitted_use") != "non-commercial research only":
        raise ValueError("E6 non-commercial use restriction changed")
    for key in (
        "raw_images_may_be_redistributed",
        "exact_one_to_one_semantic_pairing_claimed",
    ):
        _require_bool(source, key, False)
    _require_bool(source, "generator_absent_from_existing_development", True)
    uploader = source["uploader_evidence"]
    if uploader.get("repository") != "tbtiberiu/DeForge-DataHelper":
        raise ValueError("Tiny-GenImage uploader repository changed")
    if uploader.get("file") != "upload_Tiny-GenImage.py":
        raise ValueError("Tiny-GenImage uploader evidence file changed")
    if _require_revision(uploader.get("revision"), "uploader revision") != (
        EXPECTED_UPLOADER_REVISION
    ):
        raise ValueError("Tiny-GenImage uploader revision changed")

    preflight = protocol["metadata_access_preflight"]
    for key in ("image_payload_downloaded", "image_payload_decoded_or_displayed"):
        _require_bool(preflight, key, False)
    for key in (
        "repository_public",
        "range_requests_supported",
        "dataset_row_api_accessible",
        "first_biggan_pair_metadata_verified",
    ):
        _require_bool(preflight, key, True)
    _require_bool(preflight, "repository_gated", False)
    _require_bool(preflight, "row_api_reported_partial", False)
    if preflight.get("observed_repository_revision") != EXPECTED_REPOSITORY_REVISION:
        raise ValueError("E6 access preflight observed a different revision")
    if int(preflight.get("train_rows", -1)) != 28000:
        raise ValueError("Tiny-GenImage train-row count changed")
    if int(preflight.get("validation_rows", -1)) != 7000:
        raise ValueError("Tiny-GenImage validation-row count changed")
    if int(preflight.get("range_probe_status", -1)) != 206:
        raise ValueError("Tiny-GenImage selective access was not verified")

    topology = protocol["row_topology"]
    expected_generators = [
        "ADM",
        "BigGAN",
        "GLIDE",
        "Midjourney",
        "SD15",
        "VQDM",
        "Wukong",
    ]
    if topology.get("actual_generators_in_order") != expected_generators:
        raise ValueError("Tiny-GenImage generator ordering changed")
    if topology.get("declared_but_empty_generator") != "SD14":
        raise ValueError("Tiny-GenImage empty SD14 disclosure changed")
    expected_numbers = {
        "source_train_rows": 28000,
        "rows_per_cycle": 14,
        "available_cycles": 2000,
        "biggan_real_offset": 2,
        "biggan_ai_offset": 3,
    }
    for key, expected in expected_numbers.items():
        if int(topology.get(key, -1)) != expected:
            raise ValueError(f"Tiny-GenImage row topology changed: {key}")
    if topology.get("real_row_contract") != {
        "label": 0,
        "generator_id": 0,
        "generator_name": "Real",
    }:
        raise ValueError("Tiny-GenImage BigGAN real-row contract changed")
    if topology.get("ai_row_contract") != {
        "label": 1,
        "generator_id": 2,
        "generator_name": "BigGAN",
    }:
        raise ValueError("Tiny-GenImage BigGAN AI-row contract changed")
    if topology.get("pair_key") != (
        "tiny_genimage_train_biggan_cycle_<zero_based_cycle>"
    ):
        raise ValueError("Tiny-GenImage pair identity changed")
    if pair_rows(protocol, 0) != (2, 3) or pair_rows(protocol, 1999) != (27988, 27989):
        raise ValueError("Tiny-GenImage first or final pair topology changed")

    selection = protocol["selection"]
    if int(selection.get("seed", -1)) != EXPECTED_SEED:
        raise ValueError("E6 deterministic selection seed changed")
    if int(selection.get("candidate_pairs", -1)) != 2000:
        raise ValueError("E6 candidate-pair count changed")
    if int(selection.get("selected_pairs", -1)) != sum(EXPECTED_PAIR_COUNTS.values()):
        raise ValueError("E6 selected-pair count changed")
    if int(selection.get("unused_reserve_pairs", -1)) != 600:
        raise ValueError("E6 reserve-pair count changed")
    if selection.get("ranking_key") != (
        "SHA256('e6d:606:<repository_revision>:biggan:<zero_based_cycle>') "
        "followed by cycle index"
    ):
        raise ValueError("E6 deterministic ranking definition changed")
    if selection.get("roles_in_assignment_order") != list(EXPECTED_PAIR_COUNTS):
        raise ValueError("E6 role assignment order changed")
    if selection.get("pairs_per_role") != EXPECTED_PAIR_COUNTS:
        raise ValueError("E6 role pair counts changed")
    if selection.get("images_per_class_per_role") != EXPECTED_PAIR_COUNTS:
        raise ValueError("E6 role class balance changed")
    _require_bool(selection, "manual_selection_allowed", False)
    _require_bool(selection, "visual_cherry_picking_allowed", False)

    permissions = protocol["role_permissions"]
    permission_keys = (
        "fit_model_parameters",
        "select_model_or_epoch",
        "fit_probability_calibration",
        "select_decision_thresholds",
    )
    if set(permissions) != set(EXPECTED_PERMISSIONS):
        raise ValueError("E6 role permissions contain missing or extra roles")
    for role, expected in EXPECTED_PERMISSIONS.items():
        observed = tuple(permissions[role].get(key) for key in permission_keys)
        if observed != expected:
            raise ValueError(f"E6 role conflation detected for {role}")

    repartition = protocol["existing_validation_repartition"]
    if int(repartition.get("seed", -1)) != EXPECTED_SEED:
        raise ValueError("Existing validation repartition seed changed")
    _require_bool(repartition, "group_related_or_near_duplicate_items_before_assignment", True)
    if repartition.get("ranking_key") != (
        "SHA256('e6d-existing:606:<source-id>:<stable-row-identity>')"
    ):
        raise ValueError("Existing validation ranking definition changed")
    expected_existing = {
        "cifake_validation_real": (0, 1000, 400, 300, 300),
        "cifake_validation_ai": (1, 1000, 400, 300, 300),
        "sid_validation_real": (0, 1000, 400, 300, 300),
        "sid_validation_flux": (1, 1000, 400, 300, 300),
    }
    sources = repartition.get("sources")
    if not isinstance(sources, list) or {item.get("id") for item in sources} != set(
        expected_existing
    ):
        raise ValueError("E6 existing validation sources changed")
    for item in sources:
        expected = expected_existing[item["id"]]
        observed = tuple(
            int(item[key])
            for key in (
                "label",
                "available",
                "model_selection",
                "calibration",
                "threshold_selection",
            )
        )
        if observed != expected or sum(observed[2:]) != observed[1]:
            raise ValueError(f"E6 existing validation allocation changed: {item['id']}")

    isolation = protocol["isolation"]
    if set(isolation.get("consumed_test_ids_excluded", [])) != EXPECTED_CONSUMED_TESTS:
        raise ValueError("E6 consumed-test exclusion changed")
    if isolation.get("organiser_validation_id_excluded") != (
        "organiser_validation_coco_dalle"
    ):
        raise ValueError("E6 organiser validation exclusion changed")
    for key in (
        "future_lockbox_payload_accessed",
        "future_lockbox_used_for_overlap_checking",
        "aigibench_midjourney_allowed_as_e6_development",
    ):
        _require_bool(isolation, key, False)
    forbidden_generators = set(
        isolation.get("known_existing_generator_families_forbidden_as_new_generator", [])
    )
    if forbidden_generators != {"Stable Diffusion 1.4", "FLUX", "Midjourney V6"}:
        raise ValueError("E6 known-generator exclusion changed")
    if source["selected_generator"] in forbidden_generators:
        raise ValueError("E6 selected generator is already an existing development family")

    integrity = protocol["integrity"]
    for key in (
        "manual_image_inspection_before_assignment_allowed",
        "raw_images_committed_to_git",
        "cross_role_byte_duplicate_allowed",
        "cross_role_decoded_pixel_duplicate_allowed",
        "cross_role_near_duplicate_allowed",
        "conflicting_labels_allowed",
        "pair_or_parent_identity_may_cross_roles",
    ):
        _require_bool(integrity, key, False)
    for key in (
        "manifest_requires_byte_sha256",
        "manifest_requires_decoded_pixel_sha256",
        "existing_data_overlap_is_fatal",
        "future_lockbox_manifests_must_not_be_opened",
    ):
        _require_bool(integrity, key, True)
    required_existing = {
        "data/processed/train.csv",
        "data/processed/val.csv",
        "data/processed/test.csv",
        "data/processed/e4_sid_real/train.csv",
        "data/processed/e4_sid_real/val.csv",
        "data/processed/e5_sid_flux/train.csv",
        "data/processed/e5_sid_flux/val.csv",
        "data/processed/sid_set_flux_heldout.csv",
        "data/processed/e5_aigibench_midjourney.csv",
    }
    if set(integrity.get("existing_manifests_to_exclude", [])) != required_existing:
        raise ValueError("E6 overlap exclusion coverage changed")

    shortcut = protocol["shortcut_gate"]
    if shortcut.get("status") != "required_before_any_e6_model_training":
        raise ValueError("E6 shortcut gate must run before training")
    if shortcut.get("metric") != "validation_roc_auc" or float(
        shortcut.get("maximum_acceptable_auc", -1)
    ) != 0.65:
        raise ValueError("E6 metadata-shortcut acceptance gate changed")
    for key in (
        "forensic_training_allowed_before_pass",
        "semantic_training_allowed_before_pass",
        "raw_source_accuracy_is_research_evidence",
        "normalization_may_be_selected_using_consumed_tests",
    ):
        _require_bool(shortcut, key, False)
    required_metadata = {
        "original_width",
        "original_height",
        "aspect_ratio",
        "decoded_pixel_count",
        "file_format",
        "file_bytes",
    }
    if set(shortcut.get("metadata_only_inputs", [])) != required_metadata:
        raise ValueError("E6 shortcut-audit inputs changed")

    outputs = protocol["outputs"]
    if outputs != EXPECTED_OUTPUTS:
        raise ValueError("E6 output path inventory changed")
    expected_prefixes = {
        "raw_root": "data/raw/",
        "manifest_root": "data/processed/",
        "provenance": "data/processed/",
        "existing_validation_role_root": "data/processed/",
        "shortcut_audit": "reports/",
        "clip_feature_root": "data/features/",
        "forensic_feature_root": "data/features/",
    }
    for key, prefix in expected_prefixes.items():
        value = outputs[key]
        _safe_project_path(value, f"E6 output {key}")
        if not value.startswith(prefix):
            raise ValueError(f"E6 output {key} must stay under {prefix}")

    claims = protocol.get("claim_limits")
    if not isinstance(claims, list) or len(claims) < 4 or not all(
        isinstance(item, str) and item for item in claims
    ):
        raise ValueError("E6 claim limitations are missing")


def artifact_paths(protocol: dict[str, Any], project_root: Path) -> tuple[Path, ...]:
    """List outputs that must not exist when the pre-download lock is made."""

    validate_e6_development_protocol(protocol)
    return tuple(project_root / _safe_project_path(value, key) for key, value in protocol["outputs"].items())


def validate_artifacts_absent(protocol: dict[str, Any], project_root: Path) -> None:
    """Ensure no selected image, split, feature or shortcut result predates the lock."""

    present = [path for path in artifact_paths(protocol, project_root) if path.exists()]
    if present:
        joined = ", ".join(str(path) for path in present)
        raise ValueError(f"E6 development artifacts existed before protocol lock: {joined}")
