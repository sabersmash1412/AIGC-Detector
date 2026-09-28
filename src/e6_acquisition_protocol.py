"""Fail-closed validation for the frozen E6 acquisition protocol.

This module is intentionally offline.  It validates the acquisition contract,
its two upstream locks and the absence of every E6 payload path, but it never
opens a remote connection or decodes an image.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from src.e6_protocol import (
    artifact_paths,
    assignment_sha256,
    build_pair_assignment,
    load_e6_protocol,
    validate_e6_development_protocol,
)


DEFAULT_E6_ACQUISITION_PROTOCOL = Path("configs/e6_acquisition_protocol.json")
EXPECTED_PROTOCOL_ID = "e6_tiny_genimage_biggan_acquisition_v1"
EXPECTED_FROZEN_AT = "2026-09-28T08:48:00Z"
EXPECTED_DEVELOPMENT_PROTOCOL_SHA256 = (
    "1613be7082b6b8f9794e5e22ca247f1469f86adebd2e7d28f0aa44847596d551"
)
EXPECTED_DEVELOPMENT_LOCK_SHA256 = (
    "bec3ed54708ae8e9195ab01d5b5b0ae938f034e3024509d72f26705435a0cae6"
)
EXPECTED_ASSIGNMENT_SHA256 = (
    "c20aac52c0db04dfa2bf3ccda9a04ddd7858f82544442bc34cecd5b3161f4d68"
)
EXPECTED_REVISION = "89c4fe9efd0ebc7ce5c7641ef57d578ccd639c69"
EXPECTED_ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
EXPECTED_ASSET_PREFIX = (
    "/cached-assets/TheKernel01/Tiny-GenImage/--/"
    f"{EXPECTED_REVISION}/--/default/train/"
)
EXPECTED_EXISTING_MANIFESTS = [
    "data/processed/train.csv",
    "data/processed/val.csv",
    "data/processed/test.csv",
    "data/processed/e4_sid_real/train.csv",
    "data/processed/e4_sid_real/val.csv",
    "data/processed/e5_sid_flux/train.csv",
    "data/processed/e5_sid_flux/val.csv",
    "data/processed/sid_set_flux_heldout.csv",
    "data/processed/e5_aigibench_midjourney.csv",
]
EXPECTED_ROLE_MANIFESTS = {
    "development_train": (
        "data/processed/e6_tiny_genimage_biggan/development_train.csv"
    ),
    "development_model_selection": (
        "data/processed/e6_tiny_genimage_biggan/development_model_selection.csv"
    ),
    "development_calibration": (
        "data/processed/e6_tiny_genimage_biggan/development_calibration.csv"
    ),
    "development_threshold_selection": (
        "data/processed/e6_tiny_genimage_biggan/development_threshold_selection.csv"
    ),
}
EXPECTED_ABSENT_PATHS = [
    "data/raw/e6_tiny_genimage_biggan",
    "data/processed/e6_tiny_genimage_biggan",
    "data/processed/e6_tiny_genimage_biggan_provenance.json",
    "data/processed/e6_existing_validation_roles",
    "reports/e6_tiny_genimage_shortcut_audit.json",
    "data/features/e6_tiny_genimage_clip_vit_b32_quickgelu_openai",
    "data/features/e6_tiny_genimage_forensic_v1",
    "data/raw/.e6_tiny_genimage_biggan_staging",
]


def sha256_file(path: Path) -> str:
    """Hash a local file without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_e6_acquisition_protocol(
    path: Path = DEFAULT_E6_ACQUISITION_PROTOCOL,
) -> dict[str, Any]:
    """Load the E6 acquisition contract from JSON."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("E6 acquisition protocol must be a JSON object")
    return payload


def _require_exact_keys(container: dict[str, Any], expected: set[str], name: str) -> None:
    if set(container) != expected:
        raise ValueError(f"E6 acquisition {name} keys changed")


def _require_bool(container: dict[str, Any], key: str, expected: bool) -> None:
    if container.get(key) is not expected:
        raise ValueError(f"E6 acquisition requires {key}={expected}")


def _safe_project_path(value: Any, description: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{description} must be a non-empty project-relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{description} must stay inside the project")
    return path


def validate_e6_acquisition_protocol(protocol: dict[str, Any]) -> None:
    """Reject source drift, unsafe transport, weak identities or leakage."""

    _require_exact_keys(
        protocol,
        {
            "schema_version",
            "protocol_id",
            "status",
            "frozen_at_utc",
            "purpose",
            "upstream_locks",
            "source_access",
            "trusted_asset_transport",
            "resource_caps",
            "image_validation",
            "filesystem_safety",
            "identity_algorithms",
            "overlap_and_replacement",
            "outputs",
            "prohibitions",
        },
        "top-level",
    )
    if protocol.get("schema_version") != 1:
        raise ValueError("Unsupported E6 acquisition schema_version")
    if protocol.get("protocol_id") != EXPECTED_PROTOCOL_ID:
        raise ValueError("Unexpected E6 acquisition protocol_id")
    if protocol.get("status") != "frozen_before_e6_image_payload_access":
        raise ValueError("E6 acquisition must be frozen before image payload access")
    if protocol.get("frozen_at_utc") != EXPECTED_FROZEN_AT:
        raise ValueError("E6 acquisition freeze time changed")
    if not isinstance(protocol.get("purpose"), str) or not protocol["purpose"]:
        raise ValueError("E6 acquisition purpose is missing")

    upstream = protocol["upstream_locks"]
    if upstream != {
        "development_protocol": {
            "path": "configs/e6_development_protocol.json",
            "sha256": EXPECTED_DEVELOPMENT_PROTOCOL_SHA256,
        },
        "development_protocol_lock": {
            "path": "reports/e6_development_protocol_lock.json",
            "sha256": EXPECTED_DEVELOPMENT_LOCK_SHA256,
            "required_status": "PASS",
            "required_image_payload_present_at_freeze": False,
            "required_assignment_sha256": EXPECTED_ASSIGNMENT_SHA256,
        },
    }:
        raise ValueError("E6 acquisition upstream lock binding changed")

    source = protocol["source_access"]
    _require_exact_keys(
        source,
        {
            "method",
            "rows_endpoint",
            "query",
            "required_response_revision_header",
            "required_response_revision",
            "required_num_rows_total",
            "required_partial",
            "top_level_schema",
            "row_entry_schema",
            "row_value_schema",
            "row_contracts",
            "contract_drift_action",
        },
        "source-access",
    )
    if source["method"] != "GET" or source["rows_endpoint"] != EXPECTED_ROWS_ENDPOINT:
        raise ValueError("E6 acquisition rows endpoint or method changed")
    endpoint = urlsplit(source["rows_endpoint"])
    if (
        endpoint.scheme != "https"
        or endpoint.hostname != "datasets-server.huggingface.co"
        or endpoint.port is not None
        or endpoint.path != "/rows"
        or endpoint.query
        or endpoint.fragment
        or endpoint.username is not None
        or endpoint.password is not None
    ):
        raise ValueError("E6 rows endpoint is not the exact trusted HTTPS endpoint")
    if source["query"] != {
        "dataset": "TheKernel01/Tiny-GenImage",
        "config": "default",
        "split": "train",
        "offset_type": "non-negative integer",
        "length_type": "integer from 1 through 100",
        "revision_parameter_allowed": False,
    }:
        raise ValueError("E6 rows query contract changed")
    if source["required_response_revision_header"] != "X-Revision":
        raise ValueError("E6 response revision header changed")
    if source["required_response_revision"] != EXPECTED_REVISION:
        raise ValueError("E6 required source revision changed")
    if source["required_num_rows_total"] != 28000 or source["required_partial"] is not False:
        raise ValueError("E6 row-count or partial-response contract changed")
    if source["top_level_schema"] != {
        "required_keys": [
            "features",
            "rows",
            "num_rows_total",
            "num_rows_per_page",
            "partial",
        ],
        "additional_keys_allowed": False,
    }:
        raise ValueError("E6 row-response top-level schema changed")
    if source["row_entry_schema"] != {
        "required_keys": ["row_idx", "row", "truncated_cells"],
        "additional_keys_allowed": False,
        "row_idx_type": "integer",
        "truncated_cells_type": "array",
    }:
        raise ValueError("E6 row-entry schema changed")
    if source["row_value_schema"] != {
        "required_keys": ["image", "label", "generator"],
        "additional_keys_allowed": False,
        "image_required_keys": ["src", "height", "width"],
        "image_additional_keys_allowed": False,
        "src_type": "string",
        "height_type": "positive integer",
        "width_type": "positive integer",
        "label_type": "integer",
        "generator_type": "integer",
    }:
        raise ValueError("E6 row-value schema changed")
    if source["row_contracts"] != {
        "real": {
            "row_formula": "cycle * 14 + 2",
            "label": 0,
            "generator": 0,
            "generator_name": "Real",
        },
        "ai_generated": {
            "row_formula": "cycle * 14 + 3",
            "label": 1,
            "generator": 2,
            "generator_name": "BigGAN",
        },
    }:
        raise ValueError("E6 real/BigGAN row contracts changed")
    if source["contract_drift_action"] != "hard_abort_without_replacement":
        raise ValueError("E6 row-contract drift must hard abort")

    transport = protocol["trusted_asset_transport"]
    if transport != {
        "scheme": "https",
        "host": "datasets-server.huggingface.co",
        "port": 443,
        "path_prefix": EXPECTED_ASSET_PREFIX,
        "path_pattern": (
            "^/cached-assets/TheKernel01/Tiny-GenImage/--/"
            f"{EXPECTED_REVISION}/--/default/train/<expected_row_index>/image/"
            "[A-Za-z0-9._-]+$"
        ),
        "userinfo_allowed": False,
        "fragments_allowed": False,
        "redirects_allowed": False,
        "hostname_redirect_revalidation_required": True,
        "signed_query_allowed_in_memory": True,
        "signed_url_or_query_persisted": False,
        "signed_url_or_query_logged": False,
        "tls_verification_required": True,
    }:
        raise ValueError("E6 trusted asset transport policy changed")

    if protocol["resource_caps"] != {
        "connect_timeout_seconds": 10,
        "read_timeout_seconds": 60,
        "retry_attempts": 4,
        "retry_backoff_seconds": 1,
        "maximum_rows_per_metadata_request": 100,
        "maximum_metadata_response_bytes": 16777216,
        "maximum_asset_bytes": 67108864,
        "maximum_total_download_bytes": 17179869184,
        "maximum_width_pixels": 32768,
        "maximum_height_pixels": 32768,
        "maximum_decoded_pixels": 100000000,
        "maximum_candidate_pairs": 2000,
        "maximum_candidate_images": 4000,
    }:
        raise ValueError("E6 acquisition resource caps changed")

    image_validation = protocol["image_validation"]
    if image_validation != {
        "allowed_formats": {
            "JPEG": {"media_types": ["image/jpeg"], "extensions": [".jpg", ".jpeg"]},
            "PNG": {"media_types": ["image/png"], "extensions": [".png"]},
            "WEBP": {"media_types": ["image/webp"], "extensions": [".webp"]},
        },
        "generic_binary_media_types_allowed_after_magic_validation": [
            "application/octet-stream",
            "binary/octet-stream",
        ],
        "media_type_must_match_decoded_format": True,
        "extension_must_match_decoded_format": True,
        "single_frame_only": True,
        "decoder_must_load_entire_image": True,
        "truncated_images_allowed": False,
        "decompression_bomb_allowed": False,
    }:
        raise ValueError("E6 allowed image types or decoder rules changed")

    filesystem = protocol["filesystem_safety"]
    expected_filesystem = {
        "all_paths_project_relative": True,
        "remote_names_used_as_output_paths": False,
        "deterministic_local_names_only": True,
        "symlink_in_any_output_component_allowed": False,
        "temporary_file_same_filesystem_as_destination": True,
        "temporary_file_open_mode": "exclusive_create_no_follow",
        "file_fsync_before_publish": True,
        "directory_fsync_after_publish": True,
        "publish_primitive": "atomic_replace",
        "manifest_and_provenance_publish_after_full_validation_only": True,
        "existing_final_file_policy": (
            "reuse_only_after_complete_revalidation_otherwise_abort"
        ),
        "partial_download_suffix": ".part",
        "failed_pair_partial_files_retained": False,
    }
    if filesystem != expected_filesystem:
        raise ValueError("E6 atomic-write or no-symlink policy changed")

    identities = protocol["identity_algorithms"]
    _require_exact_keys(
        identities,
        {"byte_sha256", "decoded_rgba_sha256", "perceptual_hash"},
        "identity-algorithm",
    )
    if identities["byte_sha256"] != (
        "SHA-256 over the exact downloaded file bytes, emitted as 64 lowercase "
        "hexadecimal characters."
    ):
        raise ValueError("E6 byte SHA-256 definition changed")
    if identities["decoded_rgba_sha256"] != {
        "algorithm_id": "e6_rgba_sha256_v1",
        "decode": "Pillow full decode with truncated-image loading disabled",
        "orientation": "ImageOps.exif_transpose",
        "colour_mode": "RGBA",
        "preimage": (
            "ASCII bytes E6RGBA1 followed by one NUL byte, unsigned big-endian "
            "uint32 width, unsigned big-endian uint32 height, then row-major "
            "8-bit RGBA bytes"
        ),
        "digest": "SHA-256 emitted as 64 lowercase hexadecimal characters",
    }:
        raise ValueError("E6 decoded RGBA identity algorithm changed")
    if identities["perceptual_hash"] != {
        "algorithm_id": "e6_phash64_v1",
        "orientation": "ImageOps.exif_transpose",
        "alpha_handling": "RGBA alpha-composited over opaque white",
        "grayscale": "Pillow L conversion",
        "resize": "32x32 Pillow Resampling.LANCZOS",
        "transform": "two-dimensional orthonormal DCT-II over float64 pixels",
        "coefficients": "top-left 8x8 block with the DC coefficient excluded",
        "threshold": "median of the remaining 63 coefficients",
        "bit_rule": (
            "row-major coefficient strictly greater than the median is 1; "
            "otherwise 0"
        ),
        "encoding": (
            "63 bits encoded as 16 lowercase hexadecimal characters with a "
            "leading zero bit"
        ),
        "distance": "Hamming distance over the 63 meaningful bits",
        "near_duplicate_max_distance_inclusive": 4,
    }:
        raise ValueError("E6 pHash algorithm or near-duplicate threshold changed")

    overlap = protocol["overlap_and_replacement"]
    expected_overlap = {
        "existing_manifests": EXPECTED_EXISTING_MANIFESTS,
        "future_lockbox_may_be_opened": False,
        "overlap_signals": [
            "byte_sha256_equal",
            "decoded_rgba_sha256_equal",
            "perceptual_hash_hamming_distance_at_most_4",
        ],
        "precedence_highest_first": [
            "remote_row_or_transport_contract_drift_hard_abort",
            "conflicting_label_duplicate_hard_abort",
            "existing_dataset_overlap_hard_abort",
            "corrupt_remote_asset_reject_whole_pair_and_replace",
            "within_e6_same_label_duplicate_reject_later_whole_pair_and_replace",
        ],
        "conflicting_label_duplicate_action": "hard_abort_without_replacement",
        "existing_dataset_exact_decoded_or_near_overlap_action": (
            "hard_abort_without_replacement"
        ),
        "corrupt_remote_asset_action": "reject_whole_pair_and_replace",
        "within_e6_same_label_same_role_or_cross_role_duplicate_action": (
            "reject_later_whole_pair_and_replace"
        ),
        "transient_network_or_service_failure_action": (
            "stop_without_replacement_and_allow_safe_rerun"
        ),
        "replacement_source": "next unused pair in the frozen SHA-256 rank order",
        "replacement_keeps_rejected_pair_role": True,
        "replacement_is_whole_pair": True,
        "manual_replacement_allowed": False,
        "reserve_exhaustion_action": "hard_abort",
        "every_rejection_recorded_without_signed_url": True,
    }
    if overlap != expected_overlap:
        raise ValueError("E6 overlap precedence or deterministic replacement policy changed")

    outputs = protocol["outputs"]
    if outputs != {
        "raw_root": "data/raw/e6_tiny_genimage_biggan",
        "manifest_root": "data/processed/e6_tiny_genimage_biggan",
        "role_manifests": EXPECTED_ROLE_MANIFESTS,
        "provenance": "data/processed/e6_tiny_genimage_biggan_provenance.json",
        "staging_root": "data/raw/.e6_tiny_genimage_biggan_staging",
        "broader_e6_payload_paths_required_absent_at_lock": EXPECTED_ABSENT_PATHS,
    }:
        raise ValueError("E6 acquisition output inventory changed")
    for key in ("raw_root", "manifest_root", "provenance", "staging_root"):
        _safe_project_path(outputs[key], f"E6 acquisition output {key}")
    for role, path in outputs["role_manifests"].items():
        _safe_project_path(path, f"E6 acquisition role manifest {role}")
    for path in outputs["broader_e6_payload_paths_required_absent_at_lock"]:
        _safe_project_path(path, "E6 acquisition absent payload path")

    if protocol["prohibitions"] != {
        "lock_check_network_or_image_access_allowed": False,
        "image_payload_access_before_this_lock_passes_allowed": False,
        "manual_image_inspection_or_cherry_picking_allowed": False,
        "training_or_feature_extraction_during_acquisition_allowed": False,
        "evaluation_during_acquisition_allowed": False,
        "raw_image_git_commit_allowed": False,
        "signed_url_persistence_allowed": False,
        "organiser_validation_access_allowed": False,
        "consumed_test_use_for_selection_or_replacement_allowed": False,
    }:
        raise ValueError("E6 acquisition prohibitions changed")


def validate_upstream_locks(
    protocol: dict[str, Any], project_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the acquisition protocol binds the exact E6 selection and lock."""

    validate_e6_acquisition_protocol(protocol)
    upstream = protocol["upstream_locks"]
    development_path = project_root / _safe_project_path(
        upstream["development_protocol"]["path"], "E6 development protocol"
    )
    lock_path = project_root / _safe_project_path(
        upstream["development_protocol_lock"]["path"], "E6 development lock"
    )
    for path, description in (
        (development_path, "E6 development protocol"),
        (lock_path, "E6 development lock"),
    ):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"{description} must be a regular non-symlink file")

    if sha256_file(development_path) != EXPECTED_DEVELOPMENT_PROTOCOL_SHA256:
        raise ValueError("E6 development protocol SHA-256 changed")
    if sha256_file(lock_path) != EXPECTED_DEVELOPMENT_LOCK_SHA256:
        raise ValueError("E6 development lock SHA-256 changed")

    development = load_e6_protocol(development_path)
    validate_e6_development_protocol(development)
    assignments = build_pair_assignment(development)
    if assignment_sha256(assignments) != EXPECTED_ASSIGNMENT_SHA256:
        raise ValueError("E6 deterministic pair assignment changed")

    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if not isinstance(lock, dict):
        raise ValueError("E6 development lock must be a JSON object")
    if lock.get("status") != "PASS":
        raise ValueError("E6 development lock status is not PASS")
    if lock.get("image_payload_present_at_freeze") is not False:
        raise ValueError("E6 development lock did not freeze before payload access")
    if lock.get("protocol") != {
        "path": "configs/e6_development_protocol.json",
        "sha256": EXPECTED_DEVELOPMENT_PROTOCOL_SHA256,
        "schema_version": 1,
    }:
        raise ValueError("E6 development lock protocol identity changed")
    lock_source = lock.get("development_source", {})
    if lock_source.get("dataset_repository") != "TheKernel01/Tiny-GenImage":
        raise ValueError("E6 locked dataset repository changed")
    if lock_source.get("repository_revision") != EXPECTED_REVISION:
        raise ValueError("E6 locked repository revision changed")
    if lock_source.get("selected_generator") != "BigGAN":
        raise ValueError("E6 locked generator changed")
    if lock_source.get("selected_pairs") != 1400:
        raise ValueError("E6 locked selected-pair count changed")
    if lock_source.get("assignment_sha256") != EXPECTED_ASSIGNMENT_SHA256:
        raise ValueError("E6 development lock assignment identity changed")
    expected_checked = [
        path.relative_to(project_root).as_posix()
        for path in artifact_paths(development, project_root)
    ]
    if lock.get("checked_absent_paths") != expected_checked:
        raise ValueError("E6 development lock absent-path inventory changed")
    return development, lock


def acquisition_payload_paths(
    protocol: dict[str, Any], project_root: Path
) -> tuple[Path, ...]:
    """Return every payload path that must be absent when this lock is emitted."""

    validate_e6_acquisition_protocol(protocol)
    values = protocol["outputs"]["broader_e6_payload_paths_required_absent_at_lock"]
    return tuple(
        project_root / _safe_project_path(value, "E6 acquisition payload")
        for value in values
    )


def validate_acquisition_payloads_absent(
    protocol: dict[str, Any], project_root: Path
) -> None:
    """Fail if any raw image, manifest, feature or audit output predates the lock."""

    present = [
        path
        for path in acquisition_payload_paths(protocol, project_root)
        if path.exists() or path.is_symlink()
    ]
    if present:
        joined = ", ".join(str(path) for path in present)
        raise ValueError(f"E6 acquisition payload existed before protocol lock: {joined}")


def build_acquisition_lock_receipt(
    protocol: dict[str, Any],
    project_root: Path,
    protocol_path: Path = DEFAULT_E6_ACQUISITION_PROTOCOL,
) -> dict[str, Any]:
    """Build the deterministic, offline pre-download lock receipt."""

    resolved_protocol_path = (
        protocol_path if protocol_path.is_absolute() else project_root / protocol_path
    )
    development, development_lock = validate_upstream_locks(protocol, project_root)
    validate_acquisition_payloads_absent(protocol, project_root)
    assignments = build_pair_assignment(development)
    transport = protocol["trusted_asset_transport"]
    identities = protocol["identity_algorithms"]
    overlap = protocol["overlap_and_replacement"]
    return {
        "experiment": "e6_acquisition_protocol_lock",
        "status": "PASS",
        "frozen_at_utc": protocol["frozen_at_utc"],
        "acquisition_protocol": {
            "path": protocol_path.as_posix(),
            "sha256": sha256_file(resolved_protocol_path),
            "schema_version": protocol["schema_version"],
        },
        "upstream_locks": {
            "development_protocol_path": protocol["upstream_locks"][
                "development_protocol"
            ]["path"],
            "development_protocol_sha256": EXPECTED_DEVELOPMENT_PROTOCOL_SHA256,
            "development_lock_path": protocol["upstream_locks"][
                "development_protocol_lock"
            ]["path"],
            "development_lock_sha256": EXPECTED_DEVELOPMENT_LOCK_SHA256,
            "development_lock_status": development_lock["status"],
            "assignment_sha256": assignment_sha256(assignments),
        },
        "source_contract": {
            "rows_endpoint": protocol["source_access"]["rows_endpoint"],
            "dataset": protocol["source_access"]["query"]["dataset"],
            "config": protocol["source_access"]["query"]["config"],
            "split": protocol["source_access"]["query"]["split"],
            "repository_revision": protocol["source_access"][
                "required_response_revision"
            ],
            "selected_pairs": len(assignments),
            "reserve_pairs": development["selection"]["unused_reserve_pairs"],
        },
        "security_contract": {
            "asset_scheme": transport["scheme"],
            "asset_host": transport["host"],
            "asset_path_prefix": transport["path_prefix"],
            "redirects_allowed": transport["redirects_allowed"],
            "signed_urls_persisted": transport["signed_url_or_query_persisted"],
            "maximum_asset_bytes": protocol["resource_caps"]["maximum_asset_bytes"],
            "allowed_decoded_formats": sorted(
                protocol["image_validation"]["allowed_formats"]
            ),
            "generic_binary_media_types_allowed_after_magic_validation": protocol[
                "image_validation"
            ]["generic_binary_media_types_allowed_after_magic_validation"],
            "symlinks_allowed": protocol["filesystem_safety"][
                "symlink_in_any_output_component_allowed"
            ],
        },
        "identity_contract": {
            "decoded_rgba_algorithm_id": identities["decoded_rgba_sha256"][
                "algorithm_id"
            ],
            "perceptual_hash_algorithm_id": identities["perceptual_hash"][
                "algorithm_id"
            ],
            "near_duplicate_max_distance_inclusive": identities[
                "perceptual_hash"
            ]["near_duplicate_max_distance_inclusive"],
        },
        "overlap_contract": {
            "existing_manifest_count": len(overlap["existing_manifests"]),
            "existing_overlap_action": overlap[
                "existing_dataset_exact_decoded_or_near_overlap_action"
            ],
            "conflicting_label_action": overlap[
                "conflicting_label_duplicate_action"
            ],
            "eligible_replacement_reasons": [
                "corrupt_remote_asset",
                "within_e6_same_label_duplicate",
            ],
            "replacement_source": overlap["replacement_source"],
        },
        "payload_present_at_freeze": False,
        "checked_absent_paths": [
            path.relative_to(project_root).as_posix()
            for path in acquisition_payload_paths(protocol, project_root)
        ],
        "guardrails": {
            "network_or_image_access_during_lock_check": False,
            "training_or_feature_extraction_performed": False,
            "evaluation_performed": False,
            "organiser_validation_accessed": False,
            "consumed_tests_used_for_selection_or_replacement": False,
        },
    }
