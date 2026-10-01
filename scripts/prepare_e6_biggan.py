#!/usr/bin/env python3
"""Selectively acquire the frozen Tiny-GenImage real/BigGAN development pairs.

The E6 protocol was committed before any selected image payload was accessed.
This script enforces that lock, fetches only the frozen rows from the pinned
Hugging Face datasets-server revision, and writes role-separated manifests.
Signed asset URLs are used only in memory and are never written to provenance.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import struct
import sys
import tempfile
import warnings
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable
from urllib.parse import unquote, urlsplit

import numpy as np
import requests
from PIL import Image, ImageOps, UnidentifiedImageError
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from scripts.extract_clip_features import sha256_file
from src.e6_acquisition_protocol import (
    acquisition_payload_paths,
    load_e6_acquisition_protocol,
    validate_e6_acquisition_protocol,
    validate_upstream_locks,
)
from src.e6_acquisition_resume_protocol import (
    DEFAULT_RESUME_LOCK,
    DEFAULT_RESUME_PROTOCOL,
    append_rejection_event,
    find_replay_event,
    load_resume_protocol,
    validate_rejection_journal,
    validate_resume_lock_receipt,
)
from src.e6_protocol import (
    assignment_sha256,
    build_pair_assignment,
    load_e6_protocol,
    pair_rows,
    validate_e6_development_protocol,
)


DEFAULT_PROTOCOL = Path("configs/e6_development_protocol.json")
DEFAULT_LOCK_REPORT = Path("reports/e6_development_protocol_lock.json")
DEFAULT_ACQUISITION_PROTOCOL = Path("configs/e6_acquisition_protocol.json")
DEFAULT_ACQUISITION_LOCK_REPORT = Path(
    "reports/e6_acquisition_protocol_lock.json"
)
DEFAULT_TRANSPORT_PREFLIGHT_AUDIT = Path(
    "reports/e6_transport_preflight_audit.json"
)
ROWS_API = "https://datasets-server.huggingface.co/rows"
ROWS_HOST = "datasets-server.huggingface.co"
DATASET_CONFIG = "default"
ROW_PAGE_SIZE = 100
MAX_IMAGE_BYTES = 64 * 1024 * 1024
MAX_METADATA_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_TOTAL_DOWNLOAD_BYTES = 16 * 1024 * 1024 * 1024
MAX_IMAGE_WIDTH = 32_768
MAX_IMAGE_HEIGHT = 32_768
MAX_DECODED_PIXELS = 100_000_000
CONNECT_TIMEOUT_SECONDS = 10
READ_TIMEOUT_SECONDS = 60
RETRY_ATTEMPTS = 4
RETRY_BACKOFF_SECONDS = 1.0
PHASH_SIZE = 8
PHASH_HIGH_FREQUENCY_FACTOR = 4
NEAR_DUPLICATE_MAX_HAMMING = 4
SOURCE_NAME = "tiny_genimage_train_biggan_e6"
ALLOWED_DECODED_FORMATS = {"JPEG", "PNG", "WEBP"}
MIME_BY_FORMAT = {
    "JPEG": {"image/jpeg"},
    "PNG": {"image/png"},
    "WEBP": {"image/webp"},
}
EXTENSIONS_BY_FORMAT = {
    "JPEG": {".jpg", ".jpeg"},
    "PNG": {".png"},
    "WEBP": {".webp"},
}
GENERIC_BINARY_MEDIA_TYPES = {
    "application/octet-stream",
    "binary/octet-stream",
}
EXPECTED_TOP_LEVEL_ROW_KEYS = {
    "features",
    "rows",
    "num_rows_total",
    "num_rows_per_page",
    "partial",
}
EXPECTED_ROW_ENVELOPE_KEYS = {"row_idx", "row", "truncated_cells"}
EXPECTED_ROW_VALUE_KEYS = {"image", "label", "generator"}
EXPECTED_IMAGE_VALUE_KEYS = {"src", "height", "width"}
MANIFEST_FIELDS = (
    "image_path",
    "label",
    "class_name",
    "source",
    "split",
    "source_row_idx",
    "pair_key",
    "assigned_cycle",
    "accepted_cycle",
    "byte_sha256",
    "decoded_pixel_sha256",
    "original_width",
    "original_height",
    "aspect_ratio",
    "decoded_pixel_count",
    "file_format",
    "file_bytes",
    "perceptual_hash",
)


class ImageIntegrityError(ValueError):
    """A fetched asset is not a safe, decodable instance of its source row."""


class RemoteContractError(ValueError):
    """The pinned remote transport or image contract drifted."""


class TransientAcquisitionError(ValueError):
    """A retryable transfer ended inconsistently; do not replace the data row."""


@dataclass(frozen=True)
class SourceRow:
    row_idx: int
    label: int
    generator_id: int
    generator_name: str
    image_url: str
    source_width: int
    source_height: int


@dataclass(frozen=True)
class FingerprintOwner:
    identity: str
    role: str
    label: int


@dataclass(frozen=True)
class ImageVerification:
    file_bytes: int
    byte_sha256: str
    decoded_pixel_sha256: str
    perceptual_hash: str
    original_width: int
    original_height: int
    aspect_ratio: float
    decoded_pixel_count: int
    display_width: int
    display_height: int
    file_format: str


@dataclass(frozen=True)
class PreparedImage:
    row: SourceRow
    image_path: str
    verification: ImageVerification


@dataclass
class DownloadBudget:
    """Track the run-wide network payload cap across accepted and rejected pairs."""

    maximum_bytes: int = MAX_TOTAL_DOWNLOAD_BYTES
    downloaded_bytes: int = 0

    def consume(self, byte_count: int) -> None:
        if byte_count < 0:
            raise ValueError("E6 download byte count cannot be negative")
        self.downloaded_bytes += byte_count
        if self.downloaded_bytes > self.maximum_bytes:
            raise ValueError("E6 acquisition exceeded its total download byte cap")


def _strict_int(value: Any, description: str) -> int:
    """Accept JSON integers and plain decimal strings, but never booleans."""

    if isinstance(value, bool):
        raise ValueError(f"{description} must be an integer, not a boolean")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"-?(0|[1-9][0-9]*)", value):
        return int(value)
    raise ValueError(f"{description} must be an integer")


def _strict_json_int(value: Any, description: str) -> int:
    """Require an actual JSON integer for frozen remote-schema fields."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{description} must be a JSON integer")
    return value


def _project_relative(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("E6 acquisition path escaped the project") from exc


def _reject_symlink_path(path: Path, project_root: Path) -> None:
    """Reject a symlink at the target or any existing project-relative parent."""

    root_lexical = Path(os.path.abspath(project_root))
    candidate = path if path.is_absolute() else root_lexical / path
    if ".." in candidate.parts:
        raise ValueError("E6 acquisition path may not contain parent traversal")
    candidate_lexical = Path(os.path.abspath(candidate))
    try:
        relative = candidate_lexical.relative_to(root_lexical)
    except ValueError as exc:
        raise ValueError("E6 acquisition path escaped the project") from exc
    current = root_lexical
    if current.is_symlink():
        raise ValueError("E6 acquisition refuses a symlink project root")
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"E6 acquisition refuses symlink path: {relative}")
    try:
        candidate_lexical.resolve(strict=False).relative_to(project_root.resolve())
    except ValueError as exc:
        raise ValueError("E6 acquisition path resolved outside the project") from exc


def validate_lock_receipt(
    protocol_path: Path,
    lock_report_path: Path,
    protocol: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    """Require the exact pre-payload lock receipt and assignment identity."""

    receipt = json.loads(lock_report_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "PASS":
        raise ValueError("E6 development protocol lock did not pass")
    frozen = receipt.get("protocol", {})
    expected_path = _project_relative(protocol_path, project_root)
    if frozen.get("path") != expected_path:
        raise ValueError("E6 lock receipt references a different protocol")
    if frozen.get("sha256") != sha256_file(protocol_path):
        raise ValueError("E6 protocol changed after its pre-payload lock")
    if receipt.get("image_payload_present_at_freeze") is not False:
        raise ValueError("E6 image payload existed when the protocol was locked")
    frozen_source = receipt.get("development_source", {})
    if frozen_source.get("repository_revision") != protocol["development_source"][
        "repository_revision"
    ]:
        raise ValueError("E6 lock receipt uses a different repository revision")
    observed_assignment = assignment_sha256(build_pair_assignment(protocol))
    if frozen_source.get("assignment_sha256") != observed_assignment:
        raise ValueError("E6 frozen pair assignment changed after the lock")
    if receipt.get("shortcut_gate", {}).get("training_allowed_before_pass") is not False:
        raise ValueError("E6 lock improperly allowed training before the shortcut gate")
    registry = receipt.get("registry", {})
    registry_relative = registry.get("path")
    if not isinstance(registry_relative, str):
        raise ValueError("E6 lock receipt has no evaluation-registry path")
    registry_path = project_root / registry_relative
    _reject_symlink_path(registry_path, project_root)
    if not registry_path.is_file() or registry.get("sha256") != sha256_file(
        registry_path
    ):
        raise ValueError("E6 evaluation registry changed after the protocol lock")
    guardrails = receipt.get("guardrails", {})
    for key in (
        "organiser_validation_subset_used",
        "consumed_tests_used_for_e6_fitting",
        "future_lockbox_payload_accessed",
        "manual_image_selection_allowed",
    ):
        if guardrails.get(key) is not False:
            raise ValueError(f"E6 lock guardrail changed: {key}")
    return receipt


def _validate_acquisition_implementation_contract(
    acquisition: dict[str, Any], development: dict[str, Any]
) -> None:
    """Bind this downloader's executable constants to the frozen contract."""

    source = acquisition["source_access"]
    caps = acquisition["resource_caps"]
    identities = acquisition["identity_algorithms"]
    transport = acquisition["trusted_asset_transport"]
    outputs = acquisition["outputs"]
    expected = {
        "rows_api": ROWS_API,
        "rows_host": ROWS_HOST,
        "dataset_config": DATASET_CONFIG,
        "row_page_size": ROW_PAGE_SIZE,
        "maximum_image_bytes": MAX_IMAGE_BYTES,
        "maximum_metadata_response_bytes": MAX_METADATA_RESPONSE_BYTES,
        "maximum_total_download_bytes": MAX_TOTAL_DOWNLOAD_BYTES,
        "maximum_width": MAX_IMAGE_WIDTH,
        "maximum_height": MAX_IMAGE_HEIGHT,
        "maximum_decoded_pixels": MAX_DECODED_PIXELS,
        "connect_timeout": CONNECT_TIMEOUT_SECONDS,
        "read_timeout": READ_TIMEOUT_SECONDS,
        "retry_attempts": RETRY_ATTEMPTS,
        "retry_backoff": RETRY_BACKOFF_SECONDS,
        "near_duplicate_distance": NEAR_DUPLICATE_MAX_HAMMING,
    }
    observed = {
        "rows_api": source["rows_endpoint"],
        "rows_host": transport["host"],
        "dataset_config": source["query"]["config"],
        "row_page_size": caps["maximum_rows_per_metadata_request"],
        "maximum_image_bytes": caps["maximum_asset_bytes"],
        "maximum_metadata_response_bytes": caps[
            "maximum_metadata_response_bytes"
        ],
        "maximum_total_download_bytes": caps["maximum_total_download_bytes"],
        "maximum_width": caps["maximum_width_pixels"],
        "maximum_height": caps["maximum_height_pixels"],
        "maximum_decoded_pixels": caps["maximum_decoded_pixels"],
        "connect_timeout": caps["connect_timeout_seconds"],
        "read_timeout": caps["read_timeout_seconds"],
        "retry_attempts": caps["retry_attempts"],
        "retry_backoff": float(caps["retry_backoff_seconds"]),
        "near_duplicate_distance": identities["perceptual_hash"][
            "near_duplicate_max_distance_inclusive"
        ],
    }
    if observed != expected:
        raise ValueError("E6 acquisition implementation drifted from its frozen lock")
    if set(acquisition["image_validation"]["allowed_formats"]) != (
        ALLOWED_DECODED_FORMATS
    ):
        raise ValueError("E6 decoded-image allowlist drifted from its frozen lock")
    for file_format, definition in acquisition["image_validation"][
        "allowed_formats"
    ].items():
        if set(definition["media_types"]) != MIME_BY_FORMAT[file_format]:
            raise ValueError("E6 MIME allowlist drifted from its frozen lock")
        if set(definition["extensions"]) != EXTENSIONS_BY_FORMAT[file_format]:
            raise ValueError("E6 extension allowlist drifted from its frozen lock")
    if set(
        acquisition["image_validation"][
            "generic_binary_media_types_allowed_after_magic_validation"
        ]
    ) != GENERIC_BINARY_MEDIA_TYPES:
        raise ValueError("E6 generic binary MIME policy drifted from its frozen lock")
    if outputs["raw_root"] != development["outputs"]["raw_root"]:
        raise ValueError("E6 acquisition raw root disagrees with development lock")
    if outputs["manifest_root"] != development["outputs"]["manifest_root"]:
        raise ValueError("E6 acquisition manifest root disagrees with development lock")
    if outputs["provenance"] != development["outputs"]["provenance"]:
        raise ValueError("E6 acquisition provenance path disagrees with development lock")
    if acquisition["overlap_and_replacement"]["existing_manifests"] != (
        development["integrity"]["existing_manifests_to_exclude"]
    ):
        raise ValueError("E6 acquisition exclusion inventory drifted")


def validate_acquisition_lock_receipt(
    acquisition_protocol_path: Path,
    acquisition_lock_path: Path,
    acquisition: dict[str, Any],
    project_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require the exact offline acquisition/security lock before networking."""

    validate_e6_acquisition_protocol(acquisition)
    development, development_lock = validate_upstream_locks(
        acquisition, project_root
    )
    _validate_acquisition_implementation_contract(acquisition, development)
    receipt = json.loads(acquisition_lock_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "PASS":
        raise ValueError("E6 acquisition protocol lock did not pass")
    frozen = receipt.get("acquisition_protocol", {})
    expected_path = _project_relative(acquisition_protocol_path, project_root)
    if frozen != {
        "path": expected_path,
        "sha256": sha256_file(acquisition_protocol_path),
        "schema_version": 1,
    }:
        raise ValueError("E6 acquisition lock references a changed protocol")
    if receipt.get("payload_present_at_freeze") is not False:
        raise ValueError("E6 payload existed when the acquisition lock was made")
    upstream = receipt.get("upstream_locks", {})
    required_upstream = acquisition["upstream_locks"]
    if upstream.get("development_protocol_sha256") != required_upstream[
        "development_protocol"
    ]["sha256"]:
        raise ValueError("E6 acquisition lock development protocol drifted")
    if upstream.get("development_lock_sha256") != required_upstream[
        "development_protocol_lock"
    ]["sha256"]:
        raise ValueError("E6 acquisition lock development receipt drifted")
    if upstream.get("development_lock_status") != "PASS":
        raise ValueError("E6 acquisition lock did not bind a passing development lock")
    if upstream.get("assignment_sha256") != assignment_sha256(
        build_pair_assignment(development)
    ):
        raise ValueError("E6 acquisition assignment identity drifted")
    source = acquisition["source_access"]
    if receipt.get("source_contract") != {
        "rows_endpoint": source["rows_endpoint"],
        "dataset": source["query"]["dataset"],
        "config": source["query"]["config"],
        "split": source["query"]["split"],
        "repository_revision": source["required_response_revision"],
        "selected_pairs": len(build_pair_assignment(development)),
        "reserve_pairs": development["selection"]["unused_reserve_pairs"],
    }:
        raise ValueError("E6 acquisition lock source contract drifted")
    transport = acquisition["trusted_asset_transport"]
    caps = acquisition["resource_caps"]
    if receipt.get("security_contract") != {
        "asset_scheme": transport["scheme"],
        "asset_host": transport["host"],
        "asset_path_prefix": transport["path_prefix"],
        "redirects_allowed": False,
        "signed_urls_persisted": False,
        "maximum_asset_bytes": caps["maximum_asset_bytes"],
        "allowed_decoded_formats": sorted(ALLOWED_DECODED_FORMATS),
        "generic_binary_media_types_allowed_after_magic_validation": sorted(
            GENERIC_BINARY_MEDIA_TYPES
        ),
        "symlinks_allowed": False,
    }:
        raise ValueError("E6 acquisition lock security contract drifted")
    identities = acquisition["identity_algorithms"]
    if receipt.get("identity_contract") != {
        "decoded_rgba_algorithm_id": identities["decoded_rgba_sha256"][
            "algorithm_id"
        ],
        "perceptual_hash_algorithm_id": identities["perceptual_hash"][
            "algorithm_id"
        ],
        "near_duplicate_max_distance_inclusive": NEAR_DUPLICATE_MAX_HAMMING,
    }:
        raise ValueError("E6 acquisition lock identity contract drifted")
    overlap = acquisition["overlap_and_replacement"]
    if receipt.get("overlap_contract") != {
        "existing_manifest_count": len(overlap["existing_manifests"]),
        "existing_overlap_action": "hard_abort_without_replacement",
        "conflicting_label_action": "hard_abort_without_replacement",
        "eligible_replacement_reasons": [
            "corrupt_remote_asset",
            "within_e6_same_label_duplicate",
        ],
        "replacement_source": overlap["replacement_source"],
    }:
        raise ValueError("E6 acquisition lock overlap contract drifted")
    expected_absent = [
        path.relative_to(project_root).as_posix()
        for path in acquisition_payload_paths(acquisition, project_root)
    ]
    if receipt.get("checked_absent_paths") != expected_absent:
        raise ValueError("E6 acquisition lock absent-path inventory drifted")
    if receipt.get("guardrails") != {
        "network_or_image_access_during_lock_check": False,
        "training_or_feature_extraction_performed": False,
        "evaluation_performed": False,
        "organiser_validation_accessed": False,
        "consumed_tests_used_for_selection_or_replacement": False,
    }:
        raise ValueError("E6 acquisition lock guardrails changed")
    if development_lock.get("image_payload_present_at_freeze") is not False:
        raise ValueError("E6 upstream development lock was not pre-payload")
    return receipt, development


def validate_transport_preflight_audit(
    audit_path: Path,
    acquisition_protocol_path: Path,
    acquisition_lock_path: Path,
) -> dict[str, Any]:
    """Bind the disclosed post-lock/pre-commit 64-byte transport probe."""

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("schema_version") != 1
        or audit.get("status") != "DISCLOSED_NON_SELECTION_ACCESS"
    ):
        raise ValueError("E6 transport-preflight disclosure is invalid")
    frozen = audit.get("frozen_artifacts_already_present", {})
    if frozen.get("acquisition_protocol_sha256") != sha256_file(
        acquisition_protocol_path
    ) or frozen.get("acquisition_lock_sha256") != sha256_file(
        acquisition_lock_path
    ):
        raise ValueError("E6 transport-preflight disclosure hash changed")
    probe = audit.get("probe", {})
    required_probe = {
        "http_method": "GET",
        "streaming": True,
        "range_request": False,
        "source_row_index": 2,
        "frozen_pair_index": 0,
        "frozen_role": "development_train",
        "response_status": 200,
        "response_media_type": "binary/octet-stream",
        "response_body_bytes_read": 64,
        "image_decoded": False,
        "image_viewed": False,
        "image_body_saved": False,
        "signed_url_or_query_persisted": False,
        "temporary_rows_metadata_deleted": True,
    }
    if any(probe.get(key) != value for key, value in required_probe.items()):
        raise ValueError("E6 transport-preflight disclosure facts changed")
    chronology = audit.get("chronology", {})
    if chronology != {
        "development_assignment_committed_before_probe": True,
        "acquisition_protocol_and_pass_receipt_existed_before_probe": True,
        "acquisition_protocol_and_pass_receipt_changed_after_probe": False,
        "acquisition_protocol_and_pass_receipt_git_committed_before_probe": False,
    }:
        raise ValueError("E6 transport-preflight chronology changed")
    impact = audit.get("impact", {})
    for key in (
        "selection_or_replacement_influenced",
        "model_training_or_feature_extraction_performed",
        "evaluation_performed",
        "future_lockbox_accessed",
    ):
        if impact.get(key) is not False:
            raise ValueError("E6 transport-preflight impact disclosure changed")
    return audit


def ranked_cycles(protocol: dict[str, Any]) -> tuple[int, ...]:
    """Return the complete frozen pair order, including the replacement reserve."""

    validate_e6_development_protocol(protocol)
    selection = protocol["selection"]
    revision = protocol["development_source"]["repository_revision"]
    seed = int(selection["seed"])

    def key(cycle: int) -> tuple[bytes, int]:
        payload = f"e6d:{seed}:{revision}:biggan:{cycle}".encode("utf-8")
        return hashlib.sha256(payload).digest(), cycle

    ranked = tuple(sorted(range(int(selection["candidate_pairs"])), key=key))
    selected = tuple(item["pair_index"] for item in build_pair_assignment(protocol))
    if ranked[: len(selected)] != selected:
        raise ValueError("E6 acquisition ranking does not match the frozen assignment")
    return ranked


def _validate_asset_url(
    image_url: str,
    *,
    repository: str,
    revision: str,
    split: str,
    row_idx: int,
) -> str:
    """Restrict signed image URLs to the exact pinned datasets-server row."""

    parsed = urlsplit(image_url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != ROWS_HOST
        or parsed.port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ValueError(f"Unsafe datasets-server asset URL for row {row_idx}")
    decoded_path = unquote(parsed.path)
    path = PurePosixPath(decoded_path)
    if ".." in path.parts or "\x00" in decoded_path:
        raise ValueError(f"Unsafe datasets-server asset path for row {row_idx}")
    prefix = (
        f"/cached-assets/{repository}/--/{revision}/--/"
        f"{DATASET_CONFIG}/{split}/{row_idx}/image/"
    )
    if re.fullmatch(re.escape(prefix) + r"[A-Za-z0-9._-]+", decoded_path) is None:
        raise ValueError(
            f"datasets-server asset does not belong to frozen row {row_idx}"
        )
    extension = PurePosixPath(decoded_path).suffix.lower()
    if extension not in {value for values in EXTENSIONS_BY_FORMAT.values() for value in values}:
        raise ValueError(
            f"datasets-server asset has an unsupported extension for row {row_idx}"
        )
    return extension


def parse_rows_page(
    payload: dict[str, Any],
    *,
    observed_revision: str | None,
    expected_revision: str,
    expected_total_rows: int,
    repository: str,
    split: str,
    expected_offset: int | None = None,
    expected_length: int | None = None,
) -> dict[int, SourceRow]:
    """Validate one datasets-server page without decoding any image payload."""

    if observed_revision != expected_revision:
        raise ValueError(
            f"Tiny-GenImage revision drift: expected {expected_revision}, "
            f"received {observed_revision!r}"
        )
    if set(payload) != EXPECTED_TOP_LEVEL_ROW_KEYS:
        raise ValueError("Tiny-GenImage top-level row schema changed")
    if payload.get("partial") is not False:
        raise ValueError("Hugging Face returned a partial Tiny-GenImage page")
    if _strict_json_int(
        payload.get("num_rows_total"), "num_rows_total"
    ) != expected_total_rows:
        raise ValueError("Tiny-GenImage train row count changed")
    rows_per_page = _strict_json_int(
        payload.get("num_rows_per_page"), "num_rows_per_page"
    )
    if rows_per_page <= 0 or rows_per_page > ROW_PAGE_SIZE:
        raise ValueError("Tiny-GenImage num_rows_per_page is outside the lock")
    envelopes = payload.get("rows")
    if not isinstance(envelopes, list):
        raise ValueError("Tiny-GenImage page has no rows list")
    parsed: dict[int, SourceRow] = {}
    for envelope in envelopes:
        if not isinstance(envelope, dict) or not isinstance(envelope.get("row"), dict):
            raise ValueError("Tiny-GenImage row envelope is malformed")
        if set(envelope) != EXPECTED_ROW_ENVELOPE_KEYS:
            raise ValueError("Tiny-GenImage row-envelope schema changed")
        if not isinstance(envelope.get("truncated_cells"), list):
            raise ValueError("Tiny-GenImage truncated_cells must be an array")
        row_idx = _strict_json_int(envelope.get("row_idx"), "row_idx")
        row = envelope["row"]
        if set(row) != EXPECTED_ROW_VALUE_KEYS:
            raise ValueError(f"Tiny-GenImage row {row_idx} value schema changed")
        image = row.get("image")
        if not isinstance(image, dict) or not isinstance(image.get("src"), str):
            raise ValueError(f"Tiny-GenImage row {row_idx} has no image.src")
        if set(image) != EXPECTED_IMAGE_VALUE_KEYS:
            raise ValueError(f"Tiny-GenImage row {row_idx} image schema changed")
        width = _strict_json_int(image.get("width"), f"row {row_idx} image.width")
        height = _strict_json_int(image.get("height"), f"row {row_idx} image.height")
        if (
            width <= 0
            or height <= 0
            or width > MAX_IMAGE_WIDTH
            or height > MAX_IMAGE_HEIGHT
            or width * height > MAX_DECODED_PIXELS
        ):
            raise ValueError(f"Tiny-GenImage row {row_idx} has unsafe dimensions")
        image_url = image["src"]
        _validate_asset_url(
            image_url,
            repository=repository,
            revision=expected_revision,
            split=split,
            row_idx=row_idx,
        )
        if row_idx in parsed:
            raise ValueError("Tiny-GenImage page repeated a row index")
        generator_id = _strict_json_int(
            row.get("generator"), f"row {row_idx} generator"
        )
        generator_name = {0: "Real", 2: "BigGAN"}.get(
            generator_id, f"class_label_{generator_id}"
        )
        parsed[row_idx] = SourceRow(
            row_idx=row_idx,
            label=_strict_json_int(row.get("label"), f"row {row_idx} label"),
            generator_id=generator_id,
            generator_name=generator_name,
            image_url=image_url,
            source_width=width,
            source_height=height,
        )
    if expected_offset is not None or expected_length is not None:
        if expected_offset is None or expected_length is None:
            raise ValueError("Expected row-page offset and length must be supplied together")
        end = min(expected_total_rows, expected_offset + expected_length)
        expected_indices = set(range(expected_offset, end))
        if set(parsed) != expected_indices:
            raise ValueError("Tiny-GenImage page omitted, repeated, or added row indices")
    return parsed


def validate_pair_rows(
    protocol: dict[str, Any], cycle: int, rows: dict[int, SourceRow]
) -> tuple[SourceRow, SourceRow]:
    """Require the frozen real and BigGAN contracts for one cycle."""

    real_idx, ai_idx = pair_rows(protocol, cycle)
    if set(rows) != {real_idx, ai_idx}:
        raise ValueError(f"Tiny-GenImage cycle {cycle} did not return both frozen rows")
    topology = protocol["row_topology"]
    real = rows[real_idx]
    ai = rows[ai_idx]
    contracts = (
        (real, topology["real_row_contract"]),
        (ai, topology["ai_row_contract"]),
    )
    for observed, expected in contracts:
        identity = (observed.label, observed.generator_id, observed.generator_name)
        required = (
            int(expected["label"]),
            int(expected["generator_id"]),
            str(expected["generator_name"]),
        )
        if identity != required:
            raise ValueError(
                f"Tiny-GenImage row contract drift at row {observed.row_idx}: "
                f"expected {required}, received {identity}"
            )
    return real, ai


def _read_json_response_limited(
    response: requests.Response, *, maximum_bytes: int
) -> dict[str, Any]:
    """Read one JSON response without allowing an unbounded metadata body."""

    content_encoding = response.headers.get("Content-Encoding", "").lower().strip()
    if content_encoding not in {"", "identity"}:
        raise ValueError("Tiny-GenImage metadata used an encoded transport")
    content_type = response.headers.get("Content-Type", "")
    normalized_type = content_type.lower().split(";", maxsplit=1)[0].strip()
    if normalized_type not in {"application/json", "application/json-seq"}:
        raise ValueError("Tiny-GenImage metadata response was not JSON")
    declared = response.headers.get("Content-Length")
    if declared is not None:
        declared_bytes = _strict_int(declared, "metadata Content-Length")
        if declared_bytes <= 0 or declared_bytes > maximum_bytes:
            raise ValueError("Tiny-GenImage metadata response exceeded its byte cap")
    chunks: list[bytes] = []
    observed = 0
    for chunk in response.iter_content(chunk_size=64 * 1024):
        if not chunk:
            continue
        observed += len(chunk)
        if observed > maximum_bytes:
            raise ValueError("Tiny-GenImage metadata response exceeded its byte cap")
        chunks.append(chunk)
    try:
        payload = json.loads(b"".join(chunks))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Tiny-GenImage metadata response was invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Tiny-GenImage metadata response must be a JSON object")
    return payload


class RowsClient:
    """Small pinned-revision datasets-server client with page caching."""

    def __init__(
        self,
        session: requests.Session,
        protocol: dict[str, Any],
        *,
        page_size: int = ROW_PAGE_SIZE,
    ) -> None:
        if page_size <= 0 or page_size > 100:
            raise ValueError("Tiny-GenImage rows page size must be in [1, 100]")
        self.session = session
        self.protocol = protocol
        self.page_size = page_size
        self._pages: dict[int, dict[int, SourceRow]] = {}

    def _page(self, page_index: int, *, refresh: bool = False) -> dict[int, SourceRow]:
        if not refresh and page_index in self._pages:
            return self._pages[page_index]
        source = self.protocol["development_source"]
        offset = page_index * self.page_size
        response = self.session.get(
            ROWS_API,
            headers={"Accept": "application/json", "Accept-Encoding": "identity"},
            params={
                "dataset": source["dataset_repository"],
                "config": DATASET_CONFIG,
                "split": source["source_split"],
                "offset": offset,
                "length": self.page_size,
            },
            timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
            allow_redirects=False,
            stream=True,
        )
        if response.status_code != 200:
            response.close()
            raise requests.HTTPError(
                f"Tiny-GenImage rows API returned HTTP {response.status_code}"
            )
        try:
            payload = _read_json_response_limited(
                response, maximum_bytes=MAX_METADATA_RESPONSE_BYTES
            )
            page = parse_rows_page(
                payload,
                observed_revision=response.headers.get("X-Revision"),
                expected_revision=source["repository_revision"],
                expected_total_rows=int(self.protocol["row_topology"]["source_train_rows"]),
                repository=source["dataset_repository"],
                split=source["source_split"],
                expected_offset=offset,
                expected_length=self.page_size,
            )
        finally:
            response.close()
        self._pages[page_index] = page
        return page

    def pair(self, cycle: int) -> tuple[SourceRow, SourceRow]:
        required = pair_rows(self.protocol, cycle)
        rows: dict[int, SourceRow] = {}
        for row_idx in required:
            page_index = row_idx // self.page_size
            page = self._page(page_index)
            if row_idx not in page:
                raise ValueError(f"Tiny-GenImage page omitted frozen row {row_idx}")
            rows[row_idx] = page[row_idx]
        return validate_pair_rows(self.protocol, cycle, rows)

    def refresh_row(self, row_idx: int) -> SourceRow:
        """Refresh one expiring signed URL while preserving the exact source row."""

        page_index = row_idx // self.page_size
        page = self._page(page_index, refresh=True)
        if row_idx not in page:
            raise ValueError(f"Refreshed Tiny-GenImage page omitted row {row_idx}")
        return page[row_idx]


def _session() -> requests.Session:
    retry = Retry(
        total=RETRY_ATTEMPTS,
        connect=RETRY_ATTEMPTS,
        read=RETRY_ATTEMPTS,
        status=RETRY_ATTEMPTS,
        backoff_factor=RETRY_BACKOFF_SECONDS,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    session = requests.Session()
    session.headers.update({"User-Agent": "AIGC-detector-E6-development/1.0"})
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def decoded_pixel_sha256(image: Image.Image) -> str:
    """Hash orientation-normalized RGBA pixels using the frozen E6 identity."""

    rgba = ImageOps.exif_transpose(image).convert("RGBA")
    width, height = rgba.size
    digest = hashlib.sha256()
    digest.update(b"E6RGBA1\0")
    digest.update(struct.pack(">II", width, height))
    digest.update(rgba.tobytes())
    return digest.hexdigest()


def perceptual_hash64(image: Image.Image) -> int:
    """Return a deterministic 64-bit pHash for strict near-duplicate checks."""

    side = PHASH_SIZE * PHASH_HIGH_FREQUENCY_FACTOR
    rgba = ImageOps.exif_transpose(image).convert("RGBA")
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    grayscale = Image.alpha_composite(white, rgba).convert("L").resize(
        (side, side), Image.Resampling.LANCZOS
    )
    pixels = np.asarray(grayscale, dtype=np.float64)
    positions = np.arange(side, dtype=np.float64)
    frequencies = np.arange(PHASH_SIZE, dtype=np.float64)[:, None]
    basis = np.cos(math.pi * (2.0 * positions + 1.0) * frequencies / (2.0 * side))
    basis[0] *= math.sqrt(1.0 / side)
    basis[1:] *= math.sqrt(2.0 / side)
    low_frequency = basis @ pixels @ basis.T
    flattened = low_frequency.reshape(-1)
    median = float(np.median(flattened[1:]))
    bits = flattened > median
    bits[0] = False
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def inspect_image(path: Path) -> ImageVerification:
    """Fully verify and fingerprint a local image without trusting its suffix."""

    if path.is_symlink() or not path.is_file():
        raise ImageIntegrityError(f"Image cache is missing or a symlink: {path.name}")
    file_bytes = path.stat().st_size
    if not 0 < file_bytes <= MAX_IMAGE_BYTES:
        raise ImageIntegrityError(f"Image has unsafe byte size: {path.name}")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                if int(getattr(image, "n_frames", 1)) != 1:
                    raise ImageIntegrityError(
                        f"Animated or multi-frame images are forbidden: {path.name}"
                    )
                image.verify()
            with Image.open(path) as image:
                original_width, original_height = image.size
                if (
                    original_width <= 0
                    or original_height <= 0
                    or original_width > MAX_IMAGE_WIDTH
                    or original_height > MAX_IMAGE_HEIGHT
                    or original_width * original_height > MAX_DECODED_PIXELS
                ):
                    raise ImageIntegrityError(
                        f"Image has unsafe decoded dimensions: {path.name}"
                    )
                file_format = str(image.format or "unknown")
                if file_format.upper() not in ALLOWED_DECODED_FORMATS:
                    raise ImageIntegrityError(
                        f"Image format is outside frozen allowlist: {path.name}"
                    )
                display = ImageOps.exif_transpose(image).convert("RGBA")
                display.load()
                pixel_digest = decoded_pixel_sha256(image)
                perceptual = perceptual_hash64(image)
                display_width, display_height = display.size
    except (
        OSError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ImageIntegrityError(f"Image failed decode verification: {path.name}") from exc
    return ImageVerification(
        file_bytes=file_bytes,
        byte_sha256=sha256_file(path),
        decoded_pixel_sha256=pixel_digest,
        perceptual_hash=f"{perceptual:016x}",
        original_width=original_width,
        original_height=original_height,
        aspect_ratio=float(original_width / original_height),
        decoded_pixel_count=int(original_width * original_height),
        display_width=display_width,
        display_height=display_height,
        file_format=file_format.upper(),
    )


def _cached_asset_path(raw_root: Path, row: SourceRow) -> Path:
    extension = PurePosixPath(_canonical_asset_path(row.image_url)).suffix.lower()
    if extension not in {
        value for values in EXTENSIONS_BY_FORMAT.values() for value in values
    }:
        raise ValueError(f"Unsupported frozen asset extension for row {row.row_idx}")
    return raw_root / "images" / f"row-{row.row_idx:05d}{extension}"


def _cache_receipt_path(destination: Path) -> Path:
    return destination.with_name(f"{destination.name}.receipt.json")


def _cache_pending_receipt_path(destination: Path) -> Path:
    """Return the crash-recovery journal written before asset publication."""

    return destination.with_name(f"{destination.name}.receipt.pending.json")


def _discard_cached_rows(
    raw_root: Path,
    rows: Iterable[SourceRow],
    *,
    project_root: Path,
) -> None:
    """Remove only deterministic cache files for a rejected whole pair."""

    touched: set[Path] = set()
    for row in rows:
        destination = _cached_asset_path(raw_root, row)
        receipt = _cache_receipt_path(destination)
        pending_receipt = _cache_pending_receipt_path(destination)
        for path in (destination, receipt, pending_receipt):
            _reject_symlink_path(path, project_root)
            if path.exists():
                if path.is_symlink() or not path.is_file():
                    raise ValueError(f"Refusing unsafe rejected-pair cache: {path}")
                path.unlink()
                touched.add(path.parent)
    for directory in touched:
        _fsync_directory(directory)


def _discard_cached_row_indices(
    raw_root: Path, row_indices: Iterable[int], *, project_root: Path
) -> None:
    """Crash recovery for a journaled pair when signed row URLs are unavailable."""

    images_root = raw_root / "images"
    _reject_symlink_path(images_root, project_root)
    if not images_root.exists():
        return
    touched = False
    for row_idx in row_indices:
        prefix = f"row-{int(row_idx):05d}"
        asset_names = {
            f"{prefix}{extension}"
            for extensions in EXTENSIONS_BY_FORMAT.values()
            for extension in extensions
        }
        allowed = asset_names | {
            f"{name}.receipt.json" for name in asset_names
        } | {f"{name}.receipt.pending.json" for name in asset_names}
        for candidate in images_root.iterdir():
            if candidate.name not in allowed:
                continue
            _reject_symlink_path(candidate, project_root)
            if candidate.is_symlink() or not candidate.is_file():
                raise ValueError("Refusing unsafe journaled-pair cache entry")
            candidate.unlink()
            touched = True
    if touched:
        _fsync_directory(images_root)


def _canonical_asset_path(image_url: str) -> str:
    """Return the non-secret asset path, excluding every signed query value."""

    return unquote(urlsplit(image_url).path)


def _write_cache_receipt(
    receipt_path: Path,
    row: SourceRow,
    verification: ImageVerification,
) -> None:
    payload = {
        "schema_version": 1,
        "row_idx": row.row_idx,
        "label": row.label,
        "generator_id": row.generator_id,
        "generator_name": row.generator_name,
        "source_width": row.source_width,
        "source_height": row.source_height,
        "canonical_asset_path": _canonical_asset_path(row.image_url),
        "verification": asdict(verification),
        "signed_query_persisted": False,
    }
    _atomic_json_write_fsync(receipt_path, payload)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json_write_fsync(path: Path, payload: dict[str, Any]) -> None:
    """Publish JSON with an exclusive same-directory temporary and fsync."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"Refusing to overwrite symlink JSON file: {path}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _validate_cache_receipt(
    receipt_path: Path,
    row: SourceRow,
    verification: ImageVerification,
) -> None:
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError(
            f"Refusing unbound pre-existing E6 cache for row {row.row_idx}"
        )
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected_identity = {
        "row_idx": row.row_idx,
        "label": row.label,
        "generator_id": row.generator_id,
        "generator_name": row.generator_name,
        "source_width": row.source_width,
        "source_height": row.source_height,
        "canonical_asset_path": _canonical_asset_path(row.image_url),
    }
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported E6 cache receipt for row {row.row_idx}")
    if any(payload.get(key) != value for key, value in expected_identity.items()):
        raise ValueError(f"E6 cache receipt identity drift for row {row.row_idx}")
    if payload.get("verification") != asdict(verification):
        raise ValueError(f"E6 cached bytes changed for row {row.row_idx}")
    if payload.get("signed_query_persisted") is not False:
        raise ValueError(f"E6 cache receipt leaked a signed query for row {row.row_idx}")


def _validate_mime_magic(content_type: str, file_format: str, row_idx: int) -> None:
    normalized = content_type.lower().split(";", maxsplit=1)[0].strip()
    if normalized in GENERIC_BINARY_MEDIA_TYPES:
        return
    if normalized not in MIME_BY_FORMAT[file_format]:
        raise RemoteContractError(
            f"Tiny-GenImage row {row_idx} MIME/magic mismatch"
        )


def _validate_extension_magic(
    image_url: str, file_format: str, row_idx: int
) -> None:
    extension = PurePosixPath(_canonical_asset_path(image_url)).suffix.lower()
    if extension not in EXTENSIONS_BY_FORMAT[file_format]:
        raise RemoteContractError(
            f"Tiny-GenImage row {row_idx} extension/magic mismatch"
        )


def download_source_image(
    session: requests.Session,
    row: SourceRow,
    destination: Path,
    *,
    project_root: Path,
    refresh_row: Callable[[int], SourceRow] | None = None,
    budget: DownloadBudget | None = None,
) -> ImageVerification:
    """Download one exact signed asset atomically, or verify the cached bytes."""

    _reject_symlink_path(destination, project_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_path(destination.parent, project_root)
    receipt_path = _cache_receipt_path(destination)
    pending_receipt_path = _cache_pending_receipt_path(destination)
    _reject_symlink_path(receipt_path, project_root)
    _reject_symlink_path(pending_receipt_path, project_root)
    if destination.exists():
        try:
            verification = inspect_image(destination)
        except ImageIntegrityError as exc:
            raise ValueError(
                f"Cached E6 asset failed revalidation for row {row.row_idx}"
            ) from exc
        if (
            verification.original_width != row.source_width
            or verification.original_height != row.source_height
        ):
            raise ValueError(
                f"Cached Tiny-GenImage dimensions changed for row {row.row_idx}"
            )
        _validate_extension_magic(row.image_url, verification.file_format, row.row_idx)
        if receipt_path.exists():
            if pending_receipt_path.exists():
                raise ValueError(
                    f"E6 cache has conflicting receipts for row {row.row_idx}"
                )
            _validate_cache_receipt(receipt_path, row, verification)
        elif pending_receipt_path.exists():
            _validate_cache_receipt(pending_receipt_path, row, verification)
            pending_receipt_path.replace(receipt_path)
            _fsync_directory(receipt_path.parent)
        else:
            raise ValueError(
                f"Refusing unbound pre-existing E6 cache for row {row.row_idx}"
            )
        return verification
    if receipt_path.exists():
        raise ValueError(f"E6 cache receipt has no asset for row {row.row_idx}")
    if pending_receipt_path.exists():
        if pending_receipt_path.is_symlink() or not pending_receipt_path.is_file():
            raise ValueError(
                f"Refusing unsafe pending E6 receipt for row {row.row_idx}"
            )
        pending_receipt_path.unlink()
        _fsync_directory(pending_receipt_path.parent)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".part",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            current_row = row
            response: Any = None
            for request_attempt in range(2):
                try:
                    response = session.get(
                        current_row.image_url,
                        headers={"Accept-Encoding": "identity"},
                        stream=True,
                        timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
                        allow_redirects=False,
                    )
                except requests.RequestException:
                    raise TransientAcquisitionError(
                        f"Tiny-GenImage asset request failed for row {row.row_idx}"
                    ) from None
                if response.status_code not in {401, 403, 404}:
                    break
                response.close()
                if request_attempt == 0 and refresh_row is not None:
                    refreshed = refresh_row(row.row_idx)
                    if (
                        refreshed.row_idx,
                        refreshed.label,
                        refreshed.generator_id,
                        refreshed.generator_name,
                        refreshed.source_width,
                        refreshed.source_height,
                        _canonical_asset_path(refreshed.image_url),
                    ) != (
                        row.row_idx,
                        row.label,
                        row.generator_id,
                        row.generator_name,
                        row.source_width,
                        row.source_height,
                        _canonical_asset_path(row.image_url),
                    ):
                        raise ValueError(
                            f"Refreshed Tiny-GenImage row identity drift: {row.row_idx}"
                        )
                    current_row = refreshed
                    continue
                break
            try:
                if response is None or response.status_code != 200:
                    status = "no-response" if response is None else response.status_code
                    raise requests.HTTPError(
                        f"Tiny-GenImage asset row {row.row_idx} returned HTTP {status}"
                    )
                content_encoding = response.headers.get("Content-Encoding", "").lower()
                if content_encoding not in {"", "identity"}:
                    raise RemoteContractError(
                        f"Tiny-GenImage row {row.row_idx} used encoded transport"
                    )
                content_type = response.headers.get("Content-Type", "")
                normalized_type = (
                    content_type.lower().split(";", maxsplit=1)[0].strip()
                )
                if normalized_type not in {
                    item for values in MIME_BY_FORMAT.values() for item in values
                } | GENERIC_BINARY_MEDIA_TYPES:
                    raise RemoteContractError(
                        f"Tiny-GenImage row {row.row_idx} returned non-image content"
                    )
                declared_length = response.headers.get("Content-Length")
                if declared_length is None:
                    raise TransientAcquisitionError(
                        f"Tiny-GenImage row {row.row_idx} omitted Content-Length"
                    )
                length = _strict_int(declared_length, "asset Content-Length")
                if length <= 0 or length > MAX_IMAGE_BYTES:
                    raise RemoteContractError(
                        f"Tiny-GenImage row {row.row_idx} declared unsafe byte size"
                    )
                downloaded = 0
                try:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        downloaded += len(chunk)
                        if downloaded > MAX_IMAGE_BYTES:
                            raise RemoteContractError(
                                f"Tiny-GenImage row {row.row_idx} exceeded byte limit"
                            )
                        if budget is not None:
                            budget.consume(len(chunk))
                        handle.write(chunk)
                except requests.RequestException:
                    raise TransientAcquisitionError(
                        f"Tiny-GenImage asset stream failed for row {row.row_idx}"
                    ) from None
                if downloaded != length:
                    raise TransientAcquisitionError(
                        f"Tiny-GenImage row {row.row_idx} was truncated"
                    )
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                response.close()
        verification = inspect_image(temporary_path)
        _validate_mime_magic(content_type, verification.file_format, row.row_idx)
        _validate_extension_magic(
            current_row.image_url, verification.file_format, row.row_idx
        )
        if (
            verification.original_width != row.source_width
            or verification.original_height != row.source_height
        ):
            raise RemoteContractError(
                f"Downloaded Tiny-GenImage dimensions changed for row {row.row_idx}"
            )
        _write_cache_receipt(pending_receipt_path, row, verification)
        temporary_path.replace(destination)
        _fsync_directory(destination.parent)
        pending_receipt_path.replace(receipt_path)
        _fsync_directory(receipt_path.parent)
        return verification
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


class HammingBKTree:
    """Compact metric index for 64-bit perceptual hashes."""

    def __init__(self) -> None:
        self._root: dict[str, Any] | None = None

    @staticmethod
    def distance(left: int, right: int) -> int:
        return (left ^ right).bit_count()

    def add(self, value: int, owner: FingerprintOwner) -> None:
        if self._root is None:
            self._root = {"value": value, "owners": [owner], "children": {}}
            return
        node = self._root
        while True:
            distance = self.distance(value, int(node["value"]))
            if distance == 0:
                node["owners"].append(owner)
                return
            child = node["children"].get(distance)
            if child is None:
                node["children"][distance] = {
                    "value": value,
                    "owners": [owner],
                    "children": {},
                }
                return
            node = child

    def query(self, value: int, radius: int) -> list[FingerprintOwner]:
        return [owner for _, owner, _ in self.query_matches(value, radius)]

    def query_matches(
        self, value: int, radius: int
    ) -> list[tuple[int, FingerprintOwner, int]]:
        """Return every matching hash, owner and exact distance deterministically."""

        if radius < 0:
            raise ValueError("Hamming search radius must be non-negative")
        if self._root is None:
            return []
        matches: list[tuple[int, FingerprintOwner, int]] = []
        stack = [self._root]
        while stack:
            node = stack.pop()
            node_value = int(node["value"])
            distance = self.distance(value, node_value)
            if distance <= radius:
                matches.extend(
                    (node_value, owner, distance) for owner in node["owners"]
                )
            lower, upper = distance - radius, distance + radius
            stack.extend(
                child
                for edge, child in node["children"].items()
                if lower <= int(edge) <= upper
            )
        return sorted(
            matches,
            key=lambda item: (
                item[2],
                item[0],
                item[1].identity,
                item[1].role,
                item[1].label,
            ),
        )


class FingerprintIndex:
    """Exact byte/pixel and near-duplicate ownership index."""

    def __init__(self) -> None:
        self.byte_sha256: dict[str, FingerprintOwner] = {}
        self.pixel_sha256: dict[str, FingerprintOwner] = {}
        self.perceptual = HammingBKTree()
        self.count = 0

    def add(self, verification: ImageVerification, owner: FingerprintOwner) -> None:
        byte_owner = self.byte_sha256.get(verification.byte_sha256)
        pixel_owner = self.pixel_sha256.get(verification.decoded_pixel_sha256)
        for existing in (byte_owner, pixel_owner):
            if existing is not None and existing.label != owner.label:
                raise ValueError(
                    "Conflicting labels share exact image content: "
                    f"{existing.identity} and {owner.identity}"
                )
        self.byte_sha256.setdefault(verification.byte_sha256, owner)
        self.pixel_sha256.setdefault(verification.decoded_pixel_sha256, owner)
        self.perceptual.add(int(verification.perceptual_hash, 16), owner)
        self.count += 1

    def conflicts(
        self,
        verification: ImageVerification,
        *,
        role: str,
        label: int,
        all_near_duplicates: bool,
    ) -> list[dict[str, Any]]:
        conflicts: list[dict[str, Any]] = []
        byte_owner = self.byte_sha256.get(verification.byte_sha256)
        if byte_owner is not None:
            conflicts.append(
                {
                    "kind": "byte_duplicate",
                    "owner": byte_owner.identity,
                    "owner_role": byte_owner.role,
                    "owner_label": byte_owner.label,
                    "candidate_label": label,
                    "candidate_byte_sha256": verification.byte_sha256,
                    "candidate_decoded_pixel_sha256": verification.decoded_pixel_sha256,
                    "candidate_perceptual_hash": verification.perceptual_hash,
                }
            )
        pixel_owner = self.pixel_sha256.get(verification.decoded_pixel_sha256)
        if pixel_owner is not None:
            conflicts.append(
                {
                    "kind": "decoded_pixel_duplicate",
                    "owner": pixel_owner.identity,
                    "owner_role": pixel_owner.role,
                    "owner_label": pixel_owner.label,
                    "candidate_label": label,
                    "candidate_byte_sha256": verification.byte_sha256,
                    "candidate_decoded_pixel_sha256": verification.decoded_pixel_sha256,
                    "candidate_perceptual_hash": verification.perceptual_hash,
                }
            )
        near = self.perceptual.query_matches(
            int(verification.perceptual_hash, 16), NEAR_DUPLICATE_MAX_HAMMING
        )
        for owner_hash, owner, distance in near:
            if all_near_duplicates or owner.role != role or owner.label != label:
                conflicts.append(
                    {
                        "kind": "near_duplicate",
                        "owner": owner.identity,
                        "owner_role": owner.role,
                        "owner_label": owner.label,
                        "candidate_label": label,
                        "candidate_byte_sha256": verification.byte_sha256,
                        "candidate_decoded_pixel_sha256": verification.decoded_pixel_sha256,
                        "candidate_perceptual_hash": verification.perceptual_hash,
                        "owner_perceptual_hash": f"{owner_hash:016x}",
                        "hamming_distance": distance,
                    }
                )
        return conflicts


def classify_resume_conflicts(
    reasons: list[dict[str, Any]], candidate_labels_by_row: dict[int, int]
) -> str:
    """Apply the narrow v2 amendment with fatal conditions taking precedence."""

    if not reasons:
        return "accept"
    replaceable = False
    for reason in reasons:
        kind = reason.get("kind")
        if kind == "image_integrity_failure":
            replaceable = True
            continue
        if kind == "pair_label_conflict":
            return "hard_abort_without_replacement"
        row_idx = reason.get("row_idx")
        if row_idx not in candidate_labels_by_row:
            raise ValueError("E6 conflict references an unknown candidate row")
        candidate_label = candidate_labels_by_row[int(row_idx)]
        if reason.get("candidate_label") != candidate_label:
            raise ValueError("E6 conflict candidate label is inconsistent")
        if reason.get("owner_label") != candidate_label:
            return "hard_abort_without_replacement"
        scope = reason.get("scope")
        if scope == "existing_data":
            if kind in {"byte_duplicate", "decoded_pixel_duplicate"}:
                return "hard_abort_without_replacement"
            if kind != "near_duplicate":
                raise ValueError("E6 historical conflict has an unknown kind")
            replaceable = True
        elif scope == "accepted_e6":
            if kind not in {
                "byte_duplicate",
                "decoded_pixel_duplicate",
                "near_duplicate",
            }:
                raise ValueError("E6 within-run conflict has an unknown kind")
            replaceable = True
        else:
            raise ValueError("E6 conflict has an unknown scope")
    return "reject_whole_pair_and_replace" if replaceable else "accept"


def _load_manifest_image_paths(
    manifest_paths: Iterable[str], project_root: Path
) -> list[tuple[Path, int, str]]:
    observed: dict[str, tuple[Path, int, str]] = {}
    for relative_manifest in manifest_paths:
        manifest = project_root / relative_manifest
        _reject_symlink_path(manifest, project_root)
        if not manifest.is_file():
            raise ValueError(f"Required E6 exclusion manifest is missing: {relative_manifest}")
        with manifest.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not {"image_path", "label"}.issubset(reader.fieldnames or []):
                raise ValueError(f"E6 exclusion manifest is malformed: {relative_manifest}")
            for row in reader:
                image_path = project_root / row["image_path"]
                _reject_symlink_path(image_path, project_root)
                if not image_path.is_file():
                    raise ValueError(
                        f"E6 exclusion image is missing: {row['image_path']}"
                    )
                label = _strict_int(row["label"], "exclusion label")
                if label not in (0, 1):
                    raise ValueError("E6 exclusion manifest has a non-binary label")
                relative = _project_relative(image_path, project_root)
                existing = observed.get(relative)
                if existing is not None and existing[1] != label:
                    raise ValueError(f"Existing image has conflicting labels: {relative}")
                observed[relative] = (image_path, label, relative_manifest)
    return [observed[key] for key in sorted(observed)]


def validate_exclusion_manifest_hashes(
    protocol: dict[str, Any],
    development_lock: dict[str, Any],
    project_root: Path,
) -> None:
    """Bind every historical overlap manifest to the frozen registry hash."""

    registry_relative = development_lock["registry"]["path"]
    registry_path = project_root / registry_relative
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registered: dict[str, str] = {}
    for unit in registry.get("dataset_units", []):
        artifact = unit.get("artifact") if isinstance(unit, dict) else None
        if isinstance(artifact, dict):
            path = artifact.get("path")
            digest = artifact.get("sha256")
            if isinstance(path, str) and isinstance(digest, str):
                registered[path] = digest
    for relative in protocol["integrity"]["existing_manifests_to_exclude"]:
        expected = registered.get(relative)
        if expected is None:
            raise ValueError(
                f"E6 exclusion manifest is absent from frozen registry: {relative}"
            )
        path = project_root / relative
        if not path.is_file() or path.is_symlink() or sha256_file(path) != expected:
            raise ValueError(f"E6 exclusion manifest changed after registry lock: {relative}")


def load_existing_exclusions(
    protocol: dict[str, Any], project_root: Path
) -> tuple[FingerprintIndex, dict[str, Any]]:
    """Fingerprint only the explicit historical manifests, never a lockbox."""

    manifest_paths = protocol["integrity"]["existing_manifests_to_exclude"]
    images = _load_manifest_image_paths(manifest_paths, project_root)
    index = FingerprintIndex()
    for path, label, manifest in tqdm(
        images, desc="E6 existing-data fingerprints", unit="image"
    ):
        verification = inspect_image(path)
        index.add(
            verification,
            FingerprintOwner(
                identity=_project_relative(path, project_root),
                role=f"existing:{manifest}",
                label=label,
            ),
        )
    return index, {
        "manifests": list(manifest_paths),
        "unique_image_paths": len(images),
        "fingerprints_indexed": index.count,
        "future_lockbox_opened": False,
    }


def pair_conflicts(
    prepared: tuple[PreparedImage, PreparedImage],
    *,
    role: str,
    existing: FingerprintIndex,
    accepted: FingerprintIndex,
) -> list[dict[str, Any]]:
    """Return every reason a candidate pair cannot enter the frozen role."""

    real, ai = prepared
    reasons: list[dict[str, Any]] = []
    for item in prepared:
        reasons.extend(
            {
                **reason,
                "row_idx": item.row.row_idx,
                "scope": "existing_data",
            }
            for reason in existing.conflicts(
                item.verification,
                role=role,
                label=item.row.label,
                all_near_duplicates=True,
            )
        )
        reasons.extend(
            {
                **reason,
                "row_idx": item.row.row_idx,
                "scope": "accepted_e6",
            }
            for reason in accepted.conflicts(
                item.verification,
                role=role,
                label=item.row.label,
                all_near_duplicates=True,
            )
        )
    if real.verification.byte_sha256 == ai.verification.byte_sha256:
        raise ValueError("Candidate real/BigGAN pair has byte-identical conflicting labels")
    if (
        real.verification.decoded_pixel_sha256
        == ai.verification.decoded_pixel_sha256
    ):
        raise ValueError(
            "Candidate real/BigGAN pair has pixel-identical conflicting labels"
        )
    distance = HammingBKTree.distance(
        int(real.verification.perceptual_hash, 16),
        int(ai.verification.perceptual_hash, 16),
    )
    if distance <= NEAR_DUPLICATE_MAX_HAMMING:
        raise ValueError(
            "Candidate real/BigGAN pair has near-identical conflicting labels "
            f"(pHash distance {distance})"
        )
    return reasons


def _atomic_csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"Refusing to overwrite symlink manifest: {path}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _clear_publication_staging(staging_root: Path, project_root: Path) -> None:
    """Remove only the private E6 publication staging tree, never a symlink."""

    _reject_symlink_path(staging_root, project_root)
    if not staging_root.exists():
        return
    if not staging_root.is_dir():
        raise ValueError("E6 publication staging path is not a directory")
    for directory, directory_names, file_names in os.walk(
        staging_root, topdown=False, followlinks=False
    ):
        parent = Path(directory)
        for name in [*file_names, *directory_names]:
            candidate = parent / name
            if candidate.is_symlink():
                raise ValueError("E6 publication staging contains a symlink")
            if candidate.is_dir():
                candidate.rmdir()
            elif candidate.is_file():
                candidate.unlink()
            else:
                raise ValueError("E6 publication staging contains an unsafe entry")
    staging_root.rmdir()
    _fsync_directory(staging_root.parent)


def _validate_publication_bundle(
    provenance_file: Path,
    manifest_directory: Path,
    *,
    outputs: dict[str, Any],
    protocol_path: Path,
    lock_report_path: Path,
    acquisition_protocol_path: Path,
    acquisition_lock_path: Path,
    project_root: Path,
    validate_raw_assets: bool = True,
    resume_protocol_path: Path | None = None,
    resume_lock_path: Path | None = None,
    rejection_journal_path: Path | None = None,
) -> dict[str, Any]:
    """Validate a fully staged or already-published metadata bundle."""

    for path in (provenance_file, manifest_directory):
        _reject_symlink_path(path, project_root)
    if provenance_file.is_symlink() or not provenance_file.is_file():
        raise ValueError("E6 publication provenance is missing or unsafe")
    if manifest_directory.is_symlink() or not manifest_directory.is_dir():
        raise ValueError("E6 publication manifest directory is missing or unsafe")
    payload = json.loads(provenance_file.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS":
        raise ValueError("E6 publication provenance is not complete")
    if payload.get("protocol", {}).get("sha256") != sha256_file(protocol_path):
        raise ValueError("E6 publication protocol hash changed")
    if payload.get("lock_receipt", {}).get("sha256") != sha256_file(
        lock_report_path
    ):
        raise ValueError("E6 publication development-lock hash changed")
    acquisition_record = payload.get("acquisition_protocol", {})
    if acquisition_record.get("sha256") != sha256_file(acquisition_protocol_path):
        raise ValueError("E6 publication acquisition protocol hash changed")
    if acquisition_record.get("lock_sha256") != sha256_file(acquisition_lock_path):
        raise ValueError("E6 publication acquisition-lock hash changed")
    resume_paths = (
        resume_protocol_path,
        resume_lock_path,
        rejection_journal_path,
    )
    if any(path is not None for path in resume_paths):
        if not all(path is not None for path in resume_paths):
            raise ValueError("E6 publication resume validation paths are incomplete")
        assert resume_protocol_path is not None
        assert resume_lock_path is not None
        assert rejection_journal_path is not None
        _reject_symlink_path(rejection_journal_path, project_root)
        if (
            rejection_journal_path.is_symlink()
            or not rejection_journal_path.is_file()
        ):
            raise ValueError("E6 publication rejection journal changed")
        resume_protocol = load_resume_protocol(resume_protocol_path)
        resume_receipt, resume_inventory = validate_resume_lock_receipt(
            resume_protocol, resume_lock_path, project_root
        )
        journal_payload = json.loads(
            rejection_journal_path.read_text(encoding="utf-8")
        )
        validate_rejection_journal(
            journal_payload,
            resume_protocol,
            project_root,
            protocol_path=resume_protocol_path,
        )
        expected_resume_record = {
            "protocol_path": _project_relative(resume_protocol_path, project_root),
            "protocol_sha256": sha256_file(resume_protocol_path),
            "lock_path": _project_relative(resume_lock_path, project_root),
            "lock_sha256": sha256_file(resume_lock_path),
            "incident": resume_protocol["upstream_evidence"]["overlap_incident"],
            "frozen_prefix_inventory": resume_inventory,
            "journal": {
                "path": _project_relative(rejection_journal_path, project_root),
                "sha256": sha256_file(rejection_journal_path),
                "event_count": len(journal_payload["events"]),
                "seed_event_sha256": resume_receipt[
                    "initial_rejection_journal"
                ]["head_event_sha256"],
                "head_event_sha256": journal_payload["events"][-1][
                    "event_sha256"
                ],
            },
            "consumed_test_identity_metadata_used_for_integrity_filter": True,
            "consumed_test_metrics_or_visual_content_used_for_replacement": False,
        }
        if payload.get("acquisition_resume") != expected_resume_record:
            raise ValueError("E6 publication resume provenance changed")
    transport_record = payload.get("transport_preflight_audit", {})
    if transport_record.get("path") != DEFAULT_TRANSPORT_PREFLIGHT_AUDIT.as_posix():
        raise ValueError("E6 publication transport-preflight path changed")
    transport_audit_path = project_root / DEFAULT_TRANSPORT_PREFLIGHT_AUDIT
    _reject_symlink_path(transport_audit_path, project_root)
    if transport_audit_path.is_symlink() or not transport_audit_path.is_file():
        raise ValueError("E6 publication transport-preflight audit is missing")
    if transport_record.get("sha256") != sha256_file(transport_audit_path):
        raise ValueError("E6 publication transport-preflight audit changed")
    if payload.get("provenance", {}).get("path") != outputs["provenance"]:
        raise ValueError("E6 publication provenance path changed")

    summaries = payload.get("manifests")
    expected_roles = outputs["role_manifests"]
    if not isinstance(summaries, dict) or set(summaries) != set(expected_roles):
        raise ValueError("E6 publication manifest inventory changed")
    expected_names = {Path(path).name for path in expected_roles.values()}
    directory_entries = list(manifest_directory.iterdir())
    observed_names = {item.name for item in directory_entries}
    if observed_names != expected_names:
        raise ValueError("E6 publication manifest files are incomplete")
    if any(item.is_symlink() or not item.is_file() for item in directory_entries):
        raise ValueError("E6 publication manifest directory contains an unsafe entry")
    image_records = payload.get("images")
    records_by_path: dict[str, dict[str, Any]] = {}
    if validate_raw_assets:
        if not isinstance(image_records, list):
            raise ValueError("E6 publication image inventory is missing")
        for record in image_records:
            if not isinstance(record, dict) or not isinstance(
                record.get("image_path"), str
            ):
                raise ValueError("E6 publication image inventory is invalid")
            image_path = record["image_path"]
            if image_path in records_by_path:
                raise ValueError("E6 publication image inventory repeats a path")
            records_by_path[image_path] = record
    manifest_image_paths: set[str] = set()
    for role, expected_path in expected_roles.items():
        summary = summaries[role]
        if summary.get("path") != expected_path:
            raise ValueError(f"E6 publication path changed for role {role}")
        manifest_file = manifest_directory / Path(expected_path).name
        _reject_symlink_path(manifest_file, project_root)
        if manifest_file.is_symlink() or not manifest_file.is_file():
            raise ValueError(f"E6 publication manifest is unsafe for role {role}")
        if summary.get("sha256") != sha256_file(manifest_file):
            raise ValueError(f"E6 publication manifest hash changed for role {role}")
        with manifest_file.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
                raise ValueError(f"E6 publication columns changed for role {role}")
            rows = list(reader)
        if len(rows) != int(summary.get("rows", -1)):
            raise ValueError(f"E6 publication row count changed for role {role}")
        counts = Counter(int(row["label"]) for row in rows)
        if counts != Counter(
            {
                0: int(summary["class_counts"]["real_0"]),
                1: int(summary["class_counts"]["ai_generated_1"]),
            }
        ):
            raise ValueError(f"E6 publication class counts changed for role {role}")
        if validate_raw_assets:
            for row in rows:
                relative_image_path = row["image_path"]
                if relative_image_path in manifest_image_paths:
                    raise ValueError("E6 publication repeats a raw image path")
                manifest_image_paths.add(relative_image_path)
                record = records_by_path.get(relative_image_path)
                if record is None:
                    raise ValueError("E6 publication raw image lacks provenance")
                image_path = project_root / relative_image_path
                _reject_symlink_path(image_path, project_root)
                if image_path.is_symlink() or not image_path.is_file():
                    raise ValueError("E6 publication raw image is missing or unsafe")
                verification = inspect_image(image_path)
                expected_verification = {
                    "file_bytes": int(row["file_bytes"]),
                    "byte_sha256": row["byte_sha256"],
                    "decoded_pixel_sha256": row["decoded_pixel_sha256"],
                    "perceptual_hash": row["perceptual_hash"],
                    "original_width": int(row["original_width"]),
                    "original_height": int(row["original_height"]),
                    "decoded_pixel_count": int(row["decoded_pixel_count"]),
                    "file_format": row["file_format"],
                }
                observed_verification = asdict(verification)
                if any(
                    observed_verification[key] != value
                    for key, value in expected_verification.items()
                ) or not math.isclose(
                    verification.aspect_ratio,
                    float(row["aspect_ratio"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError("E6 publication raw image changed after validation")
                if any(
                    record.get(key) != value
                    for key, value in observed_verification.items()
                ):
                    raise ValueError("E6 publication raw-image provenance changed")
                if (
                    int(row["source_row_idx"]) != record.get("row_idx")
                    or int(row["label"]) != record.get("project_label")
                    or role != record.get("role")
                ):
                    raise ValueError("E6 publication row provenance changed")
                receipt_path = _cache_receipt_path(image_path)
                pending_receipt_path = _cache_pending_receipt_path(image_path)
                _reject_symlink_path(receipt_path, project_root)
                _reject_symlink_path(pending_receipt_path, project_root)
                if pending_receipt_path.exists():
                    raise ValueError("E6 publication has an unfinished cache receipt")
                if receipt_path.is_symlink() or not receipt_path.is_file():
                    raise ValueError("E6 publication cache receipt is missing or unsafe")
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                canonical_path = receipt.get("canonical_asset_path")
                if not isinstance(canonical_path, str):
                    raise ValueError("E6 publication cache receipt lost source identity")
                source_row = SourceRow(
                    row_idx=int(record["row_idx"]),
                    label=int(record["source_label"]),
                    generator_id=int(record["generator_id"]),
                    generator_name=str(record["generator_name"]),
                    image_url=f"https://{ROWS_HOST}{canonical_path}",
                    source_width=int(record["source_width"]),
                    source_height=int(record["source_height"]),
                )
                _validate_asset_url(
                    source_row.image_url,
                    repository=str(payload["source"]["repository"]),
                    revision=str(payload["source"]["repository_revision"]),
                    split=str(payload["source"]["split"]),
                    row_idx=source_row.row_idx,
                )
                _validate_extension_magic(
                    source_row.image_url,
                    verification.file_format,
                    source_row.row_idx,
                )
                _validate_cache_receipt(receipt_path, source_row, verification)
    if validate_raw_assets and manifest_image_paths != set(records_by_path):
        raise ValueError("E6 publication image inventory does not match its manifests")
    serialized = json.dumps(payload, sort_keys=True)
    if any(secret in serialized for secret in ("Signature=", "Expires=", "X-Amz-")):
        raise ValueError("E6 publication persisted a signed URL query")
    return payload


def _recover_or_reuse_publication(
    *,
    manifest_root: Path,
    provenance_path: Path,
    staging_root: Path,
    outputs: dict[str, Any],
    protocol_path: Path,
    lock_report_path: Path,
    acquisition_protocol_path: Path,
    acquisition_lock_path: Path,
    project_root: Path,
    resume_protocol_path: Path | None = None,
    resume_lock_path: Path | None = None,
    rejection_journal_path: Path | None = None,
) -> dict[str, Any] | None:
    """Finish a validated interrupted publication, or validate a completed rerun."""

    manifests_exist = manifest_root.exists()
    provenance_exists = provenance_path.exists()
    staged_provenance = staging_root / "provenance.json"
    if manifests_exist and provenance_exists:
        payload = _validate_publication_bundle(
            provenance_path,
            manifest_root,
            outputs=outputs,
            protocol_path=protocol_path,
            lock_report_path=lock_report_path,
            acquisition_protocol_path=acquisition_protocol_path,
            acquisition_lock_path=acquisition_lock_path,
            project_root=project_root,
            resume_protocol_path=resume_protocol_path,
            resume_lock_path=resume_lock_path,
            rejection_journal_path=rejection_journal_path,
        )
        _clear_publication_staging(staging_root, project_root)
        return payload
    if provenance_exists and not manifests_exist:
        raise ValueError(
            "E6 provenance exists without its manifest directory; refusing recovery"
        )
    if manifests_exist:
        if not staged_provenance.is_file():
            raise ValueError(
                "E6 manifests are incomplete and no validated recovery journal exists"
            )
        payload = _validate_publication_bundle(
            staged_provenance,
            manifest_root,
            outputs=outputs,
            protocol_path=protocol_path,
            lock_report_path=lock_report_path,
            acquisition_protocol_path=acquisition_protocol_path,
            acquisition_lock_path=acquisition_lock_path,
            project_root=project_root,
            resume_protocol_path=resume_protocol_path,
            resume_lock_path=resume_lock_path,
            rejection_journal_path=rejection_journal_path,
        )
        provenance_path.parent.mkdir(parents=True, exist_ok=True)
        _reject_symlink_path(provenance_path.parent, project_root)
        staged_provenance.replace(provenance_path)
        _fsync_directory(provenance_path.parent)
        _clear_publication_staging(staging_root, project_root)
        return payload
    return None


def _acquire_preparation_lock(raw_root: Path, project_root: Path):
    """Hold a non-blocking process lock for one E6 preparation run."""

    raw_root.mkdir(parents=True, exist_ok=True)
    _reject_symlink_path(raw_root, project_root)
    lock_path = raw_root / ".prepare.lock"
    _reject_symlink_path(lock_path, project_root)
    flags = os.O_RDWR | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    handle = os.fdopen(descriptor, "a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise ValueError("Another E6 BigGAN preparation process is running") from exc
    return handle


def _clear_stale_partial_downloads(raw_root: Path, project_root: Path) -> int:
    """Remove only interrupted downloader temp files while holding the run lock."""

    images_root = raw_root / "images"
    _reject_symlink_path(images_root, project_root)
    if not images_root.exists():
        return 0
    if images_root.is_symlink() or not images_root.is_dir():
        raise ValueError("E6 image cache directory is unsafe")
    pattern = re.compile(
        r"\.row-[0-9]{5}\.(?:jpe?g|png|webp)\.[A-Za-z0-9_-]+\.part"
    )
    removed = 0
    for entry in images_root.iterdir():
        if not entry.name.endswith(".part"):
            continue
        if pattern.fullmatch(entry.name) is None:
            raise ValueError("E6 image cache contains an unexpected partial file")
        if entry.is_symlink() or not entry.is_file():
            raise ValueError("E6 image cache contains an unsafe partial file")
        entry.unlink()
        removed += 1
    if removed:
        _fsync_directory(images_root)
    return removed


def _prepared_record(
    row: SourceRow,
    verification: ImageVerification,
    image_path: Path,
    project_root: Path,
) -> PreparedImage:
    return PreparedImage(
        row=row,
        image_path=_project_relative(image_path, project_root),
        verification=verification,
    )


def _public_image_record(
    item: PreparedImage,
    *,
    role: str,
    pair_key: str,
    assigned_cycle: int,
    accepted_cycle: int,
    replacement: bool,
) -> dict[str, Any]:
    return {
        "row_idx": item.row.row_idx,
        "source_split": "train",
        "source_label": item.row.label,
        "project_label": item.row.label,
        "generator_id": item.row.generator_id,
        "generator_name": item.row.generator_name,
        "source_width": item.row.source_width,
        "source_height": item.row.source_height,
        "role": role,
        "pair_key": pair_key,
        "assigned_cycle": assigned_cycle,
        "accepted_cycle": accepted_cycle,
        "replacement": replacement,
        "image_path": item.image_path,
        **asdict(item.verification),
    }


def _manifest_row(
    item: PreparedImage,
    role: str,
    *,
    pair_key: str,
    assigned_cycle: int,
    accepted_cycle: int,
) -> dict[str, Any]:
    verification = item.verification
    return {
        "image_path": item.image_path,
        "label": item.row.label,
        "class_name": "real" if item.row.label == 0 else "ai_generated",
        "source": SOURCE_NAME,
        "split": role,
        "source_row_idx": item.row.row_idx,
        "pair_key": pair_key,
        "assigned_cycle": assigned_cycle,
        "accepted_cycle": accepted_cycle,
        "byte_sha256": verification.byte_sha256,
        "decoded_pixel_sha256": verification.decoded_pixel_sha256,
        "original_width": verification.original_width,
        "original_height": verification.original_height,
        "aspect_ratio": verification.aspect_ratio,
        "decoded_pixel_count": verification.decoded_pixel_count,
        "file_format": verification.file_format,
        "file_bytes": verification.file_bytes,
        "perceptual_hash": verification.perceptual_hash,
    }


def _prepare_e6_biggan_locked(
    protocol_path: Path,
    lock_report_path: Path,
    acquisition_protocol_path: Path,
    acquisition_lock_path: Path,
    project_root: Path,
    resume_protocol_path: Path | None = None,
    resume_lock_path: Path | None = None,
) -> dict[str, Any]:
    """Acquire, deduplicate and publish all frozen E6 BigGAN role manifests."""

    acquisition = load_e6_acquisition_protocol(acquisition_protocol_path)
    acquisition_receipt, locked_development = validate_acquisition_lock_receipt(
        acquisition_protocol_path,
        acquisition_lock_path,
        acquisition,
        project_root,
    )
    protocol = load_e6_protocol(protocol_path)
    validate_e6_development_protocol(protocol)
    if protocol != locked_development:
        raise ValueError("E6 development protocol differs from acquisition lock")
    lock_receipt = validate_lock_receipt(
        protocol_path, lock_report_path, protocol, project_root
    )
    transport_preflight_audit_path = project_root / DEFAULT_TRANSPORT_PREFLIGHT_AUDIT
    transport_preflight_audit = validate_transport_preflight_audit(
        transport_preflight_audit_path,
        acquisition_protocol_path,
        acquisition_lock_path,
    )
    validate_exclusion_manifest_hashes(protocol, lock_receipt, project_root)
    outputs = acquisition["outputs"]
    raw_root = project_root / outputs["raw_root"]
    manifest_root = project_root / outputs["manifest_root"]
    provenance_path = project_root / outputs["provenance"]
    staging_root = project_root / outputs["staging_root"]
    for path in (raw_root, manifest_root, provenance_path, staging_root):
        _reject_symlink_path(path, project_root)
    resume_protocol_path = resume_protocol_path or project_root / DEFAULT_RESUME_PROTOCOL
    resume_lock_path = resume_lock_path or project_root / DEFAULT_RESUME_LOCK
    resume_protocol = load_resume_protocol(resume_protocol_path)
    resume_receipt, frozen_prefix_inventory = validate_resume_lock_receipt(
        resume_protocol, resume_lock_path, project_root
    )
    journal_path = project_root / resume_protocol["rejection_journal"]["path"]
    _reject_symlink_path(journal_path, project_root)
    if journal_path.is_symlink() or not journal_path.is_file():
        raise ValueError(
            "E6 seeded rejection journal is missing; rerun the offline resume checker"
        )
    rejection_journal = json.loads(journal_path.read_text(encoding="utf-8"))
    validate_rejection_journal(
        rejection_journal,
        resume_protocol,
        project_root,
        protocol_path=resume_protocol_path,
    )
    initial_journal = resume_receipt["initial_rejection_journal"]
    if (
        rejection_journal["events"][0]["event_sha256"]
        != initial_journal["head_event_sha256"]
    ):
        raise ValueError("E6 rejection journal no longer begins with its frozen seed")
    stale_partials_removed = _clear_stale_partial_downloads(raw_root, project_root)
    if stale_partials_removed:
        print(
            "Recovered interrupted E6 downloads: "
            f"removed_stale_partials={stale_partials_removed}"
        )
    recovered = _recover_or_reuse_publication(
        manifest_root=manifest_root,
        provenance_path=provenance_path,
        staging_root=staging_root,
        outputs=outputs,
        protocol_path=protocol_path,
        lock_report_path=lock_report_path,
        acquisition_protocol_path=acquisition_protocol_path,
        acquisition_lock_path=acquisition_lock_path,
        project_root=project_root,
        resume_protocol_path=resume_protocol_path,
        resume_lock_path=resume_lock_path,
        rejection_journal_path=journal_path,
    )
    if recovered is not None:
        return recovered
    _clear_publication_staging(staging_root, project_root)

    assignments = build_pair_assignment(protocol)
    ranked = ranked_cycles(protocol)
    preflight_session = _session()
    try:
        first_cycle = int(assignments[0]["pair_index"])
        preflight_rows = RowsClient(preflight_session, protocol).pair(first_cycle)
    finally:
        preflight_session.close()
    print(
        "PASS Tiny-GenImage metadata preflight: "
        f"rows={preflight_rows[0].row_idx},{preflight_rows[1].row_idx}; "
        "no image payload opened"
    )

    existing_index, exclusion_summary = load_existing_exclusions(protocol, project_root)
    accepted_index = FingerprintIndex()
    reserve_cycles = list(ranked[len(assignments) :])
    reserve_position = 0
    journal_cursor = 0
    manifests: dict[str, list[dict[str, Any]]] = {
        role: [] for role in protocol["selection"]["roles_in_assignment_order"]
    }
    image_records: list[dict[str, Any]] = []
    pair_records: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    download_budget = DownloadBudget()

    session = _session()
    rows_client = RowsClient(session, protocol)
    progress = tqdm(total=2 * len(assignments), desc="E6 BigGAN pairs", unit="image")
    try:
        for slot, assignment in enumerate(assignments):
            assigned_cycle = int(assignment["pair_index"])
            attempt_cycle = assigned_cycle
            attempted: list[int] = []
            while True:
                attempted.append(attempt_cycle)
                next_journal_event = (
                    rejection_journal["events"][journal_cursor]
                    if journal_cursor < len(rejection_journal["events"])
                    else None
                )
                if next_journal_event is not None and (
                    next_journal_event["slot"],
                    next_journal_event["attempted_cycle"],
                ) == (slot, attempt_cycle):
                    if next_journal_event["attempted_cycles"] != attempted:
                        raise ValueError(
                            "E6 journal replay attempt history disagrees with runtime"
                        )
                    _discard_cached_row_indices(
                        raw_root,
                        (
                            int(next_journal_event["real_row_index"]),
                            int(next_journal_event["ai_row_index"]),
                        ),
                        project_root=project_root,
                    )
                    rejections.append(
                        {
                            "slot": slot,
                            "role": assignment["role"],
                            "assigned_cycle": assigned_cycle,
                            "rejected_cycle": attempt_cycle,
                            "action": next_journal_event["action"],
                            "reasons": next_journal_event["reasons"],
                            "journal_event_sha256": next_journal_event[
                                "event_sha256"
                            ],
                            "replayed": True,
                        }
                    )
                    journal_cursor += 1
                    if next_journal_event["action"] == (
                        "hard_abort_without_replacement"
                    ):
                        raise ValueError(
                            "E6 replayed a journaled fatal acquisition conflict"
                        )
                    if reserve_position >= len(reserve_cycles):
                        raise ValueError("E6 exhausted the frozen reserve")
                    expected_reserve = reserve_cycles[reserve_position]
                    if next_journal_event.get("next_reserve_cycle") != expected_reserve:
                        raise ValueError("E6 journal replay changed reserve order")
                    reserve_position += 1
                    attempt_cycle = expected_reserve
                    continue
                if next_journal_event is not None and int(
                    next_journal_event["slot"]
                ) <= slot:
                    raise ValueError(
                        "E6 rejection journal is out of order with runtime acquisition"
                    )
                real_row, ai_row = rows_client.pair(attempt_cycle)
                prepared_items: list[PreparedImage] = []
                integrity_failure: str | None = None
                try:
                    for source_row in (real_row, ai_row):
                        path = _cached_asset_path(raw_root, source_row)
                        verification = download_source_image(
                            session,
                            source_row,
                            path,
                            project_root=project_root,
                            refresh_row=rows_client.refresh_row,
                            budget=download_budget,
                        )
                        prepared_items.append(
                            _prepared_record(
                                source_row, verification, path, project_root
                            )
                        )
                except ImageIntegrityError as exc:
                    integrity_failure = str(exc)

                if integrity_failure is not None:
                    conflicts = [
                        {"kind": "image_integrity_failure", "detail": integrity_failure}
                    ]
                else:
                    prepared = (prepared_items[0], prepared_items[1])
                    try:
                        conflicts = pair_conflicts(
                            prepared,
                            role=str(assignment["role"]),
                            existing=existing_index,
                            accepted=accepted_index,
                        )
                    except ValueError as exc:
                        conflicts = [
                            {
                                "kind": "pair_label_conflict",
                                "detail": str(exc),
                            }
                        ]
                if conflicts:
                    labels_by_row = {
                        real_row.row_idx: real_row.label,
                        ai_row.row_idx: ai_row.label,
                    }
                    action = classify_resume_conflicts(conflicts, labels_by_row)
                    if journal_cursor != len(rejection_journal["events"]):
                        raise ValueError(
                            "E6 runtime conflict appeared before a frozen future journal event"
                        )
                    next_reserve_cycle: int | None = None
                    if action == "reject_whole_pair_and_replace":
                        if reserve_position >= len(reserve_cycles):
                            action = "hard_abort_without_replacement"
                            conflicts = [
                                *conflicts,
                                {
                                    "kind": "pair_label_conflict",
                                    "detail": "The frozen reserve was exhausted",
                                },
                            ]
                        else:
                            next_reserve_cycle = reserve_cycles[reserve_position]
                    event_fields: dict[str, Any] = {
                        "slot": slot,
                        "role": assignment["role"],
                        "assigned_cycle": assigned_cycle,
                        "attempted_cycle": attempt_cycle,
                        "attempted_cycles": list(attempted),
                        "real_row_index": real_row.row_idx,
                        "ai_row_index": ai_row.row_idx,
                        "action": action,
                        "reasons": conflicts,
                    }
                    if next_reserve_cycle is not None:
                        event_fields["next_reserve_cycle"] = next_reserve_cycle
                    rejection_journal = append_rejection_event(
                        journal_path,
                        rejection_journal,
                        event_fields,
                        validate_before_write=lambda candidate: (
                            validate_rejection_journal(
                                candidate,
                                resume_protocol,
                                project_root,
                                protocol_path=resume_protocol_path,
                            )
                        ),
                    )
                    journal_event = rejection_journal["events"][-1]
                    journal_cursor += 1
                    _discard_cached_rows(
                        raw_root,
                        (real_row, ai_row),
                        project_root=project_root,
                    )
                    rejections.append(
                        {
                            "slot": slot,
                            "role": assignment["role"],
                            "assigned_cycle": assigned_cycle,
                            "rejected_cycle": attempt_cycle,
                            "action": action,
                            "reasons": conflicts,
                            "journal_event_sha256": journal_event[
                                "event_sha256"
                            ],
                            "replayed": False,
                        }
                    )
                    if action == "hard_abort_without_replacement":
                        raise ValueError(
                            "E6 candidate hit a fatal overlap or label conflict; "
                            "the rejection was journaled before abort"
                        )
                    reserve_position += 1
                    attempt_cycle = int(next_reserve_cycle)
                    continue

                real, ai = prepared
                accepted_cycle = attempt_cycle
                pair_key = f"tiny_genimage_train_biggan_cycle_{accepted_cycle}"
                replacement = accepted_cycle != assigned_cycle
                for item in (real, ai):
                    owner = FingerprintOwner(
                        identity=f"tiny_genimage_row_{item.row.row_idx}",
                        role=str(assignment["role"]),
                        label=item.row.label,
                    )
                    accepted_index.add(item.verification, owner)
                    manifests[str(assignment["role"])].append(
                        _manifest_row(
                            item,
                            str(assignment["role"]),
                            pair_key=pair_key,
                            assigned_cycle=assigned_cycle,
                            accepted_cycle=accepted_cycle,
                        )
                    )
                    image_records.append(
                        _public_image_record(
                            item,
                            role=str(assignment["role"]),
                            pair_key=pair_key,
                            assigned_cycle=assigned_cycle,
                            accepted_cycle=accepted_cycle,
                            replacement=replacement,
                        )
                    )
                    progress.update(1)
                pair_records.append(
                    {
                        "slot": slot,
                        "role": assignment["role"],
                        "frozen_assigned_cycle": assigned_cycle,
                        "accepted_cycle": accepted_cycle,
                        "pair_key": pair_key,
                        "replacement": replacement,
                        "attempted_cycles": attempted,
                        "real_row_index": real.row.row_idx,
                        "ai_row_index": ai.row.row_idx,
                    }
                )
                break
        if journal_cursor != len(rejection_journal["events"]):
            raise ValueError("E6 acquisition did not consume every journaled event")
    finally:
        progress.close()
        session.close()

    expected_role_pairs = protocol["selection"]["pairs_per_role"]
    for role, pair_count in expected_role_pairs.items():
        rows = manifests[role]
        counts = Counter(int(row["label"]) for row in rows)
        if len(rows) != 2 * int(pair_count) or counts != Counter(
            {0: int(pair_count), 1: int(pair_count)}
        ):
            raise ValueError(f"E6 role manifest is not pair-balanced: {role}")
        if len({row["image_path"] for row in rows}) != len(rows):
            raise ValueError(f"E6 role manifest repeats an image path: {role}")
    if len({row["accepted_cycle"] for row in pair_records}) != len(pair_records):
        raise ValueError("E6 accepted pair cycles are not unique")

    staging_manifest_root = staging_root / "manifests"
    staging_root.mkdir(parents=True, exist_ok=False)
    staging_manifest_root.mkdir()
    manifest_root.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    for path in (
        staging_root,
        staging_manifest_root,
        manifest_root.parent,
        provenance_path.parent,
    ):
        _reject_symlink_path(path, project_root)
    destination_device = os.stat(manifest_root.parent).st_dev
    if (
        os.stat(staging_root).st_dev != destination_device
        or os.stat(provenance_path.parent).st_dev != destination_device
    ):
        raise ValueError("E6 publication staging is not on the destination filesystem")
    manifest_summary: dict[str, dict[str, Any]] = {}
    for role, rows in manifests.items():
        final_path = project_root / outputs["role_manifests"][role]
        path = staging_manifest_root / final_path.name
        _atomic_csv_write(path, rows)
        manifest_summary[role] = {
            "path": _project_relative(final_path, project_root),
            "sha256": sha256_file(path),
            "rows": len(rows),
            "class_counts": {
                "real_0": sum(int(row["label"]) == 0 for row in rows),
                "ai_generated_1": sum(int(row["label"]) == 1 for row in rows),
            },
        }

    provenance = {
        "experiment": "e6_tiny_genimage_biggan_development_preparation",
        "status": "PASS",
        "protocol": {
            "path": _project_relative(protocol_path, project_root),
            "sha256": sha256_file(protocol_path),
            "assignment_sha256": assignment_sha256(assignments),
        },
        "lock_receipt": {
            "path": _project_relative(lock_report_path, project_root),
            "sha256": sha256_file(lock_report_path),
            "frozen_at_utc": lock_receipt["frozen_at_utc"],
            "image_payload_present_at_freeze": False,
        },
        "acquisition_protocol": {
            "path": _project_relative(acquisition_protocol_path, project_root),
            "sha256": sha256_file(acquisition_protocol_path),
            "lock_path": _project_relative(acquisition_lock_path, project_root),
            "lock_sha256": sha256_file(acquisition_lock_path),
            "frozen_at_utc": acquisition_receipt["frozen_at_utc"],
            "payload_present_at_freeze": False,
        },
        "acquisition_resume": {
            "protocol_path": _project_relative(resume_protocol_path, project_root),
            "protocol_sha256": sha256_file(resume_protocol_path),
            "lock_path": _project_relative(resume_lock_path, project_root),
            "lock_sha256": sha256_file(resume_lock_path),
            "incident": resume_protocol["upstream_evidence"]["overlap_incident"],
            "frozen_prefix_inventory": frozen_prefix_inventory,
            "journal": {
                "path": _project_relative(journal_path, project_root),
                "sha256": sha256_file(journal_path),
                "event_count": len(rejection_journal["events"]),
                "seed_event_sha256": rejection_journal["events"][0][
                    "event_sha256"
                ],
                "head_event_sha256": rejection_journal["events"][-1][
                    "event_sha256"
                ],
            },
            "consumed_test_identity_metadata_used_for_integrity_filter": True,
            "consumed_test_metrics_or_visual_content_used_for_replacement": False,
        },
        "transport_preflight_audit": {
            "path": _project_relative(
                transport_preflight_audit_path, project_root
            ),
            "sha256": sha256_file(transport_preflight_audit_path),
            "status": transport_preflight_audit["status"],
            "response_body_bytes_read": transport_preflight_audit["probe"][
                "response_body_bytes_read"
            ],
            "image_decoded_or_retained": False,
            "git_commit_preceded_probe": False,
        },
        "source": {
            "repository": protocol["development_source"]["dataset_repository"],
            "repository_revision": protocol["development_source"][
                "repository_revision"
            ],
            "config": DATASET_CONFIG,
            "split": protocol["development_source"]["source_split"],
            "generator": protocol["development_source"]["selected_generator"],
            "rows_api": ROWS_API,
            "signed_asset_urls_persisted": False,
            "manual_image_inspection_or_cherry_picking": False,
        },
        "integrity": {
            "byte_hash": "SHA-256 of original downloaded file bytes",
            "decoded_pixel_hash": (
                "e6_rgba_sha256_v1: SHA-256 of E6RGBA1 NUL tag, big-endian "
                "display dimensions, and EXIF-oriented RGBA bytes"
            ),
            "near_duplicate_hash": (
                "e6_phash64_v1: white-alpha-composited 32x32 grayscale DCT, "
                "lowest 8x8 block with DC excluded"
            ),
            "near_duplicate_max_hamming": NEAR_DUPLICATE_MAX_HAMMING,
            "cross_role_near_duplicates_allowed": False,
            "future_lockbox_opened": False,
        },
        "exclusions": exclusion_summary,
        "selection": {
            "selected_pair_slots": len(assignments),
            "reserve_pairs_available": int(
                protocol["selection"]["unused_reserve_pairs"]
            ),
            "reserve_pairs_consumed": len(
                {
                    row["accepted_cycle"]
                    for row in pair_records
                    if row["replacement"]
                }
            )
            + len(
                {
                    row["rejected_cycle"]
                    for row in rejections
                    if row["rejected_cycle"]
                    not in {item["pair_index"] for item in assignments}
                }
            ),
            "rejected_attempts": len(rejections),
            "pair_records": pair_records,
            "rejections": rejections,
        },
        "images": image_records,
        "counts": {
            "pairs": len(pair_records),
            "images": len(image_records),
            "real": sum(row["project_label"] == 0 for row in image_records),
            "ai_generated": sum(
                row["project_label"] == 1 for row in image_records
            ),
            "role_pairs": dict(Counter(row["role"] for row in pair_records)),
            "network_asset_bytes_this_resume_invocation": (
                download_budget.downloaded_bytes
            ),
            "frozen_prefix_asset_bytes_from_prior_invocation": (
                frozen_prefix_inventory["total_asset_bytes"]
            ),
        },
        "manifests": manifest_summary,
        "provenance": {"path": _project_relative(provenance_path, project_root)},
        "publication": {
            "raw_images_committed_to_git": False,
            "raw_images_may_be_redistributed": False,
            "aggregate_results_only": True,
        },
        "training_gate": {
            "shortcut_audit_required": True,
            "shortcut_audit_passed": False,
            "e6_training_allowed": False,
        },
    }
    staged_provenance = staging_root / "provenance.json"
    _atomic_json_write_fsync(staged_provenance, provenance)
    _validate_publication_bundle(
        staged_provenance,
        staging_manifest_root,
        outputs=outputs,
        protocol_path=protocol_path,
        lock_report_path=lock_report_path,
        acquisition_protocol_path=acquisition_protocol_path,
        acquisition_lock_path=acquisition_lock_path,
        project_root=project_root,
        validate_raw_assets=False,
        resume_protocol_path=resume_protocol_path,
        resume_lock_path=resume_lock_path,
        rejection_journal_path=journal_path,
    )
    staging_manifest_root.replace(manifest_root)
    _fsync_directory(manifest_root.parent)
    staged_provenance.replace(provenance_path)
    _fsync_directory(provenance_path.parent)
    _clear_publication_staging(staging_root, project_root)
    provenance = _validate_publication_bundle(
        provenance_path,
        manifest_root,
        outputs=outputs,
        protocol_path=protocol_path,
        lock_report_path=lock_report_path,
        acquisition_protocol_path=acquisition_protocol_path,
        acquisition_lock_path=acquisition_lock_path,
        project_root=project_root,
        validate_raw_assets=False,
        resume_protocol_path=resume_protocol_path,
        resume_lock_path=resume_lock_path,
        rejection_journal_path=journal_path,
    )
    return provenance


def prepare_e6_biggan(
    protocol_path: Path,
    lock_report_path: Path,
    acquisition_protocol_path: Path,
    acquisition_lock_path: Path,
    project_root: Path,
    resume_protocol_path: Path | None = None,
    resume_lock_path: Path | None = None,
) -> dict[str, Any]:
    """Run preparation while guaranteeing release of the process lock."""

    acquisition = load_e6_acquisition_protocol(acquisition_protocol_path)
    validate_e6_acquisition_protocol(acquisition)
    raw_relative = acquisition.get("outputs", {}).get("raw_root")
    if not isinstance(raw_relative, str) or not raw_relative:
        raise ValueError("E6 acquisition protocol has no raw root")
    raw_root = project_root / raw_relative
    _reject_symlink_path(raw_root, project_root)
    preparation_lock = _acquire_preparation_lock(raw_root, project_root)
    try:
        return _prepare_e6_biggan_locked(
            protocol_path,
            lock_report_path,
            acquisition_protocol_path,
            acquisition_lock_path,
            project_root,
            resume_protocol_path,
            resume_lock_path,
        )
    finally:
        try:
            fcntl.flock(preparation_lock.fileno(), fcntl.LOCK_UN)
        finally:
            preparation_lock.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--lock-report", type=Path, default=DEFAULT_LOCK_REPORT)
    parser.add_argument(
        "--acquisition-protocol",
        type=Path,
        default=DEFAULT_ACQUISITION_PROTOCOL,
    )
    parser.add_argument(
        "--acquisition-lock-report",
        type=Path,
        default=DEFAULT_ACQUISITION_LOCK_REPORT,
    )
    parser.add_argument(
        "--resume-protocol", type=Path, default=DEFAULT_RESUME_PROTOCOL
    )
    parser.add_argument(
        "--resume-lock-report", type=Path, default=DEFAULT_RESUME_LOCK
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    project_root = Path.cwd()
    try:
        free_gib = shutil.disk_usage(project_root).free / (1024**3)
        if free_gib < 2.0:
            raise ValueError(
                f"At least 2 GiB free space is required; found {free_gib:.2f} GiB"
            )
        result = prepare_e6_biggan(
            project_root / args.protocol,
            project_root / args.lock_report,
            project_root / args.acquisition_protocol,
            project_root / args.acquisition_lock_report,
            project_root,
            project_root / args.resume_protocol,
            project_root / args.resume_lock_report,
        )
        counts = result["counts"]
        print(
            "PASS E6 BigGAN preparation: "
            f"pairs={counts['pairs']}, real={counts['real']}, "
            f"biggan={counts['ai_generated']}, "
            f"rejected_attempts={result['selection']['rejected_attempts']}"
        )
        print(
            f"Manifest root={protocol_manifest_root(result)}; "
            f"provenance={result['provenance']['path']}"
        )
        print("Training remains blocked until the E6 metadata-only shortcut audit passes.")
        return 0
    except (
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
        requests.RequestException,
    ) as exc:
        print(f"E6 BigGAN preparation failed: {exc}", file=sys.stderr)
        return 1


def protocol_manifest_root(result: dict[str, Any]) -> str:
    """Return the common manifest directory from a preparation result."""

    paths = [Path(item["path"]) for item in result["manifests"].values()]
    parents = {path.parent.as_posix() for path in paths}
    if len(parents) != 1:
        raise ValueError("E6 result manifests do not share one directory")
    return next(iter(parents))


if __name__ == "__main__":
    raise SystemExit(main())
