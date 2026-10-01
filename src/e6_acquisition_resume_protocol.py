"""Offline validation for the post-partial-acquisition E6 resume contract."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from src.e6_acquisition_protocol import sha256_file
from src.e6_protocol import (
    assignment_sha256,
    build_pair_assignment,
    load_e6_protocol,
    pair_rows,
)


DEFAULT_RESUME_PROTOCOL = Path("configs/e6_acquisition_resume_protocol.json")
DEFAULT_RESUME_LOCK = Path("reports/e6_acquisition_resume_lock.json")
EXPECTED_PROTOCOL_ID = "e6_tiny_genimage_biggan_acquisition_resume_v2"
EXPECTED_FROZEN_AT = "2026-10-01T06:16:56Z"
EXPECTED_ASSIGNMENT_SHA256 = (
    "c20aac52c0db04dfa2bf3ccda9a04ddd7858f82544442bc34cecd5b3161f4d68"
)
EXPECTED_V1_REVISION = "9368b026a9ef98cff4059e97b1e2190aab6274b0"
EXPECTED_UPSTREAM = {
    "development_protocol": {
        "path": "configs/e6_development_protocol.json",
        "sha256": "1613be7082b6b8f9794e5e22ca247f1469f86adebd2e7d28f0aa44847596d551",
    },
    "development_lock": {
        "path": "reports/e6_development_protocol_lock.json",
        "sha256": "bec3ed54708ae8e9195ab01d5b5b0ae938f034e3024509d72f26705435a0cae6",
    },
    "acquisition_v1_protocol": {
        "path": "configs/e6_acquisition_protocol.json",
        "sha256": "ce89874897e68361876991913a741efb31d8298421632ffd2b44a55924a2707c",
    },
    "acquisition_v1_lock": {
        "path": "reports/e6_acquisition_protocol_lock.json",
        "sha256": "ebd224722da916c4b187ec1ecb4743650f10adbe5b027d177589ffde87c058dd",
    },
    "transport_preflight_audit": {
        "path": "reports/e6_transport_preflight_audit.json",
        "sha256": "aa2e0636cb401ea2599b1a6176880c030d20c34e876cab4f9aba2694f1f14598",
    },
    "overlap_incident": {
        "path": "reports/e6_acquisition_overlap_incident.json",
        "sha256": "3cee0767b1616b7a776de33aed70e7eb3b9d7d4198ee4298bc83024354444bae",
    },
    "v1_committed_revision": EXPECTED_V1_REVISION,
    "assignment_sha256": EXPECTED_ASSIGNMENT_SHA256,
}
ASSET_NAME = re.compile(r"row-([0-9]{5})\.(jpg|jpeg|png|webp)")
ZERO_EVENT_HASH = "0" * 64
FORBIDDEN_JOURNAL_TOKENS = ("Signature=", "Expires=", "X-Amz-", "image_url")


def _safe_relative(value: Any, description: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{description} must be a project-relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{description} escaped the project")
    return path


def _regular_file(project_root: Path, relative: str, description: str) -> Path:
    path = project_root / _safe_relative(relative, description)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{description} is missing or unsafe")
    return path


def load_resume_protocol(path: Path = DEFAULT_RESUME_PROTOCOL) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("E6 resume protocol must be a JSON object")
    return payload


def validate_resume_protocol(protocol: dict[str, Any]) -> None:
    expected_keys = {
        "schema_version",
        "protocol_id",
        "status",
        "frozen_at_utc",
        "purpose",
        "upstream_evidence",
        "chronology",
        "frozen_partial_cache",
        "policy_delta",
        "rejection_journal",
        "unchanged_guardrails",
    }
    if set(protocol) != expected_keys:
        raise ValueError("E6 resume protocol keys changed")
    if protocol.get("schema_version") != 1:
        raise ValueError("Unsupported E6 resume protocol schema")
    if protocol.get("protocol_id") != EXPECTED_PROTOCOL_ID:
        raise ValueError("Unexpected E6 resume protocol id")
    if protocol.get("status") != "frozen_after_partial_acquisition_before_resumption":
        raise ValueError("E6 resume chronology status changed")
    if protocol.get("frozen_at_utc") != EXPECTED_FROZEN_AT:
        raise ValueError("E6 resume freeze time changed")
    if protocol.get("upstream_evidence") != EXPECTED_UPSTREAM:
        raise ValueError("E6 resume upstream evidence changed")
    chronology = protocol["chronology"]
    if chronology != {
        "frozen_before_initial_e6_payload_access": False,
        "frozen_after_v1_abort": True,
        "frozen_before_any_resumption_access": True,
        "initial_accepted_images": 680,
        "initial_accepted_pairs": 340,
        "initial_accepted_assignment_slots": "0 through 339 inclusive",
        "post_failure_nonvisual_diagnostic_rows": [1850, 1851],
        "performance_metrics_observed_or_used": False,
        "manual_visual_inspection_used": False,
        "future_lockbox_or_organiser_validation_used": False,
        "consumed_test_identity_metadata_used_for_integrity_filter": True,
        "consumed_test_metrics_or_visual_content_used_for_replacement": False,
    }:
        raise ValueError("E6 resume chronology changed")
    cache = protocol["frozen_partial_cache"]
    if cache != {
        "raw_root": "data/raw/e6_tiny_genimage_biggan",
        "images_root": "data/raw/e6_tiny_genimage_biggan/images",
        "prefix_assignment_slots": 340,
        "expected_asset_files": 680,
        "expected_receipt_files": 680,
        "expected_partial_files": 0,
        "expected_accepted_reserve_pairs": 0,
        "required_absent_row_indices": [1850, 1851],
        "inventory_algorithm": (
            "For assignment slots 0..339 in order and real then AI row order, "
            "verify the deterministic asset and receipt, SHA-256 the asset bytes, "
            "validate the receipt identity and receipt-declared byte SHA-256, then "
            "SHA-256 canonical compact JSON of the ordered inventory entries."
        ),
        "runtime_policy": (
            "Every frozen prefix asset and receipt must continue to match this lock; "
            "later deterministic cache entries may be added only by the resumed acquisition."
        ),
    }:
        raise ValueError("E6 frozen partial-cache contract changed")
    policy = protocol["policy_delta"]
    if policy != {
        "scope": (
            "Only historical-overlap classification and rejection persistence change; "
            "all v1 source, transport, decoding, identity, role, split and threshold "
            "contracts remain unchanged."
        ),
        "precedence_highest_first": [
            "remote_row_or_transport_contract_drift_hard_abort",
            "any_conflicting_label_match_hard_abort",
            "historical_byte_duplicate_hard_abort",
            "historical_decoded_pixel_duplicate_hard_abort",
            "historical_same_label_phash_only_near_match_reject_whole_pair_and_replace",
            "corrupt_remote_asset_reject_whole_pair_and_replace",
            "within_e6_same_label_duplicate_reject_later_whole_pair_and_replace",
        ],
        "historical_byte_duplicate_action": "hard_abort_without_replacement",
        "historical_decoded_pixel_duplicate_action": "hard_abort_without_replacement",
        "historical_conflicting_label_action": "hard_abort_without_replacement",
        "historical_same_label_phash_only_action": "reject_whole_pair_and_replace",
        "within_e6_policy_changed": False,
        "perceptual_hash_algorithm_or_threshold_changed": False,
        "replacement_source": "next unused pair in the original frozen SHA-256 rank order",
        "first_available_reserve_cycle": 247,
        "first_available_reserve_rows": [3460, 3461],
        "replacement_keeps_rejected_pair_role": True,
        "replacement_is_whole_pair": True,
        "manual_replacement_allowed": False,
        "visual_review_before_replacement_allowed": False,
        "reserve_exhaustion_action": "hard_abort",
    }:
        raise ValueError("E6 resume overlap policy changed")
    journal = protocol["rejection_journal"]
    if journal.get("path") != "data/raw/e6_tiny_genimage_biggan/rejection_journal.json":
        raise ValueError("E6 resume journal path changed")
    if journal.get("schema_version") != 1:
        raise ValueError("E6 resume journal schema changed")
    if journal.get("required_event_fields") != [
        "slot",
        "role",
        "assigned_cycle",
        "attempted_cycle",
        "attempted_cycles",
        "real_row_index",
        "ai_row_index",
        "action",
        "reasons",
    ]:
        raise ValueError("E6 resume journal event contract changed")
    if journal.get("required_reason_fields") != [
        "kind",
        "scope",
        "row_idx",
        "candidate_label",
        "candidate_byte_sha256",
        "candidate_decoded_pixel_sha256",
        "candidate_perceptual_hash",
        "owner",
        "owner_role",
        "owner_label",
    ] or journal.get("near_reason_additional_fields") != [
        "owner_perceptual_hash",
        "hamming_distance",
    ]:
        raise ValueError("E6 resume journal reason contract changed")
    if journal.get("write_timing") != (
        "atomically_before_candidate_discard_or_reserve_consumption"
    ):
        raise ValueError("E6 resume journal timing changed")
    if journal.get("file_fsync_before_publish") is not True or journal.get(
        "directory_fsync_after_publish"
    ) is not True:
        raise ValueError("E6 resume journal durability changed")
    if journal.get("signed_url_or_query_allowed") is not False:
        raise ValueError("E6 resume journal could expose signed URLs")
    if journal.get("initial_event_source") != (
        "reports/e6_acquisition_overlap_incident.json"
    ):
        raise ValueError("E6 resume journal seed source changed")
    if journal.get("rerun_rule") != (
        "Validate and replay previously journaled events in exact assignment/reserve "
        "order without redownloading discarded candidates; append new events "
        "atomically with a SHA-256 hash chain."
    ):
        raise ValueError("E6 resume journal replay rule changed")
    if journal.get("non_overlap_reason_schema") != (
        "Corrupt-image reasons use a sanitized detail field and row indices; owner "
        "and fingerprint fields are required only for overlap reasons."
    ):
        raise ValueError("E6 resume journal reason schema changed")
    if protocol["unchanged_guardrails"] != {
        "organiser_validation_access_allowed": False,
        "future_lockbox_access_allowed": False,
        "consumed_test_identity_only_integrity_filter_allowed": True,
        "consumed_test_metrics_or_visual_review_for_selection_allowed": False,
        "training_feature_extraction_or_evaluation_during_acquisition_allowed": False,
        "manual_image_selection_or_cherry_picking_allowed": False,
        "raw_image_git_commit_allowed": False,
        "signed_url_persistence_or_logging_allowed": False,
    }:
        raise ValueError("E6 resume guardrails changed")


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _event_hash(event_without_hash: dict[str, Any]) -> str:
    return hashlib.sha256(
        _canonical_json(event_without_hash).encode("utf-8")
    ).hexdigest()


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    """Atomically publish and fsync a journal update."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("E6 rejection journal may not be a symlink")
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
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _initial_incident_reason(incident: dict[str, Any]) -> dict[str, Any]:
    overlap = incident["confirmed_overlap"]
    assignment = incident["frozen_assignment"]
    return {
        "kind": "near_duplicate",
        "scope": "existing_data",
        "row_idx": int(overlap["candidate_row_index"]),
        "candidate_label": int(overlap["candidate_label"]),
        "candidate_byte_sha256": overlap["candidate_byte_sha256"],
        "candidate_decoded_pixel_sha256": overlap[
            "candidate_decoded_pixel_sha256"
        ],
        "candidate_perceptual_hash": overlap["candidate_perceptual_hash"],
        "owner": overlap["owner_path"],
        "owner_role": f"existing:{overlap['owner_manifest']}",
        "owner_label": int(overlap["owner_label"]),
        "owner_perceptual_hash": overlap["owner_perceptual_hash"],
        "hamming_distance": int(overlap["perceptual_hash_hamming_distance"]),
        "candidate_generator": overlap["candidate_generator"],
        "assigned_pair_key": assignment["pair_key"],
    }


def build_initial_rejection_journal(
    protocol: dict[str, Any],
    project_root: Path,
    protocol_path: Path = DEFAULT_RESUME_PROTOCOL,
) -> dict[str, Any]:
    """Build the deterministic journal seed for the already-observed v1 abort."""

    validate_resume_upstream(protocol, project_root)
    incident_record = protocol["upstream_evidence"]["overlap_incident"]
    incident_path = project_root / incident_record["path"]
    incident = json.loads(incident_path.read_text(encoding="utf-8"))
    assignment = incident["frozen_assignment"]
    resolved_protocol = (
        protocol_path if protocol_path.is_absolute() else project_root / protocol_path
    )
    event_body = {
        "event_index": 0,
        "previous_event_sha256": ZERO_EVENT_HASH,
        "slot": int(assignment["zero_based_slot"]),
        "role": assignment["role"],
        "assigned_cycle": int(assignment["assigned_cycle"]),
        "attempted_cycle": int(assignment["assigned_cycle"]),
        "attempted_cycles": [int(assignment["assigned_cycle"])],
        "real_row_index": int(assignment["real_row_index"]),
        "ai_row_index": int(assignment["biggan_row_index"]),
        "action": "reject_whole_pair_and_replace",
        "next_reserve_cycle": int(protocol["policy_delta"]["first_available_reserve_cycle"]),
        "reasons": [_initial_incident_reason(incident)],
        "source": "post_abort_nonvisual_incident_reconstruction",
    }
    event = {**event_body, "event_sha256": _event_hash(event_body)}
    return {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "resume_protocol_sha256": sha256_file(resolved_protocol),
        "assignment_sha256": EXPECTED_ASSIGNMENT_SHA256,
        "incident_sha256": incident_record["sha256"],
        "events": [event],
    }


def write_initial_rejection_journal(
    protocol: dict[str, Any],
    project_root: Path,
    protocol_path: Path = DEFAULT_RESUME_PROTOCOL,
) -> tuple[Path, dict[str, Any]]:
    """Create the immutable seed, refusing to overwrite a changed journal."""

    payload = build_initial_rejection_journal(
        protocol, project_root, protocol_path=protocol_path
    )
    path = project_root / _safe_relative(
        protocol["rejection_journal"]["path"], "rejection journal"
    )
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise ValueError("E6 rejection journal path is unsafe")
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != payload:
            raise ValueError("E6 rejection journal already exists with different content")
        return path, payload
    _atomic_json_write(path, payload)
    return path, payload


def _validate_reason(reason: Any) -> None:
    if not isinstance(reason, dict):
        raise ValueError("E6 rejection reason must be an object")
    kind = reason.get("kind")
    if kind in {"image_integrity_failure", "pair_label_conflict"}:
        if not isinstance(reason.get("detail"), str) or not reason["detail"]:
            raise ValueError("E6 integrity rejection lacks a sanitized detail")
        return
    required = {
        "kind",
        "scope",
        "row_idx",
        "candidate_label",
        "candidate_byte_sha256",
        "candidate_decoded_pixel_sha256",
        "candidate_perceptual_hash",
        "owner",
        "owner_role",
        "owner_label",
    }
    if not required.issubset(reason):
        raise ValueError("E6 overlap rejection reason is incomplete")
    if kind not in {"byte_duplicate", "decoded_pixel_duplicate", "near_duplicate"}:
        raise ValueError("E6 overlap rejection kind is invalid")
    if reason.get("scope") not in {"existing_data", "accepted_e6"}:
        raise ValueError("E6 overlap rejection scope is invalid")
    if reason.get("candidate_label") not in {0, 1} or reason.get(
        "owner_label"
    ) not in {0, 1}:
        raise ValueError("E6 overlap rejection label is invalid")
    if kind == "near_duplicate" and not {
        "owner_perceptual_hash",
        "hamming_distance",
    }.issubset(reason):
        raise ValueError("E6 near-duplicate reason is incomplete")
    for key in (
        "candidate_byte_sha256",
        "candidate_decoded_pixel_sha256",
    ):
        if re.fullmatch(r"[0-9a-f]{64}", str(reason.get(key, ""))) is None:
            raise ValueError("E6 rejection reason has an invalid SHA-256")
    if re.fullmatch(
        r"[0-9a-f]{16}", str(reason.get("candidate_perceptual_hash", ""))
    ) is None:
        raise ValueError("E6 rejection reason has an invalid perceptual hash")
    if kind == "near_duplicate":
        if re.fullmatch(
            r"[0-9a-f]{16}", str(reason.get("owner_perceptual_hash", ""))
        ) is None or not 0 <= int(reason.get("hamming_distance", -1)) <= 4:
            raise ValueError("E6 near-duplicate reason has an invalid distance")


def _ranked_cycles(development: dict[str, Any]) -> list[int]:
    selection = development["selection"]
    revision = development["development_source"]["repository_revision"]
    seed = int(selection["seed"])

    def key(cycle: int) -> tuple[bytes, int]:
        value = f"e6d:{seed}:{revision}:biggan:{cycle}".encode("utf-8")
        return hashlib.sha256(value).digest(), cycle

    return sorted(range(int(selection["candidate_pairs"])), key=key)


def _expected_action(reasons: list[dict[str, Any]]) -> str:
    replaceable = False
    for reason in reasons:
        kind = reason.get("kind")
        if kind == "image_integrity_failure":
            replaceable = True
            continue
        if kind == "pair_label_conflict":
            return "hard_abort_without_replacement"
        if reason.get("owner_label") != reason.get("candidate_label"):
            return "hard_abort_without_replacement"
        if reason.get("scope") == "existing_data" and kind in {
            "byte_duplicate",
            "decoded_pixel_duplicate",
        }:
            return "hard_abort_without_replacement"
        replaceable = True
    return "reject_whole_pair_and_replace" if replaceable else "hard_abort_without_replacement"


def validate_rejection_journal(
    payload: dict[str, Any],
    protocol: dict[str, Any],
    project_root: Path,
    protocol_path: Path = DEFAULT_RESUME_PROTOCOL,
) -> dict[str, Any]:
    """Validate journal identity, seed, hash chain, and secret-free events."""

    initial = build_initial_rejection_journal(
        protocol, project_root, protocol_path=protocol_path
    )
    for key in (
        "schema_version",
        "protocol_id",
        "resume_protocol_sha256",
        "assignment_sha256",
        "incident_sha256",
    ):
        if payload.get(key) != initial[key]:
            raise ValueError(f"E6 rejection journal identity changed: {key}")
    events = payload.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("E6 rejection journal has no seed event")
    if events[0] != initial["events"][0]:
        raise ValueError("E6 rejection journal seed changed")
    previous = ZERO_EVENT_HASH
    seen: set[tuple[int, int]] = set()
    development = load_e6_protocol(
        project_root / protocol["upstream_evidence"]["development_protocol"]["path"]
    )
    assignments = build_pair_assignment(development)
    ranked = _ranked_cycles(development)
    reserves = ranked[len(assignments) :]
    reserve_index = 0
    prior_slot = -1
    expected_attempt_for_slot: int | None = None
    expected_attempted_cycles: list[int] = []
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError("E6 rejection journal event must be an object")
        body = {key: value for key, value in event.items() if key != "event_sha256"}
        if event.get("event_index") != index:
            raise ValueError("E6 rejection journal event order changed")
        if event.get("previous_event_sha256") != previous:
            raise ValueError("E6 rejection journal chain changed")
        if event.get("event_sha256") != _event_hash(body):
            raise ValueError("E6 rejection journal event hash changed")
        required_event_keys = {
            "event_index",
            "previous_event_sha256",
            "event_sha256",
            "slot",
            "role",
            "assigned_cycle",
            "attempted_cycle",
            "attempted_cycles",
            "real_row_index",
            "ai_row_index",
            "action",
            "reasons",
        }
        allowed_event_keys = required_event_keys | {"next_reserve_cycle", "source"}
        if set(event) - allowed_event_keys or not required_event_keys.issubset(event):
            raise ValueError("E6 rejection journal event fields changed")
        identity = (int(event.get("slot", -1)), int(event.get("attempted_cycle", -1)))
        if identity in seen:
            raise ValueError("E6 rejection journal repeats an attempted cycle")
        seen.add(identity)
        if event.get("action") not in {
            "reject_whole_pair_and_replace",
            "hard_abort_without_replacement",
        }:
            raise ValueError("E6 rejection journal has an invalid action")
        reasons = event.get("reasons")
        if not isinstance(reasons, list) or not reasons:
            raise ValueError("E6 rejection journal event has no reasons")
        for reason in reasons:
            _validate_reason(reason)
        slot = int(event["slot"])
        if not 0 <= slot < len(assignments) or slot < prior_slot:
            raise ValueError("E6 rejection journal slot order changed")
        assignment = assignments[slot]
        assigned_cycle = int(assignment["pair_index"])
        if slot != prior_slot:
            expected_attempt_for_slot = assigned_cycle
            expected_attempted_cycles = []
        if (
            event["role"] != assignment["role"]
            or int(event["assigned_cycle"]) != assigned_cycle
            or int(event["attempted_cycle"]) != expected_attempt_for_slot
        ):
            raise ValueError("E6 rejection journal assignment order changed")
        expected_attempted_cycles = [
            *expected_attempted_cycles,
            int(event["attempted_cycle"]),
        ]
        if event["attempted_cycles"] != expected_attempted_cycles:
            raise ValueError("E6 rejection journal attempt history changed")
        real_row, ai_row = pair_rows(development, int(event["attempted_cycle"]))
        if (
            int(event["real_row_index"]) != real_row
            or int(event["ai_row_index"]) != ai_row
        ):
            raise ValueError("E6 rejection journal row identity changed")
        expected_labels = {real_row: 0, ai_row: 1}
        for reason in reasons:
            if reason.get("kind") in {
                "image_integrity_failure",
                "pair_label_conflict",
            }:
                continue
            reason_row = reason.get("row_idx")
            if reason_row not in expected_labels or reason.get(
                "candidate_label"
            ) != expected_labels[reason_row]:
                raise ValueError(
                    "E6 rejection reason disagrees with its frozen candidate row"
                )
        action = event["action"]
        if action != _expected_action(reasons):
            raise ValueError("E6 rejection journal action disagrees with its reasons")
        if action == "reject_whole_pair_and_replace":
            if reserve_index >= len(reserves):
                raise ValueError("E6 rejection journal exhausted the frozen reserve")
            next_cycle = int(event.get("next_reserve_cycle", -1))
            if next_cycle != reserves[reserve_index]:
                raise ValueError("E6 rejection journal reserve order changed")
            reserve_index += 1
            expected_attempt_for_slot = next_cycle
        elif "next_reserve_cycle" in event:
            raise ValueError("E6 fatal rejection journal event consumed a reserve")
        if action == "hard_abort_without_replacement" and index != len(events) - 1:
            raise ValueError("E6 rejection journal continued after a fatal event")
        prior_slot = slot
        serialized = _canonical_json(event)
        if any(token in serialized for token in FORBIDDEN_JOURNAL_TOKENS):
            raise ValueError("E6 rejection journal contains a signed URL or query")
        previous = str(event["event_sha256"])
    return payload


def append_rejection_event(
    path: Path,
    payload: dict[str, Any],
    event_fields: dict[str, Any],
    *,
    validate_before_write: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Idempotently append one event and durably publish it before side effects."""

    events = payload.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("E6 rejection journal is not initialized")
    identity = (event_fields.get("slot"), event_fields.get("attempted_cycle"))
    for existing in events:
        if (existing.get("slot"), existing.get("attempted_cycle")) == identity:
            comparable = {
                key: value
                for key, value in existing.items()
                if key not in {"event_index", "previous_event_sha256", "event_sha256"}
            }
            if comparable != event_fields:
                raise ValueError("E6 rejection journal replay disagrees with prior event")
            return payload
    body = {
        "event_index": len(events),
        "previous_event_sha256": events[-1]["event_sha256"],
        **event_fields,
    }
    event = {**body, "event_sha256": _event_hash(body)}
    updated = {**payload, "events": [*events, event]}
    serialized = _canonical_json(updated)
    if any(token in serialized for token in FORBIDDEN_JOURNAL_TOKENS):
        raise ValueError("E6 rejection journal contains a signed URL or query")
    if validate_before_write is not None:
        validate_before_write(updated)
    _atomic_json_write(path, updated)
    return updated


def find_replay_event(
    payload: dict[str, Any], slot: int, attempted_cycle: int
) -> dict[str, Any] | None:
    matches = [
        event
        for event in payload.get("events", [])
        if event.get("slot") == slot and event.get("attempted_cycle") == attempted_cycle
    ]
    if len(matches) > 1:
        raise ValueError("E6 rejection journal has ambiguous replay events")
    return matches[0] if matches else None


def validate_resume_upstream(protocol: dict[str, Any], project_root: Path) -> None:
    validate_resume_protocol(protocol)
    for name, record in protocol["upstream_evidence"].items():
        if name in {"v1_committed_revision", "assignment_sha256"}:
            continue
        path = _regular_file(project_root, record["path"], f"E6 resume {name}")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"E6 resume upstream evidence changed: {name}")
    development_path = project_root / protocol["upstream_evidence"][
        "development_protocol"
    ]["path"]
    development = load_e6_protocol(development_path)
    if assignment_sha256(build_pair_assignment(development)) != EXPECTED_ASSIGNMENT_SHA256:
        raise ValueError("E6 resume assignment changed")


def _asset_inventory(images_root: Path) -> tuple[dict[int, Path], dict[int, Path], list[Path]]:
    assets: dict[int, Path] = {}
    receipts: dict[int, Path] = {}
    partials: list[Path] = []
    for path in images_root.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ValueError("E6 resume image cache contains an unsafe entry")
        if path.name.endswith(".part") or path.name.endswith(".pending.json"):
            partials.append(path)
            continue
        if path.name.endswith(".receipt.json"):
            asset_name = path.name[: -len(".receipt.json")]
            match = ASSET_NAME.fullmatch(asset_name)
            if match is None:
                raise ValueError("E6 resume cache contains an invalid receipt name")
            row_idx = int(match.group(1))
            if row_idx in receipts:
                raise ValueError("E6 resume cache repeats a row receipt")
            receipts[row_idx] = path
            continue
        match = ASSET_NAME.fullmatch(path.name)
        if match is None:
            raise ValueError("E6 resume cache contains an unexpected image entry")
        row_idx = int(match.group(1))
        if row_idx in assets:
            raise ValueError("E6 resume cache repeats an asset row")
        assets[row_idx] = path
    return assets, receipts, partials


def build_frozen_prefix_inventory(
    protocol: dict[str, Any],
    project_root: Path,
    *,
    require_exact_cache: bool,
) -> dict[str, Any]:
    """Hash the frozen 340-pair prefix without decoding image pixels."""

    validate_resume_upstream(protocol, project_root)
    cache = protocol["frozen_partial_cache"]
    images_root = project_root / _safe_relative(cache["images_root"], "images root")
    if images_root.is_symlink() or not images_root.is_dir():
        raise ValueError("E6 resume images root is missing or unsafe")
    assets, receipts, partials = _asset_inventory(images_root)
    development = load_e6_protocol(
        project_root / protocol["upstream_evidence"]["development_protocol"]["path"]
    )
    assignments = build_pair_assignment(development)
    expected_rows: list[tuple[int, str, int, int]] = []
    prefix_assignments: list[dict[str, Any]] = []
    for slot, assignment in enumerate(assignments[: cache["prefix_assignment_slots"]]):
        prefix_assignments.append(
            {
                "slot": slot,
                "role": str(assignment["role"]),
                "assigned_cycle": int(assignment["pair_index"]),
                "real_row_index": int(assignment["real_row_index"]),
                "ai_row_index": int(assignment["ai_row_index"]),
            }
        )
        expected_rows.extend(
            [
                (slot, str(assignment["role"]), int(assignment["real_row_index"]), 0),
                (slot, str(assignment["role"]), int(assignment["ai_row_index"]), 1),
            ]
        )
    expected_indices = {item[2] for item in expected_rows}
    if require_exact_cache:
        if set(assets) != expected_indices or set(receipts) != expected_indices:
            raise ValueError("E6 resume cache is not the exact frozen 340-pair prefix")
        if partials:
            raise ValueError("E6 resume cache has unfinished partial files")
    entries: list[dict[str, Any]] = []
    total_bytes = 0
    for slot, role, row_idx, label in expected_rows:
        asset = assets.get(row_idx)
        receipt_path = receipts.get(row_idx)
        if asset is None or receipt_path is None:
            raise ValueError(f"E6 frozen prefix row is missing: {row_idx}")
        if receipt_path.name != f"{asset.name}.receipt.json":
            raise ValueError(f"E6 frozen prefix receipt/asset mismatch: {row_idx}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        verification = receipt.get("verification", {})
        canonical_path = receipt.get("canonical_asset_path")
        expected_suffix = PurePosixPath(str(canonical_path)).suffix.lower()
        row_contract = development["row_topology"][
            "real_row_contract" if label == 0 else "ai_row_contract"
        ]
        expected_canonical_prefix = (
            "/cached-assets/"
            f"{development['development_source']['dataset_repository']}/--/"
            f"{development['development_source']['repository_revision']}/--/"
            f"default/{development['development_source']['source_split']}/"
            f"{row_idx}/image/"
        )
        if (
            receipt.get("schema_version") != 1
            or receipt.get("row_idx") != row_idx
            or receipt.get("label") != label
            or receipt.get("generator_id") != row_contract["generator_id"]
            or receipt.get("generator_name") != row_contract["generator_name"]
            or receipt.get("signed_query_persisted") is not False
            or not isinstance(canonical_path, str)
            or not canonical_path.startswith(expected_canonical_prefix)
            or expected_suffix != asset.suffix.lower()
        ):
            raise ValueError(f"E6 frozen prefix receipt identity changed: {row_idx}")
        asset_hash = sha256_file(asset)
        if verification.get("byte_sha256") != asset_hash:
            raise ValueError(f"E6 frozen prefix asset bytes changed: {row_idx}")
        file_bytes = asset.stat().st_size
        if verification.get("file_bytes") != file_bytes:
            raise ValueError(f"E6 frozen prefix asset size changed: {row_idx}")
        if (
            re.fullmatch(
                r"[0-9a-f]{64}", str(verification.get("decoded_pixel_sha256", ""))
            )
            is None
            or re.fullmatch(
                r"[0-9a-f]{16}", str(verification.get("perceptual_hash", ""))
            )
            is None
            or verification.get("original_width") != receipt.get("source_width")
            or verification.get("original_height") != receipt.get("source_height")
        ):
            raise ValueError(f"E6 frozen prefix decoded identity changed: {row_idx}")
        total_bytes += file_bytes
        entries.append(
            {
                "slot": slot,
                "role": role,
                "row_idx": row_idx,
                "label": label,
                "asset_path": asset.relative_to(project_root).as_posix(),
                "asset_sha256": asset_hash,
                "asset_bytes": file_bytes,
                "receipt_path": receipt_path.relative_to(project_root).as_posix(),
                "receipt_sha256": sha256_file(receipt_path),
                "decoded_pixel_sha256": verification.get("decoded_pixel_sha256"),
                "perceptual_hash": verification.get("perceptual_hash"),
            }
        )
    if require_exact_cache:
        for row_idx in cache["required_absent_row_indices"]:
            if row_idx in assets or row_idx in receipts:
                raise ValueError("E6 failed candidate unexpectedly remained in cache")
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    return {
        "prefix_slots": cache["prefix_assignment_slots"],
        "asset_files": len(entries),
        "receipt_files": len(entries),
        "total_asset_bytes": total_bytes,
        "inventory_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "prefix_assignment_sha256": hashlib.sha256(
            _canonical_json(prefix_assignments).encode("utf-8")
        ).hexdigest(),
        "first_row_index": entries[0]["row_idx"],
        "last_inventory_row_index": entries[-1]["row_idx"],
    }


def build_resume_lock_receipt(
    protocol: dict[str, Any],
    project_root: Path,
    protocol_path: Path = DEFAULT_RESUME_PROTOCOL,
) -> dict[str, Any]:
    validate_resume_upstream(protocol, project_root)
    resolved = protocol_path if protocol_path.is_absolute() else project_root / protocol_path
    inventory = build_frozen_prefix_inventory(
        protocol, project_root, require_exact_cache=True
    )
    initial_journal = build_initial_rejection_journal(
        protocol, project_root, protocol_path=protocol_path
    )
    for relative in (
        "data/processed/e6_tiny_genimage_biggan",
        "data/processed/e6_tiny_genimage_biggan_provenance.json",
        "data/raw/.e6_tiny_genimage_biggan_staging",
    ):
        path = project_root / relative
        if path.exists() or path.is_symlink():
            raise ValueError("E6 final or staging output existed before resume freeze")
    return {
        "experiment": "e6_acquisition_resume_lock",
        "status": "PASS",
        "frozen_at_utc": protocol["frozen_at_utc"],
        "resume_protocol": {
            "path": protocol_path.as_posix(),
            "sha256": sha256_file(resolved),
            "schema_version": protocol["schema_version"],
        },
        "chronology": protocol["chronology"],
        "assignment_sha256": EXPECTED_ASSIGNMENT_SHA256,
        "frozen_partial_cache": inventory,
        "incident": protocol["upstream_evidence"]["overlap_incident"],
        "initial_rejection_journal": {
            "path": protocol["rejection_journal"]["path"],
            "sha256": hashlib.sha256(
                (json.dumps(initial_journal, indent=2) + "\n").encode("utf-8")
            ).hexdigest(),
            "event_count": 1,
            "head_event_sha256": initial_journal["events"][0]["event_sha256"],
        },
        "policy": {
            "same_label_phash_only_action": "reject_whole_pair_and_replace",
            "exact_byte_or_pixel_action": "hard_abort_without_replacement",
            "conflicting_label_action": "hard_abort_without_replacement",
            "first_available_reserve_cycle": 247,
            "journal_required_before_replacement": True,
        },
        "guardrails": protocol["unchanged_guardrails"],
    }


def validate_resume_lock_receipt(
    protocol: dict[str, Any],
    lock_path: Path,
    project_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the committed resume receipt and unchanged frozen prefix."""

    validate_resume_upstream(protocol, project_root)
    receipt = json.loads(lock_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "PASS":
        raise ValueError("E6 resume lock did not pass")
    expected_protocol = {
        "path": DEFAULT_RESUME_PROTOCOL.as_posix(),
        "sha256": sha256_file(project_root / DEFAULT_RESUME_PROTOCOL),
        "schema_version": 1,
    }
    if receipt.get("resume_protocol") != expected_protocol:
        raise ValueError("E6 resume lock protocol identity changed")
    inventory = build_frozen_prefix_inventory(
        protocol, project_root, require_exact_cache=False
    )
    if receipt.get("frozen_partial_cache") != inventory:
        raise ValueError("E6 frozen prefix differs from its resume lock")
    if receipt.get("assignment_sha256") != EXPECTED_ASSIGNMENT_SHA256:
        raise ValueError("E6 resume lock assignment changed")
    initial = build_initial_rejection_journal(protocol, project_root)
    expected_initial_record = {
        "path": protocol["rejection_journal"]["path"],
        "sha256": hashlib.sha256(
            (json.dumps(initial, indent=2) + "\n").encode("utf-8")
        ).hexdigest(),
        "event_count": 1,
        "head_event_sha256": initial["events"][0]["event_sha256"],
    }
    if receipt.get("initial_rejection_journal") != expected_initial_record:
        raise ValueError("E6 resume lock journal seed changed")
    if receipt.get("chronology") != protocol["chronology"]:
        raise ValueError("E6 resume lock chronology changed")
    if receipt.get("incident") != protocol["upstream_evidence"]["overlap_incident"]:
        raise ValueError("E6 resume lock incident changed")
    if receipt.get("policy") != {
        "same_label_phash_only_action": "reject_whole_pair_and_replace",
        "exact_byte_or_pixel_action": "hard_abort_without_replacement",
        "conflicting_label_action": "hard_abort_without_replacement",
        "first_available_reserve_cycle": 247,
        "journal_required_before_replacement": True,
    }:
        raise ValueError("E6 resume lock policy changed")
    if receipt.get("guardrails") != protocol["unchanged_guardrails"]:
        raise ValueError("E6 resume lock guardrails changed")
    return receipt, inventory
