"""Conservative C2PA provenance inspection for image files.

This module deliberately keeps provenance evidence separate from visual model
predictions. A missing C2PA manifest is an absence of evidence, not proof that
an image is authentic.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import c2pa
from c2pa import Context, Reader


SCHEMA_VERSION = 1
MAX_FILE_BYTES = 100 * 1024 * 1024
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRUST_DIR = PROJECT_ROOT / "configs" / "c2pa_trust"
MANIFEST_TRUST_LIST = DEFAULT_TRUST_DIR / "C2PA-TRUST-LIST.pem"
TSA_TRUST_LIST = DEFAULT_TRUST_DIR / "C2PA-TSA-TRUST-LIST.pem"
MANIFEST_TRUST_SHA256 = "75cacc98b79ecac33713c7ecfb58d4a0ef383f3c1f886e7409f9e37e8664aea5"
TSA_TRUST_SHA256 = "c688d3555f4a2f1f8d663472bbd37888ff234abdd234c25934c0f9292e4eb5c9"
TRUST_LIST_COMMIT = "5a94626972e693dcfef53d59da56183ea8d3f8e5"
SUPPORTED_IMAGE_EXTENSIONS = {
    ".avif",
    ".dng",
    ".gif",
    ".heic",
    ".heif",
    ".jpeg",
    ".jpg",
    ".jxl",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

AI_GENERATED_SOURCE_TYPES = {"trainedalgorithmicmedia"}
AI_EDITED_SOURCE_TYPES = {"compositewithtrainedalgorithmicmedia"}
ALGORITHMIC_EDIT_SOURCE_TYPES = {"algorithmicallyenhanced"}
CAPTURE_SOURCE_TYPES = {"computationalcapture", "digitalcapture"}
SYNTHETIC_COMPOSITE_SOURCE_TYPES = {"compositesynthetic"}
SUPPORTED_ACTION_LABELS = {"c2pa.actions", "c2pa.actions.v2"}
SUPPORTED_AI_DISCLOSURE_LABELS = {"c2pa.ai-disclosure"}
IPTC_SOURCE_TYPE_PREFIXES = (
    "http://cv.iptc.org/newscodes/digitalsourcetype/",
    "https://cv.iptc.org/newscodes/digitalsourcetype/",
)


class ProvenanceInspectionError(RuntimeError):
    """Raised when a file cannot be safely inspected."""


def _read_pinned_trust_list(path: Path, expected_sha256: str) -> str:
    if not path.is_file():
        raise ProvenanceInspectionError(f"Pinned C2PA trust list is missing: {path}")
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise ProvenanceInspectionError(
            f"Pinned C2PA trust list hash changed: {path}; "
            f"expected={expected_sha256}, actual={actual_sha256}"
        )
    return path.read_text(encoding="utf-8")


def _offline_context_settings() -> dict[str, Any]:
    manifest_anchors = _read_pinned_trust_list(
        MANIFEST_TRUST_LIST, MANIFEST_TRUST_SHA256
    )
    tsa_anchors = _read_pinned_trust_list(TSA_TRUST_LIST, TSA_TRUST_SHA256)
    source_root = (
        "https://github.com/c2pa-org/conformance-public/blob/"
        f"{TRUST_LIST_COMMIT}/trust-list"
    )
    return {
        "version": 1,
        "trust": {
            "anchors": [
                {
                    "trust_uri": f"{source_root}/C2PA-TRUST-LIST.pem",
                    "trust_kind": "manifest",
                    "trust_anchors": manifest_anchors,
                },
                {
                    "trust_uri": f"{source_root}/C2PA-TSA-TRUST-LIST.pem",
                    "trust_kind": "tsa",
                    "trust_anchors": tsa_anchors,
                },
            ]
        },
        "core": {
            "allowed_network_hosts": [],
            "decode_identity_assertions": False,
        },
        "verify": {
            "verify_after_reading": True,
            "verify_trust": True,
            "verify_timestamp_trust": True,
            "remote_manifest_fetch": False,
            "ocsp_fetch": False,
        },
    }


def _offline_context() -> Context:
    """Create a verifying C2PA context that performs no network fetches."""

    return Context.from_dict(_offline_context_settings())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _as_nonempty_string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _source_type_name(source_type: str) -> str | None:
    """Return a token only for a canonical IPTC digital-source-type URI."""

    for prefix in IPTC_SOURCE_TYPE_PREFIXES:
        if source_type.startswith(prefix):
            token = source_type.removeprefix(prefix)
            if token and "/" not in token and "?" not in token and "#" not in token:
                return token.casefold()
    return None


def _unique_strings(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _extract_actions(assertions: Any) -> list[dict[str, str]]:
    actions: list[dict[str, str]] = []
    if not isinstance(assertions, list):
        return actions

    for assertion in assertions:
        if not isinstance(assertion, dict):
            continue
        label = _as_nonempty_string(assertion.get("label"))
        data = assertion.get("data")
        if label not in SUPPORTED_ACTION_LABELS or not isinstance(data, dict):
            continue
        raw_actions = data.get("actions")
        if not isinstance(raw_actions, list):
            continue

        templates_by_action: dict[str, dict[str, Any]] = {}
        raw_templates = data.get("templates")
        if isinstance(raw_templates, list):
            for raw_template in raw_templates:
                if not isinstance(raw_template, dict):
                    continue
                template_action = _as_nonempty_string(raw_template.get("action"))
                if template_action is not None:
                    templates_by_action[template_action] = raw_template

        pending_actions = list(raw_actions)
        while pending_actions:
            raw_action = pending_actions.pop(0)
            if not isinstance(raw_action, dict):
                continue
            action_name = _as_nonempty_string(raw_action.get("action"))
            merged_action = dict(templates_by_action.get(action_name or "", {}))
            merged_action.update(raw_action)
            action: dict[str, str] = {}
            action_name = _as_nonempty_string(merged_action.get("action"))
            source_type = _as_nonempty_string(merged_action.get("digitalSourceType"))
            if action_name is not None:
                action["action"] = action_name
            if source_type is not None:
                action["digital_source_type"] = source_type
            if action:
                actions.append(action)
            related = raw_action.get("related")
            if isinstance(related, list):
                pending_actions.extend(related)
    return actions


def _extract_ai_disclosures(assertions: Any) -> list[dict[str, str]]:
    disclosures: list[dict[str, str]] = []
    if not isinstance(assertions, list):
        return disclosures

    for assertion in assertions:
        if not isinstance(assertion, dict):
            continue
        label = _as_nonempty_string(assertion.get("label"))
        data = assertion.get("data")
        if label not in SUPPORTED_AI_DISCLOSURE_LABELS:
            continue
        if not isinstance(data, dict):
            continue

        disclosure: dict[str, str] = {}
        for source_key, output_key in (
            ("modelType", "model_type"),
            ("modelName", "model_name"),
            ("modelIdentifier", "model_identifier"),
        ):
            value = _as_nonempty_string(data.get(source_key))
            if value is not None:
                disclosure[output_key] = value

        content_profile = data.get("contentProfile")
        if isinstance(content_profile, dict):
            oversight = _as_nonempty_string(
                content_profile.get("humanOversightLevel")
            )
            if oversight is not None:
                disclosure["human_oversight_level"] = oversight
        if disclosure:
            disclosures.append(disclosure)
    return disclosures


def _collect_validation_codes(value: Any) -> list[str]:
    codes: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            if key == "code":
                code = _as_nonempty_string(nested)
                if code is not None:
                    codes.append(code)
            else:
                codes.extend(_collect_validation_codes(nested))
    elif isinstance(value, list):
        for nested in value:
            codes.extend(_collect_validation_codes(nested))
    return _unique_strings(codes)


def _active_manifest(manifest_store: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    active_id = _as_nonempty_string(manifest_store.get("active_manifest"))
    manifests = manifest_store.get("manifests")
    if active_id is None or not isinstance(manifests, dict):
        return active_id, {}
    manifest = manifests.get(active_id)
    return active_id, manifest if isinstance(manifest, dict) else {}


def _manifest_evidence(manifest_id: str, manifest: dict[str, Any]) -> dict[str, Any]:
    assertions = manifest.get("assertions", [])
    actions = _extract_actions(assertions)
    ai_disclosures = _extract_ai_disclosures(assertions)
    digital_source_types = _unique_strings(
        [
            action["digital_source_type"]
            for action in actions
            if "digital_source_type" in action
        ]
    )
    return {
        "manifest_id": manifest_id,
        "actions": actions,
        "digital_source_types": digital_source_types,
        "ai_disclosures": ai_disclosures,
        "content_signal": _content_signal(actions, ai_disclosures),
    }


def _parent_validation_state(
    manifest_id: str, validation_results: dict[str, Any] | None
) -> tuple[str, list[str]]:
    """Derive one linked parent's state from its own SDK validation records."""

    if not isinstance(validation_results, dict):
        return "unknown", []
    deltas = validation_results.get("ingredientDeltas")
    if not isinstance(deltas, list):
        return "unknown", []

    codes_by_bucket: dict[str, list[str]] = {
        "success": [],
        "informational": [],
        "failure": [],
    }
    manifest_marker = f"/{manifest_id}/"
    for delta in deltas:
        if not isinstance(delta, dict):
            continue
        validation_deltas = delta.get("validationDeltas")
        if not isinstance(validation_deltas, dict):
            continue
        for bucket in codes_by_bucket:
            records = validation_deltas.get(bucket)
            if not isinstance(records, list):
                continue
            for record in records:
                if not isinstance(record, dict):
                    continue
                url = _as_nonempty_string(record.get("url"))
                code = _as_nonempty_string(record.get("code"))
                if url is not None and code is not None and manifest_marker in url:
                    codes_by_bucket[bucket].append(code)

    success = set(codes_by_bucket["success"])
    failure = set(codes_by_bucket["failure"])
    all_codes = _unique_strings(
        codes_by_bucket["success"]
        + codes_by_bucket["informational"]
        + codes_by_bucket["failure"]
    )
    if not all_codes:
        return "unknown", []
    substantive_failures = failure.difference({"signingCredential.untrusted"})
    if substantive_failures:
        return "invalid", all_codes
    signature_valid = bool(
        success.intersection(
            {"claimSignature.validated", "ingredient.claimSignature.validated"}
        )
    )
    if not signature_valid:
        return "unknown", all_codes
    if "signingCredential.trusted" in success:
        return "trusted", all_codes
    return "valid_untrusted", all_codes


def _linked_parent_history(
    manifest_store: dict[str, Any],
    active_id: str | None,
    validation_results: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Follow only explicit parentOf links reachable from the active manifest."""

    manifests = manifest_store.get("manifests")
    if active_id is None or not isinstance(manifests, dict):
        return []

    history: list[dict[str, Any]] = []
    visited = {active_id}
    pending = [active_id]
    while pending:
        current_id = pending.pop(0)
        current = manifests.get(current_id)
        if not isinstance(current, dict):
            continue
        ingredients = current.get("ingredients")
        if not isinstance(ingredients, list):
            continue
        for ingredient in ingredients:
            if not isinstance(ingredient, dict):
                continue
            if ingredient.get("relationship") != "parentOf":
                continue
            parent_id = _as_nonempty_string(ingredient.get("active_manifest"))
            if parent_id is None or parent_id in visited:
                continue
            parent = manifests.get(parent_id)
            if not isinstance(parent, dict):
                continue
            visited.add(parent_id)
            evidence = _manifest_evidence(parent_id, parent)
            parent_state, parent_codes = _parent_validation_state(
                parent_id, validation_results
            )
            evidence["relationship"] = "parentOf"
            evidence["linked_from_manifest_id"] = current_id
            evidence["validation_state"] = parent_state
            evidence["integrity_and_signer_verified"] = parent_state == "trusted"
            evidence["validation_codes"] = parent_codes
            history.append(evidence)
            pending.append(parent_id)
    return history


def _content_signal(
    actions: list[dict[str, str]], ai_disclosures: list[dict[str, str]]
) -> str:
    action_sources = [
        (action.get("action"), _source_type_name(action["digital_source_type"]))
        for action in actions
        if "digital_source_type" in action
    ]
    created_sources = {
        source_type
        for action_name, source_type in action_sources
        if action_name == "c2pa.created" and source_type is not None
    }
    all_sources = {
        source_type for _, source_type in action_sources if source_type is not None
    }
    if created_sources.intersection(AI_GENERATED_SOURCE_TYPES):
        return "ai_generated"
    if all_sources.intersection(AI_EDITED_SOURCE_TYPES):
        return "ai_edited"
    if all_sources.intersection(AI_GENERATED_SOURCE_TYPES):
        return "ai_edited"
    if ai_disclosures:
        return "ai_involvement"
    if all_sources.intersection(ALGORITHMIC_EDIT_SOURCE_TYPES):
        return "algorithmically_enhanced"
    if all_sources.intersection(SYNTHETIC_COMPOSITE_SOURCE_TYPES):
        return "synthetic_composite"
    if created_sources.intersection(CAPTURE_SOURCE_TYPES):
        return "capture"
    if all_sources.intersection(CAPTURE_SOURCE_TYPES):
        return "capture_component"
    return "other"


def _chain_content_signal(
    active_signal: str, parent_history: list[dict[str, Any]]
) -> str:
    trusted_parent_signals = {
        item["content_signal"]
        for item in parent_history
        if item["validation_state"] == "trusted"
    }
    unverified_parent_signals = {
        item["content_signal"]
        for item in parent_history
        if item["validation_state"] != "trusted"
    }
    if active_signal in {"ai_generated", "ai_edited", "ai_involvement"}:
        return active_signal
    if trusted_parent_signals.intersection(
        {"ai_generated", "ai_edited", "ai_involvement"}
    ):
        return "ai_history"
    if unverified_parent_signals.intersection(
        {"ai_generated", "ai_edited", "ai_involvement"}
    ):
        return "ai_history_unverified"
    if active_signal in {"algorithmically_enhanced", "synthetic_composite"}:
        return active_signal
    if trusted_parent_signals.intersection(
        {"algorithmically_enhanced", "synthetic_composite"}
    ):
        return "synthetic_history"
    if unverified_parent_signals.intersection(
        {"algorithmically_enhanced", "synthetic_composite"}
    ):
        return "synthetic_history_unverified"
    if active_signal in {"capture", "capture_component"}:
        return active_signal
    if trusted_parent_signals.intersection({"capture", "capture_component"}):
        return "capture_history"
    if unverified_parent_signals.intersection({"capture", "capture_component"}):
        return "capture_history_unverified"
    return "other"


def _decision(validation_state: str, content_signal: str) -> tuple[str, str]:
    if validation_state == "invalid":
        return (
            "invalid_provenance",
            "The manifest did not validate; its content claims must be disregarded.",
        )
    if validation_state not in {"trusted", "valid"}:
        return (
            "unknown_provenance_state",
            "A manifest exists, but its validation state could not be interpreted.",
        )

    if validation_state == "trusted":
        decisions = {
            "ai_generated": (
                "trusted_claim_ai_generated",
                "A trusted C2PA manifest claims the asset was created as "
                "trained-algorithmic media.",
            ),
            "ai_edited": (
                "trusted_claim_ai_edited",
                "A trusted C2PA manifest claims an AI-generated component or edit.",
            ),
            "ai_involvement": (
                "trusted_claim_ai_involvement",
                "A trusted C2PA manifest makes an AI-involvement disclosure.",
            ),
            "ai_history": (
                "trusted_claim_ai_history",
                "A trusted C2PA chain contains a parent manifest claiming AI involvement.",
            ),
            "ai_history_unverified": (
                "linked_ai_history_requires_review",
                "A linked parent claims AI involvement but is not independently trusted.",
            ),
            "algorithmically_enhanced": (
                "trusted_claim_algorithmic_edit",
                "A trusted C2PA manifest claims algorithmic enhancement.",
            ),
            "synthetic_composite": (
                "trusted_claim_synthetic_composite",
                "A trusted C2PA manifest claims a composite with synthetic elements.",
            ),
            "synthetic_history": (
                "trusted_claim_synthetic_history",
                "A trusted C2PA chain contains a parent synthetic-media claim.",
            ),
            "synthetic_history_unverified": (
                "linked_synthetic_history_requires_review",
                "A linked parent makes a synthetic-media claim but is not trusted.",
            ),
            "capture": (
                "trusted_claim_capture",
                "A trusted C2PA manifest claims the asset was created by digital "
                "or computational capture.",
            ),
            "capture_component": (
                "trusted_claim_capture_component",
                "A trusted C2PA manifest reports capture for a component, not the whole asset.",
            ),
            "capture_history": (
                "trusted_claim_capture_history",
                "A trusted C2PA chain contains a parent capture claim.",
            ),
            "capture_history_unverified": (
                "linked_capture_history_requires_review",
                "A linked parent makes a capture claim but is not independently trusted.",
            ),
            "other": (
                "trusted_provenance_other",
                "The C2PA manifest is trusted but contains no supported AI or capture signal.",
            ),
        }
        return decisions[content_signal]

    if validation_state == "valid":
        if content_signal in {
            "ai_generated",
            "ai_edited",
            "ai_involvement",
            "ai_history",
            "ai_history_unverified",
        }:
            return (
                "valid_untrusted_ai_claim",
                "The manifest is structurally valid, but its signer is not trusted.",
            )
        if content_signal in {
            "capture",
            "capture_component",
            "capture_history",
            "capture_history_unverified",
        }:
            return (
                "valid_untrusted_capture_claim",
                "The capture claim is validly signed, but its signer is not trusted.",
            )
        if content_signal in {
            "synthetic_composite",
            "synthetic_history",
            "synthetic_history_unverified",
        }:
            return (
                "valid_untrusted_synthetic_claim",
                "The synthetic-media claim is validly signed, but its signer is not trusted.",
            )
        return (
            "valid_untrusted_provenance",
            "The manifest is structurally valid, but its signer is not trusted.",
        )

    raise AssertionError(f"Unhandled validation state: {validation_state}")


def summarise_manifest_store(
    manifest_store: dict[str, Any],
    *,
    validation_state: str | None,
    validation_results: dict[str, Any] | None,
    embedded: bool,
    remote_url: str | None,
) -> dict[str, Any]:
    """Convert a C2PA manifest store into a small, conservative evidence result."""

    normalised_state = (validation_state or "unknown").casefold()
    if normalised_state not in {"invalid", "valid", "trusted"}:
        normalised_state = "unknown"

    active_id, active_manifest = _active_manifest(manifest_store)
    if active_id is None or not active_manifest:
        normalised_state = "unknown"
    active_evidence = _manifest_evidence(active_id or "", active_manifest)
    actions = active_evidence["actions"]
    ai_disclosures = active_evidence["ai_disclosures"]
    digital_source_types = active_evidence["digital_source_types"]
    parent_history = _linked_parent_history(
        manifest_store, active_id, validation_results
    )
    content_signal = _chain_content_signal(
        active_evidence["content_signal"], parent_history
    )
    decision, summary = _decision(normalised_state, content_signal)

    signature_info = active_manifest.get("signature_info")
    if not isinstance(signature_info, dict):
        signature_info = {}

    return {
        "manifest_present": True,
        "manifest_status": (
            "manifest_trusted"
            if normalised_state == "trusted"
            else "manifest_valid_untrusted"
            if normalised_state == "valid"
            else "manifest_invalid"
            if normalised_state == "invalid"
            else "manifest_state_unknown"
        ),
        "validation_state": normalised_state,
        "decision": decision,
        "summary": summary,
        "content_signal": content_signal,
        "active_manifest_integrity_and_signer_verified": normalised_state == "trusted",
        "embedded_manifest": embedded,
        "remote_manifest_url": remote_url,
        "network_fetch_enabled": False,
        "active_manifest_id": active_id,
        "claim_generator": _as_nonempty_string(active_manifest.get("claim_generator")),
        "issuer": _as_nonempty_string(signature_info.get("issuer")),
        "signature_time": _as_nonempty_string(signature_info.get("time")),
        "actions": actions,
        "digital_source_types": digital_source_types,
        "ai_disclosures": ai_disclosures,
        "linked_parent_history": parent_history,
        "validation_codes": _collect_validation_codes(validation_results or {}),
    }


def _absent_result() -> dict[str, Any]:
    return {
        "manifest_present": False,
        "manifest_status": "manifest_absent",
        "validation_state": None,
        "decision": "no_supported_signal",
        "summary": (
            "No embedded C2PA manifest was found. This does not prove that the image is real."
        ),
        "content_signal": "none",
        "active_manifest_integrity_and_signer_verified": False,
        "embedded_manifest": None,
        "remote_manifest_url": None,
        "network_fetch_enabled": False,
        "active_manifest_id": None,
        "claim_generator": None,
        "issuer": None,
        "signature_time": None,
        "actions": [],
        "digital_source_types": [],
        "ai_disclosures": [],
        "linked_parent_history": [],
        "validation_codes": [],
    }


def _remote_manifest_result() -> dict[str, Any]:
    return {
        "manifest_present": True,
        "manifest_status": "remote_manifest_unavailable",
        "validation_state": None,
        "decision": "no_local_provenance",
        "summary": (
            "The image points to a remote manifest, but network retrieval is disabled. "
            "No authenticity conclusion was made."
        ),
        "content_signal": "none",
        "active_manifest_integrity_and_signer_verified": False,
        "embedded_manifest": False,
        "remote_manifest_url": None,
        "network_fetch_enabled": False,
        "active_manifest_id": None,
        "claim_generator": None,
        "issuer": None,
        "signature_time": None,
        "actions": [],
        "digital_source_types": [],
        "ai_disclosures": [],
        "linked_parent_history": [],
        "validation_codes": [],
    }


def _validate_container_signature(path: Path) -> None:
    """Reject obvious extension/content mismatches before native parsing."""

    with path.open("rb") as handle:
        header = handle.read(32)
    suffix = path.suffix.casefold()

    signatures = {
        ".gif": header.startswith((b"GIF87a", b"GIF89a")),
        ".jpeg": header.startswith(b"\xff\xd8\xff"),
        ".jpg": header.startswith(b"\xff\xd8\xff"),
        ".png": header.startswith(b"\x89PNG\r\n\x1a\n"),
        ".tif": header.startswith((b"II*\x00", b"MM\x00*")),
        ".tiff": header.startswith((b"II*\x00", b"MM\x00*")),
        ".dng": header.startswith((b"II*\x00", b"MM\x00*")),
        ".webp": header.startswith(b"RIFF") and header[8:12] == b"WEBP",
        ".avif": header[4:8] == b"ftyp",
        ".heic": header[4:8] == b"ftyp",
        ".heif": header[4:8] == b"ftyp",
        ".jxl": header.startswith(b"\xff\x0a")
        or header.startswith(b"\x00\x00\x00\x0cJXL \r\n\x87\n"),
    }
    if not signatures.get(suffix, False):
        raise ProvenanceInspectionError(
            f"Image content does not match the {suffix} container signature: {path}"
        )


def inspect_provenance_file(
    path: Path, *, max_file_bytes: int = MAX_FILE_BYTES
) -> dict[str, Any]:
    """Inspect one image without modifying it or performing network requests."""

    if max_file_bytes <= 0:
        raise ValueError("max_file_bytes must be positive")
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    if not path.is_file():
        raise ProvenanceInspectionError(f"Input must be a file: {path}")
    if path.suffix.casefold() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ProvenanceInspectionError(
            f"Unsupported image extension {path.suffix!r}; "
            f"supported={sorted(SUPPORTED_IMAGE_EXTENSIONS)}"
        )

    byte_count = path.stat().st_size
    if byte_count > max_file_bytes:
        raise ProvenanceInspectionError(
            f"Image is too large for provenance inspection: {byte_count} bytes; "
            f"limit={max_file_bytes}"
        )
    _validate_container_signature(path)

    file_details = {
        "path": path.as_posix(),
        "name": path.name,
        "bytes": byte_count,
        "sha256": _sha256(path),
    }

    try:
        with _offline_context() as context:
            reader = Reader.try_create(path, context=context)
            if reader is None:
                provenance = _absent_result()
            else:
                with reader:
                    raw_store = json.loads(reader.json())
                    if not isinstance(raw_store, dict):
                        raise ProvenanceInspectionError(
                            "C2PA SDK returned a non-object manifest store"
                        )
                    provenance = summarise_manifest_store(
                        raw_store,
                        validation_state=reader.get_validation_state(),
                        validation_results=reader.get_validation_results(),
                        embedded=reader.is_embedded(),
                        remote_url=reader.get_remote_url(),
                    )
    except c2pa.C2paError.RemoteManifest:
        provenance = _remote_manifest_result()
    except ProvenanceInspectionError:
        raise
    except Exception as exc:
        raise ProvenanceInspectionError(f"C2PA inspection failed: {exc}") from exc

    return {
        "schema_version": SCHEMA_VERSION,
        "file": file_details,
        "provenance": provenance,
        "software": {
            "c2pa_python_version": c2pa.__version__,
            "c2pa_sdk_version": c2pa.sdk_version(),
            "trust_list_commit": TRUST_LIST_COMMIT,
            "manifest_trust_list_sha256": MANIFEST_TRUST_SHA256,
            "tsa_trust_list_sha256": TSA_TRUST_SHA256,
        },
        "limitations": [
            "No manifest is not evidence that an image is real.",
            "A valid-but-untrusted manifest is a claim, not verified identity.",
            "Trusted status verifies integrity and signer trust, not the factual truth of a claim.",
            "Remote manifests and online certificate checks are disabled.",
            "Provenance can be stripped when an image is copied or re-encoded.",
        ],
    }


def write_provenance_report(
    report: dict[str, Any], output_path: Path, *, source_path: Path | None = None
) -> None:
    """Write a JSON report atomically."""

    if source_path is not None and output_path.resolve() == source_path.resolve():
        raise ProvenanceInspectionError("JSON output path must not replace the input image")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(json.dumps(report, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
