from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

import src.provenance as provenance
from src.provenance import (
    MANIFEST_TRUST_SHA256,
    TSA_TRUST_SHA256,
    ProvenanceInspectionError,
    _offline_context_settings,
    inspect_provenance_file,
    summarise_manifest_store,
    write_provenance_report,
)


ACTIVE_ID = "urn:uuid:test-manifest"


def _store(*assertions: dict) -> dict:
    return {
        "active_manifest": ACTIVE_ID,
        "manifests": {
            ACTIVE_ID: {
                "claim_generator": "test-generator/1.0",
                "signature_info": {
                    "issuer": "Example Issuer",
                    "time": "2026-09-28T00:00:00Z",
                },
                "assertions": list(assertions),
            }
        },
    }


def _actions(source_type: str, *, action: str = "c2pa.created") -> dict:
    return {
        "label": "c2pa.actions.v2",
        "data": {
            "actions": [
                {
                    "action": action,
                    "digitalSourceType": (
                        "https://cv.iptc.org/newscodes/digitalsourcetype/"
                        f"{source_type}"
                    ),
                }
            ]
        },
    }


def _summary(
    store: dict, state: str, validation_results: dict | None = None
) -> dict:
    if validation_results is None:
        validation_results = {
            "activeManifest": {
                "success": [{"code": "claimSignature.validated"}]
            }
        }
    return summarise_manifest_store(
        store,
        validation_state=state,
        validation_results=validation_results,
        embedded=True,
        remote_url=None,
    )


def test_trusted_ai_generation_is_verified() -> None:
    result = _summary(_store(_actions("trainedAlgorithmicMedia")), "Trusted")

    assert result["manifest_status"] == "manifest_trusted"
    assert result["decision"] == "trusted_claim_ai_generated"
    assert result["content_signal"] == "ai_generated"
    assert result["active_manifest_integrity_and_signer_verified"] is True
    assert result["validation_codes"] == ["claimSignature.validated"]


def test_trusted_ai_composite_is_not_mislabeled_as_fully_generated() -> None:
    result = _summary(
        _store(_actions("compositeWithTrainedAlgorithmicMedia")), "Trusted"
    )

    assert result["decision"] == "trusted_claim_ai_edited"
    assert result["content_signal"] == "ai_edited"


def test_ai_source_type_on_non_created_action_is_an_edit_not_whole_generation() -> None:
    result = _summary(
        _store(_actions("trainedAlgorithmicMedia", action="c2pa.placed")),
        "Trusted",
    )

    assert result["decision"] == "trusted_claim_ai_edited"
    assert result["content_signal"] == "ai_edited"


def test_actions_v2_template_is_merged_before_interpretation() -> None:
    templated_actions = {
        "label": "c2pa.actions.v2",
        "data": {
            "actions": [{"action": "com.example.synthetic-filter"}],
            "templates": [
                {
                    "action": "com.example.synthetic-filter",
                    "digitalSourceType": (
                        "http://cv.iptc.org/newscodes/digitalsourcetype/"
                        "compositeSynthetic"
                    ),
                }
            ],
        },
    }

    result = _summary(_store(templated_actions), "Trusted")

    assert result["decision"] == "trusted_claim_synthetic_composite"
    assert result["content_signal"] == "synthetic_composite"


def test_trusted_capture_is_capture_provenance() -> None:
    result = _summary(_store(_actions("digitalCapture")), "Trusted")

    assert result["decision"] == "trusted_claim_capture"
    assert result["content_signal"] == "capture"


def test_capture_source_on_placed_component_does_not_label_whole_asset_capture() -> None:
    result = _summary(
        _store(_actions("digitalCapture", action="c2pa.placed")), "Trusted"
    )

    assert result["decision"] == "trusted_claim_capture_component"
    assert result["content_signal"] == "capture_component"


def test_valid_unknown_signer_is_not_called_verified() -> None:
    result = _summary(_store(_actions("trainedAlgorithmicMedia")), "Valid")

    assert result["manifest_status"] == "manifest_valid_untrusted"
    assert result["decision"] == "valid_untrusted_ai_claim"


def test_invalid_manifest_claim_is_disregarded() -> None:
    result = _summary(_store(_actions("trainedAlgorithmicMedia")), "Invalid")

    assert result["manifest_status"] == "manifest_invalid"
    assert result["decision"] == "invalid_provenance"


def test_missing_validation_state_is_not_mislabeled_as_invalid() -> None:
    result = _summary(_store(_actions("trainedAlgorithmicMedia")), "")

    assert result["manifest_status"] == "manifest_state_unknown"
    assert result["decision"] == "unknown_provenance_state"
    assert result["active_manifest_integrity_and_signer_verified"] is False


def test_missing_active_manifest_cannot_be_reported_as_trusted() -> None:
    result = _summary(
        {"active_manifest": "urn:uuid:missing", "manifests": {}}, "Trusted"
    )

    assert result["manifest_status"] == "manifest_state_unknown"
    assert result["decision"] == "unknown_provenance_state"


def test_ai_disclosure_is_extracted_without_inventing_generation_claim() -> None:
    disclosure = {
        "label": "c2pa.ai-disclosure",
        "data": {
            "modelType": "generative_model",
            "modelName": "Example Model",
            "modelIdentifier": "example:model:1",
            "contentProfile": {"humanOversightLevel": "prompt_guided"},
        },
    }

    result = _summary(_store(disclosure), "Trusted")

    assert result["decision"] == "trusted_claim_ai_involvement"
    assert result["ai_disclosures"] == [
        {
            "model_type": "generative_model",
            "model_name": "Example Model",
            "model_identifier": "example:model:1",
            "human_oversight_level": "prompt_guided",
        }
    ]


def test_similarly_named_assertion_and_noncanonical_uri_are_ignored() -> None:
    misleading = {
        "label": "c2pa.actions.evil",
        "data": {
            "actions": [
                {
                    "action": "c2pa.created",
                    "digitalSourceType": "https://evil.example/trainedAlgorithmicMedia",
                }
            ]
        },
    }

    result = _summary(_store(misleading), "Trusted")

    assert result["decision"] == "trusted_provenance_other"
    assert result["digital_source_types"] == []

    noncanonical_uri = {
        "label": "c2pa.actions.v2",
        "data": {
            "actions": [
                {
                    "action": "c2pa.created",
                    "digitalSourceType": "https://evil.example/trainedAlgorithmicMedia",
                }
            ]
        },
    }
    result = _summary(_store(noncanonical_uri), "Trusted")

    assert result["decision"] == "trusted_provenance_other"
    assert result["content_signal"] == "other"


def test_linked_parent_ai_history_is_preserved_without_calling_current_edit_generated() -> None:
    parent_id = "urn:uuid:parent-ai"
    store = _store()
    store["manifests"][ACTIVE_ID]["ingredients"] = [
        {
            "relationship": "parentOf",
            "active_manifest": parent_id,
            "label": "c2pa.ingredient",
        }
    ]
    store["manifests"][parent_id] = {
        "assertions": [_actions("trainedAlgorithmicMedia")],
        "signature_info": {},
    }

    parent_signature_url = f"self#jumbf=/c2pa/{parent_id}/c2pa.signature"
    validation_results = {
        "activeManifest": {"success": []},
        "ingredientDeltas": [
            {
                "validationDeltas": {
                    "success": [
                        {
                            "code": "claimSignature.validated",
                            "url": parent_signature_url,
                        },
                        {
                            "code": "signingCredential.trusted",
                            "url": parent_signature_url,
                        },
                    ],
                    "informational": [],
                    "failure": [],
                }
            }
        ],
    }
    result = _summary(store, "Trusted", validation_results)

    assert result["decision"] == "trusted_claim_ai_history"
    assert result["content_signal"] == "ai_history"
    assert result["linked_parent_history"][0]["manifest_id"] == parent_id
    assert result["linked_parent_history"][0]["validation_state"] == "trusted"


def test_untrusted_parent_ai_history_is_never_promoted_to_trusted() -> None:
    parent_id = "urn:uuid:parent-ai"
    store = _store()
    store["manifests"][ACTIVE_ID]["ingredients"] = [
        {"relationship": "parentOf", "active_manifest": parent_id}
    ]
    store["manifests"][parent_id] = {
        "assertions": [_actions("trainedAlgorithmicMedia")]
    }
    parent_signature_url = f"self#jumbf=/c2pa/{parent_id}/c2pa.signature"
    validation_results = {
        "ingredientDeltas": [
            {
                "validationDeltas": {
                    "success": [
                        {
                            "code": "claimSignature.validated",
                            "url": parent_signature_url,
                        }
                    ],
                    "informational": [],
                    "failure": [
                        {
                            "code": "signingCredential.untrusted",
                            "url": parent_signature_url,
                        }
                    ],
                }
            }
        ]
    }

    result = _summary(store, "Trusted", validation_results)

    assert result["decision"] == "linked_ai_history_requires_review"
    assert result["content_signal"] == "ai_history_unverified"
    assert result["linked_parent_history"][0]["validation_state"] == "valid_untrusted"


def test_unlinked_manifest_is_not_treated_as_provenance_history() -> None:
    store = _store()
    store["manifests"]["urn:uuid:unlinked"] = {
        "assertions": [_actions("trainedAlgorithmicMedia")]
    }

    result = _summary(store, "Trusted")

    assert result["decision"] == "trusted_provenance_other"
    assert result["linked_parent_history"] == []


def test_offline_context_pins_both_trust_lists_and_disables_network_paths() -> None:
    settings = _offline_context_settings()

    assert settings["core"] == {
        "allowed_network_hosts": [],
        "decode_identity_assertions": False,
    }
    assert settings["verify"]["remote_manifest_fetch"] is False
    assert settings["verify"]["ocsp_fetch"] is False
    assert [anchor["trust_kind"] for anchor in settings["trust"]["anchors"]] == [
        "manifest",
        "tsa",
    ]
    assert len(MANIFEST_TRUST_SHA256) == 64
    assert len(TSA_TRUST_SHA256) == 64


def test_modified_trust_list_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    modified = tmp_path / "modified.pem"
    modified.write_text("not the pinned trust list\n", encoding="utf-8")
    monkeypatch.setattr(provenance, "MANIFEST_TRUST_LIST", modified)

    with pytest.raises(ProvenanceInspectionError, match="hash changed"):
        _offline_context_settings()


def test_real_png_without_manifest_returns_no_signal(tmp_path: Path) -> None:
    image_path = tmp_path / "plain.png"
    Image.new("RGB", (8, 8), color=(12, 34, 56)).save(image_path)

    result = inspect_provenance_file(image_path)

    assert result["provenance"]["manifest_status"] == "manifest_absent"
    assert result["provenance"]["decision"] == "no_supported_signal"
    assert "does not prove" in result["provenance"]["summary"]
    assert len(result["file"]["sha256"]) == 64


def test_unsupported_input_is_rejected(tmp_path: Path) -> None:
    input_path = tmp_path / "notes.txt"
    input_path.write_text("not an image", encoding="utf-8")

    with pytest.raises(ProvenanceInspectionError, match="Unsupported image extension"):
        inspect_provenance_file(input_path)


def test_extension_header_mismatch_is_rejected(tmp_path: Path) -> None:
    input_path = tmp_path / "pretend.jpg"
    input_path.write_text("not really a JPEG", encoding="utf-8")

    with pytest.raises(ProvenanceInspectionError, match="container signature"):
        inspect_provenance_file(input_path)


def test_json_report_write_is_atomic_and_readable(tmp_path: Path) -> None:
    output_path = tmp_path / "reports" / "provenance.json"
    report = {"schema_version": 1, "provenance": {"decision": "no_supported_signal"}}

    write_provenance_report(report, output_path)

    assert output_path.read_text(encoding="utf-8").endswith("\n")
    assert not list(output_path.parent.glob(".provenance.json.*.tmp"))


def test_json_report_cannot_replace_input_image(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    Image.new("RGB", (8, 8), color=(12, 34, 56)).save(image_path)
    original_bytes = image_path.read_bytes()

    with pytest.raises(ProvenanceInspectionError, match="must not replace"):
        write_provenance_report(
            {"schema_version": 1}, image_path, source_path=image_path
        )

    assert image_path.read_bytes() == original_bytes
