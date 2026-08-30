"""Tests for SID-Set held-out protocol and deterministic sampling."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.prepare_sid_set_heldout import (
    Candidate,
    deterministic_page_order,
    is_allowed_image_content_type,
    register_unique_content,
    select_balanced_candidates,
    validate_protocol,
)


PROTOCOL_PATH = Path("configs/section4b_held_out_generator.json")


def _protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _candidate(row_idx: int, label: int) -> Candidate:
    return Candidate(
        row_idx=row_idx,
        img_id=f"image-{row_idx}",
        source_label=label,
        project_label=label,
        class_name="real" if label == 0 else "ai_generated",
        image_url=f"https://example.test/{row_idx}.jpg",
        source_width=1024,
        source_height=768,
    )


def test_checked_in_protocol_passes_all_guardrails() -> None:
    validate_protocol(_protocol())


def test_protocol_keeps_individual_sid_set_images_out_of_public_demo() -> None:
    protocol = copy.deepcopy(_protocol())
    protocol["download"]["individual_images_allowed_in_public_demo"] = True

    with pytest.raises(ValueError, match="cannot be shown in the public demo"):
        validate_protocol(protocol)


@pytest.mark.parametrize(
    "rule",
    [
        "retraining_allowed",
        "threshold_changes_allowed",
        "model_reselection_allowed",
        "organiser_validation_subset_used",
    ],
)
def test_protocol_rejects_relaxed_frozen_evaluation_rules(rule: str) -> None:
    protocol = copy.deepcopy(_protocol())
    protocol["frozen_evaluation"][rule] = True

    with pytest.raises(ValueError, match="Frozen Section 4B rule changed"):
        validate_protocol(protocol)


def test_protocol_rejects_including_tampered_as_fully_synthetic() -> None:
    protocol = copy.deepcopy(_protocol())
    protocol["task"]["included_source_labels"]["2"] = {
        "source_meaning": "tampered",
        "project_label": 1,
        "project_class_name": "ai_generated",
    }

    with pytest.raises(ValueError, match="Only SID-Set source labels 0 and 1"):
        validate_protocol(protocol)


def test_page_order_is_complete_deterministic_and_seeded() -> None:
    first = deterministic_page_order(300, 42)
    second = deterministic_page_order(300, 42)

    assert first == second
    assert sorted(first) == list(range(300))
    assert first != deterministic_page_order(300, 43)


def test_balanced_selection_is_deterministic_and_order_independent() -> None:
    candidates = [_candidate(index, index % 2) for index in range(20)]

    selected = select_balanced_candidates(candidates, target_per_class=4, seed=42)
    reversed_selected = select_balanced_candidates(
        list(reversed(candidates)), target_per_class=4, seed=42
    )

    assert selected == reversed_selected
    assert len(selected) == 8
    assert sum(row.project_label == 0 for row in selected) == 4
    assert sum(row.project_label == 1 for row in selected) == 4
    assert len({row.row_idx for row in selected}) == 8


def test_balanced_selection_rejects_insufficient_class_pool() -> None:
    candidates = [_candidate(index, 0) for index in range(10)] + [_candidate(99, 1)]

    with pytest.raises(ValueError, match="need 2"):
        select_balanced_candidates(candidates, target_per_class=2, seed=42)


@pytest.mark.parametrize(
    "content_type",
    ["image/jpeg", "application/octet-stream", "binary/octet-stream"],
)
def test_expected_sid_set_asset_content_types_are_documented(
    content_type: str,
) -> None:
    assert is_allowed_image_content_type(content_type)


def test_non_image_text_response_remains_rejected() -> None:
    assert not is_allowed_image_content_type("text/html; charset=utf-8")


def test_content_registration_detects_same_label_duplicate() -> None:
    first = _candidate(1, 1)
    second = _candidate(2, 1)
    owners: dict[str, Candidate] = {}
    digest = "a" * 64

    assert register_unique_content(first, digest, owners) is None
    assert register_unique_content(second, digest, owners) == first
    assert owners[digest] == first


def test_content_registration_rejects_conflicting_labels() -> None:
    owners: dict[str, Candidate] = {}
    register_unique_content(_candidate(1, 0), "b" * 64, owners)

    with pytest.raises(ValueError, match="conflicting project labels"):
        register_unique_content(_candidate(2, 1), "b" * 64, owners)
