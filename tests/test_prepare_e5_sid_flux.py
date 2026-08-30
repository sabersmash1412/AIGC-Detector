from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.prepare_e5_sid_flux import (
    allocate_manifest_rows,
    candidate_rank,
    deterministic_page_order,
    load_development_exclusions,
    parse_flux_page,
)
from scripts.prepare_sid_set_heldout import Candidate
from src.e5_protocol import validate_e5_protocol


PROTOCOL_PATH = Path("configs/e5_source_matched_adaptation.json")


def _protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _candidate(index: int, image_id: str | None = None) -> Candidate:
    return Candidate(
        row_idx=index,
        img_id=image_id or f"flux-{index}",
        source_label=1,
        project_label=1,
        class_name="ai_generated",
        image_url=f"https://example.test/{index}.jpg",
        source_width=100,
        source_height=80,
    )


def test_checked_in_e5_protocol_is_flux_only_and_audit_excluded() -> None:
    protocol = _protocol()
    validate_e5_protocol(protocol)
    flux = protocol["development_data"]["sid_set_train_flux"]
    assert flux["source_split"] == "train"
    assert flux["allowed_source_label"] == 1
    assert set(flux["forbidden_source_labels"]) == {"0", "2"}
    audit = protocol["forbidden_development_data"]["section4_sid_flux_audit"]
    assert audit["allowed_for_training"] is False
    assert audit["allowed_for_validation"] is False


def test_e5_page_order_is_deterministic_complete_and_namespaced() -> None:
    first = deterministic_page_order(30, 505)
    assert first == deterministic_page_order(30, 505)
    assert sorted(first) == list(range(30))
    assert first != deterministic_page_order(30, 506)


def test_e5_candidate_rank_is_stable_and_uses_identity() -> None:
    candidate = _candidate(7)
    assert candidate_rank(candidate, 505) == candidate_rank(candidate, 505)
    assert candidate_rank(candidate, 505) != candidate_rank(_candidate(8), 505)
    assert candidate_rank(candidate, 505) != candidate_rank(candidate, 506)


def test_parse_flux_page_retains_only_label_one() -> None:
    revision = "a" * 40
    rows = []
    for index, label in enumerate((0, 1, 2), start=10):
        rows.append(
            {
                "row_idx": index,
                "row": {
                    "img_id": f"image-{index}",
                    "image": {"src": f"https://example.test/{index}.jpg"},
                    "width": 100,
                    "height": 80,
                    "label": label,
                },
            }
        )
    result = parse_flux_page(
        {"partial": False, "num_rows_total": 210000, "rows": rows},
        expected_rows=210000,
        expected_revision=revision,
        observed_revision=revision,
    )
    assert len(result) == 1
    assert result[0].row_idx == 11
    assert result[0].source_label == result[0].project_label == 1
    assert result[0].class_name == "ai_generated"


def test_parse_flux_page_rejects_revision_partial_or_row_count_drift() -> None:
    payload = {"partial": False, "num_rows_total": 210000, "rows": []}
    with pytest.raises(ValueError, match="revision drift"):
        parse_flux_page(
            payload,
            expected_rows=210000,
            expected_revision="a" * 40,
            observed_revision="b" * 40,
        )
    with pytest.raises(ValueError, match="partial"):
        parse_flux_page(
            {**payload, "partial": True},
            expected_rows=210000,
            expected_revision="a" * 40,
            observed_revision="a" * 40,
        )
    with pytest.raises(ValueError, match="row count changed"):
        parse_flux_page(
            {**payload, "num_rows_total": 1},
            expected_rows=210000,
            expected_revision="a" * 40,
            observed_revision="a" * 40,
        )


def test_allocate_manifest_rows_creates_disjoint_flux_only_splits(
    tmp_path: Path,
) -> None:
    accepted = [
        (
            _candidate(index),
            tmp_path / "data" / f"{index}.jpg",
            {
                "bytes": 10,
                "sha256": f"{index:064x}",
                "width": 100,
                "height": 80,
                "format": "JPEG",
            },
        )
        for index in range(4)
    ]
    manifests, provenance = allocate_manifest_rows(
        accepted,
        train_count=3,
        validation_count=1,
        project_root=tmp_path,
    )
    assert len(manifests["train"]) == 3
    assert len(manifests["val"]) == 1
    assert {row["label"] for rows in manifests.values() for row in rows} == {1}
    assert {row["source"] for rows in manifests.values() for row in rows} == {
        "sid_set_train_flux_e5"
    }
    assert {row["e5_split"] for row in provenance} == {"train", "val"}
    assert len({row["image_path"] for row in provenance}) == 4


def test_frozen_real_and_audit_exclusions_are_disjoint_and_complete() -> None:
    exclusions = load_development_exclusions(_protocol(), Path.cwd())
    assert len(exclusions.real_image_ids) == 4000
    assert len(exclusions.real_content_sha256) == 4000
    assert len(exclusions.audit_image_ids) == 2000
    assert len(exclusions.audit_content_sha256) == 2000
    assert len(exclusions.all_image_ids) == 6000
    assert len(exclusions.all_content_sha256) == 6000
