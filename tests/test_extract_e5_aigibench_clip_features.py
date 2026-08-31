"""Input-validation tests for the fresh AIGIBench CLIP extraction."""

from __future__ import annotations

import copy

import pytest

from scripts.extract_e5_aigibench_clip_features import (
    validate_manifest_rows,
    validate_preparation_provenance,
)


def _rows() -> list[dict[str, str]]:
    return [
        {
            "image_path": "data/real-a.png",
            "label": "0",
            "class_name": "real",
            "source": "aigibench_midjourney_v6_external",
            "split": "external_test",
        },
        {
            "image_path": "data/real-b.png",
            "label": "0",
            "class_name": "real",
            "source": "aigibench_midjourney_v6_external",
            "split": "external_test",
        },
        {
            "image_path": "data/ai-a.png",
            "label": "1",
            "class_name": "ai_generated",
            "source": "aigibench_midjourney_v6_external",
            "split": "external_test",
        },
        {
            "image_path": "data/ai-b.png",
            "label": "1",
            "class_name": "ai_generated",
            "source": "aigibench_midjourney_v6_external",
            "split": "external_test",
        },
    ]


def _amendment() -> dict:
    return {
        "base_protocol": {"path": "base.json", "sha256": "base-hash"},
        "duplicate_audit": {"path": "audit.json", "sha256": "audit-hash"},
        "deduplication_selection": {
            "target_unique_images_per_class": 2,
            "algorithm": "frozen-dedup",
        },
    }


def _provenance() -> dict:
    rows = _rows()
    return {
        "amendment": {"sha256": "amendment-hash"},
        "manifest": {"sha256": "manifest-hash"},
        "base_protocol": {"path": "base.json", "sha256": "base-hash"},
        "duplicate_audit": {"path": "audit.json", "sha256": "audit-hash"},
        "dataset": {
            "organiser_validation_subset_used": False,
            "manual_image_inspection_before_scoring": False,
        },
        "counts": {
            "total_selected_unique": 4,
            "real_selected_unique": 2,
            "ai_selected_unique": 2,
            "cross_class_duplicate_groups": 0,
            "duplicates_excluded": 4,
            "development_or_prior_audit_overlap": 0,
        },
        "selection": {
            "algorithm": "frozen-dedup",
            "real_candidates_examined": 1000,
            "ai_candidates_examined": 1004,
            "real_duplicates_excluded": [],
            "ai_duplicates_excluded": [{"sha256": str(i)} for i in range(4)],
            "final_selected_member_list_sha256": "selection-hash",
        },
        "images": [
            {
                "image_path": row["image_path"],
                "label": int(row["label"]),
                "sha256": f"{index:064x}",
            }
            for index, row in enumerate(rows)
        ],
    }


def test_manifest_accepts_balanced_unique_external_rows() -> None:
    result = validate_manifest_rows(_rows(), target_per_class=2)
    assert result["samples"] == 4
    assert result["class_counts"] == {"real_0": 2, "ai_generated_1": 2}


def test_manifest_rejects_duplicate_path() -> None:
    rows = _rows()
    rows[-1]["image_path"] = rows[0]["image_path"]
    with pytest.raises(ValueError, match="paths are not unique"):
        validate_manifest_rows(rows, target_per_class=2)


def test_provenance_accepts_frozen_unique_selection() -> None:
    result = validate_preparation_provenance(
        _amendment(),
        _provenance(),
        _rows(),
        amendment_sha256="amendment-hash",
        manifest_sha256="manifest-hash",
    )
    assert result["unique_image_sha256"] == 4
    assert result["duplicates_excluded"] == 4


def test_provenance_rejects_duplicate_selected_content() -> None:
    provenance = _provenance()
    provenance["images"][-1]["sha256"] = provenance["images"][0]["sha256"]
    with pytest.raises(ValueError, match="duplicate content"):
        validate_preparation_provenance(
            _amendment(),
            provenance,
            _rows(),
            amendment_sha256="amendment-hash",
            manifest_sha256="manifest-hash",
        )


def test_provenance_rejects_observed_model_access() -> None:
    provenance = copy.deepcopy(_provenance())
    provenance["dataset"]["manual_image_inspection_before_scoring"] = True
    with pytest.raises(ValueError, match="manually inspected"):
        validate_preparation_provenance(
            _amendment(),
            provenance,
            _rows(),
            amendment_sha256="amendment-hash",
            manifest_sha256="manifest-hash",
        )
