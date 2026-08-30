"""Tests for held-out SID-Set feature-extraction input validation."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.extract_heldout_clip_features import (
    _relative_artifact_path,
    validate_manifest_rows,
    validate_preparation_provenance,
)


PROTOCOL_PATH = Path("configs/section4b_held_out_generator.json")


def _rows() -> list[dict[str, str]]:
    return [
        {
            "image_path": "data/real-a.jpg",
            "label": "0",
            "class_name": "real",
            "source": "sid_set_flux_heldout",
            "split": "heldout",
        },
        {
            "image_path": "data/real-b.jpg",
            "label": "0",
            "class_name": "real",
            "source": "sid_set_flux_heldout",
            "split": "heldout",
        },
        {
            "image_path": "data/ai-a.jpg",
            "label": "1",
            "class_name": "ai_generated",
            "source": "sid_set_flux_heldout",
            "split": "heldout",
        },
        {
            "image_path": "data/ai-b.jpg",
            "label": "1",
            "class_name": "ai_generated",
            "source": "sid_set_flux_heldout",
            "split": "heldout",
        },
    ]


def test_manifest_validation_accepts_balanced_unique_rows() -> None:
    result = validate_manifest_rows(_rows(), target_per_class=2)

    assert result["samples"] == 4
    assert result["class_counts"] == {"real_0": 2, "ai_generated_1": 2}
    assert result["unique_paths"] == 4


def test_manifest_validation_rejects_duplicate_paths() -> None:
    rows = _rows()
    rows[3]["image_path"] = rows[0]["image_path"]

    with pytest.raises(ValueError, match="paths are not unique"):
        validate_manifest_rows(rows, target_per_class=2)


def test_manifest_validation_rejects_label_imbalance() -> None:
    rows = _rows()
    rows[3]["label"] = "0"

    with pytest.raises(ValueError, match="not label balanced"):
        validate_manifest_rows(rows, target_per_class=2)


def _provenance(protocol: dict) -> dict:
    return {
        "protocol": {"sha256": "protocol-hash"},
        "manifest": {"sha256": "manifest-hash"},
        "dataset": {"source_revision": protocol["dataset"]["source_revision"]},
        "sampling": {"excluded_source_label_2": True},
        "download": {
            "content_unique_image_sha256": 4,
            "excluded_exact_duplicates": [{"row_idx": 9}],
        },
        "images": [
            {"row_idx": index, "sha256": f"{index:064x}"} for index in range(4)
        ],
        "organiser_validation_subset_used": False,
    }


def test_provenance_validation_requires_unique_content_and_frozen_source() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol = copy.deepcopy(protocol)
    protocol["sample"]["total_target"] = 4
    result = validate_preparation_provenance(
        protocol,
        _provenance(protocol),
        protocol_sha256="protocol-hash",
        manifest_sha256="manifest-hash",
    )

    assert result["unique_source_rows"] == 4
    assert result["unique_image_sha256"] == 4
    assert result["excluded_exact_duplicates"] == 1


def test_provenance_validation_rejects_duplicate_content() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol = copy.deepcopy(protocol)
    protocol["sample"]["total_target"] = 4
    provenance = _provenance(protocol)
    provenance["images"][3]["sha256"] = provenance["images"][0]["sha256"]

    with pytest.raises(ValueError, match="duplicate image content"):
        validate_preparation_provenance(
            protocol,
            provenance,
            protocol_sha256="protocol-hash",
            manifest_sha256="manifest-hash",
        )


def test_artifact_paths_are_repository_relative(tmp_path: Path) -> None:
    artifact = tmp_path / "data" / "features" / "heldout.npz"

    assert _relative_artifact_path(artifact, tmp_path) == "data/features/heldout.npz"
