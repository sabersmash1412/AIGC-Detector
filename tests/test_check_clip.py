"""Unit tests for the Section 2A frozen-CLIP sanity check."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch

from scripts.check_clip import select_real_and_fake, validate_features


MANIFEST_FIELDS = ["image_path", "label", "class_name", "source", "split"]


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_select_real_and_fake_returns_fixed_label_order(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    write_manifest(
        manifest,
        [
            {
                "image_path": "fake.jpg",
                "label": "1",
                "class_name": "ai_generated",
                "source": "test",
                "split": "test",
            },
            {
                "image_path": "real.jpg",
                "label": "0",
                "class_name": "real",
                "source": "test",
                "split": "test",
            },
        ],
    )

    selected = select_real_and_fake(manifest)

    assert [row["label"] for row in selected] == ["0", "1"]
    assert [row["image_path"] for row in selected] == ["real.jpg", "fake.jpg"]


def test_validate_features_accepts_finite_unit_vectors() -> None:
    features = torch.zeros(2, 512)
    features[0, 0] = 1.0
    features[1, 1] = 1.0

    diagnostics = validate_features(features)

    assert diagnostics["shape"] == [2, 512]
    assert diagnostics["all_finite"] is True
    assert diagnostics["l2_norms"] == [1.0, 1.0]
    assert diagnostics["real_fake_cosine_similarity"] == 0.0


@pytest.mark.parametrize("shape", [(1, 512), (2, 511)])
def test_validate_features_rejects_wrong_shape(shape: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="Expected CLIP feature shape"):
        validate_features(torch.zeros(shape))


def test_validate_features_rejects_non_finite_values() -> None:
    features = torch.zeros(2, 512)
    features[:, 0] = 1.0
    features[1, 1] = float("nan")

    with pytest.raises(ValueError, match="NaN or infinite"):
        validate_features(features)


def test_validate_features_rejects_unnormalized_vectors() -> None:
    with pytest.raises(ValueError, match="not L2-normalized"):
        validate_features(torch.ones(2, 512))
