"""Tests for paired transformed-feature cache provenance and alignment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.linear_probe import FeatureCache
from src.transformed_features import (
    atomic_transformed_feature_cache_write,
    load_transformed_feature_cache,
)


def _reference() -> FeatureCache:
    features = np.zeros((2, 512), dtype=np.float32)
    features[0, 0] = 1.0
    features[1, 1] = 1.0
    return FeatureCache(
        features=features,
        labels=np.asarray([0, 1], dtype=np.int64),
        image_paths=np.asarray(["real.jpg", "fake.jpg"]),
        split="train",
        model_name="ViT-B-32-quickgelu",
        pretrained="openai",
        manifest_sha256="manifest-123",
    )


def _write(path: Path, reference: FeatureCache) -> None:
    atomic_transformed_feature_cache_write(
        path,
        features=reference.features,
        reference=reference,
        condition="jpeg_q50",
        transform_seed=42,
        clean_cache_sha256="clean-123",
    )


def test_transformed_cache_round_trip_verifies_pairing(tmp_path: Path) -> None:
    path = tmp_path / "jpeg_q50.npz"
    reference = _reference()
    _write(path, reference)

    cache = load_transformed_feature_cache(
        path,
        expected_split="train",
        expected_condition="jpeg_q50",
        expected_seed=42,
        expected_clean_cache_sha256="clean-123",
        reference=reference,
    )

    assert cache.condition == "jpeg_q50"
    assert cache.transform_seed == 42
    assert np.array_equal(cache.image_paths, reference.image_paths)
    assert np.array_equal(cache.labels, reference.labels)


def test_transformed_cache_rejects_wrong_seed(tmp_path: Path) -> None:
    path = tmp_path / "jpeg_q50.npz"
    reference = _reference()
    _write(path, reference)

    with pytest.raises(ValueError, match="Expected transformation seed"):
        load_transformed_feature_cache(
            path,
            expected_split="train",
            expected_condition="jpeg_q50",
            expected_seed=7,
            expected_clean_cache_sha256="clean-123",
            reference=reference,
        )


def test_transformed_cache_rejects_misaligned_reference(tmp_path: Path) -> None:
    path = tmp_path / "jpeg_q50.npz"
    reference = _reference()
    _write(path, reference)
    misaligned = FeatureCache(
        features=reference.features,
        labels=reference.labels[::-1],
        image_paths=reference.image_paths[::-1],
        split=reference.split,
        model_name=reference.model_name,
        pretrained=reference.pretrained,
        manifest_sha256=reference.manifest_sha256,
    )

    with pytest.raises(ValueError, match="labels are not aligned"):
        load_transformed_feature_cache(
            path,
            expected_split="train",
            expected_condition="jpeg_q50",
            expected_seed=42,
            expected_clean_cache_sha256="clean-123",
            reference=misaligned,
        )


def test_transformed_writer_rejects_clean_condition(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-clean condition"):
        atomic_transformed_feature_cache_write(
            tmp_path / "clean.npz",
            features=_reference().features,
            reference=_reference(),
            condition="clean",
            transform_seed=42,
            clean_cache_sha256="clean-123",
        )
