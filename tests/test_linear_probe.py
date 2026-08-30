"""Tests for safe CLIP linear-probe caches, probabilities, and checkpoints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.extract_clip_features import atomic_feature_cache_write
from src.linear_probe import (
    LinearProbeCheckpoint,
    linear_probe_probabilities,
    load_feature_cache,
    load_linear_probe_checkpoint,
    save_linear_probe_checkpoint,
)


def test_linear_probe_probabilities_follow_logistic_equation() -> None:
    features = np.zeros((3, 512), dtype=np.float32)
    features[:, 0] = np.asarray([-1.0, 0.0, 1.0])
    coefficients = np.zeros(512, dtype=np.float64)
    coefficients[0] = 2.0

    probabilities = linear_probe_probabilities(features, coefficients, intercept=0.0)

    assert probabilities[0] == pytest.approx(0.11920292)
    assert probabilities[1] == pytest.approx(0.5)
    assert probabilities[2] == pytest.approx(0.88079708)


def test_linear_probe_checkpoint_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "linear_probe.npz"
    coefficients = np.linspace(-1.0, 1.0, 512)
    original = LinearProbeCheckpoint(
        coefficients=coefficients,
        intercept=0.25,
        regularization_c=0.1,
        threshold=0.5,
        seed=42,
        selected_validation_roc_auc=0.9,
        train_cache_sha256="train-hash",
        validation_cache_sha256="val-hash",
    )

    save_linear_probe_checkpoint(path, original)
    loaded = load_linear_probe_checkpoint(path)

    assert np.array_equal(loaded.coefficients, coefficients)
    assert loaded.intercept == 0.25
    assert loaded.regularization_c == 0.1
    assert loaded.threshold == 0.5
    assert loaded.train_cache_sha256 == "train-hash"


def test_load_feature_cache_preserves_alignment(tmp_path: Path) -> None:
    path = tmp_path / "train.npz"
    features = np.zeros((2, 512), dtype=np.float32)
    features[0, 0] = 1.0
    features[1, 1] = 1.0
    atomic_feature_cache_write(
        path,
        features=features,
        labels=np.asarray([0, 1], dtype=np.int64),
        image_paths=np.asarray(["real.jpg", "fake.jpg"]),
        split="train",
        model_name="ViT-B-32-quickgelu",
        pretrained="openai",
        manifest_sha256="manifest-hash",
    )

    cache = load_feature_cache(path, "train")

    assert np.array_equal(cache.features, features)
    assert cache.labels.tolist() == [0, 1]
    assert cache.image_paths.tolist() == ["real.jpg", "fake.jpg"]
    assert cache.manifest_sha256 == "manifest-hash"


def test_load_feature_cache_rejects_wrong_split(tmp_path: Path) -> None:
    path = tmp_path / "val.npz"
    features = np.zeros((2, 512), dtype=np.float32)
    features[0, 0] = 1.0
    features[1, 1] = 1.0
    atomic_feature_cache_write(
        path,
        features=features,
        labels=np.asarray([0, 1], dtype=np.int64),
        image_paths=np.asarray(["real.jpg", "fake.jpg"]),
        split="val",
        model_name="ViT-B-32-quickgelu",
        pretrained="openai",
        manifest_sha256="manifest-hash",
    )

    with pytest.raises(ValueError, match="Expected split"):
        load_feature_cache(path, "test")


def test_load_feature_cache_requires_explicit_real_only_mode(tmp_path: Path) -> None:
    path = tmp_path / "real-only.npz"
    features = np.zeros((2, 512), dtype=np.float32)
    features[:, 0] = 1.0
    atomic_feature_cache_write(
        path,
        features=features,
        labels=np.asarray([0, 0], dtype=np.int64),
        image_paths=np.asarray(["real-a.jpg", "real-b.jpg"]),
        split="train",
        model_name="ViT-B-32-quickgelu",
        pretrained="openai",
        manifest_sha256="manifest-hash",
    )

    with pytest.raises(ValueError, match="must contain both binary labels"):
        load_feature_cache(path, "train")

    cache = load_feature_cache(path, "train", require_both_labels=False)
    assert cache.labels.tolist() == [0, 0]


def test_linear_probe_rejects_wrong_feature_dimension() -> None:
    with pytest.raises(ValueError, match="Expected features shaped"):
        linear_probe_probabilities(
            np.zeros((2, 511)), np.zeros(512), intercept=0.0
        )
