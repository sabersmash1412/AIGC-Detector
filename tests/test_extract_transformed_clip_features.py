"""Tests for cumulative transformed-feature extraction summaries."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.extract_transformed_clip_features import (
    _limited_reference,
    ordered_recorded_conditions,
)
from src.linear_probe import FeatureCache


def test_recorded_conditions_union_uses_registry_order() -> None:
    result = ordered_recorded_conditions(
        ["jpeg_q50", "gaussian_blur_sigma1"],
        ["jpeg_q90", "jpeg_q30", "gaussian_blur_sigma1"],
    )

    assert result == [
        "jpeg_q90",
        "jpeg_q50",
        "jpeg_q30",
        "gaussian_blur_sigma1",
    ]


def test_recorded_conditions_reject_unknown_values() -> None:
    with pytest.raises(ValueError, match="unknown transformation conditions"):
        ordered_recorded_conditions(["unknown"], ["jpeg_q90"])


def test_debug_prefix_rejects_single_class_before_extraction() -> None:
    reference = FeatureCache(
        features=np.zeros((3, 512), dtype=np.float32),
        labels=np.asarray([1, 1, 0], dtype=np.int64),
        image_paths=np.asarray(["a.jpg", "b.jpg", "c.jpg"]),
        split="test",
        model_name="ViT-B-32-quickgelu",
        pretrained="openai",
        manifest_sha256="manifest",
    )

    with pytest.raises(ValueError, match="must contain both labels"):
        _limited_reference(reference, 2)

    limited = _limited_reference(reference, 3)
    assert limited.labels.tolist() == [1, 1, 0]


def test_debug_prefix_accepts_explicit_real_only_mode() -> None:
    reference = FeatureCache(
        features=np.zeros((3, 512), dtype=np.float32),
        labels=np.asarray([0, 0, 0], dtype=np.int64),
        image_paths=np.asarray(["a.jpg", "b.jpg", "c.jpg"]),
        split="train",
        model_name="ViT-B-32-quickgelu",
        pretrained="openai",
        manifest_sha256="manifest",
    )

    limited = _limited_reference(
        reference, 2, single_class_label=0
    )
    assert limited.labels.tolist() == [0, 0]


def test_debug_prefix_accepts_explicit_ai_only_mode() -> None:
    reference = FeatureCache(
        features=np.zeros((3, 512), dtype=np.float32),
        labels=np.asarray([1, 1, 1], dtype=np.int64),
        image_paths=np.asarray(["a.jpg", "b.jpg", "c.jpg"]),
        split="train",
        model_name="ViT-B-32-quickgelu",
        pretrained="openai",
        manifest_sha256="manifest",
    )

    limited = _limited_reference(reference, 2, single_class_label=1)
    assert limited.labels.tolist() == [1, 1]


def test_single_class_modes_reject_the_opposite_label() -> None:
    real = FeatureCache(
        features=np.zeros((2, 512), dtype=np.float32),
        labels=np.asarray([0, 0], dtype=np.int64),
        image_paths=np.asarray(["a.jpg", "b.jpg"]),
        split="train",
        model_name="ViT-B-32-quickgelu",
        pretrained="openai",
        manifest_sha256="manifest",
    )
    ai = FeatureCache(
        features=np.zeros((2, 512), dtype=np.float32),
        labels=np.asarray([1, 1], dtype=np.int64),
        image_paths=np.asarray(["a.jpg", "b.jpg"]),
        split="train",
        model_name="ViT-B-32-quickgelu",
        pretrained="openai",
        manifest_sha256="manifest",
    )

    with pytest.raises(ValueError, match="only label 1"):
        _limited_reference(real, 2, single_class_label=1)
    with pytest.raises(ValueError, match="only label 0"):
        _limited_reference(ai, 2, single_class_label=0)


def test_single_class_mode_rejects_invalid_label_configuration() -> None:
    reference = FeatureCache(
        features=np.zeros((1, 512), dtype=np.float32),
        labels=np.asarray([1], dtype=np.int64),
        image_paths=np.asarray(["a.jpg"]),
        split="train",
        model_name="ViT-B-32-quickgelu",
        pretrained="openai",
        manifest_sha256="manifest",
    )
    with pytest.raises(ValueError, match="None, 0, or 1"):
        _limited_reference(reference, 1, single_class_label=2)
