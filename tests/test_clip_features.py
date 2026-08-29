"""Tests for frozen-CLIP encoding and feature-cache validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from scripts.extract_clip_features import (
    atomic_feature_cache_write,
    validate_feature_cache,
)
from src.clip_features import encode_normalized_images


class FakeClip(nn.Module):
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        features = torch.zeros(images.shape[0], 512, device=images.device)
        features[:, 0] = 3.0
        features[:, 1] = 4.0
        return features


def test_encode_normalized_images_returns_unit_float32_features() -> None:
    images = torch.zeros(3, 3, 8, 8)

    features = encode_normalized_images(FakeClip(), images)

    assert features.shape == (3, 512)
    assert features.dtype == torch.float32
    assert torch.allclose(torch.linalg.vector_norm(features, dim=1), torch.ones(3))
    assert torch.allclose(features[:, :2], torch.tensor([[0.6, 0.8]]).repeat(3, 1))


def test_encode_normalized_images_rejects_wrong_input_shape() -> None:
    with pytest.raises(ValueError, match="Expected image batch shape"):
        encode_normalized_images(FakeClip(), torch.zeros(3, 8, 8))


def test_round_trip_feature_cache_is_safe_and_valid(tmp_path: Path) -> None:
    cache_path = tmp_path / "train.npz"
    features = np.zeros((2, 512), dtype=np.float32)
    features[0, 0] = 1.0
    features[1, 1] = 1.0
    paths = ["real.jpg", "fake.jpg"]
    labels = [0, 1]

    atomic_feature_cache_write(
        cache_path,
        features=features,
        labels=np.asarray(labels, dtype=np.int64),
        image_paths=np.asarray(paths),
        split="train",
        model_name="ViT-B-32-quickgelu",
        pretrained="openai",
        manifest_sha256="abc123",
    )
    result = validate_feature_cache(
        cache_path,
        expected_split="train",
        expected_manifest_sha256="abc123",
        expected_paths=paths,
        expected_labels=labels,
    )

    assert result["samples"] == 2
    assert result["feature_shape"] == [2, 512]
    assert result["class_counts"] == {"real_0": 1, "ai_generated_1": 1}
    assert result["maximum_l2_norm_error"] == 0.0


def test_feature_cache_rejects_manifest_mismatch(tmp_path: Path) -> None:
    cache_path = tmp_path / "val.npz"
    features = np.zeros((1, 512), dtype=np.float32)
    features[0, 0] = 1.0
    atomic_feature_cache_write(
        cache_path,
        features=features,
        labels=np.asarray([0], dtype=np.int64),
        image_paths=np.asarray(["real.jpg"]),
        split="val",
        model_name="ViT-B-32-quickgelu",
        pretrained="openai",
        manifest_sha256="original",
    )

    with pytest.raises(ValueError, match="different manifest"):
        validate_feature_cache(
            cache_path,
            expected_split="val",
            expected_manifest_sha256="changed",
            expected_paths=["real.jpg"],
            expected_labels=[0],
        )


def test_feature_cache_rejects_unnormalized_embeddings(tmp_path: Path) -> None:
    cache_path = tmp_path / "test.npz"
    atomic_feature_cache_write(
        cache_path,
        features=np.ones((1, 512), dtype=np.float32),
        labels=np.asarray([1], dtype=np.int64),
        image_paths=np.asarray(["fake.jpg"]),
        split="test",
        model_name="ViT-B-32-quickgelu",
        pretrained="openai",
        manifest_sha256="abc123",
    )

    with pytest.raises(ValueError, match="not L2-normalized"):
        validate_feature_cache(
            cache_path,
            expected_split="test",
            expected_manifest_sha256="abc123",
            expected_paths=["fake.jpg"],
            expected_labels=[1],
        )
