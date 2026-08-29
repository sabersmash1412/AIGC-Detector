"""Tests for deterministic robustness-evaluation image transformations."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.image_transforms import (
    DEFAULT_ROBUSTNESS_CONDITIONS,
    apply_evaluation_transform,
)


def patterned_image() -> Image.Image:
    values = np.arange(32 * 32 * 3, dtype=np.uint16).reshape(32, 32, 3)
    return Image.fromarray((values % 256).astype(np.uint8), mode="RGB")


@pytest.mark.parametrize("condition", DEFAULT_ROBUSTNESS_CONDITIONS)
def test_evaluation_transforms_preserve_rgb_size(condition: str) -> None:
    image = patterned_image()

    transformed = apply_evaluation_transform(
        image, condition, image_path="example/image.jpg", seed=42
    )

    assert transformed.mode == "RGB"
    assert transformed.size == (32, 32)


@pytest.mark.parametrize(
    "condition", ["gaussian_noise_sigma0_05", "color_jitter_seeded_20pct"]
)
def test_seeded_transforms_are_repeatable(condition: str) -> None:
    image = patterned_image()

    first = apply_evaluation_transform(
        image, condition, image_path="same/path.jpg", seed=42
    )
    second = apply_evaluation_transform(
        image, condition, image_path="same/path.jpg", seed=42
    )

    assert np.array_equal(np.asarray(first), np.asarray(second))


def test_noise_changes_when_path_changes() -> None:
    image = patterned_image()
    first = apply_evaluation_transform(
        image, "gaussian_noise_sigma0_05", image_path="first.jpg", seed=42
    )
    second = apply_evaluation_transform(
        image, "gaussian_noise_sigma0_05", image_path="second.jpg", seed=42
    )

    assert not np.array_equal(np.asarray(first), np.asarray(second))


def test_clean_transform_preserves_pixels() -> None:
    image = patterned_image()
    clean = apply_evaluation_transform(
        image, "clean", image_path="image.jpg", seed=42
    )

    assert np.array_equal(np.asarray(clean), np.asarray(image))


def test_unknown_transform_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown transform condition"):
        apply_evaluation_transform(
            patterned_image(), "unknown", image_path="image.jpg", seed=42
        )
