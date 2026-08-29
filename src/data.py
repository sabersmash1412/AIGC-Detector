"""Shared image preprocessing for the CIFAKE smoke-test pipeline."""

from __future__ import annotations

from collections.abc import Sequence

from torchvision import transforms


DEFAULT_IMAGE_SIZE = 32
DEFAULT_MEAN = (0.5, 0.5, 0.5)
DEFAULT_STD = (0.5, 0.5, 0.5)


def build_image_transform(
    image_size: int = DEFAULT_IMAGE_SIZE,
    mean: Sequence[float] = DEFAULT_MEAN,
    std: Sequence[float] = DEFAULT_STD,
) -> transforms.Compose:
    """Build deterministic RGB preprocessing shared by training and inference."""

    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if len(mean) != 3 or len(std) != 3:
        raise ValueError("mean and std must each contain three RGB values")
    if any(value <= 0 for value in std):
        raise ValueError("every standard-deviation value must be positive")

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=tuple(mean), std=tuple(std)),
        ]
    )


def preprocessing_config(
    image_size: int = DEFAULT_IMAGE_SIZE,
    mean: Sequence[float] = DEFAULT_MEAN,
    std: Sequence[float] = DEFAULT_STD,
) -> dict[str, object]:
    """Return JSON/checkpoint-safe preprocessing metadata."""

    return {
        "image_size": int(image_size),
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
        "colour_mode": "RGB",
    }
