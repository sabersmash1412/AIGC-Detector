"""Shared frozen-OpenCLIP loading and image-feature encoding."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn


CLIP_MODEL_NAME = "ViT-B-32-quickgelu"
CLIP_PRETRAINED = "openai"
CLIP_FEATURE_DIMENSION = 512


def load_frozen_clip(
    device: torch.device,
    cache_dir: Path,
    *,
    model_name: str = CLIP_MODEL_NAME,
    pretrained: str = CLIP_PRETRAINED,
) -> tuple[nn.Module, Callable[[Image.Image], torch.Tensor]]:
    """Load OpenCLIP in evaluation mode with every parameter frozen."""

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        cache_dir=cache_dir,
    )
    model.requires_grad_(False)
    model.eval()
    model.to(device)

    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable_parameters:
        raise ValueError(
            f"CLIP must be frozen, but {trainable_parameters:,} parameters can train"
        )
    return model, preprocess


def encode_normalized_images(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Return finite, L2-normalized float32 CLIP image embeddings."""

    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"Expected image batch shape (N, 3, H, W), got {tuple(images.shape)}")

    with torch.inference_mode():
        features = model.encode_image(images)  # type: ignore[attr-defined]
        features = F.normalize(features.float(), dim=1)

    if features.ndim != 2 or features.shape[1] != CLIP_FEATURE_DIMENSION:
        raise ValueError(
            "Expected CLIP features with shape "
            f"(N, {CLIP_FEATURE_DIMENSION}), got {tuple(features.shape)}"
        )
    if not bool(torch.isfinite(features).all()):
        raise ValueError("CLIP features contain NaN or infinite values")
    return features
