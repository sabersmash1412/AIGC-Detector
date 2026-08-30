"""Deterministic image transformations for robustness evaluation."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageFilter
from torchvision.transforms import functional as transform_functional


@dataclass(frozen=True)
class EvaluationTransformSpec:
    """JSON-safe identity and parameters for one evaluation condition."""

    name: str
    display_name: str
    parameters: dict[str, Any]


TRANSFORM_SPECS = {
    "clean": EvaluationTransformSpec(
        name="clean",
        display_name="Clean",
        parameters={},
    ),
    "jpeg_q90": EvaluationTransformSpec(
        name="jpeg_q90",
        display_name="JPEG Q90",
        parameters={"quality": 90, "subsampling": 2},
    ),
    "jpeg_q70": EvaluationTransformSpec(
        name="jpeg_q70",
        display_name="JPEG Q70",
        parameters={"quality": 70, "subsampling": 2},
    ),
    "jpeg_q50": EvaluationTransformSpec(
        name="jpeg_q50",
        display_name="JPEG Q50",
        parameters={"quality": 50, "subsampling": 2},
    ),
    "jpeg_q30": EvaluationTransformSpec(
        name="jpeg_q30",
        display_name="JPEG Q30",
        parameters={"quality": 30, "subsampling": 2},
    ),
    "gaussian_blur_sigma0_5": EvaluationTransformSpec(
        name="gaussian_blur_sigma0_5",
        display_name="Blur σ=0.5",
        parameters={"sigma": 0.5},
    ),
    "gaussian_blur_sigma1": EvaluationTransformSpec(
        name="gaussian_blur_sigma1",
        display_name="Blur σ=1",
        parameters={"sigma": 1.0},
    ),
    "gaussian_blur_sigma2": EvaluationTransformSpec(
        name="gaussian_blur_sigma2",
        display_name="Blur σ=2",
        parameters={"sigma": 2.0},
    ),
    "resize_0_5x": EvaluationTransformSpec(
        name="resize_0_5x",
        display_name="Resize 0.5×",
        parameters={"downscale_factor": 0.5, "upscale_to_original": True},
    ),
    "resize_0_25x": EvaluationTransformSpec(
        name="resize_0_25x",
        display_name="Resize 0.25×",
        parameters={"downscale_factor": 0.25, "upscale_to_original": True},
    ),
    "gaussian_noise_sigma0_02": EvaluationTransformSpec(
        name="gaussian_noise_sigma0_02",
        display_name="Noise σ=0.02",
        parameters={"sigma": 0.02, "pixel_range": [0.0, 1.0]},
    ),
    "gaussian_noise_sigma0_05": EvaluationTransformSpec(
        name="gaussian_noise_sigma0_05",
        display_name="Noise σ=0.05",
        parameters={"sigma": 0.05, "pixel_range": [0.0, 1.0]},
    ),
    "gaussian_noise_sigma0_10": EvaluationTransformSpec(
        name="gaussian_noise_sigma0_10",
        display_name="Noise σ=0.10",
        parameters={"sigma": 0.10, "pixel_range": [0.0, 1.0]},
    ),
    "color_jitter_seeded_20pct": EvaluationTransformSpec(
        name="color_jitter_seeded_20pct",
        display_name="Colour jitter ±20%",
        parameters={
            "brightness_factor_range": [0.8, 1.2],
            "contrast_factor_range": [0.8, 1.2],
            "saturation_factor_range": [0.8, 1.2],
            "factor_and_operation_order": "deterministic per image path",
        },
    ),
    "center_crop_80pct": EvaluationTransformSpec(
        name="center_crop_80pct",
        display_name="Centre crop 80%",
        parameters={"retained_fraction": 0.8, "resize_to_original": True},
    ),
}

FULL_ROBUSTNESS_CONDITIONS = tuple(TRANSFORM_SPECS)
DEFAULT_ROBUSTNESS_CONDITIONS = (
    "clean",
    "jpeg_q50",
    "gaussian_blur_sigma1",
    "resize_0_5x",
    "gaussian_noise_sigma0_05",
    "color_jitter_seeded_20pct",
    "center_crop_80pct",
)


def _stable_rng(image_path: str, seed: int, salt: str) -> np.random.Generator:
    payload = f"{seed}:{salt}:{image_path}".encode("utf-8")
    stable_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return np.random.default_rng(stable_seed)


def _jpeg(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(
        buffer,
        format="JPEG",
        quality=quality,
        subsampling=2,
        optimize=False,
        progressive=False,
    )
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB").copy()


def _resize_round_trip(image: Image.Image, factor: float) -> Image.Image:
    original_size = image.size
    downsampled_size = tuple(max(1, round(dimension * factor)) for dimension in original_size)
    downsampled = image.resize(downsampled_size, resample=Image.Resampling.BICUBIC)
    return downsampled.resize(original_size, resample=Image.Resampling.BICUBIC)


def _gaussian_noise(
    image: Image.Image, image_path: str, seed: int, sigma: float
) -> Image.Image:
    array = np.asarray(image, dtype=np.float32) / 255.0
    rng = _stable_rng(image_path, seed, "gaussian_noise")
    noise = rng.normal(0.0, sigma, size=array.shape).astype(np.float32)
    transformed = np.clip(array + noise, 0.0, 1.0)
    return Image.fromarray(np.rint(transformed * 255.0).astype(np.uint8), mode="RGB")


def _seeded_color_jitter(image: Image.Image, image_path: str, seed: int) -> Image.Image:
    rng = _stable_rng(image_path, seed, "color_jitter")
    operations = [
        (transform_functional.adjust_brightness, float(rng.uniform(0.8, 1.2))),
        (transform_functional.adjust_contrast, float(rng.uniform(0.8, 1.2))),
        (transform_functional.adjust_saturation, float(rng.uniform(0.8, 1.2))),
    ]
    transformed = image
    for operation_index in rng.permutation(len(operations)):
        operation, factor = operations[int(operation_index)]
        transformed = operation(transformed, factor)
    return transformed.convert("RGB")


def _center_crop_round_trip(image: Image.Image, retained_fraction: float) -> Image.Image:
    original_width, original_height = image.size
    crop_width = max(1, round(original_width * retained_fraction))
    crop_height = max(1, round(original_height * retained_fraction))
    left = (original_width - crop_width) // 2
    top = (original_height - crop_height) // 2
    cropped = image.crop((left, top, left + crop_width, top + crop_height))
    return cropped.resize(image.size, resample=Image.Resampling.BICUBIC)


def apply_evaluation_transform(
    image: Image.Image,
    condition: str,
    *,
    image_path: str,
    seed: int = 42,
) -> Image.Image:
    """Apply one named transform deterministically and preserve RGB dimensions."""

    if condition not in TRANSFORM_SPECS:
        raise ValueError(
            f"Unknown transform condition {condition!r}; choose from {tuple(TRANSFORM_SPECS)}"
        )
    rgb_image = image.convert("RGB")
    parameters = TRANSFORM_SPECS[condition].parameters
    if condition == "clean":
        transformed = rgb_image.copy()
    elif condition.startswith("jpeg_q"):
        transformed = _jpeg(rgb_image, quality=int(parameters["quality"]))
    elif condition.startswith("gaussian_blur_sigma"):
        transformed = rgb_image.filter(
            ImageFilter.GaussianBlur(radius=float(parameters["sigma"]))
        )
    elif condition.startswith("resize_"):
        transformed = _resize_round_trip(
            rgb_image, factor=float(parameters["downscale_factor"])
        )
    elif condition.startswith("gaussian_noise_sigma"):
        transformed = _gaussian_noise(
            rgb_image, image_path, seed, sigma=float(parameters["sigma"])
        )
    elif condition == "color_jitter_seeded_20pct":
        transformed = _seeded_color_jitter(rgb_image, image_path, seed)
    elif condition == "center_crop_80pct":
        transformed = _center_crop_round_trip(
            rgb_image, retained_fraction=float(parameters["retained_fraction"])
        )
    else:  # pragma: no cover - exhaustive guard for future registry edits
        raise AssertionError(f"Transform is registered but not implemented: {condition}")

    if transformed.mode != "RGB" or transformed.size != rgb_image.size:
        raise ValueError(
            f"Transform {condition} changed mode/size unexpectedly: "
            f"{transformed.mode}, {transformed.size}"
        )
    return transformed
