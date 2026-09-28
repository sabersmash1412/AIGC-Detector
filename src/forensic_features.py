"""Deterministic E6 high-pass and radial-frequency image features.

This module intentionally contains no classifier, threshold, or real-versus-AI
decision.  It turns decoded pixels into the fixed descriptor frozen by
``configs/e6_forensic_features.json`` so that a later, separately reviewed E6
training stage can consume the features without changing their definition.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter


DEFAULT_SPEC_PATH = Path("configs/e6_forensic_features.json")
MAX_INPUT_PIXELS = 50_000_000

FORENSIC_SPEC_SCHEMA_VERSION = 1
FORENSIC_FEATURE_VERSION = 1
FORENSIC_EXTRACTOR_ID = "e6_forensic_v1_32d"
FORENSIC_STATUS = "features_only_untrained"
FORENSIC_FEATURE_DIMENSION = 32
RADIAL_FEATURE_PREFIX = "luminance_residual_radial_log_power_band"
SPATIAL_STATISTICS = (
    "residual_rms",
    "absolute_residual_q50",
    "absolute_residual_q75",
    "absolute_residual_q90",
    "absolute_residual_q95",
    "absolute_residual_q99",
    "horizontal_neighbor_correlation",
    "vertical_neighbor_correlation",
)


class ForensicFeatureError(ValueError):
    """Raised when an image or frozen feature specification is invalid."""


@dataclass(frozen=True)
class ForensicFeatureResult:
    """One fixed forensic descriptor and non-predictive analysis diagnostics."""

    values: np.ndarray
    feature_names: tuple[str, ...]
    original_size: tuple[int, int]
    analysis_size: tuple[int, int]
    downscaled: bool


def _require_mapping(container: dict[str, Any], key: str) -> dict[str, Any]:
    value = container.get(key)
    if not isinstance(value, dict):
        raise ForensicFeatureError(f"Forensic spec {key!r} must be an object")
    return value


def _require_finite_number(container: dict[str, Any], key: str) -> float:
    value = container.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ForensicFeatureError(f"Forensic spec {key!r} must be numeric")
    number = float(value)
    if not np.isfinite(number):
        raise ForensicFeatureError(f"Forensic spec {key!r} must be finite")
    return number


def _feature_names(radial_bands: int) -> tuple[str, ...]:
    width = max(2, len(str(radial_bands - 1)))
    radial = tuple(
        f"{RADIAL_FEATURE_PREFIX}_{index:0{width}d}"
        for index in range(radial_bands)
    )
    return radial + SPATIAL_STATISTICS


def validate_forensic_spec(spec: dict[str, Any]) -> tuple[str, ...]:
    """Validate the frozen extractor contract and return its feature names."""

    if not isinstance(spec, dict):
        raise ForensicFeatureError("Forensic spec must be a JSON object")
    if spec.get("schema_version") != FORENSIC_SPEC_SCHEMA_VERSION:
        raise ForensicFeatureError("Unsupported forensic spec schema_version")
    if spec.get("extractor_id") != FORENSIC_EXTRACTOR_ID:
        raise ForensicFeatureError("Unsupported forensic extractor_id")
    if spec.get("status") != FORENSIC_STATUS:
        raise ForensicFeatureError("Forensic spec must remain features-only and untrained")

    input_spec = _require_mapping(spec, "input")
    fixed_input_values = {
        "exif_orientation": "transpose",
        "rgb_conversion": "PIL_RGB",
        "luminance_standard": "BT.601",
        "value_range": [0.0, 1.0],
        "oversize_resample": "lanczos",
        "upscale_small_images": False,
    }
    for key, expected in fixed_input_values.items():
        if input_spec.get(key) != expected:
            raise ForensicFeatureError(f"Unsupported forensic input setting: {key}")

    weights = input_spec.get("luminance_weights")
    if (
        not isinstance(weights, list)
        or len(weights) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(float(value))
            or float(value) < 0.0
            for value in weights
        )
        or not np.isclose(sum(float(value) for value in weights), 1.0, atol=1e-12)
    ):
        raise ForensicFeatureError(
            "Forensic luminance_weights must be three non-negative finite values summing to 1"
        )

    minimum_side = input_spec.get("minimum_side_pixels")
    maximum_side = input_spec.get("maximum_analysis_side_pixels")
    if isinstance(minimum_side, bool) or not isinstance(minimum_side, int):
        raise ForensicFeatureError("minimum_side_pixels must be an integer")
    if isinstance(maximum_side, bool) or not isinstance(maximum_side, int):
        raise ForensicFeatureError("maximum_analysis_side_pixels must be an integer")
    if minimum_side < 2 or maximum_side < minimum_side:
        raise ForensicFeatureError("Forensic analysis side limits are invalid")

    residual_spec = _require_mapping(spec, "residual")
    if residual_spec.get("method") != "input_minus_gaussian_blur":
        raise ForensicFeatureError("Unsupported forensic residual method")
    if residual_spec.get("boundary_mode") != "reflect":
        raise ForensicFeatureError("Unsupported forensic residual boundary mode")
    if _require_finite_number(residual_spec, "gaussian_sigma") <= 0.0:
        raise ForensicFeatureError("Forensic gaussian_sigma must be positive")

    spectrum_spec = _require_mapping(spec, "spectrum")
    fixed_spectrum_values = {
        "window": "separable_hann",
        "window_rms_normalized": True,
        "fft_normalization": "ortho",
        "exclude_dc": True,
        "band_value": (
            "mean_power_divided_by_global_valid_frequency_mean_then_log10"
        ),
        "zero_residual_profile": "all_zero",
    }
    for key, expected in fixed_spectrum_values.items():
        if spectrum_spec.get(key) != expected:
            raise ForensicFeatureError(f"Unsupported forensic spectrum setting: {key}")

    radial_min = _require_finite_number(
        spectrum_spec, "radial_min_cycles_per_pixel"
    )
    radial_max = _require_finite_number(
        spectrum_spec, "radial_max_cycles_per_pixel"
    )
    radial_bands = spectrum_spec.get("radial_bands")
    epsilon = _require_finite_number(spectrum_spec, "log10_epsilon")
    if radial_min != 0.0 or radial_max != 0.5:
        raise ForensicFeatureError("Frozen radial frequency range must be (0, 0.5]")
    if isinstance(radial_bands, bool) or not isinstance(radial_bands, int):
        raise ForensicFeatureError("radial_bands must be an integer")
    if radial_bands <= 0 or epsilon <= 0.0:
        raise ForensicFeatureError("Forensic radial_bands and epsilon must be positive")

    spatial_spec = _require_mapping(spec, "spatial")
    if spatial_spec.get("statistics") != list(SPATIAL_STATISTICS):
        raise ForensicFeatureError("Frozen forensic spatial statistics changed")
    if spatial_spec.get("constant_signal_correlation") != 0.0:
        raise ForensicFeatureError("Constant-signal correlation must remain zero")

    output_spec = _require_mapping(spec, "output")
    names = _feature_names(radial_bands)
    if output_spec.get("dimension") != len(names):
        raise ForensicFeatureError("Forensic output dimension does not match its features")
    if output_spec.get("dimension") != FORENSIC_FEATURE_DIMENSION:
        raise ForensicFeatureError("Unsupported forensic output dimension")
    if output_spec.get("dtype") != "float32":
        raise ForensicFeatureError("Forensic output dtype must remain float32")
    if output_spec.get("prediction") is not None:
        raise ForensicFeatureError("E6C forensic extraction must not contain a prediction")

    guardrails = _require_mapping(spec, "guardrails")
    for key in (
        "labels_used_by_extractor",
        "file_metadata_in_feature_vector",
        "dimensions_in_feature_vector",
        "c2pa_in_feature_vector",
        "trained_classifier_present",
        "consumed_tests_permitted_during_e6c",
        "lockbox_access_permitted_during_e6c",
    ):
        if guardrails.get(key) is not False:
            raise ForensicFeatureError(f"Forensic guardrail changed: {key}")
    if guardrails.get("permitted_data_stage") != "approved_development_data_only":
        raise ForensicFeatureError("Forensic permitted_data_stage changed")

    limitations = spec.get("limitations")
    if not isinstance(limitations, list) or not limitations or not all(
        isinstance(item, str) and item for item in limitations
    ):
        raise ForensicFeatureError("Forensic spec limitations must be non-empty text")
    if len(names) != len(set(names)):
        raise ForensicFeatureError("Forensic feature names must be unique")
    return names


def load_forensic_spec(path: Path = DEFAULT_SPEC_PATH) -> dict[str, Any]:
    """Load and validate a JSON forensic feature specification."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ForensicFeatureError(f"Could not load forensic spec: {path}") from exc
    if not isinstance(payload, dict):
        raise ForensicFeatureError("Forensic spec must be a JSON object")
    validate_forensic_spec(payload)
    return payload


def canonical_forensic_spec_json(spec: dict[str, Any]) -> str:
    """Return stable JSON for a validated extractor specification."""

    validate_forensic_spec(spec)
    return json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def forensic_spec_sha256(spec: dict[str, Any]) -> str:
    """Return the SHA-256 identity of a validated canonical specification."""

    return hashlib.sha256(canonical_forensic_spec_json(spec).encode("utf-8")).hexdigest()


def _prepare_luminance(
    image: Image.Image, spec: dict[str, Any]
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int], bool]:
    if not isinstance(image, Image.Image):
        raise ForensicFeatureError("Forensic input must be a Pillow image")

    # Pillow exposes container dimensions before decoding pixel storage.  Check
    # them first so a hostile oversized header cannot force a large allocation
    # merely because the caller requested forensic features.
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ForensicFeatureError("Image dimensions must be positive")
    if width * height > MAX_INPUT_PIXELS:
        raise ForensicFeatureError(
            f"Image is too large ({width}x{height}); maximum is {MAX_INPUT_PIXELS:,} pixels"
        )
    try:
        image.load()
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise ForensicFeatureError("Image data is incomplete or could not be decoded") from exc

    try:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise ForensicFeatureError("Image could not be converted to RGB safely") from exc

    input_spec = spec["input"]
    original_size = rgb.size
    minimum_side = int(input_spec["minimum_side_pixels"])
    if min(original_size) < minimum_side:
        raise ForensicFeatureError(
            f"Image minimum side must be at least {minimum_side} pixels"
        )

    maximum_side = int(input_spec["maximum_analysis_side_pixels"])
    downscaled = max(original_size) > maximum_side
    if downscaled:
        scale = maximum_side / max(original_size)
        analysis_size = tuple(max(1, round(dimension * scale)) for dimension in original_size)
        if min(analysis_size) < minimum_side:
            raise ForensicFeatureError(
                "Image aspect ratio is too extreme for the frozen analysis-size limit"
            )
        rgb = rgb.resize(analysis_size, resample=Image.Resampling.LANCZOS)
    else:
        analysis_size = original_size

    pixels = np.asarray(rgb, dtype=np.float64) / 255.0
    weights = np.asarray(input_spec["luminance_weights"], dtype=np.float64)
    luminance = np.tensordot(pixels, weights, axes=([2], [0]))
    if luminance.shape != (analysis_size[1], analysis_size[0]):
        raise ForensicFeatureError("Forensic luminance conversion produced the wrong shape")
    if not bool(np.isfinite(luminance).all()):
        raise ForensicFeatureError("Forensic luminance contains non-finite values")
    return luminance, original_size, analysis_size, downscaled


def _safe_neighbor_correlation(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=np.float64).reshape(-1)
    right = np.asarray(second, dtype=np.float64).reshape(-1)
    left = left - np.mean(left)
    right = right - np.mean(right)
    denominator = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
    if denominator <= np.finfo(np.float64).eps:
        return 0.0
    correlation = float(np.dot(left, right) / denominator)
    return float(np.clip(correlation, -1.0, 1.0))


def _radial_log_power(residual: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    spectrum_spec = spec["spectrum"]
    bands = int(spectrum_spec["radial_bands"])
    epsilon = float(spectrum_spec["log10_epsilon"])
    height, width = residual.shape

    window = np.outer(np.hanning(height), np.hanning(width))
    window_rms = float(np.sqrt(np.mean(np.square(window))))
    if not np.isfinite(window_rms) or window_rms <= 0.0:
        raise ForensicFeatureError("Could not construct the frozen Hann analysis window")
    window = window / window_rms
    power = np.square(np.abs(np.fft.fft2(residual * window, norm="ortho")))

    y_frequency = np.fft.fftfreq(height)
    x_frequency = np.fft.fftfreq(width)
    radius = np.hypot(y_frequency[:, None], x_frequency[None, :])
    radial_min = float(spectrum_spec["radial_min_cycles_per_pixel"])
    radial_max = float(spectrum_spec["radial_max_cycles_per_pixel"])
    valid = (radius > radial_min) & (radius <= radial_max)
    valid_power = power[valid]
    if len(valid_power) == 0:
        raise ForensicFeatureError("Image has no frequencies in the frozen radial range")
    if not bool(np.isfinite(valid_power).all()):
        raise ForensicFeatureError("Forensic power spectrum contains non-finite values")
    if not bool(np.any(valid_power > 0.0)):
        return np.zeros(bands, dtype=np.float64)

    global_mean = float(np.mean(valid_power))
    normalized_power = power / global_mean
    edges = np.linspace(radial_min, radial_max, bands + 1)
    values = np.zeros(bands, dtype=np.float64)
    for index in range(bands):
        lower = edges[index]
        upper = edges[index + 1]
        in_band = valid & (radius >= lower)
        if index == bands - 1:
            in_band &= radius <= upper
        else:
            in_band &= radius < upper
        if bool(np.any(in_band)):
            values[index] = float(
                np.log10(float(np.mean(normalized_power[in_band])) + epsilon)
            )
        else:
            # Native 32px inputs undersample the lowest of 24 frozen bands.
            # Zero is neutral in the normalized log-power representation and,
            # unlike interpolation, does not invent unobserved spectral energy.
            values[index] = 0.0
    return values


def extract_forensic_features(
    image: Image.Image, spec: dict[str, Any]
) -> ForensicFeatureResult:
    """Extract the frozen descriptor from decoded pixels without predicting a label."""

    feature_names = validate_forensic_spec(spec)
    luminance, original_size, analysis_size, downscaled = _prepare_luminance(
        image, spec
    )
    residual_spec = spec["residual"]
    blurred = gaussian_filter(
        luminance,
        sigma=float(residual_spec["gaussian_sigma"]),
        mode=str(residual_spec["boundary_mode"]),
    )
    residual = luminance - blurred
    absolute_residual = np.abs(residual)

    radial = _radial_log_power(residual, spec)
    quantiles = np.quantile(absolute_residual, [0.50, 0.75, 0.90, 0.95, 0.99])
    spatial = np.asarray(
        [
            float(np.sqrt(np.mean(np.square(residual)))),
            *[float(value) for value in quantiles],
            _safe_neighbor_correlation(residual[:, :-1], residual[:, 1:]),
            _safe_neighbor_correlation(residual[:-1, :], residual[1:, :]),
        ],
        dtype=np.float64,
    )
    values = np.concatenate([radial, spatial]).astype(np.float32)
    if values.shape != (len(feature_names),):
        raise ForensicFeatureError(
            f"Expected {len(feature_names)} forensic features, got {values.shape}"
        )
    if not bool(np.isfinite(values).all()):
        raise ForensicFeatureError("Forensic features contain NaN or infinite values")
    values.setflags(write=False)
    return ForensicFeatureResult(
        values=values,
        feature_names=feature_names,
        original_size=original_size,
        analysis_size=analysis_size,
        downscaled=downscaled,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_forensic_file(
    path: Path, spec_path: Path = DEFAULT_SPEC_PATH
) -> dict[str, Any]:
    """Inspect one image and return a JSON-safe, explicitly non-predictive report."""

    if not path.is_file():
        raise ForensicFeatureError(f"Image file not found: {path}")
    spec = load_forensic_spec(spec_path)
    try:
        with Image.open(path) as image:
            decoded_format = image.format
            decoded_mode = image.mode
            result = extract_forensic_features(image, spec)
    except ForensicFeatureError:
        raise
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise ForensicFeatureError(f"Could not decode image: {path}") from exc

    named_values = {
        name: float(value)
        for name, value in zip(
            result.feature_names, result.values.tolist(), strict=True
        )
    }
    status = str(spec["status"])
    prediction = spec["output"]["prediction"]
    extractor = {
        "id": str(spec["extractor_id"]),
        "version": FORENSIC_FEATURE_VERSION,
        "spec_path": spec_path.as_posix(),
        "spec_sha256": forensic_spec_sha256(spec),
        "dimension": len(result.feature_names),
        "dtype": str(result.values.dtype),
        "status": status,
        "prediction": prediction,
    }
    return {
        "schema_version": 1,
        "file": {
            "path": path.as_posix(),
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
            "decoded_format": decoded_format,
            "decoded_mode": decoded_mode,
        },
        "extractor": extractor,
        "forensic_features": {
            "status": status,
            "prediction": prediction,
            "original_size": {
                "width": result.original_size[0],
                "height": result.original_size[1],
            },
            "analysis_size": {
                "width": result.analysis_size[0],
                "height": result.analysis_size[1],
            },
            "downscaled": result.downscaled,
            "feature_names": list(result.feature_names),
            "values": [float(value) for value in result.values.tolist()],
            "named_values": named_values,
        },
        "limitations": list(spec["limitations"]),
    }


def write_forensic_report(
    report: dict[str, Any], output_path: Path, *, source_path: Path | None = None
) -> None:
    """Atomically write a report without ever replacing the inspected image."""

    if source_path is not None and output_path.resolve() == source_path.resolve():
        raise ForensicFeatureError("Forensic JSON report must not replace the input image")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
            temporary_path = Path(handle.name)
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
