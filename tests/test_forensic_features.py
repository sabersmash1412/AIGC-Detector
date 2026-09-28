from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.forensic_features import (
    DEFAULT_SPEC_PATH,
    FORENSIC_EXTRACTOR_ID,
    FORENSIC_FEATURE_DIMENSION,
    FORENSIC_STATUS,
    MAX_INPUT_PIXELS,
    ForensicFeatureError,
    extract_forensic_features,
    forensic_spec_sha256,
    inspect_forensic_file,
    load_forensic_spec,
    validate_forensic_spec,
    write_forensic_report,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / DEFAULT_SPEC_PATH


@pytest.fixture(scope="module")
def spec() -> dict:
    return load_forensic_spec(SPEC_PATH)


def _grayscale_array(values: np.ndarray) -> Image.Image:
    pixels = np.clip(np.rint(values * 255.0), 0, 255).astype(np.uint8)
    return Image.fromarray(pixels, mode="L").convert("RGB")


def _patterned_image(size: int = 128) -> Image.Image:
    y, x = np.mgrid[:size, :size]
    red = (3 * x + 5 * y) % 256
    green = (7 * x + 2 * y + 31) % 256
    blue = (11 * x + 13 * y + 17) % 256
    return Image.fromarray(
        np.stack([red, green, blue], axis=2).astype(np.uint8), mode="RGB"
    )


def test_checked_in_spec_fixes_unique_32d_float32_contract(spec: dict) -> None:
    names = validate_forensic_spec(spec)

    assert spec["extractor_id"] == FORENSIC_EXTRACTOR_ID
    assert spec["status"] == FORENSIC_STATUS
    assert len(names) == FORENSIC_FEATURE_DIMENSION == 32
    assert len(set(names)) == len(names)
    assert names[:2] == (
        "luminance_residual_radial_log_power_band_00",
        "luminance_residual_radial_log_power_band_01",
    )
    assert names[-2:] == (
        "horizontal_neighbor_correlation",
        "vertical_neighbor_correlation",
    )
    assert len(forensic_spec_sha256(spec)) == 64
    assert not any(
        forbidden in name
        for name in names
        for forbidden in ("width", "height", "aspect", "codec", "format", "exif")
    )


def test_spec_drift_is_rejected(spec: dict) -> None:
    modified = copy.deepcopy(spec)
    modified["output"]["prediction"] = "ai_generated"

    with pytest.raises(ForensicFeatureError, match="must not contain a prediction"):
        validate_forensic_spec(modified)


def test_load_spec_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(ForensicFeatureError, match="Could not load"):
        load_forensic_spec(path)


def test_extraction_is_deterministic_finite_read_only_float32(spec: dict) -> None:
    image = _patterned_image()

    first = extract_forensic_features(image, spec)
    second = extract_forensic_features(image, spec)

    assert first.values.shape == (FORENSIC_FEATURE_DIMENSION,)
    assert first.values.dtype == np.float32
    assert np.isfinite(first.values).all()
    assert np.array_equal(first.values, second.values)
    assert first.feature_names == second.feature_names
    assert first.values.flags.writeable is False
    assert first.original_size == first.analysis_size == (128, 128)
    assert first.downscaled is False


@pytest.mark.parametrize("mode", ["L", "RGBA"])
def test_grayscale_and_rgba_inputs_are_supported(spec: dict, mode: str) -> None:
    base = _patterned_image(64)
    image = base.convert(mode)

    result = extract_forensic_features(image, spec)

    assert result.values.shape == (32,)
    assert np.isfinite(result.values).all()


def test_constant_image_has_zero_profile_and_safe_correlations(spec: dict) -> None:
    image = Image.new("RGB", (64, 64), color=(127, 127, 127))

    result = extract_forensic_features(image, spec)

    assert np.allclose(result.values, 0.0, rtol=0.0, atol=1e-12)


def test_known_sinusoid_peaks_in_expected_radial_band(spec: dict) -> None:
    size = 256
    cycles = 32
    x = np.arange(size, dtype=np.float64)
    row = 0.5 + 0.45 * np.sin(2.0 * np.pi * cycles * x / size)
    image = _grayscale_array(np.repeat(row[None, :], size, axis=0))

    result = extract_forensic_features(image, spec)
    radial = result.values[:24]

    # 32/256 = 0.125 cycles/pixel, on the boundary of frozen bands 5 and 6.
    assert int(np.argmax(radial)) in {5, 6}


def test_high_frequency_stripes_have_more_residual_energy_than_gradient(
    spec: dict,
) -> None:
    size = 128
    gradient = np.repeat(
        np.linspace(0.0, 1.0, size, dtype=np.float64)[None, :], size, axis=0
    )
    stripes = np.repeat((np.arange(size) % 2)[None, :], size, axis=0).astype(
        np.float64
    )
    smooth_result = extract_forensic_features(_grayscale_array(gradient), spec)
    stripe_result = extract_forensic_features(_grayscale_array(stripes), spec)
    rms_index = smooth_result.feature_names.index("residual_rms")

    assert stripe_result.values[rms_index] > 20.0 * smooth_result.values[rms_index]
    assert stripe_result.values[23] > smooth_result.values[23]


def test_transpose_preserves_radial_profile_and_swaps_neighbor_correlations(
    spec: dict,
) -> None:
    image = _patterned_image(128)
    transposed = image.transpose(Image.Transpose.TRANSPOSE)

    original = extract_forensic_features(image, spec)
    rotated = extract_forensic_features(transposed, spec)
    horizontal = original.feature_names.index("horizontal_neighbor_correlation")
    vertical = original.feature_names.index("vertical_neighbor_correlation")

    assert np.allclose(original.values[:24], rotated.values[:24], atol=2e-5)
    assert np.isclose(original.values[horizontal], rotated.values[vertical], atol=2e-5)
    assert np.isclose(original.values[vertical], rotated.values[horizontal], atol=2e-5)


def test_small_images_are_not_upscaled_and_oversize_images_are_bounded(
    spec: dict,
) -> None:
    small = extract_forensic_features(_patterned_image(32), spec)
    wide_array = np.resize(np.asarray(_patterned_image(64)), (80, 1100, 3))
    wide = extract_forensic_features(
        Image.fromarray(wide_array.astype(np.uint8), mode="RGB"), spec
    )

    assert small.original_size == small.analysis_size == (32, 32)
    assert small.downscaled is False
    assert wide.original_size == (1100, 80)
    assert wide.analysis_size == (1024, 74)
    assert wide.downscaled is True


def test_tiny_image_is_rejected(spec: dict) -> None:
    with pytest.raises(ForensicFeatureError, match="at least 32"):
        extract_forensic_features(Image.new("RGB", (31, 64)), spec)


def test_oversize_header_is_rejected_before_pixel_decode(
    spec: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = Image.new("RGB", (32, 32))
    image._size = (MAX_INPUT_PIXELS + 1, 1)

    def fail_if_loaded() -> None:
        raise AssertionError("oversize image pixels must not be decoded")

    monkeypatch.setattr(image, "load", fail_if_loaded)

    with pytest.raises(ForensicFeatureError, match="Image is too large"):
        extract_forensic_features(image, spec)


def test_exif_orientation_is_applied_before_analysis(spec: dict) -> None:
    image = _patterned_image(64).resize((64, 96))
    exif = image.getexif()
    exif[274] = 6
    image.info["exif"] = exif.tobytes()

    result = extract_forensic_features(image, spec)

    assert result.original_size == (96, 64)


def test_lossless_container_metadata_does_not_change_features(
    tmp_path: Path, spec: dict
) -> None:
    image = _patterned_image(96)
    png = tmp_path / "same-pixels.png"
    bmp = tmp_path / "different-name.bmp"
    image.save(png)
    image.save(bmp)

    with Image.open(png) as png_image, Image.open(bmp) as bmp_image:
        png_result = extract_forensic_features(png_image, spec)
        bmp_result = extract_forensic_features(bmp_image, spec)

    assert np.array_equal(png_result.values, bmp_result.values)


def test_inspection_report_is_hashed_and_explicitly_non_predictive(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "input.png"
    _patterned_image(64).save(image_path)

    report = inspect_forensic_file(image_path, spec_path=SPEC_PATH)

    assert report["extractor"]["id"] == FORENSIC_EXTRACTOR_ID
    assert report["extractor"]["dimension"] == 32
    assert report["extractor"]["status"] == FORENSIC_STATUS
    assert report["extractor"]["prediction"] is None
    assert len(report["file"]["sha256"]) == 64
    assert report["forensic_features"]["prediction"] is None
    assert len(report["forensic_features"]["values"]) == 32
    assert len(report["forensic_features"]["named_values"]) == 32


def test_malformed_image_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "pretend.png"
    path.write_text("not an image", encoding="utf-8")

    with pytest.raises(ForensicFeatureError, match="Could not decode"):
        inspect_forensic_file(path, spec_path=SPEC_PATH)


def test_report_write_is_atomic_and_cannot_replace_source(tmp_path: Path) -> None:
    source = tmp_path / "input.png"
    _patterned_image(64).save(source)
    report = inspect_forensic_file(source, spec_path=SPEC_PATH)
    output = tmp_path / "reports" / "forensic.json"

    write_forensic_report(report, output, source_path=source)

    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert output.read_text(encoding="utf-8").endswith("\n")
    assert not list(output.parent.glob(".forensic.json.*.tmp"))
    original = source.read_bytes()
    with pytest.raises(ForensicFeatureError, match="must not replace"):
        write_forensic_report(report, source, source_path=source)
    assert source.read_bytes() == original
