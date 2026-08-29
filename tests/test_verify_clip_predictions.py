"""Tests for strict prediction-JSON and cached-probability verification."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.extract_clip_features import atomic_feature_cache_write
from scripts.verify_clip_predictions import load_prediction_json, verify_predictions
from src.linear_probe import LinearProbeCheckpoint, save_linear_probe_checkpoint


def _reference_artifacts(tmp_path: Path) -> tuple[Path, Path, list[dict[str, object]]]:
    cache_path = tmp_path / "test.npz"
    features = np.zeros((2, 512), dtype=np.float32)
    features[0, 0] = 1.0
    features[1, 1] = 1.0
    paths = ["real.jpg", "fake.jpg"]
    atomic_feature_cache_write(
        cache_path,
        features=features,
        labels=np.asarray([0, 1], dtype=np.int64),
        image_paths=np.asarray(paths),
        split="test",
        model_name="ViT-B-32-quickgelu",
        pretrained="openai",
        manifest_sha256="test-manifest",
    )
    checkpoint_path = tmp_path / "linear.npz"
    coefficients = np.zeros(512, dtype=np.float64)
    coefficients[1] = 2.0
    checkpoint = LinearProbeCheckpoint(
        coefficients=coefficients,
        intercept=0.0,
        regularization_c=1.0,
        threshold=0.5,
        seed=42,
        selected_validation_roc_auc=0.9,
        train_cache_sha256="train",
        validation_cache_sha256="val",
    )
    save_linear_probe_checkpoint(checkpoint_path, checkpoint)
    probabilities = checkpoint.probabilities(features)
    predictions = [
        {"image_path": path, "pred": round(float(probability), 6)}
        for path, probability in zip(paths, probabilities, strict=True)
    ]
    return cache_path, checkpoint_path, predictions


def test_load_prediction_json_requires_exact_schema(tmp_path: Path) -> None:
    path = tmp_path / "predictions.json"
    path.write_text(
        json.dumps([{"image_path": "image.jpg", "pred": 0.5, "extra": 1}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly"):
        load_prediction_json(path)


def test_verify_predictions_matches_cached_probabilities(tmp_path: Path) -> None:
    cache_path, checkpoint_path, predictions = _reference_artifacts(tmp_path)

    result = verify_predictions(predictions, cache_path, checkpoint_path, tolerance=1e-5)

    assert result["status"] == "passed"
    assert result["json_predictions"] == 2
    assert result["reference_predictions_compared"] == 2
    assert result["maximum_absolute_probability_difference"] <= 1e-5


def test_verify_predictions_rejects_missing_reference_path(tmp_path: Path) -> None:
    cache_path, checkpoint_path, predictions = _reference_artifacts(tmp_path)

    with pytest.raises(ValueError, match="missing 1 reference paths"):
        verify_predictions(predictions[:1], cache_path, checkpoint_path, tolerance=1e-5)


def test_verify_predictions_rejects_probability_mismatch(tmp_path: Path) -> None:
    cache_path, checkpoint_path, predictions = _reference_artifacts(tmp_path)
    predictions[0]["pred"] = 0.9

    with pytest.raises(ValueError, match="differ from cached-model"):
        verify_predictions(predictions, cache_path, checkpoint_path, tolerance=1e-5)
