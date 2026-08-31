"""Validation tests for the fixed AIGIBench transformed-feature matrix."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.extract_clip_features import sha256_file
from scripts.extract_e5_aigibench_transformed_clip_features import (
    FROZEN_CONDITIONS,
    validate_clean_reference,
)
from src.image_transforms import FULL_ROBUSTNESS_CONDITIONS


def _summary(cache_sha256: str, manifest_sha256: str) -> dict:
    return {
        "amendment": {"sha256": "amendment-hash"},
        "preparation_validation": {"manifest_sha256": manifest_sha256},
        "feature_cache": {
            "cache_sha256": cache_sha256,
            "manifest_sha256": manifest_sha256,
            "feature_shape": [2000, 512],
            "class_counts": {"real_0": 1000, "ai_generated_1": 1000},
        },
        "frozen_guardrails": {
            "e5_checkpoint_loaded": False,
            "classifier_training_performed": False,
            "threshold_selection_performed": False,
            "model_selection_performed": False,
            "predictions_or_metrics_computed": False,
            "organiser_validation_subset_used": False,
        },
    }


def test_frozen_conditions_are_the_complete_nonclean_matrix() -> None:
    assert FROZEN_CONDITIONS == tuple(
        condition for condition in FULL_ROBUSTNESS_CONDITIONS if condition != "clean"
    )
    assert len(FROZEN_CONDITIONS) == 14


def test_clean_reference_rejects_observed_predictions(tmp_path: Path) -> None:
    amendment_path = tmp_path / "amendment.json"
    amendment_path.write_text("{}\n", encoding="utf-8")
    cache_path = tmp_path / "clean.npz"
    cache_path.write_bytes(b"cache")
    summary = _summary("unused", "manifest")
    summary["amendment"]["sha256"] = sha256_file(amendment_path)
    summary["feature_cache"]["cache_sha256"] = sha256_file(cache_path)
    summary["frozen_guardrails"]["predictions_or_metrics_computed"] = True
    with pytest.raises(ValueError, match="guardrail failed"):
        validate_clean_reference(
            amendment={"candidate_population": {"selection_seed": 42}},
            amendment_path=amendment_path,
            clean_summary=summary,
            clean_cache_path=cache_path,
        )


def test_clean_reference_rejects_class_count_drift(tmp_path: Path) -> None:
    amendment_path = tmp_path / "amendment.json"
    amendment_path.write_text("{}\n", encoding="utf-8")
    cache_path = tmp_path / "clean.npz"
    cache_path.write_bytes(b"cache")
    summary = _summary(sha256_file(cache_path), "manifest")
    summary["amendment"]["sha256"] = sha256_file(amendment_path)
    summary["feature_cache"]["class_counts"] = {
        "real_0": 900,
        "ai_generated_1": 1100,
    }
    with pytest.raises(ValueError, match="class counts changed"):
        validate_clean_reference(
            amendment={"candidate_population": {"selection_seed": 42}},
            amendment_path=amendment_path,
            clean_summary=summary,
            clean_cache_path=cache_path,
        )
