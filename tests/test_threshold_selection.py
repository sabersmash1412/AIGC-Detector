"""Tests for validation-only Section 3 operating-threshold selection."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.linear_probe import (
    LinearProbeCheckpoint,
    load_linear_probe_checkpoint,
    save_linear_probe_checkpoint,
)
from src.threshold_selection import (
    select_balanced_accuracy_threshold,
    write_thresholded_checkpoint,
)


def test_threshold_selection_maximises_mean_balanced_accuracy() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    probabilities = {
        "clean": np.asarray([0.1, 0.4, 0.6, 0.9]),
        "jpeg_q50": np.asarray([0.2, 0.45, 0.55, 0.8]),
    }

    result = select_balanced_accuracy_threshold(
        labels, probabilities, np.asarray([0.4, 0.5, 0.6])
    )

    assert result.threshold == 0.5
    assert result.objective_score == 1.0
    assert result.selected_metrics["mean_balanced_accuracy"] == 1.0


def test_threshold_tie_break_prefers_closest_then_lower() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    probabilities = {
        "clean": np.asarray([0.1, 0.2, 0.8, 0.9]),
        "jpeg_q50": np.asarray([0.1, 0.2, 0.8, 0.9]),
    }

    result = select_balanced_accuracy_threshold(
        labels, probabilities, np.asarray([0.4, 0.6])
    )

    assert result.threshold == 0.4


def test_thresholded_checkpoint_preserves_weights_and_changes_threshold(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.npz"
    destination = tmp_path / "thresholded.npz"
    checkpoint = LinearProbeCheckpoint(
        coefficients=np.arange(512, dtype=np.float64),
        intercept=0.25,
        regularization_c=100.0,
        threshold=0.5,
        seed=42,
        selected_validation_roc_auc=0.9,
        train_cache_sha256="train",
        validation_cache_sha256="val",
    )
    save_linear_probe_checkpoint(source, checkpoint)

    write_thresholded_checkpoint(
        source,
        destination,
        threshold=0.321,
        objective_score=0.88,
        protocol_sha256="protocol",
        source_checkpoint_sha256="source",
    )

    loaded = load_linear_probe_checkpoint(destination)
    assert loaded.threshold == 0.321
    assert np.array_equal(loaded.coefficients, checkpoint.coefficients)
    assert loaded.intercept == checkpoint.intercept
    with np.load(destination, allow_pickle=False) as archive:
        assert archive["threshold_selection_split"].item() == "validation"
        assert archive["threshold_selection_score"].item() == 0.88
