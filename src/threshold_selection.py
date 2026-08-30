"""Validation-only threshold selection and safe checkpoint copying."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.linear_probe import load_linear_probe_checkpoint
from src.metrics import binary_classification_metrics


THRESHOLD_SELECTION_KIND = "section3_validation_balanced_accuracy_v1"


@dataclass(frozen=True)
class ThresholdSelectionResult:
    """Selected threshold, full objective curve, and validation diagnostics."""

    threshold: float
    objective_score: float
    worst_condition_balanced_accuracy: float
    curve: np.ndarray
    selected_metrics: dict[str, Any]
    baseline_0_5_metrics: dict[str, Any]


def _condition_metrics(
    labels: np.ndarray,
    probabilities_by_condition: dict[str, np.ndarray],
    threshold: float,
) -> dict[str, Any]:
    by_condition = {
        condition: binary_classification_metrics(labels, probabilities, threshold)
        for condition, probabilities in probabilities_by_condition.items()
    }
    balanced_accuracies = [
        metrics["balanced_accuracy"] for metrics in by_condition.values()
    ]
    return {
        "threshold": threshold,
        "mean_balanced_accuracy": float(np.mean(balanced_accuracies)),
        "worst_condition_balanced_accuracy": float(np.min(balanced_accuracies)),
        "worst_condition": min(
            by_condition,
            key=lambda condition: by_condition[condition]["balanced_accuracy"],
        ),
        "by_condition": by_condition,
    }


def select_balanced_accuracy_threshold(
    labels: np.ndarray,
    probabilities_by_condition: dict[str, np.ndarray],
    candidate_thresholds: np.ndarray,
) -> ThresholdSelectionResult:
    """Maximise mean condition-balanced accuracy with deterministic tie-breaking."""

    label_array = np.asarray(labels, dtype=np.int64)
    thresholds = np.asarray(candidate_thresholds, dtype=np.float64)
    if label_array.ndim != 1 or set(np.unique(label_array).tolist()) != {0, 1}:
        raise ValueError("Threshold selection requires one-dimensional binary labels")
    if not probabilities_by_condition or "clean" not in probabilities_by_condition:
        raise ValueError("Threshold selection requires clean and transformed probabilities")
    if thresholds.ndim != 1 or len(thresholds) == 0:
        raise ValueError("Candidate thresholds must be a non-empty vector")
    if not bool(np.all(np.diff(thresholds) > 0.0)):
        raise ValueError("Candidate thresholds must be strictly increasing")
    if np.any((thresholds <= 0.0) | (thresholds >= 1.0)):
        raise ValueError("Candidate thresholds must lie strictly between zero and one")

    positive_mask = label_array == 1
    negative_mask = label_array == 0
    curves: list[np.ndarray] = []
    for condition, probabilities in probabilities_by_condition.items():
        probability_array = np.asarray(probabilities, dtype=np.float64)
        if probability_array.shape != label_array.shape:
            raise ValueError(f"Probability shape mismatch for condition {condition!r}")
        if not bool(np.isfinite(probability_array).all()) or np.any(
            (probability_array < 0.0) | (probability_array > 1.0)
        ):
            raise ValueError(f"Invalid probabilities for condition {condition!r}")
        predictions = probability_array[:, None] >= thresholds[None, :]
        true_positive_rate = predictions[positive_mask].mean(axis=0)
        true_negative_rate = (~predictions[negative_mask]).mean(axis=0)
        curves.append(0.5 * (true_positive_rate + true_negative_rate))

    condition_curves = np.stack(curves)
    mean_curve = condition_curves.mean(axis=0)
    maximum_score = float(mean_curve.max())
    tied_indices = np.flatnonzero(
        np.isclose(mean_curve, maximum_score, rtol=0.0, atol=1e-12)
    )
    tied_distances = np.abs(thresholds[tied_indices] - 0.5)
    closest_distance = tied_distances.min()
    closest_indices = tied_indices[
        np.isclose(tied_distances, closest_distance, rtol=0.0, atol=1e-12)
    ]
    selected_index = int(closest_indices[0])
    selected_threshold = float(thresholds[selected_index])
    selected_metrics = _condition_metrics(
        label_array, probabilities_by_condition, selected_threshold
    )
    baseline_metrics = _condition_metrics(
        label_array, probabilities_by_condition, 0.5
    )

    curve = np.column_stack(
        [thresholds, mean_curve, condition_curves.min(axis=0)]
    )
    return ThresholdSelectionResult(
        threshold=selected_threshold,
        objective_score=maximum_score,
        worst_condition_balanced_accuracy=float(
            condition_curves[:, selected_index].min()
        ),
        curve=curve,
        selected_metrics=selected_metrics,
        baseline_0_5_metrics=baseline_metrics,
    )


def write_thresholded_checkpoint(
    source_path: Path,
    destination_path: Path,
    *,
    threshold: float,
    objective_score: float,
    protocol_sha256: str,
    source_checkpoint_sha256: str,
) -> None:
    """Copy all safe checkpoint metadata and replace only the operating threshold."""

    if not 0.0 < threshold < 1.0:
        raise ValueError("Selected threshold must lie strictly between zero and one")
    load_linear_probe_checkpoint(source_path)
    with np.load(source_path, allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in source.files}
    if any(array.dtype.hasobject for array in arrays.values()):
        raise ValueError("Source checkpoint contains unsafe object arrays")
    arrays.update(
        {
            "threshold": np.asarray(threshold, dtype=np.float64),
            "threshold_selection_kind": np.asarray(THRESHOLD_SELECTION_KIND),
            "threshold_selection_split": np.asarray("validation"),
            "threshold_selection_objective": np.asarray(
                "mean_balanced_accuracy_across_clean_and_six_transforms"
            ),
            "threshold_selection_score": np.asarray(
                objective_score, dtype=np.float64
            ),
            "threshold_selection_protocol_sha256": np.asarray(protocol_sha256),
            "threshold_source_checkpoint_sha256": np.asarray(
                source_checkpoint_sha256
            ),
        }
    )

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=destination_path.parent
    ) as temporary_directory:
        temporary_path = Path(temporary_directory) / destination_path.name
        np.savez(temporary_path, **arrays)
        temporary_path.replace(destination_path)
    loaded = load_linear_probe_checkpoint(destination_path)
    if loaded.threshold != threshold:
        raise ValueError("Thresholded checkpoint did not preserve the selected threshold")
