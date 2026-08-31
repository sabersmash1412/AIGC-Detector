"""Source-matched E5 training with risk-controlled abstention selection."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from src.clip_features import CLIP_FEATURE_DIMENSION, CLIP_MODEL_NAME, CLIP_PRETRAINED
from src.linear_probe import (
    LINEAR_PROBE_CHECKPOINT_VERSION,
    LINEAR_PROBE_NAME,
    LinearProbeCheckpoint,
    linear_probe_probabilities,
    load_linear_probe_checkpoint,
)
from src.robust_linear_training import PairedFeatureSet, _transformed_batch


E5_CHECKPOINT_KIND = "e5_source_matched_risk_controlled_linear_head_v1"
GROUP_NAMES = (
    "cifake_real",
    "cifake_ai_generated",
    "sid_set_train_real",
    "sid_set_train_flux",
)
REAL_GROUPS = ("cifake_real", "sid_set_train_real")
AI_GROUPS = ("cifake_ai_generated", "sid_set_train_flux")


@dataclass(frozen=True)
class E5CandidateResult:
    anchor_weight: float
    best_epoch: int | None
    coefficients: np.ndarray | None
    intercept: float | None
    best_validation: dict[str, Any] | None
    history: list[dict[str, Any]]
    epochs_completed: int
    stopped_early: bool


@dataclass(frozen=True)
class E5TrainingOutcome:
    accepted: bool
    selected: E5CandidateResult | None
    candidates: tuple[E5CandidateResult, ...]
    initial_validation: dict[str, Any]


def _condition_features(dataset: PairedFeatureSet) -> dict[str, np.ndarray]:
    return {
        "clean": dataset.clean.features,
        **{
            condition: cache.features
            for condition, cache in zip(
                dataset.conditions, dataset.transformed, strict=True
            )
        },
    }


def _condition_probabilities(
    dataset: PairedFeatureSet, coefficients: np.ndarray, intercept: float
) -> dict[str, np.ndarray]:
    return {
        condition: linear_probe_probabilities(features, coefficients, intercept)
        for condition, features in _condition_features(dataset).items()
    }


def validation_probability_groups(
    cifake: PairedFeatureSet,
    sid_real: PairedFeatureSet,
    sid_flux: PairedFeatureSet,
    coefficients: np.ndarray,
    intercept: float,
) -> dict[str, dict[str, np.ndarray]]:
    """Return four single-label validation groups over identical conditions."""

    if not cifake.conditions == sid_real.conditions == sid_flux.conditions:
        raise ValueError("E5 validation conditions must align")
    if set(np.unique(cifake.clean.labels).tolist()) != {0, 1}:
        raise ValueError("E5 CIFAKE validation must contain both labels")
    if set(np.unique(sid_real.clean.labels).tolist()) != {0}:
        raise ValueError("E5 SID-real validation must contain only label 0")
    if set(np.unique(sid_flux.clean.labels).tolist()) != {1}:
        raise ValueError("E5 SID-FLUX validation must contain only label 1")
    cifake_probabilities = _condition_probabilities(cifake, coefficients, intercept)
    real_mask = cifake.clean.labels == 0
    ai_mask = cifake.clean.labels == 1
    return {
        "cifake_real": {
            condition: values[real_mask]
            for condition, values in cifake_probabilities.items()
        },
        "cifake_ai_generated": {
            condition: values[ai_mask]
            for condition, values in cifake_probabilities.items()
        },
        "sid_set_train_real": _condition_probabilities(
            sid_real, coefficients, intercept
        ),
        "sid_set_train_flux": _condition_probabilities(
            sid_flux, coefficients, intercept
        ),
    }


def _wilson_upper_counts(
    successes: np.ndarray, trials: int, confidence_level: float
) -> np.ndarray:
    counts = np.asarray(successes, dtype=np.float64)
    if trials <= 0 or np.any(counts < 0) or np.any(counts > trials):
        raise ValueError("Wilson counts are invalid")
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    observed = counts / trials
    z_squared = z * z
    denominator = 1.0 + z_squared / trials
    centre = (observed + z_squared / (2.0 * trials)) / denominator
    radius = (
        z
        * np.sqrt(
            observed * (1.0 - observed) / trials
            + z_squared / (4.0 * trials * trials)
        )
        / denominator
    )
    return np.minimum(1.0, centre + radius)


def _validate_group_probabilities(
    probabilities: dict[str, dict[str, np.ndarray]],
) -> tuple[str, ...]:
    if tuple(probabilities) != GROUP_NAMES:
        raise ValueError(f"E5 validation groups must be ordered as {GROUP_NAMES}")
    condition_names = tuple(probabilities[GROUP_NAMES[0]])
    if not condition_names or condition_names[0] != "clean":
        raise ValueError("E5 validation conditions must begin with clean")
    for group in GROUP_NAMES:
        if tuple(probabilities[group]) != condition_names:
            raise ValueError("E5 validation group conditions do not align")
        for values in probabilities[group].values():
            array = np.asarray(values, dtype=np.float64)
            if array.ndim != 1 or len(array) == 0:
                raise ValueError("E5 group probabilities must be non-empty vectors")
            if not bool(np.isfinite(array).all()) or np.any((array < 0) | (array > 1)):
                raise ValueError("E5 probabilities must be finite values in [0, 1]")
    return condition_names


def _pair_statistics(
    probabilities: dict[str, dict[str, np.ndarray]],
    *,
    real_threshold: float,
    ai_threshold: float,
    confidence_level: float,
) -> dict[str, Any]:
    if not 0.0 <= real_threshold < ai_threshold <= 1.0:
        raise ValueError("E5 thresholds must satisfy real < AI")
    conditions = _validate_group_probabilities(probabilities)
    rows: dict[str, Any] = {}
    coverages: list[float] = []
    decisive_accuracies: list[float] = []
    real_error_uppers: list[float] = []
    ai_error_uppers: list[float] = []
    clean_coverages: list[float] = []
    for group in GROUP_NAMES:
        is_real = group in REAL_GROUPS
        group_rows: dict[str, Any] = {}
        for condition in conditions:
            values = np.asarray(probabilities[group][condition], dtype=np.float64)
            called_real = int(np.sum(values <= real_threshold))
            called_ai = int(np.sum(values >= ai_threshold))
            uncertain = int(len(values) - called_real - called_ai)
            decisive = called_real + called_ai
            correct = called_real if is_real else called_ai
            incorrect = called_ai if is_real else called_real
            coverage = decisive / len(values)
            decisive_accuracy = correct / decisive if decisive else 0.0
            error_upper = float(
                _wilson_upper_counts(
                    np.asarray([incorrect]), len(values), confidence_level
                )[0]
            )
            group_rows[condition] = {
                "samples": len(values),
                "called_real": called_real,
                "uncertain": uncertain,
                "called_ai_generated": called_ai,
                "decisive_coverage": float(coverage),
                "decisive_accuracy": float(decisive_accuracy),
                "confident_error_rate": float(incorrect / len(values)),
                "confident_error_wilson_upper": error_upper,
            }
            coverages.append(float(coverage))
            decisive_accuracies.append(float(decisive_accuracy))
            if condition == "clean":
                clean_coverages.append(float(coverage))
            if is_real:
                real_error_uppers.append(error_upper)
            else:
                ai_error_uppers.append(error_upper)
        rows[group] = group_rows
    return {
        "real_threshold": float(real_threshold),
        "ai_threshold": float(ai_threshold),
        "uncertainty_width": float(ai_threshold - real_threshold),
        "worst_real_called_ai_wilson_upper": float(max(real_error_uppers)),
        "worst_ai_called_real_wilson_upper": float(max(ai_error_uppers)),
        "minimum_clean_decisive_coverage": float(min(clean_coverages)),
        "worst_source_condition_decisive_coverage": float(min(coverages)),
        "mean_source_condition_decisive_coverage": float(np.mean(coverages)),
        "worst_decisive_accuracy": float(min(decisive_accuracies)),
        "by_group_condition": rows,
    }


def _source_condition_auc(
    probabilities: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    conditions = _validate_group_probabilities(probabilities)
    rows: dict[str, dict[str, float]] = {"cifake": {}, "sid_set_train": {}}
    for condition in conditions:
        for source, real_group, ai_group in (
            ("cifake", "cifake_real", "cifake_ai_generated"),
            ("sid_set_train", "sid_set_train_real", "sid_set_train_flux"),
        ):
            real = probabilities[real_group][condition]
            ai = probabilities[ai_group][condition]
            labels = np.concatenate(
                [np.zeros(len(real), dtype=np.int64), np.ones(len(ai), dtype=np.int64)]
            )
            values = np.concatenate([real, ai])
            rows[source][condition] = float(roc_auc_score(labels, values))
    flattened = [value for source in rows.values() for value in source.values()]
    return {
        "worst_source_condition_roc_auc": float(min(flattened)),
        "mean_source_condition_roc_auc": float(np.mean(flattened)),
        "by_source_condition": rows,
    }


def select_risk_controlled_thresholds(
    probabilities: dict[str, dict[str, np.ndarray]],
    *,
    thresholds: np.ndarray,
    confidence_level: float,
    maximum_real_called_ai_upper: float,
    maximum_ai_called_real_upper: float,
    minimum_clean_coverage: float,
    minimum_worst_coverage: float,
    minimum_mean_coverage: float,
) -> dict[str, Any]:
    """Select a two-threshold E5 decision or return a documented rejection."""

    conditions = _validate_group_probabilities(probabilities)
    candidates = np.asarray(thresholds, dtype=np.float64)
    if candidates.ndim != 1 or len(candidates) == 0 or not np.all(np.diff(candidates) > 0):
        raise ValueError("E5 thresholds must be a non-empty increasing vector")
    if np.any((candidates <= 0) | (candidates >= 1)):
        raise ValueError("E5 threshold grid must lie strictly inside (0, 1)")

    rows = [(group, condition) for group in GROUP_NAMES for condition in conditions]
    lower_counts = np.asarray(
        [
            np.sum(
                probabilities[group][condition][:, None] <= candidates[None, :],
                axis=0,
            )
            for group, condition in rows
        ],
        dtype=np.int64,
    )
    upper_counts = np.asarray(
        [
            np.sum(
                probabilities[group][condition][:, None] >= candidates[None, :],
                axis=0,
            )
            for group, condition in rows
        ],
        dtype=np.int64,
    )
    trials = np.asarray([len(probabilities[g][c]) for g, c in rows], dtype=np.int64)
    real_rows = np.asarray([g in REAL_GROUPS for g, _ in rows])
    ai_rows = ~real_rows
    real_threshold_ok = np.ones(len(candidates), dtype=bool)
    ai_threshold_ok = np.ones(len(candidates), dtype=bool)
    for row_index in np.flatnonzero(ai_rows):
        real_threshold_ok &= (
            _wilson_upper_counts(
                lower_counts[row_index], int(trials[row_index]), confidence_level
            )
            <= maximum_ai_called_real_upper + 1e-12
        )
    for row_index in np.flatnonzero(real_rows):
        ai_threshold_ok &= (
            _wilson_upper_counts(
                upper_counts[row_index], int(trials[row_index]), confidence_level
            )
            <= maximum_real_called_ai_upper + 1e-12
        )

    real_indices = np.flatnonzero(real_threshold_ok)
    ai_indices = np.flatnonzero(ai_threshold_ok)
    error_feasible_pairs = 0
    coverage_feasible_pairs = 0
    best: tuple[tuple[float, ...], int, int] | None = None
    clean_rows = np.asarray([condition == "clean" for _, condition in rows])
    for real_index in real_indices:
        eligible_ai = ai_indices[candidates[ai_indices] > candidates[real_index]]
        if len(eligible_ai) == 0:
            continue
        error_feasible_pairs += len(eligible_ai)
        coverage = (
            lower_counts[:, real_index, None]
            + upper_counts[:, eligible_ai]
        ) / trials[:, None]
        clean_min = np.min(coverage[clean_rows], axis=0)
        worst = np.min(coverage, axis=0)
        mean = np.mean(coverage, axis=0)
        feasible = (
            (clean_min >= minimum_clean_coverage - 1e-12)
            & (worst >= minimum_worst_coverage - 1e-12)
            & (mean >= minimum_mean_coverage - 1e-12)
        )
        coverage_feasible_pairs += int(np.sum(feasible))
        for local_index in np.flatnonzero(feasible):
            ai_index = int(eligible_ai[local_index])
            decisive = lower_counts[:, real_index] + upper_counts[:, ai_index]
            correct = np.where(
                real_rows,
                lower_counts[:, real_index],
                upper_counts[:, ai_index],
            )
            decisive_accuracy = np.divide(
                correct,
                decisive,
                out=np.zeros_like(correct, dtype=np.float64),
                where=decisive > 0,
            )
            rank = (
                float(worst[local_index]),
                float(mean[local_index]),
                float(np.min(decisive_accuracy)),
                -float(candidates[ai_index] - candidates[real_index]),
            )
            if best is None or rank > best[0]:
                best = (rank, int(real_index), ai_index)

    auc = _source_condition_auc(probabilities)
    if best is None:
        fallback_real = int(real_indices[0]) if len(real_indices) else 0
        fallback_ai = int(ai_indices[-1]) if len(ai_indices) else len(candidates) - 1
        if candidates[fallback_real] >= candidates[fallback_ai]:
            fallback_real, fallback_ai = 0, len(candidates) - 1
        selected = _pair_statistics(
            probabilities,
            real_threshold=float(candidates[fallback_real]),
            ai_threshold=float(candidates[fallback_ai]),
            confidence_level=confidence_level,
        )
        selected.update(
            {
                "constraint_satisfied": False,
                "rejected": True,
                "rejection_reason": (
                    "No threshold pair satisfied every frozen Wilson-error and "
                    "anti-trivial-abstention coverage constraint."
                ),
                "error_feasible_pairs": int(error_feasible_pairs),
                "coverage_feasible_pairs": int(coverage_feasible_pairs),
                **auc,
            }
        )
        return selected

    _, real_index, ai_index = best
    selected = _pair_statistics(
        probabilities,
        real_threshold=float(candidates[real_index]),
        ai_threshold=float(candidates[ai_index]),
        confidence_level=confidence_level,
    )
    selected.update(
        {
            "constraint_satisfied": True,
            "rejected": False,
            "rejection_reason": None,
            "error_feasible_pairs": int(error_feasible_pairs),
            "coverage_feasible_pairs": int(coverage_feasible_pairs),
            **auc,
        }
    )
    return selected


def select_binary_benchmark_threshold(
    probabilities: dict[str, dict[str, np.ndarray]], thresholds: np.ndarray
) -> dict[str, Any]:
    """Select a validation-only binary benchmark with no deployment-safety claim."""

    conditions = _validate_group_probabilities(probabilities)
    candidates = np.asarray(thresholds, dtype=np.float64)
    curves: dict[str, dict[str, np.ndarray]] = {"cifake": {}, "sid_set_train": {}}
    for condition in conditions:
        for source, real_group, ai_group in (
            ("cifake", "cifake_real", "cifake_ai_generated"),
            ("sid_set_train", "sid_set_train_real", "sid_set_train_flux"),
        ):
            real = probabilities[real_group][condition]
            ai = probabilities[ai_group][condition]
            real_recall = np.mean(real[:, None] < candidates[None, :], axis=0)
            ai_recall = np.mean(ai[:, None] >= candidates[None, :], axis=0)
            curves[source][condition] = 0.5 * (real_recall + ai_recall)
    matrix = np.stack(
        [curves[source][condition] for source in curves for condition in conditions]
    )
    worst = np.min(matrix, axis=0)
    mean = np.mean(matrix, axis=0)
    eligible = np.flatnonzero(np.isclose(worst, np.max(worst), rtol=0, atol=1e-12))
    eligible = eligible[
        np.isclose(mean[eligible], np.max(mean[eligible]), rtol=0, atol=1e-12)
    ]
    index = int(eligible[0])
    return {
        "threshold": float(candidates[index]),
        "worst_source_condition_balanced_accuracy": float(worst[index]),
        "mean_source_condition_balanced_accuracy": float(mean[index]),
        "by_source_condition": {
            source: {
                condition: float(values[index])
                for condition, values in condition_rows.items()
            }
            for source, condition_rows in curves.items()
        },
        "deployment_safety_claim": False,
    }


def evaluate_e5_validation(
    cifake: PairedFeatureSet,
    sid_real: PairedFeatureSet,
    sid_flux: PairedFeatureSet,
    coefficients: np.ndarray,
    intercept: float,
    *,
    thresholds: np.ndarray,
    confidence_level: float,
    maximum_real_called_ai_upper: float,
    maximum_ai_called_real_upper: float,
    minimum_clean_coverage: float,
    minimum_worst_coverage: float,
    minimum_mean_coverage: float,
) -> dict[str, Any]:
    groups = validation_probability_groups(
        cifake, sid_real, sid_flux, coefficients, intercept
    )
    risk_controlled = select_risk_controlled_thresholds(
        groups,
        thresholds=thresholds,
        confidence_level=confidence_level,
        maximum_real_called_ai_upper=maximum_real_called_ai_upper,
        maximum_ai_called_real_upper=maximum_ai_called_real_upper,
        minimum_clean_coverage=minimum_clean_coverage,
        minimum_worst_coverage=minimum_worst_coverage,
        minimum_mean_coverage=minimum_mean_coverage,
    )
    return {
        "risk_controlled": risk_controlled,
        "binary_benchmark": select_binary_benchmark_threshold(groups, thresholds),
    }


def validation_rank(
    validation: dict[str, Any], *, anchor_weight: float, epoch: int
) -> tuple[float, ...] | None:
    risk = validation["risk_controlled"]
    if not risk["constraint_satisfied"]:
        return None
    return (
        float(risk["worst_source_condition_decisive_coverage"]),
        float(risk["mean_source_condition_decisive_coverage"]),
        float(risk["worst_decisive_accuracy"]),
        float(risk["worst_source_condition_roc_auc"]),
        -float(anchor_weight),
        -float(epoch),
        -float(risk["uncertainty_width"]),
    )


def _pair_coordinates(dataset: PairedFeatureSet, label: int) -> tuple[np.ndarray, np.ndarray]:
    samples = np.flatnonzero(dataset.clean.labels == label)
    conditions = np.repeat(np.arange(len(dataset.conditions), dtype=np.int64), len(samples))
    sample_indices = np.tile(samples.astype(np.int64), len(dataset.conditions))
    return conditions, sample_indices


def build_four_group_schedule(
    cifake: PairedFeatureSet,
    sid_real: PairedFeatureSet,
    sid_flux: PairedFeatureSet,
    *,
    examples_per_epoch: int,
    seed: int,
    epoch: int,
) -> dict[str, np.ndarray]:
    """Build one exact 25/25/25/25 E5 source-label schedule."""

    if not cifake.conditions == sid_real.conditions == sid_flux.conditions:
        raise ValueError("E5 training conditions must align")
    if set(np.unique(cifake.clean.labels).tolist()) != {0, 1}:
        raise ValueError("E5 CIFAKE training must contain both labels")
    if set(np.unique(sid_real.clean.labels).tolist()) != {0}:
        raise ValueError("E5 SID-real training must contain label 0 only")
    if set(np.unique(sid_flux.clean.labels).tolist()) != {1}:
        raise ValueError("E5 SID-FLUX training must contain label 1 only")
    if examples_per_epoch <= 0 or examples_per_epoch % 4:
        raise ValueError("E5 examples per epoch must be positive and divisible by four")
    target = examples_per_epoch // 4
    coordinates = (
        _pair_coordinates(cifake, 0),
        _pair_coordinates(cifake, 1),
        _pair_coordinates(sid_real, 0),
        _pair_coordinates(sid_flux, 1),
    )
    if any(len(samples) < target for _, samples in coordinates):
        raise ValueError("An E5 source-label group does not contain enough pairs")
    rng = np.random.default_rng(seed + epoch)
    condition_parts: list[np.ndarray] = []
    sample_parts: list[np.ndarray] = []
    for condition, sample in coordinates:
        pick = rng.permutation(len(sample))[:target]
        condition_parts.append(condition[pick])
        sample_parts.append(sample[pick])
    group = np.repeat(np.arange(4, dtype=np.int64), target)
    condition = np.concatenate(condition_parts)
    sample = np.concatenate(sample_parts)
    source = np.repeat(np.asarray([0, 0, 1, 2], dtype=np.int64), target)
    order = rng.permutation(examples_per_epoch)
    return {
        "source": source[order],
        "condition": condition[order],
        "sample": sample[order],
        "group": group[order],
    }


def _gather_batch(
    cifake: PairedFeatureSet,
    sid_real: PairedFeatureSet,
    sid_flux: PairedFeatureSet,
    source: np.ndarray,
    condition: np.ndarray,
    sample: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch_size = len(source)
    clean = np.empty((batch_size, CLIP_FEATURE_DIMENSION), dtype=np.float32)
    transformed = np.empty_like(clean)
    labels = np.empty(batch_size, dtype=np.float32)
    for source_index, dataset in ((0, cifake), (1, sid_real), (2, sid_flux)):
        positions = np.flatnonzero(source == source_index)
        clean[positions] = dataset.clean.features[sample[positions]]
        transformed[positions] = _transformed_batch(
            dataset, condition[positions], sample[positions]
        )
        labels[positions] = dataset.clean.labels[sample[positions]]
    return clean, transformed, labels


def _compact_validation(validation: dict[str, Any]) -> dict[str, Any]:
    risk = validation["risk_controlled"]
    binary = validation["binary_benchmark"]
    return {
        "feasible": risk["constraint_satisfied"],
        "real_threshold": risk["real_threshold"],
        "ai_threshold": risk["ai_threshold"],
        "worst_real_error_upper": risk["worst_real_called_ai_wilson_upper"],
        "worst_ai_error_upper": risk["worst_ai_called_real_wilson_upper"],
        "minimum_clean_coverage": risk["minimum_clean_decisive_coverage"],
        "worst_coverage": risk["worst_source_condition_decisive_coverage"],
        "mean_coverage": risk["mean_source_condition_decisive_coverage"],
        "worst_decisive_accuracy": risk["worst_decisive_accuracy"],
        "worst_roc_auc": risk["worst_source_condition_roc_auc"],
        "binary_threshold": binary["threshold"],
        "binary_worst_balanced_accuracy": binary[
            "worst_source_condition_balanced_accuracy"
        ],
    }


def train_e5_candidates(
    *,
    cifake_train: PairedFeatureSet,
    sid_real_train: PairedFeatureSet,
    sid_flux_train: PairedFeatureSet,
    cifake_validation: PairedFeatureSet,
    sid_real_validation: PairedFeatureSet,
    sid_flux_validation: PairedFeatureSet,
    initialization: LinearProbeCheckpoint,
    device: torch.device,
    anchor_weights: tuple[float, ...],
    examples_per_epoch: int,
    seed: int,
    batch_size: int,
    maximum_epochs: int,
    learning_rate: float,
    weight_decay: float,
    early_stopping_patience: int,
    consistency_weight: float,
    thresholds: np.ndarray,
    confidence_level: float,
    maximum_real_called_ai_upper: float,
    maximum_ai_called_real_upper: float,
    minimum_clean_coverage: float,
    minimum_worst_coverage: float,
    minimum_mean_coverage: float,
) -> E5TrainingOutcome:
    """Train every frozen E5 anchor candidate and select only feasible results."""

    datasets = (
        cifake_train,
        sid_real_train,
        sid_flux_train,
        cifake_validation,
        sid_real_validation,
        sid_flux_validation,
    )
    if any(dataset.conditions != datasets[0].conditions for dataset in datasets[1:]):
        raise ValueError("All E5 train/validation conditions must match")
    if not anchor_weights or any(weight < 0 for weight in anchor_weights):
        raise ValueError("E5 anchor weights must be a non-empty non-negative tuple")
    if min(batch_size, maximum_epochs, early_stopping_patience) <= 0:
        raise ValueError("E5 training counts must be positive")
    if learning_rate <= 0 or weight_decay < 0 or consistency_weight < 0:
        raise ValueError("E5 optimizer and loss weights are invalid")

    validation_kwargs = {
        "thresholds": thresholds,
        "confidence_level": confidence_level,
        "maximum_real_called_ai_upper": maximum_real_called_ai_upper,
        "maximum_ai_called_real_upper": maximum_ai_called_real_upper,
        "minimum_clean_coverage": minimum_clean_coverage,
        "minimum_worst_coverage": minimum_worst_coverage,
        "minimum_mean_coverage": minimum_mean_coverage,
    }
    initial_validation = evaluate_e5_validation(
        cifake_validation,
        sid_real_validation,
        sid_flux_validation,
        initialization.coefficients,
        initialization.intercept,
        **validation_kwargs,
    )
    candidate_results: list[E5CandidateResult] = []
    global_best: tuple[tuple[float, ...], E5CandidateResult] | None = None
    initial_coefficients = torch.as_tensor(
        initialization.coefficients, dtype=torch.float32, device=device
    )
    initial_intercept = torch.tensor(
        initialization.intercept, dtype=torch.float32, device=device
    )

    for candidate_index, anchor_weight in enumerate(anchor_weights):
        torch.manual_seed(seed)
        coefficients = torch.nn.Parameter(initial_coefficients.clone())
        intercept = torch.nn.Parameter(initial_intercept.clone())
        optimizer = torch.optim.AdamW(
            [
                {"params": [coefficients], "weight_decay": weight_decay},
                {"params": [intercept], "weight_decay": 0.0},
            ],
            lr=learning_rate,
        )
        best: tuple[tuple[float, ...], int, np.ndarray, float, dict[str, Any]] | None = None
        history: list[dict[str, Any]] = []
        epochs_without_improvement = 0
        for epoch in range(1, maximum_epochs + 1):
            schedule = build_four_group_schedule(
                cifake_train,
                sid_real_train,
                sid_flux_train,
                examples_per_epoch=examples_per_epoch,
                seed=seed,
                epoch=epoch,
            )
            totals = {"supervised": 0.0, "consistency": 0.0, "anchor": 0.0}
            observed = 0
            for start in range(0, examples_per_epoch, batch_size):
                stop = start + batch_size
                clean_np, transformed_np, labels_np = _gather_batch(
                    cifake_train,
                    sid_real_train,
                    sid_flux_train,
                    schedule["source"][start:stop],
                    schedule["condition"][start:stop],
                    schedule["sample"][start:stop],
                )
                clean = torch.from_numpy(clean_np).to(device)
                transformed = torch.from_numpy(transformed_np).to(device)
                labels = torch.from_numpy(labels_np).to(device)
                clean_logits = clean @ coefficients + intercept
                transformed_logits = transformed @ coefficients + intercept
                supervised = 0.5 * (
                    F.binary_cross_entropy_with_logits(clean_logits, labels)
                    + F.binary_cross_entropy_with_logits(transformed_logits, labels)
                )
                consistency = F.mse_loss(
                    torch.sigmoid(clean_logits), torch.sigmoid(transformed_logits)
                )
                anchor = torch.mean((coefficients - initial_coefficients) ** 2) + (
                    (intercept - initial_intercept) ** 2 / CLIP_FEATURE_DIMENSION
                )
                loss = (
                    supervised
                    + consistency_weight * consistency
                    + float(anchor_weight) * anchor
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                size = len(labels_np)
                observed += size
                totals["supervised"] += float(supervised.detach().cpu()) * size
                totals["consistency"] += float(consistency.detach().cpu()) * size
                totals["anchor"] += float(anchor.detach().cpu()) * size

            current_coefficients = coefficients.detach().cpu().numpy().astype(np.float64)
            current_intercept = float(intercept.detach().cpu())
            validation = evaluate_e5_validation(
                cifake_validation,
                sid_real_validation,
                sid_flux_validation,
                current_coefficients,
                current_intercept,
                **validation_kwargs,
            )
            rank = validation_rank(
                validation, anchor_weight=float(anchor_weight), epoch=epoch
            )
            improved = rank is not None and (best is None or rank > best[0])
            if improved:
                best = (
                    rank,
                    epoch,
                    current_coefficients.copy(),
                    current_intercept,
                    validation,
                )
                epochs_without_improvement = 0
            elif best is not None:
                epochs_without_improvement += 1
            compact = _compact_validation(validation)
            group_counts = np.bincount(schedule["group"], minlength=4)
            history.append(
                {
                    "epoch": epoch,
                    "supervised_loss": totals["supervised"] / observed,
                    "consistency_loss": totals["consistency"] / observed,
                    "anchor_loss": totals["anchor"] / observed,
                    "total_loss": (
                        totals["supervised"]
                        + consistency_weight * totals["consistency"]
                        + float(anchor_weight) * totals["anchor"]
                    )
                    / observed,
                    "examples": observed,
                    "group_counts": {
                        name: int(group_counts[index])
                        for index, name in enumerate(GROUP_NAMES)
                    },
                    "validation": compact,
                    "selected_as_candidate_best": improved,
                }
            )
            print(
                f"anchor={anchor_weight:g} epoch={epoch:02d} "
                f"loss={history[-1]['total_loss']:.6f} "
                f"feasible={compact['feasible']} "
                f"thresholds={compact['real_threshold']:.3f}/{compact['ai_threshold']:.3f} "
                f"errors={compact['worst_real_error_upper']:.3f}/"
                f"{compact['worst_ai_error_upper']:.3f} "
                f"coverage={compact['worst_coverage']:.3f}/{compact['mean_coverage']:.3f} "
                f"candidate_best={best[1] if best else 'none'}",
                flush=True,
            )
            if best is not None and epochs_without_improvement >= early_stopping_patience:
                break

        result = E5CandidateResult(
            anchor_weight=float(anchor_weight),
            best_epoch=None if best is None else best[1],
            coefficients=None if best is None else best[2],
            intercept=None if best is None else best[3],
            best_validation=None if best is None else best[4],
            history=history,
            epochs_completed=len(history),
            stopped_early=len(history) < maximum_epochs,
        )
        candidate_results.append(result)
        if best is not None and (global_best is None or best[0] > global_best[0]):
            global_best = (best[0], result)

    return E5TrainingOutcome(
        accepted=global_best is not None,
        selected=None if global_best is None else global_best[1],
        candidates=tuple(candidate_results),
        initial_validation=initial_validation,
    )


def combined_cache_digest(datasets: dict[str, PairedFeatureSet], split: str) -> str:
    payload = {
        name: {
            "clean": dataset.clean_cache_sha256,
            "transformed": dict(
                zip(dataset.conditions, dataset.transformed_cache_sha256, strict=True)
            ),
        }
        for name, dataset in sorted(datasets.items())
    }
    return hashlib.sha256(
        json.dumps({"split": split, "datasets": payload}, sort_keys=True).encode()
    ).hexdigest()


def save_e5_checkpoint(
    path: Path,
    *,
    outcome: E5TrainingOutcome,
    initialization: LinearProbeCheckpoint,
    protocol_sha256: str,
    initial_checkpoint_sha256: str,
    train_cache_sha256: str,
    validation_cache_sha256: str,
    seed: int,
    consistency_weight: float,
) -> None:
    """Save an inference-compatible E5 checkpoint only after protocol acceptance."""

    if not outcome.accepted or outcome.selected is None:
        raise ValueError("A rejected E5 outcome cannot produce a deployable checkpoint")
    selected = outcome.selected
    if (
        selected.coefficients is None
        or selected.intercept is None
        or selected.best_epoch is None
        or selected.best_validation is None
    ):
        raise ValueError("Accepted E5 outcome is incomplete")
    risk = selected.best_validation["risk_controlled"]
    binary = selected.best_validation["binary_benchmark"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as temporary_directory:
        temporary_path = Path(temporary_directory) / path.name
        np.savez(
            temporary_path,
            checkpoint_version=np.asarray(LINEAR_PROBE_CHECKPOINT_VERSION, dtype=np.int64),
            classifier_name=np.asarray(LINEAR_PROBE_NAME),
            coefficients=np.asarray(selected.coefficients, dtype=np.float64),
            intercept=np.asarray(selected.intercept, dtype=np.float64),
            classes=np.asarray([0, 1], dtype=np.int64),
            regularization_c=np.asarray(initialization.regularization_c, dtype=np.float64),
            threshold=np.asarray(binary["threshold"], dtype=np.float64),
            feature_dimension=np.asarray(CLIP_FEATURE_DIMENSION, dtype=np.int64),
            feature_model_name=np.asarray(CLIP_MODEL_NAME),
            feature_pretrained=np.asarray(CLIP_PRETRAINED),
            features_normalized=np.asarray(True),
            seed=np.asarray(seed, dtype=np.int64),
            selected_validation_roc_auc=np.asarray(
                risk["mean_source_condition_roc_auc"], dtype=np.float64
            ),
            train_cache_sha256=np.asarray(train_cache_sha256),
            validation_cache_sha256=np.asarray(validation_cache_sha256),
            robust_checkpoint_kind=np.asarray(E5_CHECKPOINT_KIND),
            experiment=np.asarray("E5_source_matched_domain_adaptation"),
            protocol_sha256=np.asarray(protocol_sha256),
            initial_checkpoint_sha256=np.asarray(initial_checkpoint_sha256),
            consistency_weight=np.asarray(consistency_weight, dtype=np.float64),
            anchor_weight=np.asarray(selected.anchor_weight, dtype=np.float64),
            selected_best_epoch=np.asarray(selected.best_epoch, dtype=np.int64),
            threshold_selection_kind=np.asarray("validation_risk_controlled_abstention"),
            real_threshold=np.asarray(risk["real_threshold"], dtype=np.float64),
            ai_threshold=np.asarray(risk["ai_threshold"], dtype=np.float64),
            binary_benchmark_threshold=np.asarray(binary["threshold"], dtype=np.float64),
            selected_validation_json=np.asarray(
                json.dumps(selected.best_validation, sort_keys=True)
            ),
        )
        temporary_path.replace(path)
    loaded = load_linear_probe_checkpoint(path)
    if loaded.threshold != binary["threshold"]:
        raise ValueError("Saved E5 binary benchmark threshold changed")
