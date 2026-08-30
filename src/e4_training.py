"""Source-balanced E4 linear-head training and constrained validation."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.clip_features import CLIP_FEATURE_DIMENSION, CLIP_MODEL_NAME, CLIP_PRETRAINED
from src.linear_probe import (
    LINEAR_PROBE_CHECKPOINT_VERSION,
    LINEAR_PROBE_NAME,
    LinearProbeCheckpoint,
    linear_probe_probabilities,
    load_linear_probe_checkpoint,
)
from src.metrics import binary_classification_metrics
from src.robust_linear_training import PairedFeatureSet, _transformed_batch


E4_CHECKPOINT_KIND = "e4_domain_adapted_linear_head_v1"


@dataclass(frozen=True)
class E4TrainingResult:
    """Selected E4 weights, threshold, validation trace, and training history."""

    coefficients: np.ndarray
    intercept: float
    threshold: float
    best_epoch: int
    best_validation: dict[str, Any]
    initial_validation: dict[str, Any]
    history: list[dict[str, Any]]
    stopped_early: bool
    epochs_completed: int


def _condition_probabilities(
    dataset: PairedFeatureSet,
    coefficients: np.ndarray,
    intercept: float,
) -> dict[str, np.ndarray]:
    conditions = {
        "clean": dataset.clean.features,
        **{
            condition: cache.features
            for condition, cache in zip(
                dataset.conditions, dataset.transformed, strict=True
            )
        },
    }
    return {
        condition: linear_probe_probabilities(features, coefficients, intercept)
        for condition, features in conditions.items()
    }


def select_constrained_threshold(
    labels: np.ndarray,
    cifake_probabilities: dict[str, np.ndarray],
    sid_real_probabilities: dict[str, np.ndarray],
    *,
    thresholds: np.ndarray,
    sid_real_fpr_constraint: float,
) -> dict[str, Any]:
    """Select a validation threshold under the frozen SID-real FPR constraint."""

    label_array = np.asarray(labels, dtype=np.int64)
    candidates = np.asarray(thresholds, dtype=np.float64)
    if set(np.unique(label_array).tolist()) != {0, 1}:
        raise ValueError("E4 CIFAKE validation labels must contain both classes")
    if candidates.ndim != 1 or len(candidates) == 0:
        raise ValueError("E4 threshold candidates must be a non-empty vector")
    if not bool(np.all(np.diff(candidates) > 0.0)):
        raise ValueError("E4 threshold candidates must be strictly increasing")
    if not 0.0 <= sid_real_fpr_constraint <= 1.0:
        raise ValueError("E4 false-positive constraint must lie in [0, 1]")
    if tuple(cifake_probabilities) != tuple(sid_real_probabilities):
        raise ValueError("CIFAKE and SID validation conditions must align")

    real = label_array == 0
    generated = label_array == 1
    condition_names = tuple(cifake_probabilities)
    cifake_balanced_curves: list[np.ndarray] = []
    sid_fpr_curves: list[np.ndarray] = []
    for condition in condition_names:
        cifake = np.asarray(cifake_probabilities[condition], dtype=np.float64)
        sid = np.asarray(sid_real_probabilities[condition], dtype=np.float64)
        if cifake.shape != label_array.shape:
            raise ValueError("E4 CIFAKE probabilities and labels must align")
        if sid.ndim != 1 or len(sid) == 0:
            raise ValueError("E4 SID probabilities must be non-empty vectors")
        predictions = cifake[:, None] >= candidates[None, :]
        real_recall = np.mean(~predictions[real], axis=0)
        generated_recall = np.mean(predictions[generated], axis=0)
        cifake_balanced_curves.append(0.5 * (real_recall + generated_recall))
        sid_fpr_curves.append(
            np.mean(sid[:, None] >= candidates[None, :], axis=0)
        )

    balanced = np.stack(cifake_balanced_curves)
    sid_fpr = np.stack(sid_fpr_curves)
    mean_balanced = balanced.mean(axis=0)
    worst_balanced = balanced.min(axis=0)
    worst_sid_fpr = sid_fpr.max(axis=0)
    feasible = worst_sid_fpr <= sid_real_fpr_constraint + 1e-12
    fallback_used = not bool(np.any(feasible))
    eligible = np.flatnonzero(feasible)
    if fallback_used:
        minimum_fpr = float(np.min(worst_sid_fpr))
        eligible = np.flatnonzero(
            np.isclose(worst_sid_fpr, minimum_fpr, rtol=0.0, atol=1e-12)
        )

    maximum_mean = float(np.max(mean_balanced[eligible]))
    eligible = eligible[
        np.isclose(mean_balanced[eligible], maximum_mean, rtol=0.0, atol=1e-12)
    ]
    maximum_worst = float(np.max(worst_balanced[eligible]))
    eligible = eligible[
        np.isclose(worst_balanced[eligible], maximum_worst, rtol=0.0, atol=1e-12)
    ]
    selected_index = int(eligible[0])
    selected_threshold = float(candidates[selected_index])
    return {
        "threshold": selected_threshold,
        "constraint_satisfied": bool(feasible[selected_index]),
        "fallback_used": fallback_used,
        "sid_real_fpr_constraint": float(sid_real_fpr_constraint),
        "worst_sid_real_fpr": float(worst_sid_fpr[selected_index]),
        "worst_sid_real_fpr_condition": condition_names[
            int(np.argmax(sid_fpr[:, selected_index]))
        ],
        "mean_cifake_balanced_accuracy": float(mean_balanced[selected_index]),
        "worst_cifake_balanced_accuracy": float(worst_balanced[selected_index]),
        "worst_cifake_condition": condition_names[
            int(np.argmin(balanced[:, selected_index]))
        ],
        "by_condition": {
            condition: {
                "cifake_balanced_accuracy": float(balanced[row, selected_index]),
                "sid_real_false_positive_rate": float(sid_fpr[row, selected_index]),
            }
            for row, condition in enumerate(condition_names)
        },
    }


def evaluate_e4_validation(
    cifake: PairedFeatureSet,
    sid_real: PairedFeatureSet,
    coefficients: np.ndarray,
    intercept: float,
    *,
    thresholds: np.ndarray,
    sid_real_fpr_constraint: float,
) -> dict[str, Any]:
    """Select the operating threshold and return full multi-domain metrics."""

    if cifake.conditions != sid_real.conditions:
        raise ValueError("E4 CIFAKE and SID validation conditions must match")
    if set(np.unique(cifake.clean.labels).tolist()) != {0, 1}:
        raise ValueError("E4 CIFAKE validation must contain both classes")
    if set(np.unique(sid_real.clean.labels).tolist()) != {0}:
        raise ValueError("E4 SID validation must contain only real label 0")
    cifake_probabilities = _condition_probabilities(cifake, coefficients, intercept)
    sid_probabilities = _condition_probabilities(sid_real, coefficients, intercept)
    selection = select_constrained_threshold(
        cifake.clean.labels,
        cifake_probabilities,
        sid_probabilities,
        thresholds=thresholds,
        sid_real_fpr_constraint=sid_real_fpr_constraint,
    )
    threshold = float(selection["threshold"])
    by_condition: dict[str, Any] = {}
    for condition in cifake_probabilities:
        cifake_metrics = binary_classification_metrics(
            cifake.clean.labels,
            cifake_probabilities[condition],
            threshold=threshold,
        )
        sid_values = sid_probabilities[condition]
        by_condition[condition] = {
            "cifake": cifake_metrics,
            "sid_real": {
                "samples": len(sid_values),
                "false_positives": int(np.sum(sid_values >= threshold)),
                "false_positive_rate": float(np.mean(sid_values >= threshold)),
                "real_recall": float(np.mean(sid_values < threshold)),
                "mean_ai_probability": float(np.mean(sid_values)),
                "q95_ai_probability": float(np.quantile(sid_values, 0.95)),
            },
        }
    selection["mean_cifake_roc_auc"] = float(
        np.mean([row["cifake"]["roc_auc"] for row in by_condition.values()])
    )
    selection["by_condition_metrics"] = by_condition
    return selection


def validation_rank(validation: dict[str, Any]) -> tuple[float, ...]:
    """Return the frozen lexicographic epoch-selection rank."""

    if validation["constraint_satisfied"]:
        return (
            1.0,
            float(validation["mean_cifake_balanced_accuracy"]),
            float(validation["worst_cifake_balanced_accuracy"]),
        )
    return (
        0.0,
        -float(validation["worst_sid_real_fpr"]),
        float(validation["mean_cifake_balanced_accuracy"]),
        float(validation["worst_cifake_balanced_accuracy"]),
    )


def _pair_coordinates(dataset: PairedFeatureSet, label: int) -> tuple[np.ndarray, np.ndarray]:
    samples = np.flatnonzero(dataset.clean.labels == label)
    conditions = np.repeat(np.arange(len(dataset.conditions), dtype=np.int64), len(samples))
    sample_indices = np.tile(samples.astype(np.int64), len(dataset.conditions))
    return conditions, sample_indices


def build_source_balanced_schedule(
    cifake: PairedFeatureSet,
    sid_real: PairedFeatureSet,
    *,
    seed: int,
    epoch: int,
) -> dict[str, np.ndarray]:
    """Build one exact 50% fake, 25% CIFAKE-real, 25% SID-real epoch."""

    if cifake.conditions != sid_real.conditions:
        raise ValueError("E4 training conditions must align")
    if set(np.unique(cifake.clean.labels).tolist()) != {0, 1}:
        raise ValueError("E4 CIFAKE training must contain both classes")
    if set(np.unique(sid_real.clean.labels).tolist()) != {0}:
        raise ValueError("E4 SID training must contain only real label 0")
    ai_condition, ai_sample = _pair_coordinates(cifake, 1)
    cifake_real_condition, cifake_real_sample = _pair_coordinates(cifake, 0)
    sid_condition, sid_sample = _pair_coordinates(sid_real, 0)
    ai_target = len(ai_sample)
    real_source_target = ai_target // 2
    if ai_target % 2 != 0:
        raise ValueError("E4 CIFAKE AI pair count must be even")
    if len(cifake_real_sample) < real_source_target or len(sid_sample) < real_source_target:
        raise ValueError("E4 source groups do not contain enough pairs")
    rng = np.random.default_rng(seed + epoch)
    ai_pick = rng.permutation(len(ai_sample))[:ai_target]
    cifake_real_pick = rng.permutation(len(cifake_real_sample))[:real_source_target]
    sid_pick = rng.permutation(len(sid_sample))[:real_source_target]
    source = np.concatenate(
        [
            np.zeros(ai_target + real_source_target, dtype=np.int64),
            np.ones(real_source_target, dtype=np.int64),
        ]
    )
    condition = np.concatenate(
        [
            ai_condition[ai_pick],
            cifake_real_condition[cifake_real_pick],
            sid_condition[sid_pick],
        ]
    )
    sample = np.concatenate(
        [ai_sample[ai_pick], cifake_real_sample[cifake_real_pick], sid_sample[sid_pick]]
    )
    group = np.concatenate(
        [
            np.full(ai_target, 0, dtype=np.int64),
            np.full(real_source_target, 1, dtype=np.int64),
            np.full(real_source_target, 2, dtype=np.int64),
        ]
    )
    order = rng.permutation(len(source))
    return {
        "source": source[order],
        "condition": condition[order],
        "sample": sample[order],
        "group": group[order],
    }


def _gather_batch(
    cifake: PairedFeatureSet,
    sid_real: PairedFeatureSet,
    source: np.ndarray,
    condition: np.ndarray,
    sample: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch_size = len(source)
    clean = np.empty((batch_size, CLIP_FEATURE_DIMENSION), dtype=np.float32)
    transformed = np.empty_like(clean)
    labels = np.empty(batch_size, dtype=np.float32)
    for source_index, dataset in ((0, cifake), (1, sid_real)):
        positions = np.flatnonzero(source == source_index)
        clean[positions] = dataset.clean.features[sample[positions]]
        transformed[positions] = _transformed_batch(
            dataset, condition[positions], sample[positions]
        )
        labels[positions] = dataset.clean.labels[sample[positions]]
    return clean, transformed, labels


def train_e4_linear_head(
    *,
    cifake_train: PairedFeatureSet,
    sid_train_real: PairedFeatureSet,
    cifake_validation: PairedFeatureSet,
    sid_validation_real: PairedFeatureSet,
    initialization: LinearProbeCheckpoint,
    device: torch.device,
    seed: int,
    batch_size: int,
    maximum_epochs: int,
    learning_rate: float,
    weight_decay: float,
    early_stopping_patience: int,
    consistency_weight: float,
    thresholds: np.ndarray,
    sid_real_fpr_constraint: float,
) -> E4TrainingResult:
    """Train E4 with exact source/class balance and validation-only selection."""

    if batch_size <= 0 or maximum_epochs <= 0 or early_stopping_patience <= 0:
        raise ValueError("E4 batch, epoch, and patience values must be positive")
    if learning_rate <= 0.0 or weight_decay < 0.0 or consistency_weight < 0.0:
        raise ValueError("E4 learning rate and loss weights are invalid")
    conditions = cifake_train.conditions
    if not (
        conditions
        == sid_train_real.conditions
        == cifake_validation.conditions
        == sid_validation_real.conditions
    ):
        raise ValueError("All E4 datasets must use identical transformed conditions")

    torch.manual_seed(seed)
    coefficients = torch.nn.Parameter(
        torch.as_tensor(initialization.coefficients, dtype=torch.float32, device=device).clone()
    )
    intercept = torch.nn.Parameter(
        torch.tensor(initialization.intercept, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": [coefficients], "weight_decay": weight_decay},
            {"params": [intercept], "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )
    initial_validation = evaluate_e4_validation(
        cifake_validation,
        sid_validation_real,
        initialization.coefficients,
        initialization.intercept,
        thresholds=thresholds,
        sid_real_fpr_constraint=sid_real_fpr_constraint,
    )
    best_rank: tuple[float, ...] | None = None
    best_epoch = 0
    best_coefficients: np.ndarray | None = None
    best_intercept = 0.0
    best_validation: dict[str, Any] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, maximum_epochs + 1):
        schedule = build_source_balanced_schedule(
            cifake_train, sid_train_real, seed=seed, epoch=epoch
        )
        observed = 0
        supervised_total = 0.0
        consistency_total = 0.0
        for start in range(0, len(schedule["source"]), batch_size):
            stop = start + batch_size
            clean_np, transformed_np, labels_np = _gather_batch(
                cifake_train,
                sid_train_real,
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
            loss = supervised + consistency_weight * consistency
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            size = len(labels_np)
            observed += size
            supervised_total += float(supervised.detach().cpu()) * size
            consistency_total += float(consistency.detach().cpu()) * size

        current_coefficients = coefficients.detach().cpu().numpy().astype(np.float64)
        current_intercept = float(intercept.detach().cpu())
        validation = evaluate_e4_validation(
            cifake_validation,
            sid_validation_real,
            current_coefficients,
            current_intercept,
            thresholds=thresholds,
            sid_real_fpr_constraint=sid_real_fpr_constraint,
        )
        rank = validation_rank(validation)
        improved = best_rank is None or rank > best_rank
        if improved:
            best_rank = rank
            best_epoch = epoch
            best_coefficients = current_coefficients.copy()
            best_intercept = current_intercept
            best_validation = validation
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        group_counts = np.bincount(schedule["group"], minlength=3)
        history.append(
            {
                "epoch": epoch,
                "supervised_loss": supervised_total / observed,
                "consistency_loss": consistency_total / observed,
                "total_loss": (
                    supervised_total + consistency_weight * consistency_total
                )
                / observed,
                "examples": observed,
                "group_counts": {
                    "cifake_ai_generated": int(group_counts[0]),
                    "cifake_real": int(group_counts[1]),
                    "sid_set_train_real": int(group_counts[2]),
                },
                "validation": validation,
                "selected_as_best": improved,
            }
        )
        print(
            f"epoch={epoch:02d} loss={history[-1]['total_loss']:.6f} "
            f"threshold={validation['threshold']:.3f} "
            f"sid_worst_fpr={validation['worst_sid_real_fpr']:.3f} "
            f"cifake_mean_ba={validation['mean_cifake_balanced_accuracy']:.6f} "
            f"cifake_worst_ba={validation['worst_cifake_balanced_accuracy']:.6f} "
            f"feasible={validation['constraint_satisfied']} best_epoch={best_epoch}",
            flush=True,
        )
        if epochs_without_improvement >= early_stopping_patience:
            break

    if best_coefficients is None or best_validation is None:
        raise RuntimeError("No E4 epoch was selected")
    return E4TrainingResult(
        coefficients=best_coefficients,
        intercept=best_intercept,
        threshold=float(best_validation["threshold"]),
        best_epoch=best_epoch,
        best_validation=best_validation,
        initial_validation=initial_validation,
        history=history,
        stopped_early=len(history) < maximum_epochs,
        epochs_completed=len(history),
    )


def combined_cache_digest(datasets: dict[str, PairedFeatureSet], split: str) -> str:
    """Hash canonical cache provenance for one E4 train or validation group."""

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


def save_e4_checkpoint(
    path: Path,
    *,
    result: E4TrainingResult,
    initialization: LinearProbeCheckpoint,
    protocol_sha256: str,
    initial_checkpoint_sha256: str,
    train_cache_sha256: str,
    validation_cache_sha256: str,
    seed: int,
    consistency_weight: float,
    sid_real_fpr_constraint: float,
) -> None:
    """Save an inference-compatible, object-free E4 checkpoint."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as temporary_directory:
        temporary_path = Path(temporary_directory) / path.name
        np.savez(
            temporary_path,
            checkpoint_version=np.asarray(LINEAR_PROBE_CHECKPOINT_VERSION, dtype=np.int64),
            classifier_name=np.asarray(LINEAR_PROBE_NAME),
            coefficients=np.asarray(result.coefficients, dtype=np.float64),
            intercept=np.asarray(result.intercept, dtype=np.float64),
            classes=np.asarray([0, 1], dtype=np.int64),
            regularization_c=np.asarray(initialization.regularization_c, dtype=np.float64),
            threshold=np.asarray(result.threshold, dtype=np.float64),
            feature_dimension=np.asarray(CLIP_FEATURE_DIMENSION, dtype=np.int64),
            feature_model_name=np.asarray(CLIP_MODEL_NAME),
            feature_pretrained=np.asarray(CLIP_PRETRAINED),
            features_normalized=np.asarray(True),
            seed=np.asarray(seed, dtype=np.int64),
            selected_validation_roc_auc=np.asarray(
                result.best_validation["mean_cifake_roc_auc"], dtype=np.float64
            ),
            train_cache_sha256=np.asarray(train_cache_sha256),
            validation_cache_sha256=np.asarray(validation_cache_sha256),
            robust_checkpoint_kind=np.asarray(E4_CHECKPOINT_KIND),
            experiment=np.asarray("E4_sid_real_domain_adaptation"),
            protocol_sha256=np.asarray(protocol_sha256),
            initial_checkpoint_sha256=np.asarray(initial_checkpoint_sha256),
            consistency_weight=np.asarray(consistency_weight, dtype=np.float64),
            selected_best_epoch=np.asarray(result.best_epoch, dtype=np.int64),
            sid_real_fpr_constraint=np.asarray(sid_real_fpr_constraint, dtype=np.float64),
            threshold_selection_kind=np.asarray("validation_sid_real_fpr_constrained"),
            selected_validation_json=np.asarray(
                json.dumps(result.best_validation, sort_keys=True)
            ),
            source_group_weights_json=np.asarray(
                json.dumps(
                    {
                        "cifake_ai_generated": 0.5,
                        "cifake_real": 0.25,
                        "sid_set_train_real": 0.25,
                    },
                    sort_keys=True,
                )
            ),
        )
        temporary_path.replace(path)
    loaded = load_linear_probe_checkpoint(path)
    if loaded.threshold != result.threshold:
        raise ValueError("Saved E4 checkpoint threshold changed")
