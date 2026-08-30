"""Controlled paired-feature training utilities for Section 3."""

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
    FeatureCache,
    LinearProbeCheckpoint,
    linear_probe_probabilities,
    load_feature_cache,
    load_linear_probe_checkpoint,
)
from src.metrics import binary_classification_metrics
from src.transformed_features import (
    TransformedFeatureCache,
    load_transformed_feature_cache,
)


ROBUST_CHECKPOINT_KIND = "section3_robust_linear_head_v1"


def sha256_file(path: Path) -> str:
    """Return a file's SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PairedFeatureSet:
    """Clean features and aligned transformed views for one data split."""

    clean: FeatureCache
    transformed: tuple[TransformedFeatureCache, ...]
    conditions: tuple[str, ...]
    clean_cache_path: Path
    transformed_cache_paths: tuple[Path, ...]
    clean_cache_sha256: str
    transformed_cache_sha256: tuple[str, ...]

    @property
    def samples(self) -> int:
        return len(self.clean.labels)

    @property
    def pairs(self) -> int:
        return self.samples * len(self.conditions)


def load_paired_feature_set(
    *,
    split: str,
    clean_cache_path: Path,
    transformed_feature_dir: Path,
    conditions: tuple[str, ...],
    seed: int,
) -> PairedFeatureSet:
    """Load and validate every clean/transformed pair for one split."""

    if not conditions:
        raise ValueError("At least one transformed condition is required")
    if len(set(conditions)) != len(conditions):
        raise ValueError("Transformed conditions must be unique")

    clean = load_feature_cache(clean_cache_path, split)
    clean_digest = sha256_file(clean_cache_path)
    transformed_caches: list[TransformedFeatureCache] = []
    transformed_paths: list[Path] = []
    transformed_digests: list[str] = []
    for condition in conditions:
        cache_path = transformed_feature_dir / split / f"{condition}.npz"
        transformed_caches.append(
            load_transformed_feature_cache(
                cache_path,
                expected_split=split,
                expected_condition=condition,
                expected_seed=seed,
                expected_clean_cache_sha256=clean_digest,
                reference=clean,
            )
        )
        transformed_paths.append(cache_path)
        transformed_digests.append(sha256_file(cache_path))

    return PairedFeatureSet(
        clean=clean,
        transformed=tuple(transformed_caches),
        conditions=conditions,
        clean_cache_path=clean_cache_path,
        transformed_cache_paths=tuple(transformed_paths),
        clean_cache_sha256=clean_digest,
        transformed_cache_sha256=tuple(transformed_digests),
    )


def validation_metrics(
    dataset: PairedFeatureSet,
    coefficients: np.ndarray,
    intercept: float,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate clean and transformed validation views without combining rows."""

    condition_features = {
        "clean": dataset.clean.features,
        **{
            condition: cache.features
            for condition, cache in zip(
                dataset.conditions, dataset.transformed, strict=True
            )
        },
    }
    by_condition: dict[str, Any] = {}
    for condition, features in condition_features.items():
        probabilities = linear_probe_probabilities(features, coefficients, intercept)
        metrics = binary_classification_metrics(
            dataset.clean.labels, probabilities, threshold=threshold
        )
        metrics["samples"] = dataset.samples
        by_condition[condition] = metrics

    auc_values = [row["roc_auc"] for row in by_condition.values()]
    balanced_accuracy_values = [
        row["balanced_accuracy"] for row in by_condition.values()
    ]
    return {
        "selection_mean_roc_auc": float(np.mean(auc_values)),
        "mean_balanced_accuracy_at_0_5": float(
            np.mean(balanced_accuracy_values)
        ),
        "worst_condition_roc_auc": float(np.min(auc_values)),
        "worst_condition": min(
            by_condition, key=lambda name: by_condition[name]["roc_auc"]
        ),
        "by_condition": by_condition,
    }


def _transformed_batch(
    dataset: PairedFeatureSet,
    condition_indices: np.ndarray,
    sample_indices: np.ndarray,
) -> np.ndarray:
    """Gather mixed-condition transformed rows without stacking all feature caches."""

    batch = np.empty(
        (len(sample_indices), CLIP_FEATURE_DIMENSION), dtype=np.float32
    )
    for condition_index in np.unique(condition_indices):
        positions = np.flatnonzero(condition_indices == condition_index)
        batch[positions] = dataset.transformed[int(condition_index)].features[
            sample_indices[positions]
        ]
    return batch


@dataclass(frozen=True)
class RobustTrainingResult:
    """Best E2/E3 linear parameters plus a complete training trace."""

    coefficients: np.ndarray
    intercept: float
    best_epoch: int
    best_validation: dict[str, Any]
    initial_validation: dict[str, Any]
    history: list[dict[str, Any]]
    stopped_early: bool
    epochs_completed: int


def train_paired_linear_head(
    *,
    train: PairedFeatureSet,
    validation: PairedFeatureSet,
    initialization: LinearProbeCheckpoint,
    device: torch.device,
    seed: int,
    batch_size: int,
    maximum_epochs: int,
    learning_rate: float,
    weight_decay: float,
    early_stopping_patience: int,
    consistency_weight: float,
) -> RobustTrainingResult:
    """Train a paired linear head; E2 uses a consistency weight of zero."""

    if train.conditions != validation.conditions:
        raise ValueError("Train and validation conditions must be identical")
    if batch_size <= 0 or maximum_epochs <= 0 or early_stopping_patience <= 0:
        raise ValueError("Batch size, epochs, and patience must be positive")
    if learning_rate <= 0.0 or weight_decay < 0.0 or consistency_weight < 0.0:
        raise ValueError("Learning rate must be positive and loss weights non-negative")

    torch.manual_seed(seed)
    coefficients = torch.nn.Parameter(
        torch.as_tensor(
            initialization.coefficients, dtype=torch.float32, device=device
        ).clone()
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

    initial_validation = validation_metrics(
        validation,
        initialization.coefficients,
        initialization.intercept,
    )
    best_score = float("-inf")
    best_epoch = 0
    best_coefficients: np.ndarray | None = None
    best_intercept = 0.0
    best_validation: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    epochs_without_improvement = 0
    pair_count = train.pairs
    sample_count = train.samples

    for epoch in range(1, maximum_epochs + 1):
        order = np.random.default_rng(seed + epoch).permutation(pair_count)
        epoch_supervised_loss = 0.0
        epoch_consistency_loss = 0.0
        observed_pairs = 0

        for start in range(0, pair_count, batch_size):
            pair_indices = order[start : start + batch_size]
            condition_indices = pair_indices // sample_count
            sample_indices = pair_indices % sample_count
            clean_features = torch.from_numpy(
                train.clean.features[sample_indices]
            ).to(device)
            transformed_features = torch.from_numpy(
                _transformed_batch(train, condition_indices, sample_indices)
            ).to(device)
            labels = torch.from_numpy(
                train.clean.labels[sample_indices].astype(np.float32, copy=False)
            ).to(device)

            clean_logits = clean_features @ coefficients + intercept
            transformed_logits = transformed_features @ coefficients + intercept
            supervised_loss = 0.5 * (
                F.binary_cross_entropy_with_logits(clean_logits, labels)
                + F.binary_cross_entropy_with_logits(transformed_logits, labels)
            )
            consistency_loss = F.mse_loss(
                torch.sigmoid(clean_logits), torch.sigmoid(transformed_logits)
            )
            loss = supervised_loss + consistency_weight * consistency_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            current_batch_size = len(pair_indices)
            observed_pairs += current_batch_size
            epoch_supervised_loss += float(supervised_loss.detach().cpu()) * current_batch_size
            epoch_consistency_loss += float(consistency_loss.detach().cpu()) * current_batch_size

        current_coefficients = coefficients.detach().cpu().numpy().astype(np.float64)
        current_intercept = float(intercept.detach().cpu())
        current_validation = validation_metrics(
            validation, current_coefficients, current_intercept
        )
        current_score = float(current_validation["selection_mean_roc_auc"])
        improved = current_score > best_score + 1e-12
        if improved:
            best_score = current_score
            best_epoch = epoch
            best_coefficients = current_coefficients.copy()
            best_intercept = current_intercept
            best_validation = current_validation
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        history.append(
            {
                "epoch": epoch,
                "supervised_loss": epoch_supervised_loss / observed_pairs,
                "consistency_loss_diagnostic": epoch_consistency_loss / observed_pairs,
                "total_loss": (
                    epoch_supervised_loss
                    + consistency_weight * epoch_consistency_loss
                )
                / observed_pairs,
                "validation": current_validation,
                "selected_as_best": improved,
            }
        )
        print(
            f"epoch={epoch:02d} supervised_loss={epoch_supervised_loss / observed_pairs:.6f} "
            f"val_mean_auc={current_score:.6f} "
            f"val_worst_auc={current_validation['worst_condition_roc_auc']:.6f} "
            f"best_epoch={best_epoch}",
            flush=True,
        )
        if epochs_without_improvement >= early_stopping_patience:
            break

    if best_coefficients is None or best_validation is None:
        raise RuntimeError("No robust linear-head epoch was selected")
    return RobustTrainingResult(
        coefficients=best_coefficients,
        intercept=best_intercept,
        best_epoch=best_epoch,
        best_validation=best_validation,
        initial_validation=initial_validation,
        history=history,
        stopped_early=len(history) < maximum_epochs,
        epochs_completed=len(history),
    )


def save_robust_linear_checkpoint(
    path: Path,
    *,
    result: RobustTrainingResult,
    initialization: LinearProbeCheckpoint,
    experiment: str,
    protocol_sha256: str,
    train: PairedFeatureSet,
    validation: PairedFeatureSet,
    seed: int,
    consistency_weight: float,
) -> None:
    """Save an inference-compatible object-free checkpoint with robust provenance."""

    transformed_train_hashes = dict(
        zip(train.conditions, train.transformed_cache_sha256, strict=True)
    )
    transformed_validation_hashes = dict(
        zip(
            validation.conditions,
            validation.transformed_cache_sha256,
            strict=True,
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as temporary_directory:
        temporary_path = Path(temporary_directory) / path.name
        np.savez(
            temporary_path,
            checkpoint_version=np.asarray(
                LINEAR_PROBE_CHECKPOINT_VERSION, dtype=np.int64
            ),
            classifier_name=np.asarray(LINEAR_PROBE_NAME),
            coefficients=np.asarray(result.coefficients, dtype=np.float64),
            intercept=np.asarray(result.intercept, dtype=np.float64),
            classes=np.asarray([0, 1], dtype=np.int64),
            regularization_c=np.asarray(
                initialization.regularization_c, dtype=np.float64
            ),
            threshold=np.asarray(0.5, dtype=np.float64),
            feature_dimension=np.asarray(CLIP_FEATURE_DIMENSION, dtype=np.int64),
            feature_model_name=np.asarray(CLIP_MODEL_NAME),
            feature_pretrained=np.asarray(CLIP_PRETRAINED),
            features_normalized=np.asarray(True),
            seed=np.asarray(seed, dtype=np.int64),
            selected_validation_roc_auc=np.asarray(
                result.best_validation["selection_mean_roc_auc"], dtype=np.float64
            ),
            train_cache_sha256=np.asarray(train.clean_cache_sha256),
            validation_cache_sha256=np.asarray(validation.clean_cache_sha256),
            robust_checkpoint_kind=np.asarray(ROBUST_CHECKPOINT_KIND),
            experiment=np.asarray(experiment),
            protocol_sha256=np.asarray(protocol_sha256),
            consistency_weight=np.asarray(consistency_weight, dtype=np.float64),
            selected_best_epoch=np.asarray(result.best_epoch, dtype=np.int64),
            transformed_train_cache_sha256_json=np.asarray(
                json.dumps(transformed_train_hashes, sort_keys=True)
            ),
            transformed_validation_cache_sha256_json=np.asarray(
                json.dumps(transformed_validation_hashes, sort_keys=True)
            ),
        )
        temporary_path.replace(path)
    load_linear_probe_checkpoint(path)
