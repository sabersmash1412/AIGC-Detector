"""Safe feature-cache and linear-probe checkpoint interfaces."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import expit

from src.clip_features import CLIP_FEATURE_DIMENSION, CLIP_MODEL_NAME, CLIP_PRETRAINED


LINEAR_PROBE_CHECKPOINT_VERSION = 1
LINEAR_PROBE_NAME = "clip_logistic_regression_v1"
REQUIRED_FEATURE_CACHE_KEYS = {
    "format_version",
    "features",
    "labels",
    "image_paths",
    "split",
    "model_name",
    "pretrained",
    "manifest_sha256",
    "normalized",
}
REQUIRED_CHECKPOINT_KEYS = {
    "checkpoint_version",
    "classifier_name",
    "coefficients",
    "intercept",
    "classes",
    "regularization_c",
    "threshold",
    "feature_dimension",
    "feature_model_name",
    "feature_pretrained",
    "features_normalized",
    "seed",
    "selected_validation_roc_auc",
    "train_cache_sha256",
    "validation_cache_sha256",
}


def _scalar(container: Any, key: str) -> Any:
    value = container[key]
    if value.ndim != 0:
        raise ValueError(f"Metadata {key!r} must be a scalar")
    return value.item()


@dataclass(frozen=True)
class FeatureCache:
    """In-memory arrays and provenance loaded from a safe NPZ feature cache."""

    features: np.ndarray
    labels: np.ndarray
    image_paths: np.ndarray
    split: str
    model_name: str
    pretrained: str
    manifest_sha256: str


def load_feature_cache(path: Path, expected_split: str) -> FeatureCache:
    """Load and validate an embedding cache without permitting pickle objects."""

    if not path.is_file():
        raise FileNotFoundError(f"Feature cache not found: {path}")

    with np.load(path, allow_pickle=False) as cache:
        missing = REQUIRED_FEATURE_CACHE_KEYS.difference(cache.files)
        if missing:
            raise ValueError(f"Feature cache is missing keys: {sorted(missing)}")

        split = str(_scalar(cache, "split"))
        model_name = str(_scalar(cache, "model_name"))
        pretrained = str(_scalar(cache, "pretrained"))
        manifest_sha256 = str(_scalar(cache, "manifest_sha256"))
        normalized = bool(_scalar(cache, "normalized"))
        features = cache["features"]
        labels = cache["labels"]
        image_paths = cache["image_paths"]

    if split != expected_split:
        raise ValueError(f"Expected split {expected_split!r}, found {split!r}")
    if model_name != CLIP_MODEL_NAME or pretrained != CLIP_PRETRAINED:
        raise ValueError("Feature cache does not match the configured CLIP representation")
    if not normalized:
        raise ValueError("Feature cache must contain L2-normalized embeddings")
    if features.ndim != 2 or features.shape[1] != CLIP_FEATURE_DIMENSION:
        raise ValueError(f"Invalid feature shape: {features.shape}")
    if features.dtype != np.float32:
        raise ValueError(f"Features must be float32, found {features.dtype}")
    if labels.shape != (len(features),) or labels.dtype != np.int64:
        raise ValueError("Labels have the wrong shape or dtype")
    if image_paths.shape != (len(features),) or image_paths.dtype.kind not in {"U", "S"}:
        raise ValueError("Image paths have the wrong shape or dtype")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("Feature cache must contain both binary labels 0 and 1")
    if len(set(image_paths.tolist())) != len(image_paths):
        raise ValueError("Feature-cache paths must be unique")
    if not bool(np.isfinite(features).all()):
        raise ValueError("Features contain NaN or infinite values")
    maximum_norm_error = float(
        np.max(np.abs(np.linalg.vector_norm(features, axis=1) - 1.0))
    )
    if maximum_norm_error > 2e-5:
        raise ValueError("Feature cache is not L2-normalized")

    return FeatureCache(
        features=features,
        labels=labels,
        image_paths=image_paths,
        split=split,
        model_name=model_name,
        pretrained=pretrained,
        manifest_sha256=manifest_sha256,
    )


@dataclass(frozen=True)
class LinearProbeCheckpoint:
    """Portable logistic-regression weights and required inference metadata."""

    coefficients: np.ndarray
    intercept: float
    regularization_c: float
    threshold: float
    seed: int
    selected_validation_roc_auc: float
    train_cache_sha256: str
    validation_cache_sha256: str

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        """Return the probability of label 1 (AI-generated) for each row."""

        return linear_probe_probabilities(features, self.coefficients, self.intercept)


def linear_probe_probabilities(
    features: np.ndarray, coefficients: np.ndarray, intercept: float
) -> np.ndarray:
    """Apply a binary logistic-regression head to CLIP features."""

    feature_array = np.asarray(features)
    coefficient_array = np.asarray(coefficients)
    if feature_array.ndim != 2 or feature_array.shape[1] != CLIP_FEATURE_DIMENSION:
        raise ValueError(
            f"Expected features shaped (N, {CLIP_FEATURE_DIMENSION}), "
            f"got {feature_array.shape}"
        )
    if coefficient_array.shape != (CLIP_FEATURE_DIMENSION,):
        raise ValueError(
            f"Expected {CLIP_FEATURE_DIMENSION} coefficients, got {coefficient_array.shape}"
        )
    if not bool(np.isfinite(feature_array).all()) or not bool(
        np.isfinite(coefficient_array).all()
    ):
        raise ValueError("Features and coefficients must be finite")
    if not np.isfinite(intercept):
        raise ValueError("Intercept must be finite")
    return expit(feature_array @ coefficient_array + float(intercept))


def save_linear_probe_checkpoint(path: Path, checkpoint: LinearProbeCheckpoint) -> None:
    """Atomically save the portable checkpoint as an object-free NPZ archive."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as temporary_directory:
        temporary_path = Path(temporary_directory) / path.name
        np.savez(
            temporary_path,
            checkpoint_version=np.asarray(LINEAR_PROBE_CHECKPOINT_VERSION, dtype=np.int64),
            classifier_name=np.asarray(LINEAR_PROBE_NAME),
            coefficients=np.asarray(checkpoint.coefficients, dtype=np.float64),
            intercept=np.asarray(checkpoint.intercept, dtype=np.float64),
            classes=np.asarray([0, 1], dtype=np.int64),
            regularization_c=np.asarray(checkpoint.regularization_c, dtype=np.float64),
            threshold=np.asarray(checkpoint.threshold, dtype=np.float64),
            feature_dimension=np.asarray(CLIP_FEATURE_DIMENSION, dtype=np.int64),
            feature_model_name=np.asarray(CLIP_MODEL_NAME),
            feature_pretrained=np.asarray(CLIP_PRETRAINED),
            features_normalized=np.asarray(True),
            seed=np.asarray(checkpoint.seed, dtype=np.int64),
            selected_validation_roc_auc=np.asarray(
                checkpoint.selected_validation_roc_auc, dtype=np.float64
            ),
            train_cache_sha256=np.asarray(checkpoint.train_cache_sha256),
            validation_cache_sha256=np.asarray(checkpoint.validation_cache_sha256),
        )
        temporary_path.replace(path)


def load_linear_probe_checkpoint(path: Path) -> LinearProbeCheckpoint:
    """Load and strictly validate an object-free linear-probe checkpoint."""

    if not path.is_file():
        raise FileNotFoundError(f"Linear-probe checkpoint not found: {path}")

    with np.load(path, allow_pickle=False) as archive:
        missing = REQUIRED_CHECKPOINT_KEYS.difference(archive.files)
        if missing:
            raise ValueError(f"Linear-probe checkpoint is missing keys: {sorted(missing)}")
        if int(_scalar(archive, "checkpoint_version")) != LINEAR_PROBE_CHECKPOINT_VERSION:
            raise ValueError("Unsupported linear-probe checkpoint version")
        if str(_scalar(archive, "classifier_name")) != LINEAR_PROBE_NAME:
            raise ValueError("Unsupported linear-probe classifier")
        if int(_scalar(archive, "feature_dimension")) != CLIP_FEATURE_DIMENSION:
            raise ValueError("Checkpoint feature dimension does not match CLIP")
        if str(_scalar(archive, "feature_model_name")) != CLIP_MODEL_NAME:
            raise ValueError("Checkpoint CLIP model does not match")
        if str(_scalar(archive, "feature_pretrained")) != CLIP_PRETRAINED:
            raise ValueError("Checkpoint pretrained tag does not match")
        if bool(_scalar(archive, "features_normalized")) is not True:
            raise ValueError("Checkpoint requires normalized CLIP features")
        classes = archive["classes"]
        coefficients = archive["coefficients"]
        if not np.array_equal(classes, np.asarray([0, 1], dtype=np.int64)):
            raise ValueError("Checkpoint class mapping must be [0, 1]")
        if coefficients.shape != (CLIP_FEATURE_DIMENSION,):
            raise ValueError("Checkpoint coefficients have the wrong shape")

        checkpoint = LinearProbeCheckpoint(
            coefficients=coefficients,
            intercept=float(_scalar(archive, "intercept")),
            regularization_c=float(_scalar(archive, "regularization_c")),
            threshold=float(_scalar(archive, "threshold")),
            seed=int(_scalar(archive, "seed")),
            selected_validation_roc_auc=float(
                _scalar(archive, "selected_validation_roc_auc")
            ),
            train_cache_sha256=str(_scalar(archive, "train_cache_sha256")),
            validation_cache_sha256=str(_scalar(archive, "validation_cache_sha256")),
        )

    if checkpoint.regularization_c <= 0.0:
        raise ValueError("Checkpoint regularization C must be positive")
    if not 0.0 <= checkpoint.threshold <= 1.0:
        raise ValueError("Checkpoint threshold must be in [0, 1]")
    if not bool(np.isfinite(checkpoint.coefficients).all()):
        raise ValueError("Checkpoint coefficients must be finite")
    return checkpoint
