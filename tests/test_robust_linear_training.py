"""Tests for controlled Section 3 paired linear-head training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.linear_probe import FeatureCache, LinearProbeCheckpoint, load_linear_probe_checkpoint
from src.robust_linear_training import (
    PairedFeatureSet,
    RobustTrainingResult,
    save_robust_linear_checkpoint,
    train_paired_linear_head,
)
from src.transformed_features import TransformedFeatureCache


def _unit_features(values: list[tuple[float, float]]) -> np.ndarray:
    features = np.zeros((len(values), 512), dtype=np.float32)
    for index, (first, second) in enumerate(values):
        vector = np.asarray([first, second], dtype=np.float32)
        vector /= np.linalg.vector_norm(vector)
        features[index, :2] = vector
    return features


def _paired(split: str) -> PairedFeatureSet:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    paths = np.asarray([f"{split}-{index}.jpg" for index in range(4)])
    clean_features = _unit_features([(-1, 0.2), (-1, -0.2), (1, 0.2), (1, -0.2)])
    transformed_features = _unit_features(
        [(-0.8, 0.4), (-0.9, -0.3), (0.8, 0.4), (0.9, -0.3)]
    )
    clean = FeatureCache(
        features=clean_features,
        labels=labels,
        image_paths=paths,
        split=split,
        model_name="ViT-B-32-quickgelu",
        pretrained="openai",
        manifest_sha256=f"{split}-manifest",
    )
    transformed = TransformedFeatureCache(
        features=transformed_features,
        labels=labels,
        image_paths=paths,
        split=split,
        condition="jpeg_q50",
        transform_seed=42,
        clean_cache_sha256=f"{split}-clean",
    )
    return PairedFeatureSet(
        clean=clean,
        transformed=(transformed,),
        conditions=("jpeg_q50",),
        clean_cache_path=Path(f"{split}.npz"),
        transformed_cache_paths=(Path(f"{split}-jpeg.npz"),),
        clean_cache_sha256=f"{split}-clean",
        transformed_cache_sha256=(f"{split}-jpeg",),
    )


def _initialization() -> LinearProbeCheckpoint:
    return LinearProbeCheckpoint(
        coefficients=np.zeros(512, dtype=np.float64),
        intercept=0.0,
        regularization_c=100.0,
        threshold=0.5,
        seed=42,
        selected_validation_roc_auc=0.5,
        train_cache_sha256="old-train",
        validation_cache_sha256="old-val",
    )


def test_e2_training_is_repeatable_and_improves_toy_validation_auc() -> None:
    arguments = {
        "train": _paired("train"),
        "validation": _paired("val"),
        "initialization": _initialization(),
        "device": torch.device("cpu"),
        "seed": 42,
        "batch_size": 4,
        "maximum_epochs": 4,
        "learning_rate": 0.05,
        "weight_decay": 0.0,
        "early_stopping_patience": 2,
        "consistency_weight": 0.0,
    }

    first = train_paired_linear_head(**arguments)
    second = train_paired_linear_head(**arguments)

    assert first.best_validation["selection_mean_roc_auc"] == 1.0
    assert np.array_equal(first.coefficients, second.coefficients)
    assert first.intercept == second.intercept
    assert first.best_epoch == second.best_epoch


def test_consistency_term_changes_training_while_preserving_valid_outputs() -> None:
    initialization = _initialization()
    initialization.coefficients[1] = 1.0
    common_arguments = {
        "train": _paired("train"),
        "validation": _paired("val"),
        "initialization": initialization,
        "device": torch.device("cpu"),
        "seed": 42,
        "batch_size": 4,
        "maximum_epochs": 4,
        "learning_rate": 0.05,
        "weight_decay": 0.0,
        "early_stopping_patience": 2,
    }

    e2 = train_paired_linear_head(**common_arguments, consistency_weight=0.0)
    e3 = train_paired_linear_head(**common_arguments, consistency_weight=1.0)

    assert e3.history[0]["consistency_loss_diagnostic"] > 0.0
    assert e3.history[0]["total_loss"] > e3.history[0]["supervised_loss"]
    assert e2.history[0]["total_loss"] == e2.history[0]["supervised_loss"]
    assert e2.history[2]["supervised_loss"] != e3.history[2]["supervised_loss"]
    assert np.isfinite(e3.coefficients).all()
    assert 0.0 <= e3.best_validation["selection_mean_roc_auc"] <= 1.0


def test_robust_checkpoint_remains_compatible_with_inference_loader(
    tmp_path: Path,
) -> None:
    path = tmp_path / "e2.npz"
    result = RobustTrainingResult(
        coefficients=np.zeros(512, dtype=np.float64),
        intercept=0.0,
        best_epoch=3,
        best_validation={"selection_mean_roc_auc": 0.8},
        initial_validation={"selection_mean_roc_auc": 0.7},
        history=[],
        stopped_early=False,
        epochs_completed=3,
    )

    save_robust_linear_checkpoint(
        path,
        result=result,
        initialization=_initialization(),
        experiment="E2_supervised_clean_plus_transformed",
        protocol_sha256="protocol-hash",
        train=_paired("train"),
        validation=_paired("val"),
        seed=42,
        consistency_weight=0.0,
    )

    loaded = load_linear_probe_checkpoint(path)
    assert loaded.selected_validation_roc_auc == 0.8
    assert loaded.threshold == 0.5
    with np.load(path, allow_pickle=False) as archive:
        assert archive["experiment"].item() == "E2_supervised_clean_plus_transformed"
        assert archive["selected_best_epoch"].item() == 3
