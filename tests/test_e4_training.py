from pathlib import Path

import numpy as np
import pytest
import torch

from src.e4_training import (
    E4TrainingResult,
    build_source_balanced_schedule,
    save_e4_checkpoint,
    select_constrained_threshold,
    train_e4_linear_head,
    validation_rank,
)
from src.linear_probe import FeatureCache, LinearProbeCheckpoint, load_linear_probe_checkpoint
from src.robust_linear_training import PairedFeatureSet
from src.transformed_features import TransformedFeatureCache


CONDITIONS = ("jpeg_q50", "gaussian_blur_sigma1")


def _unit_features(values: list[tuple[float, float]]) -> np.ndarray:
    features = np.zeros((len(values), 512), dtype=np.float32)
    for index, value in enumerate(values):
        vector = np.asarray(value, dtype=np.float32)
        vector /= np.linalg.vector_norm(vector)
        features[index, :2] = vector
    return features


def _paired(split: str, *, real_only: bool) -> PairedFeatureSet:
    if real_only:
        labels = np.asarray([0, 0], dtype=np.int64)
        clean_features = _unit_features([(-1.0, 0.2), (-1.0, -0.2)])
    else:
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        clean_features = _unit_features(
            [(-1.0, 0.2), (-1.0, -0.2), (1.0, 0.2), (1.0, -0.2)]
        )
    paths = np.asarray([f"{split}-{index}.jpg" for index in range(len(labels))])
    clean = FeatureCache(
        features=clean_features,
        labels=labels,
        image_paths=paths,
        split=split,
        model_name="ViT-B-32-quickgelu",
        pretrained="openai",
        manifest_sha256=f"{split}-manifest",
    )
    transformed = tuple(
        TransformedFeatureCache(
            features=_unit_features(
                [(float(row[0]), float(row[1]) + 0.1 * (condition_index + 1)) for row in clean_features[:, :2]]
            ),
            labels=labels,
            image_paths=paths,
            split=split,
            condition=condition,
            transform_seed=42,
            clean_cache_sha256=f"{split}-clean",
        )
        for condition_index, condition in enumerate(CONDITIONS)
    )
    return PairedFeatureSet(
        clean=clean,
        transformed=transformed,
        conditions=CONDITIONS,
        clean_cache_path=Path(f"{split}.npz"),
        transformed_cache_paths=tuple(
            Path(f"{split}-{condition}.npz") for condition in CONDITIONS
        ),
        clean_cache_sha256=f"{split}-clean",
        transformed_cache_sha256=tuple(
            f"{split}-{condition}-hash" for condition in CONDITIONS
        ),
    )


def _initialization() -> LinearProbeCheckpoint:
    coefficients = np.zeros(512, dtype=np.float64)
    coefficients[0] = 1.0
    return LinearProbeCheckpoint(
        coefficients=coefficients,
        intercept=0.0,
        regularization_c=100.0,
        threshold=0.437,
        seed=42,
        selected_validation_roc_auc=0.9,
        train_cache_sha256="old-train",
        validation_cache_sha256="old-val",
    )


def test_constrained_threshold_selects_lowest_tied_feasible_threshold() -> None:
    labels = np.asarray([0, 0, 1, 1])
    cifake = {"clean": np.asarray([0.1, 0.2, 0.8, 0.9])}
    sid = {"clean": np.asarray([0.1, 0.2, 0.3, 0.4])}

    result = select_constrained_threshold(
        labels,
        cifake,
        sid,
        thresholds=np.asarray([0.2, 0.3, 0.4, 0.5, 0.6]),
        sid_real_fpr_constraint=0.25,
    )

    assert result["constraint_satisfied"] is True
    assert result["fallback_used"] is False
    assert result["threshold"] == pytest.approx(0.4)
    assert result["worst_sid_real_fpr"] == pytest.approx(0.25)
    assert result["mean_cifake_balanced_accuracy"] == pytest.approx(1.0)


def test_constrained_threshold_reports_infeasible_fallback() -> None:
    labels = np.asarray([0, 0, 1, 1])
    cifake = {"clean": np.asarray([0.1, 0.2, 0.8, 0.9])}
    sid = {"clean": np.asarray([0.8, 0.8, 0.9, 0.9])}

    result = select_constrained_threshold(
        labels,
        cifake,
        sid,
        thresholds=np.asarray([0.1, 0.2]),
        sid_real_fpr_constraint=0.05,
    )

    assert result["constraint_satisfied"] is False
    assert result["fallback_used"] is True
    assert result["threshold"] == pytest.approx(0.2)
    assert result["worst_sid_real_fpr"] == pytest.approx(1.0)


def test_source_balanced_schedule_has_exact_frozen_group_weights() -> None:
    schedule = build_source_balanced_schedule(
        _paired("train", real_only=False),
        _paired("sid-train", real_only=True),
        seed=404,
        epoch=1,
    )

    counts = np.bincount(schedule["group"], minlength=3)
    assert counts.tolist() == [4, 2, 2]
    assert np.sum(schedule["source"] == 0) == 6
    assert np.sum(schedule["source"] == 1) == 2
    assert all(len(values) == 8 for values in schedule.values())
    repeated = build_source_balanced_schedule(
        _paired("train", real_only=False),
        _paired("sid-train", real_only=True),
        seed=404,
        epoch=1,
    )
    assert all(np.array_equal(schedule[key], repeated[key]) for key in schedule)


def test_validation_rank_prefers_constraint_then_utility() -> None:
    infeasible = {
        "constraint_satisfied": False,
        "worst_sid_real_fpr": 0.01,
        "mean_cifake_balanced_accuracy": 1.0,
        "worst_cifake_balanced_accuracy": 1.0,
    }
    feasible = {
        "constraint_satisfied": True,
        "worst_sid_real_fpr": 0.05,
        "mean_cifake_balanced_accuracy": 0.7,
        "worst_cifake_balanced_accuracy": 0.6,
    }

    assert validation_rank(feasible) > validation_rank(infeasible)


def test_e4_training_runs_with_source_balanced_toy_data() -> None:
    result = train_e4_linear_head(
        cifake_train=_paired("train", real_only=False),
        sid_train_real=_paired("sid-train", real_only=True),
        cifake_validation=_paired("val", real_only=False),
        sid_validation_real=_paired("sid-val", real_only=True),
        initialization=_initialization(),
        device=torch.device("cpu"),
        seed=404,
        batch_size=4,
        maximum_epochs=2,
        learning_rate=0.01,
        weight_decay=0.0,
        early_stopping_patience=2,
        consistency_weight=1.0,
        thresholds=np.asarray([0.4, 0.5, 0.6]),
        sid_real_fpr_constraint=0.5,
    )

    assert result.epochs_completed == 2
    assert result.history[0]["group_counts"] == {
        "cifake_ai_generated": 4,
        "cifake_real": 2,
        "sid_set_train_real": 2,
    }
    assert np.isfinite(result.coefficients).all()
    assert result.threshold in {0.4, 0.5, 0.6}


def test_e4_checkpoint_remains_inference_compatible(tmp_path: Path) -> None:
    result = E4TrainingResult(
        coefficients=np.zeros(512, dtype=np.float64),
        intercept=0.1,
        threshold=0.7,
        best_epoch=3,
        best_validation={"mean_cifake_roc_auc": 0.9},
        initial_validation={},
        history=[],
        stopped_early=False,
        epochs_completed=3,
    )
    path = tmp_path / "e4.npz"

    save_e4_checkpoint(
        path,
        result=result,
        initialization=_initialization(),
        protocol_sha256="protocol-hash",
        initial_checkpoint_sha256="initial-hash",
        train_cache_sha256="train-hash",
        validation_cache_sha256="val-hash",
        seed=404,
        consistency_weight=1.0,
        sid_real_fpr_constraint=0.05,
    )

    loaded = load_linear_probe_checkpoint(path)
    assert loaded.threshold == pytest.approx(0.7)
    assert loaded.intercept == pytest.approx(0.1)
    with np.load(path, allow_pickle=False) as archive:
        assert archive["robust_checkpoint_kind"].item() == "e4_domain_adapted_linear_head_v1"
        assert archive["selected_best_epoch"].item() == 3
