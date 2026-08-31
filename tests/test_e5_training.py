from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.e5_training import (
    E5CandidateResult,
    E5TrainingOutcome,
    build_four_group_schedule,
    save_e5_checkpoint,
    select_risk_controlled_thresholds,
    train_e5_candidates,
    validation_rank,
)
from src.linear_probe import FeatureCache, LinearProbeCheckpoint, load_linear_probe_checkpoint
from src.robust_linear_training import PairedFeatureSet
from src.transformed_features import TransformedFeatureCache


CONDITIONS = ("jpeg_q50",)


def _probability_groups(real: np.ndarray, ai: np.ndarray) -> dict:
    return {
        "cifake_real": {"clean": real, "jpeg_q50": real},
        "cifake_ai_generated": {"clean": ai, "jpeg_q50": ai},
        "sid_set_train_real": {"clean": real, "jpeg_q50": real},
        "sid_set_train_flux": {"clean": ai, "jpeg_q50": ai},
    }


def _selection(groups: dict) -> dict:
    return select_risk_controlled_thresholds(
        groups,
        thresholds=np.arange(0.1, 1.0, 0.1),
        confidence_level=0.95,
        maximum_real_called_ai_upper=0.05,
        maximum_ai_called_real_upper=0.1,
        minimum_clean_coverage=0.6,
        minimum_worst_coverage=0.25,
        minimum_mean_coverage=0.5,
    )


def test_risk_controlled_selection_finds_safe_decisive_pair() -> None:
    result = _selection(
        _probability_groups(np.full(100, 0.05), np.full(100, 0.95))
    )
    assert result["constraint_satisfied"] is True
    assert result["real_threshold"] < result["ai_threshold"]
    assert result["worst_real_called_ai_wilson_upper"] < 0.05
    assert result["worst_ai_called_real_wilson_upper"] < 0.1
    assert result["worst_source_condition_decisive_coverage"] == 1.0
    assert result["worst_decisive_accuracy"] == 1.0


def test_risk_controlled_selection_rejects_all_uncertain_solution() -> None:
    values = np.full(100, 0.5)
    result = _selection(_probability_groups(values, values))
    assert result["constraint_satisfied"] is False
    assert result["rejected"] is True
    assert result["coverage_feasible_pairs"] == 0


def _paired(labels: np.ndarray, *, split: str) -> PairedFeatureSet:
    labels = np.asarray(labels, dtype=np.int64)
    features = np.zeros((len(labels), 512), dtype=np.float32)
    features[:, 0] = np.where(labels == 1, 1.0, -1.0)
    paths = np.asarray([f"{split}-{index}.jpg" for index in range(len(labels))])
    clean = FeatureCache(
        features=features,
        labels=labels,
        image_paths=paths,
        split=split,
        model_name="ViT-B-32-quickgelu",
        pretrained="openai",
        manifest_sha256=f"{split}-manifest",
    )
    transformed = TransformedFeatureCache(
        features=features.copy(),
        labels=labels.copy(),
        image_paths=paths.copy(),
        split=split,
        condition="jpeg_q50",
        transform_seed=42,
        clean_cache_sha256=f"{split}-clean",
    )
    return PairedFeatureSet(
        clean=clean,
        transformed=(transformed,),
        conditions=CONDITIONS,
        clean_cache_path=Path(f"{split}.npz"),
        transformed_cache_paths=(Path(f"{split}/jpeg_q50.npz"),),
        clean_cache_sha256=f"{split}-clean",
        transformed_cache_sha256=(f"{split}-jpeg",),
    )


def test_four_group_schedule_is_exactly_balanced() -> None:
    cifake = _paired(np.asarray([0] * 10 + [1] * 10), split="train")
    sid_real = _paired(np.zeros(10, dtype=np.int64), split="train")
    sid_flux = _paired(np.ones(10, dtype=np.int64), split="train")
    schedule = build_four_group_schedule(
        cifake,
        sid_real,
        sid_flux,
        examples_per_epoch=32,
        seed=505,
        epoch=1,
    )
    assert np.bincount(schedule["group"], minlength=4).tolist() == [8, 8, 8, 8]
    assert len(schedule["source"]) == 32


def test_validation_rank_rejects_infeasible_candidate() -> None:
    infeasible = {
        "risk_controlled": {"constraint_satisfied": False}
    }
    assert validation_rank(infeasible, anchor_weight=0.0, epoch=1) is None


def _initialization() -> LinearProbeCheckpoint:
    coefficients = np.zeros(512, dtype=np.float64)
    coefficients[0] = 5.0
    return LinearProbeCheckpoint(
        coefficients=coefficients,
        intercept=0.0,
        regularization_c=100.0,
        threshold=0.5,
        seed=42,
        selected_validation_roc_auc=1.0,
        train_cache_sha256="train",
        validation_cache_sha256="val",
    )


def test_e5_toy_training_runs_all_anchor_candidates_and_saves(tmp_path: Path) -> None:
    cifake_train = _paired(np.asarray([0] * 100 + [1] * 100), split="train")
    real_train = _paired(np.zeros(100, dtype=np.int64), split="train")
    flux_train = _paired(np.ones(100, dtype=np.int64), split="train")
    cifake_val = _paired(np.asarray([0] * 100 + [1] * 100), split="val")
    real_val = _paired(np.zeros(100, dtype=np.int64), split="val")
    flux_val = _paired(np.ones(100, dtype=np.int64), split="val")
    outcome = train_e5_candidates(
        cifake_train=cifake_train,
        sid_real_train=real_train,
        sid_flux_train=flux_train,
        cifake_validation=cifake_val,
        sid_real_validation=real_val,
        sid_flux_validation=flux_val,
        initialization=_initialization(),
        device=torch.device("cpu"),
        anchor_weights=(0.0, 0.1),
        examples_per_epoch=400,
        seed=505,
        batch_size=64,
        maximum_epochs=1,
        learning_rate=0.001,
        weight_decay=0.0,
        early_stopping_patience=1,
        consistency_weight=1.0,
        thresholds=np.arange(0.1, 1.0, 0.1),
        confidence_level=0.95,
        maximum_real_called_ai_upper=0.05,
        maximum_ai_called_real_upper=0.1,
        minimum_clean_coverage=0.6,
        minimum_worst_coverage=0.25,
        minimum_mean_coverage=0.5,
    )
    assert outcome.accepted is True
    assert len(outcome.candidates) == 2
    assert all(candidate.epochs_completed == 1 for candidate in outcome.candidates)
    checkpoint_path = tmp_path / "e5.npz"
    save_e5_checkpoint(
        checkpoint_path,
        outcome=outcome,
        initialization=_initialization(),
        protocol_sha256="protocol",
        initial_checkpoint_sha256="initial",
        train_cache_sha256="train",
        validation_cache_sha256="val",
        seed=505,
        consistency_weight=1.0,
    )
    loaded = load_linear_probe_checkpoint(checkpoint_path)
    assert loaded.coefficients.shape == (512,)
    with np.load(checkpoint_path, allow_pickle=False) as archive:
        assert archive["robust_checkpoint_kind"].item().startswith("e5_")
        assert archive["real_threshold"].item() < archive["ai_threshold"].item()


def test_rejected_outcome_cannot_save_checkpoint(tmp_path: Path) -> None:
    rejected = E5TrainingOutcome(
        accepted=False,
        selected=None,
        candidates=(
            E5CandidateResult(
                anchor_weight=0.0,
                best_epoch=None,
                coefficients=None,
                intercept=None,
                best_validation=None,
                history=[],
                epochs_completed=1,
                stopped_early=False,
            ),
        ),
        initial_validation={},
    )
    with pytest.raises(ValueError, match="rejected"):
        save_e5_checkpoint(
            tmp_path / "rejected.npz",
            outcome=rejected,
            initialization=_initialization(),
            protocol_sha256="protocol",
            initial_checkpoint_sha256="initial",
            train_cache_sha256="train",
            validation_cache_sha256="val",
            seed=505,
            consistency_weight=1.0,
        )
