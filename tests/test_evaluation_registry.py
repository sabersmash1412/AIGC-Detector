from __future__ import annotations

import copy
from pathlib import Path

import pytest

from src.evaluation_registry import (
    load_evaluation_registry,
    validate_evaluation_registry,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "configs/evaluation_registry.json"


def _registry() -> dict:
    return load_evaluation_registry(REGISTRY_PATH)


def _unit(registry: dict, unit_id: str) -> dict:
    return next(unit for unit in registry["dataset_units"] if unit["id"] == unit_id)


def test_checked_in_evaluation_registry_and_artifact_hashes_pass() -> None:
    roles = validate_evaluation_registry(_registry(), ROOT)

    assert roles["development_train"] == 3
    assert roles["development_validation"] == 3
    assert roles["consumed_test"] == 3
    assert roles["locked_test"] == 0
    assert roles["prohibited"] == 1


def test_aigibench_is_consumed_and_cannot_be_a_fresh_e6_claim() -> None:
    registry = _registry()
    aigibench = _unit(registry, "aigibench_midjourney_v6")

    assert aigibench["role"] == "consumed_test"
    assert aigibench["results_viewed"] is True
    assert aigibench["eligible_for_fresh_external_claim"] is False
    assert aigibench["allowed_for_training"] is False
    assert aigibench["allowed_for_threshold_selection"] is False
    assert aigibench["real_source_novelty_vs_development"] is False
    assert aigibench["generator_novelty_vs_development"] is True


def test_registry_detects_aigibench_open_images_source_overlap() -> None:
    registry = copy.deepcopy(_registry())
    aigibench = _unit(registry, "aigibench_midjourney_v6")
    aigibench["real_source_novelty_vs_development"] = True

    with pytest.raises(ValueError, match="real-source novelty is incorrect"):
        validate_evaluation_registry(registry, ROOT, verify_artifacts=False)


def test_consumed_test_cannot_be_relabelled_as_training_data() -> None:
    registry = copy.deepcopy(_registry())
    aigibench = _unit(registry, "aigibench_midjourney_v6")
    aigibench["allowed_for_training"] = True

    with pytest.raises(ValueError, match="cannot be reused"):
        validate_evaluation_registry(registry, ROOT, verify_artifacts=False)


def test_viewed_data_cannot_be_marked_as_a_locked_test() -> None:
    registry = copy.deepcopy(_registry())
    aigibench = _unit(registry, "aigibench_midjourney_v6")
    aigibench["role"] = "locked_test"
    aigibench["eligible_for_fresh_external_claim"] = True
    aigibench["preregistered_before_access"] = True
    registry["current_locked_test_ids"] = ["aigibench_midjourney_v6"]

    with pytest.raises(ValueError, match="cannot have viewed"):
        validate_evaluation_registry(registry, ROOT, verify_artifacts=False)


def test_organiser_validation_subset_must_remain_prohibited() -> None:
    registry = copy.deepcopy(_registry())
    organiser = _unit(registry, "organiser_validation_coco_dalle")
    organiser["used"] = True

    with pytest.raises(ValueError, match="must remain unused"):
        validate_evaluation_registry(registry, ROOT, verify_artifacts=False)


def test_registry_detects_frozen_manifest_hash_drift() -> None:
    registry = copy.deepcopy(_registry())
    cifake_train = _unit(registry, "cifake_train")
    cifake_train["artifact"]["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="artifact hash changed"):
        validate_evaluation_registry(registry, ROOT)
