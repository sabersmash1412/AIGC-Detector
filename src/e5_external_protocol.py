"""Validation helpers for the frozen E5 fresh-external evaluation lock."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from src.image_transforms import (
    DEFAULT_ROBUSTNESS_CONDITIONS,
    FULL_ROBUSTNESS_CONDITIONS,
)
from src.robust_linear_training import sha256_file


EXPECTED_PRIMARY_CONDITIONS = DEFAULT_ROBUSTNESS_CONDITIONS
EXPECTED_STRESS_CONDITIONS = tuple(
    condition
    for condition in FULL_ROBUSTNESS_CONDITIONS
    if condition not in DEFAULT_ROBUSTNESS_CONDITIONS
)


def _require_sha256(value: str, description: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{description} must be a SHA-256 hash")


def _require_md5(value: str, description: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{32}", value):
        raise ValueError(f"{description} must be an MD5 hash")


def validate_e5_external_protocol(protocol: dict[str, Any]) -> None:
    """Reject source, model, threshold, evaluation, or claim drift."""

    if protocol["status"] != "frozen_before_external_download_or_feature_inspection":
        raise ValueError("E5 external protocol must be frozen before data access")
    if "universal" not in protocol["research_claim_limit"].lower():
        raise ValueError("E5 external claim limitation must remain explicit")

    frozen = protocol["frozen_e5"]
    for key in (
        "development_protocol_sha256",
        "training_report_sha256",
        "checkpoint_sha256",
    ):
        _require_sha256(frozen[key], f"frozen E5 {key}")
    if frozen["checkpoint_kind"] != "e5_source_matched_risk_controlled_linear_head_v1":
        raise ValueError("Unexpected E5 checkpoint kind")
    expected_selection = {
        "selected_anchor_weight": 0.01,
        "selected_epoch": 40,
        "real_threshold": 0.23700000000000002,
        "ai_threshold": 0.8170000000000001,
        "binary_benchmark_threshold": 0.52,
    }
    for key, expected in expected_selection.items():
        if float(frozen[key]) != expected:
            raise ValueError(f"Frozen E5 selection changed: {key}")
    if not float(frozen["real_threshold"]) < float(frozen["ai_threshold"]):
        raise ValueError("E5 external evaluation requires an uncertainty interval")

    dataset = protocol["external_dataset"]
    if dataset["benchmark_doi"] != "10.5281/zenodo.10066460":
        raise ValueError("Unexpected E5 external benchmark")
    synthetic = dataset["synthetic"]
    real = dataset["real"]
    if synthetic["selected_generator"] != "Adobe Firefly":
        raise ValueError("The one-time external generator must remain Adobe Firefly")
    if synthetic["selected_generator_key"] != "firefly":
        raise ValueError("Unexpected Firefly archive key")
    if real["source_name"] != "RAISE-1k":
        raise ValueError("The fresh authentic source must remain RAISE-1k")
    if int(synthetic["expected_images"]) != 1000 or int(real["expected_images"]) != 1000:
        raise ValueError("The fresh external evaluation must retain 1,000 images per class")
    if int(synthetic["project_label"]) != 1 or int(real["project_label"]) != 0:
        raise ValueError("External class mapping changed")
    _require_md5(synthetic["source_archive_md5"], "Synthbuster archive")
    _require_md5(real["source_archive_md5"], "RAISE-1k archive")
    if synthetic["generator_family_absent_from_e5_development"] is not True:
        raise ValueError("External generator is no longer held out")
    if real["authentic_source_absent_from_e5_development"] is not True:
        raise ValueError("External authentic source is no longer held out")
    for key in (
        "organiser_validation_subset_used",
        "organiser_coco_val2017_forbidden",
        "organiser_dalle_advanced_forbidden",
    ):
        expected = False if key == "organiser_validation_subset_used" else True
        if dataset[key] is not expected:
            raise ValueError(f"Organiser validation exclusion changed: {key}")

    integrity = protocol["acquisition_and_integrity"]
    if integrity["download_only_after_protocol_commit"] is not True:
        raise ValueError("External data may only be downloaded after the lock is committed")
    for key in (
        "manual_image_exclusion_allowed",
        "manual_image_inspection_before_scoring_allowed",
        "raw_images_committed_to_git",
        "individual_images_allowed_in_public_demo",
    ):
        if integrity[key] is not False:
            raise ValueError(f"External integrity/privacy rule changed: {key}")
    if "exactly 1000" not in integrity["count_rule"]:
        raise ValueError("External class-count rule changed")
    if "fatal" not in integrity["deduplication_rule"]:
        raise ValueError("External overlap must remain a fatal violation")

    representation = protocol["representation"]
    expected_representation = {
        "model_name": "ViT-B-32-quickgelu",
        "pretrained": "openai",
        "feature_dimension": 512,
        "l2_normalized": True,
        "encoder_trainable_parameters": 0,
        "extra_source_alignment_or_resizing_allowed": False,
    }
    for key, expected in expected_representation.items():
        if representation[key] != expected:
            raise ValueError(f"External representation changed: {key}")

    conditions = protocol["evaluation_conditions"]
    if tuple(conditions["full_matrix"]) != FULL_ROBUSTNESS_CONDITIONS:
        raise ValueError("External full transformation matrix changed")
    if tuple(conditions["primary_claim_conditions"]) != EXPECTED_PRIMARY_CONDITIONS:
        raise ValueError("External primary claim conditions changed")
    if tuple(conditions["additional_stress_test_conditions"]) != EXPECTED_STRESS_CONDITIONS:
        raise ValueError("External stress-test conditions changed")
    if int(conditions["transform_seed"]) != 42:
        raise ValueError("External transform seed changed")

    criteria = protocol["frozen_success_criteria"]
    expected_criteria = {
        "confidence_level": 0.95,
        "maximum_wilson_upper_real_called_ai": 0.05,
        "maximum_wilson_upper_ai_called_real": 0.1,
        "minimum_clean_decisive_coverage_per_class": 0.6,
        "minimum_worst_primary_condition_decisive_coverage": 0.25,
        "minimum_mean_primary_condition_decisive_coverage": 0.5,
        "minimum_worst_primary_condition_roc_auc": 0.8,
        "all_criteria_required_for_pass": True,
    }
    for key, expected in expected_criteria.items():
        if criteria[key] != expected:
            raise ValueError(f"External success criterion changed: {key}")
    if not criteria["failure_rule"].startswith("Report E5 as not externally validated"):
        raise ValueError("External failure action changed")

    reporting = protocol["reporting"]
    if reporting["single_use"] is not True:
        raise ValueError("Fresh external evaluation must remain single-use")
    for key in (
        "comparison_models_can_change_e5_pass_decision",
        "retraining_after_results_allowed",
        "threshold_changes_after_results_allowed",
        "model_reselection_after_results_allowed",
    ):
        if reporting[key] is not False:
            raise ValueError(f"Post-test guardrail changed: {key}")


def validate_frozen_e5_external_inputs(
    protocol: dict[str, Any], project_root: Path
) -> None:
    """Verify the exact E5 protocol, report, and checkpoint frozen for the test."""

    frozen = protocol["frozen_e5"]
    records = (
        (frozen["development_protocol"], frozen["development_protocol_sha256"]),
        (frozen["training_report"], frozen["training_report_sha256"]),
        (frozen["checkpoint"], frozen["checkpoint_sha256"]),
    )
    for relative, expected in records:
        path = project_root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Frozen E5 external input changed or is missing: {path}")

    checkpoint_path = project_root / frozen["checkpoint"]
    with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
        if checkpoint["robust_checkpoint_kind"].item() != frozen["checkpoint_kind"]:
            raise ValueError("E5 checkpoint metadata kind changed")
        checkpoint_values = {
            "anchor_weight": float(checkpoint["anchor_weight"].item()),
            "selected_best_epoch": int(checkpoint["selected_best_epoch"].item()),
            "real_threshold": float(checkpoint["real_threshold"].item()),
            "ai_threshold": float(checkpoint["ai_threshold"].item()),
            "binary_benchmark_threshold": float(
                checkpoint["binary_benchmark_threshold"].item()
            ),
        }
    expected_values = {
        "anchor_weight": float(frozen["selected_anchor_weight"]),
        "selected_best_epoch": int(frozen["selected_epoch"]),
        "real_threshold": float(frozen["real_threshold"]),
        "ai_threshold": float(frozen["ai_threshold"]),
        "binary_benchmark_threshold": float(frozen["binary_benchmark_threshold"]),
    }
    if checkpoint_values != expected_values:
        raise ValueError("E5 checkpoint metadata does not match the external lock")

    training_report = json.loads(
        (project_root / frozen["training_report"]).read_text(encoding="utf-8")
    )
    if training_report["accepted_for_fresh_external_evaluation"] is not True:
        raise ValueError("E5 was not accepted for fresh external evaluation")
    if training_report["development_data"]["test_images_or_features_loaded"] is not False:
        raise ValueError("E5 training report records test-data leakage")
    if training_report["development_data"]["prior_audit_images_or_features_loaded"] is not False:
        raise ValueError("E5 training report records prior-audit leakage")
    if training_report["development_data"]["organiser_validation_subset_used"] is not False:
        raise ValueError("E5 training report records organiser-data leakage")


def external_artifact_paths(protocol: dict[str, Any], project_root: Path) -> tuple[Path, ...]:
    """Return all external data/result paths that must be absent at lock time."""

    integrity = protocol["acquisition_and_integrity"]
    return tuple(
        project_root / integrity[key]
        for key in (
            "raw_root",
            "manifest",
            "provenance",
            "clean_feature_cache",
            "transformed_feature_root",
            "evaluation_report",
            "probability_output",
        )
    )


def validate_external_artifacts_absent(
    protocol: dict[str, Any], project_root: Path
) -> None:
    """Prove that target external data/results did not exist when the lock was made."""

    present = [path for path in external_artifact_paths(protocol, project_root) if path.exists()]
    if present:
        joined = ", ".join(str(path) for path in present)
        raise ValueError(f"External artifacts existed before protocol freeze: {joined}")
