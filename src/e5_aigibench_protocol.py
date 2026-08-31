"""Validation helpers for the frozen AIGIBench Midjourney E5 audit."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.e5_external_protocol import (
    EXPECTED_PRIMARY_CONDITIONS,
    EXPECTED_STRESS_CONDITIONS,
    validate_frozen_e5_external_inputs,
)
from src.image_transforms import FULL_ROBUSTNESS_CONDITIONS


EXPECTED_ARCHIVE_COMMIT = "e44ec40efe5117a5ccdaa6ff0e89ed934d03d310"
EXPECTED_ARCHIVE_BYTES = 8_999_861_437
EXPECTED_MEMBER_LIST_SHA256 = (
    "2e6edd984c20835a5c870d8a965822e5168e1858617f10c1be18426f970ba113"
)


def _require_sha256(value: str, description: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{description} must be a SHA-256 hash")


def validate_e5_aigibench_protocol(protocol: dict[str, Any]) -> None:
    """Reject changes to the replacement dataset, E5 model, or audit gates."""

    if protocol["status"] != (
        "frozen_before_selected_image_payload_download_or_feature_inspection"
    ):
        raise ValueError("AIGIBench audit must be frozen before image payload access")
    if "universal" not in protocol["research_claim_limit"].lower():
        raise ValueError("AIGIBench claim limitation must remain explicit")
    replacement = protocol["replacement_context"]
    if replacement["supersedes_results_from_prior_protocol"] is not False:
        raise ValueError("The blocked Synthbuster protocol produced no result to supersede")
    if "before any image download" not in replacement["prior_protocol_outcome"]:
        raise ValueError("The pre-result replacement reason changed")

    frozen = protocol["frozen_e5"]
    for key in (
        "development_protocol_sha256",
        "training_report_sha256",
        "checkpoint_sha256",
    ):
        _require_sha256(frozen[key], f"frozen E5 {key}")
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
    if frozen["checkpoint_kind"] != "e5_source_matched_risk_controlled_linear_head_v1":
        raise ValueError("Unexpected E5 checkpoint kind")

    dataset = protocol["external_dataset"]
    if dataset["benchmark_repository"] != "HorizonTEL/AIGIBench":
        raise ValueError("Unexpected replacement benchmark repository")
    if dataset["license"] != "CC-BY-NC-SA-4.0":
        raise ValueError("AIGIBench license record changed")
    archive = dataset["archive"]
    if archive["repository_commit"] != EXPECTED_ARCHIVE_COMMIT:
        raise ValueError("AIGIBench archive commit changed")
    if int(archive["remote_bytes"]) != EXPECTED_ARCHIVE_BYTES:
        raise ValueError("AIGIBench archive byte size changed")
    if archive["byte_ranges_verified"] is not True:
        raise ValueError("AIGIBench byte-range acquisition was not verified")
    if int(archive["archive_real_count"]) != 3000 or int(
        archive["archive_ai_count"]
    ) != 3000:
        raise ValueError("AIGIBench source class counts changed")

    selection = dataset["selection"]
    if int(selection["selection_seed"]) != 42:
        raise ValueError("AIGIBench deterministic selection seed changed")
    if selection["canonical_member_list_sha256"] != EXPECTED_MEMBER_LIST_SHA256:
        raise ValueError("AIGIBench selected member list changed")
    if selection["manual_selection_allowed"] is not False:
        raise ValueError("Manual AIGIBench selection is forbidden")
    real, synthetic = dataset["real"], dataset["synthetic"]
    if real["source_name"] != "Open Images V7":
        raise ValueError("Fresh authentic source must remain Open Images V7")
    if synthetic["selected_generator"] != "Midjourney V6":
        raise ValueError("Fresh generator must remain Midjourney V6")
    if real["expected_archive_prefix"] != "Midjourney/0_real/":
        raise ValueError("AIGIBench real prefix changed")
    if synthetic["expected_archive_prefix"] != "Midjourney/1_fake/":
        raise ValueError("AIGIBench AI prefix changed")
    if int(real["selected_images"]) != 1000 or int(
        synthetic["selected_images"]
    ) != 1000:
        raise ValueError("AIGIBench audit must retain 1,000 images per class")
    if int(real["project_label"]) != 0 or int(synthetic["project_label"]) != 1:
        raise ValueError("AIGIBench class mapping changed")
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
        raise ValueError("Image payload may only be downloaded after lock commit")
    for key in (
        "manual_image_exclusion_allowed",
        "manual_image_inspection_before_scoring_allowed",
        "raw_images_committed_to_git",
        "individual_images_allowed_in_public_demo",
    ):
        if integrity[key] is not False:
            raise ValueError(f"External integrity/privacy rule changed: {key}")
    if "exactly 1000" not in integrity["count_rule"]:
        raise ValueError("AIGIBench class-count rule changed")
    if "fatal" not in integrity["deduplication_rule"]:
        raise ValueError("AIGIBench overlap must remain fatal")

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
            raise ValueError(f"AIGIBench representation changed: {key}")

    conditions = protocol["evaluation_conditions"]
    if tuple(conditions["full_matrix"]) != FULL_ROBUSTNESS_CONDITIONS:
        raise ValueError("AIGIBench full transformation matrix changed")
    if tuple(conditions["primary_claim_conditions"]) != EXPECTED_PRIMARY_CONDITIONS:
        raise ValueError("AIGIBench primary claim conditions changed")
    if tuple(conditions["additional_stress_test_conditions"]) != EXPECTED_STRESS_CONDITIONS:
        raise ValueError("AIGIBench stress-test conditions changed")
    if int(conditions["transform_seed"]) != 42:
        raise ValueError("AIGIBench transform seed changed")

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
            raise ValueError(f"AIGIBench success criterion changed: {key}")
    if not criteria["failure_rule"].startswith("Report E5 as not externally validated"):
        raise ValueError("AIGIBench failure action changed")

    reporting = protocol["reporting"]
    if reporting["single_use"] is not True:
        raise ValueError("AIGIBench external audit must remain single-use")
    for key in (
        "comparison_models_can_change_e5_pass_decision",
        "retraining_after_results_allowed",
        "threshold_changes_after_results_allowed",
        "model_reselection_after_results_allowed",
    ):
        if reporting[key] is not False:
            raise ValueError(f"AIGIBench post-test guardrail changed: {key}")


def validate_frozen_inputs(protocol: dict[str, Any], project_root: Path) -> None:
    """Delegate exact E5 artifact validation to the established validator."""

    validate_frozen_e5_external_inputs(protocol, project_root)


def artifact_paths(protocol: dict[str, Any], project_root: Path) -> tuple[Path, ...]:
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


def validate_artifacts_absent(protocol: dict[str, Any], project_root: Path) -> None:
    present = [path for path in artifact_paths(protocol, project_root) if path.exists()]
    if present:
        joined = ", ".join(str(path) for path in present)
        raise ValueError(f"AIGIBench artifacts existed before protocol freeze: {joined}")
