"""Validation and statistical primitives for the frozen E5 protocol."""

from __future__ import annotations

import re
from pathlib import Path
from statistics import NormalDist
from typing import Any

from src.image_transforms import DEFAULT_ROBUSTNESS_CONDITIONS
from src.robust_linear_training import sha256_file


EXPECTED_GROUPS = {
    "cifake_real": 0.25,
    "cifake_ai_generated": 0.25,
    "sid_set_train_real": 0.25,
    "sid_set_train_flux": 0.25,
}
EXPECTED_ANCHOR_WEIGHTS = (0.0, 0.001, 0.01, 0.1)


def _require_sha256(value: str, description: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{description} must be a SHA-256 hash")


def validate_e5_protocol(protocol: dict[str, Any]) -> None:
    """Reject E5 leakage, source-label confounding, or selection-rule drift."""

    if protocol["status"] != "frozen_before_e5_sid_flux_download":
        raise ValueError("E5 protocol must be frozen before SID FLUX download")
    positioning = protocol["research_positioning"]
    required_positioning = {
        "E3_remains_original_primary_binary_baseline": True,
        "E4_remains_documented_failed_replacement": True,
        "E5_is_new_post_hoc_research_candidate": True,
        "universal_detector_claim_allowed": False,
        "fresh_external_test_required": True,
        "existing_E3_E4_results_may_be_overwritten": False,
    }
    for key, expected in required_positioning.items():
        if positioning[key] is not expected:
            raise ValueError(f"E5 research positioning changed: {key}")

    e4 = protocol["frozen_e4_failure_evidence"]
    _require_sha256(e4["report_sha256"], "E4 failure report hash")
    expected_e4_metrics = {
        "sid_real_false_positive_rate_E3": 0.646,
        "sid_real_false_positive_rate_E4": 0.006,
        "flux_recall_E3": 0.997,
        "flux_recall_E4": 0.205,
    }
    for key, expected in expected_e4_metrics.items():
        if float(e4[key]) != expected:
            raise ValueError(f"Frozen E4 failure evidence changed: {key}")

    data = protocol["development_data"]
    cifake = data["cifake"]
    if (
        int(cifake["train_per_class"]) * 2 != int(cifake["train_samples"])
        or int(cifake["validation_per_class"]) * 2
        != int(cifake["validation_samples"])
    ):
        raise ValueError("CIFAKE development counts are no longer class-balanced")
    sid_real = data["sid_set_train_real"]
    if sid_real["source_split"] != "train" or int(sid_real["source_label"]) != 0:
        raise ValueError("E5 real supervision must reuse SID train label 0")
    for key in (
        "train_manifest_sha256",
        "validation_manifest_sha256",
        "provenance_sha256",
        "clean_embedding_summary_sha256",
        "transformed_embedding_summary_sha256",
    ):
        _require_sha256(sid_real[key], f"SID real {key}")

    sid_flux = data["sid_set_train_flux"]
    if sid_flux["repository"] != "saberzl/SID_Set":
        raise ValueError("Unexpected E5 SID FLUX repository")
    if not re.fullmatch(r"[0-9a-f]{40}", sid_flux["source_revision"]):
        raise ValueError("E5 SID source revision must be a full Git hash")
    if sid_flux["source_split"] != "train" or int(sid_flux["source_rows"]) != 210000:
        raise ValueError("E5 FLUX development data must come from SID train")
    if int(sid_flux["allowed_source_label"]) != 1:
        raise ValueError("E5 SID synthetic supervision must be FLUX label 1")
    if set(sid_flux["forbidden_source_labels"]) != {"0", "2"}:
        raise ValueError("E5 must exclude SID real and tampered rows from FLUX sampling")
    if sid_flux["declared_license"] != "CC-BY-4.0":
        raise ValueError("Unexpected SID-Set declared license")

    sampling = protocol["sid_flux_sampling"]
    expected_pages = (
        int(sid_flux["source_rows"]) + int(sampling["page_size"]) - 1
    ) // int(sampling["page_size"])
    if int(sampling["page_count"]) != expected_pages:
        raise ValueError("E5 page count does not cover the pinned SID train split")
    if int(sampling["train_count"]) + int(sampling["validation_count"]) != int(
        sampling["total_count"]
    ):
        raise ValueError("E5 FLUX split counts do not match the total")
    if min(
        int(sampling["train_count"]),
        int(sampling["validation_count"]),
        int(sampling["candidate_buffer"]),
    ) <= 0:
        raise ValueError("E5 sampling counts and buffer must be positive")
    if sampling["manual_cherry_picking_allowed"] is not False:
        raise ValueError("Manual E5 image cherry-picking is forbidden")

    forbidden = protocol["forbidden_development_data"]
    audit = forbidden["section4_sid_flux_audit"]
    if audit["dataset_split"] != "SID-Set validation":
        raise ValueError("Frozen audit split changed")
    for key in ("manifest_sha256", "provenance_sha256"):
        _require_sha256(audit[key], f"Frozen audit {key}")
    for key in (
        "allowed_for_training",
        "allowed_for_validation",
        "allowed_for_model_threshold_or_hyperparameter_selection",
    ):
        if audit[key] is not False:
            raise ValueError(f"E5 audit leakage rule changed: {key}")
    if audit["allowed_as_posthoc_regression_diagnostic_only"] is not True:
        raise ValueError("Prior audit may only remain a post-hoc diagnostic")
    organiser = forbidden["organiser_validation_subset"]
    for key in (
        "allowed_for_training",
        "allowed_for_validation",
        "allowed_for_model_threshold_or_hyperparameter_selection",
        "used",
    ):
        if organiser[key] is not False:
            raise ValueError(f"Organiser exclusion changed: {key}")

    initialization = protocol["initialization"]
    if "E3" not in initialization["model"]:
        raise ValueError("E5 must initialise from E3")
    _require_sha256(initialization["checkpoint_sha256"], "E3 initialization hash")
    if tuple(protocol["representative_conditions"]) != DEFAULT_ROBUSTNESS_CONDITIONS:
        raise ValueError("E5 representative transformations changed")

    training = protocol["training"]
    if training["group_sampling_per_epoch"] != EXPECTED_GROUPS:
        raise ValueError("E5 source-label group balance changed")
    if training["class_balance_per_epoch"] != {"real": 0.5, "ai_generated": 0.5}:
        raise ValueError("E5 class balance changed")
    if training["source_balance_per_epoch"] != {
        "cifake": 0.5,
        "sid_set_train": 0.5,
    }:
        raise ValueError("E5 source balance changed")
    if int(training["examples_per_epoch"]) != 60000:
        raise ValueError("E5 examples per epoch changed")
    if tuple(float(value) for value in training["anchor_weights"]) != EXPECTED_ANCHOR_WEIGHTS:
        raise ValueError("E5 anchor candidate grid changed")
    if training["all_anchor_candidates_must_run"] is not True:
        raise ValueError("All frozen E5 anchor candidates must run")
    if min(
        int(training["batch_size"]),
        int(training["maximum_epochs"]),
        int(training["early_stopping_patience"]),
    ) <= 0:
        raise ValueError("E5 training counts must be positive")
    if min(
        float(training["learning_rate"]),
        float(training["consistency_weight"]),
    ) <= 0.0 or float(training["weight_decay"]) < 0.0:
        raise ValueError("E5 training loss or optimizer settings are invalid")

    selection = protocol["validation_and_decision_selection"]
    if "no test, prior audit, or organiser data" not in selection["data"]:
        raise ValueError("E5 selection data exclusion changed")
    minimum = float(selection["score_grid_minimum"])
    maximum = float(selection["score_grid_maximum"])
    step = float(selection["score_grid_step"])
    if not 0.0 < minimum < maximum < 1.0 or step <= 0.0:
        raise ValueError("E5 score grid is invalid")
    decision = selection["deployment_decision"]
    if decision["threshold_order_required"] != "real_threshold < ai_threshold":
        raise ValueError("E5 must retain an explicit uncertainty interval")
    if float(selection["confidence_level"]) != 0.95:
        raise ValueError("E5 Wilson confidence level changed")
    errors = selection["confident_error_constraints"]
    if float(errors["maximum_wilson_upper_real_called_ai"]) != 0.05:
        raise ValueError("E5 confident false-accusation constraint changed")
    if float(errors["maximum_wilson_upper_ai_called_real"]) != 0.1:
        raise ValueError("E5 confident AI-miss constraint changed")
    coverage = selection["anti_trivial_abstention_constraints"]
    expected_coverage = {
        "minimum_clean_decisive_coverage_per_group": 0.6,
        "minimum_worst_source_condition_decisive_coverage": 0.25,
        "minimum_mean_source_condition_decisive_coverage": 0.5,
    }
    if coverage != expected_coverage:
        raise ValueError("E5 anti-trivial-abstention constraints changed")
    if not selection["no_feasible_candidate_rule"].startswith("Reject E5"):
        raise ValueError("E5 must be rejected when no candidate is feasible")

    gates = protocol["pre_evaluation_acceptance_gates"]
    for key in (
        "source_and_class_balance_verified",
        "all_four_anchor_candidates_completed",
        "risk_controlled_threshold_pair_feasible",
        "minimum_validation_coverage_constraints_met",
    ):
        if gates[key] is not True:
            raise ValueError(f"E5 pre-evaluation gate changed: {key}")
    if gates["test_audit_and_organiser_data_loaded_during_development"] is not False:
        raise ValueError("E5 development may not load test, audit, or organiser data")

    external = protocol["fresh_external_evaluation"]
    for key in (
        "required_before_any_E5_success_claim",
        "separate_protocol_must_be_frozen_before_download_or_feature_inspection",
        "real_source_must_be_absent_from_E5_development",
        "generator_family_must_be_absent_from_E5_development",
        "organiser_validation_subset_forbidden",
        "single_use_no_threshold_changes_after_results",
    ):
        if external[key] is not True:
            raise ValueError(f"E5 fresh-test rule changed: {key}")
    if external["prior_section4_sid_flux_audit_qualifies_as_fresh_test"] is not False:
        raise ValueError("The inspected SID/FLUX audit cannot become E5's fresh test")
    if min(int(external["minimum_real_images"]), int(external["minimum_ai_images"])) < 1000:
        raise ValueError("E5 fresh test must contain at least 1,000 images per class")

    privacy = protocol["privacy_and_publication"]
    if privacy["raw_sid_images_must_not_be_committed"] is not True:
        raise ValueError("Raw SID images must remain outside Git")
    if privacy["individual_sid_images_allowed_in_public_demo"] is not False:
        raise ValueError("SID images may not appear in the public demo")
    if privacy["source_urls_persisted"] is not False:
        raise ValueError("SID source URLs may not be persisted")


def validate_frozen_e5_inputs(protocol: dict[str, Any], project_root: Path) -> None:
    """Verify that every existing input frozen into E5A still has the same bytes."""

    records = [
        (
            protocol["frozen_e4_failure_evidence"]["report"],
            protocol["frozen_e4_failure_evidence"]["report_sha256"],
        ),
        (
            protocol["initialization"]["checkpoint"],
            protocol["initialization"]["checkpoint_sha256"],
        ),
    ]
    real = protocol["development_data"]["sid_set_train_real"]
    records.extend(
        [
            (real["train_manifest"], real["train_manifest_sha256"]),
            (real["validation_manifest"], real["validation_manifest_sha256"]),
            (real["provenance"], real["provenance_sha256"]),
            (real["clean_embedding_summary"], real["clean_embedding_summary_sha256"]),
            (
                real["transformed_embedding_summary"],
                real["transformed_embedding_summary_sha256"],
            ),
        ]
    )
    audit = protocol["forbidden_development_data"]["section4_sid_flux_audit"]
    records.extend(
        [
            (audit["manifest"], audit["manifest_sha256"]),
            (audit["provenance"], audit["provenance_sha256"]),
        ]
    )
    for relative, expected in records:
        path = project_root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Frozen E5 input changed or is missing: {path}")


def wilson_interval(successes: int, trials: int, confidence_level: float = 0.95) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a binomial rate."""

    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("Wilson counts must satisfy 0 <= successes <= positive trials")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("Wilson confidence level must lie in (0, 1)")
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    observed = successes / trials
    z_squared = z * z
    denominator = 1.0 + z_squared / trials
    centre = (observed + z_squared / (2.0 * trials)) / denominator
    radius = (
        z
        * (
            observed * (1.0 - observed) / trials
            + z_squared / (4.0 * trials * trials)
        )
        ** 0.5
        / denominator
    )
    return max(0.0, centre - radius), min(1.0, centre + radius)
