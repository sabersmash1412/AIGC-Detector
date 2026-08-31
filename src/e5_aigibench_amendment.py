"""Validation for the pre-feature AIGIBench content-deduplication amendment."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.e5_aigibench_protocol import (
    validate_e5_aigibench_protocol,
    validate_frozen_inputs,
)
from src.robust_linear_training import sha256_file


EXPECTED_BASE_PROTOCOL_SHA256 = (
    "479dedced86c27cd8093a27477b0c2037ee2fc15b60ee109837b69079d1426f5"
)
EXPECTED_BASE_LOCK_SHA256 = (
    "ac81536c42edec82effefd7bf9a71c8ef0e75b397711dd8f3facb80dcaa405fa"
)
EXPECTED_DUPLICATE_AUDIT_SHA256 = (
    "9b730984292cc69bb5a9e8aa72d8f42b0c6c55fc9276dc9d0ab78ed4359d7e80"
)
EXPECTED_REAL_ORDER_SHA256 = (
    "fde08251c9f7509940cbda0681f50e18680995f7a2e1b4233c2f5e414f1cb7b4"
)
EXPECTED_AI_ORDER_SHA256 = (
    "df9309b6fce707ece74c754316a5129c9687335032f0adf45f6aa4953c33890c"
)


def _load_and_verify(record: dict[str, str], project_root: Path) -> dict[str, Any]:
    path = project_root / record["path"]
    if not path.is_file() or sha256_file(path) != record["sha256"]:
        raise ValueError(f"Frozen amendment dependency changed or is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_amendment(amendment: dict[str, Any], project_root: Path) -> None:
    if amendment["status"] != (
        "frozen_before_additional_candidate_payload_access_or_feature_extraction"
    ):
        raise ValueError("Deduplication amendment must be frozen before continuation")
    if amendment["base_protocol"]["sha256"] != EXPECTED_BASE_PROTOCOL_SHA256:
        raise ValueError("Base AIGIBench protocol hash changed")
    if amendment["base_lock_report"]["sha256"] != EXPECTED_BASE_LOCK_SHA256:
        raise ValueError("Base AIGIBench lock hash changed")
    if amendment["duplicate_audit"]["sha256"] != EXPECTED_DUPLICATE_AUDIT_SHA256:
        raise ValueError("Duplicate audit hash changed")

    base = _load_and_verify(amendment["base_protocol"], project_root)
    validate_e5_aigibench_protocol(base)
    validate_frozen_inputs(base, project_root)
    lock = _load_and_verify(amendment["base_lock_report"], project_root)
    if lock.get("status") != "PASS":
        raise ValueError("Base AIGIBench lock did not pass")
    audit = _load_and_verify(amendment["duplicate_audit"], project_root)
    if audit.get("status") != "BLOCKED_DUPLICATE_CONTENT_BEFORE_FEATURE_EXTRACTION":
        raise ValueError("Duplicate audit does not document a pre-feature block")
    if audit["acquisition"]["features_extracted"] is not False:
        raise ValueError("Features existed before the amendment")
    if audit["acquisition"]["predictions_generated"] is not False:
        raise ValueError("Predictions existed before the amendment")

    population = amendment["candidate_population"]
    expected_population = {
        "archive_repository_commit": "e44ec40efe5117a5ccdaa6ff0e89ed934d03d310",
        "archive_remote_bytes": 8_999_861_437,
        "selection_seed": 42,
        "real_prefix": "Midjourney/0_real/",
        "ai_prefix": "Midjourney/1_fake/",
        "real_candidates": 3000,
        "ai_candidates": 3000,
        "real_candidate_order_sha256": EXPECTED_REAL_ORDER_SHA256,
        "ai_candidate_order_sha256": EXPECTED_AI_ORDER_SHA256,
    }
    for key, expected in expected_population.items():
        if population[key] != expected:
            raise ValueError(f"Frozen candidate population changed: {key}")

    selection = amendment["deduplication_selection"]
    if int(selection["target_unique_images_per_class"]) != 1000:
        raise ValueError("Deduplicated audit must retain 1,000 unique images per class")
    if selection["duplicate_tie_break"] != "The lower frozen candidate rank is retained.":
        raise ValueError("Duplicate tie-break changed")
    for key in (
        "manual_replacement_allowed",
        "filename_or_visual_quality_selection_allowed",
        "model_score_selection_allowed",
    ):
        if selection[key] is not False:
            raise ValueError(f"Forbidden amendment selection enabled: {key}")
    if selection["cross_class_duplicate_action"] != "fatal_abort":
        raise ValueError("Cross-class duplicates must remain fatal")
    if selection["development_or_prior_audit_overlap_action"] != "fatal_abort":
        raise ValueError("Development overlap must remain fatal")

    frozen = amendment["frozen_e5"]
    expected_e5 = {
        "checkpoint_sha256": base["frozen_e5"]["checkpoint_sha256"],
        "selected_anchor_weight": 0.01,
        "selected_epoch": 40,
        "real_threshold": 0.23700000000000002,
        "ai_threshold": 0.8170000000000001,
        "binary_benchmark_threshold": 0.52,
        "model_or_threshold_change": False,
    }
    for key, expected in expected_e5.items():
        if frozen[key] != expected:
            raise ValueError(f"E5 changed in deduplication amendment: {key}")

    for key, expected in {
        "source_and_generator_unchanged": True,
        "organiser_validation_subset_used": False,
        "full_transformation_matrix_unchanged": True,
        "success_criteria_unchanged": True,
        "single_use_model_evaluation": True,
        "retraining_after_results_allowed": False,
        "threshold_changes_after_results_allowed": False,
        "model_reselection_after_results_allowed": False,
        "manual_image_inspection_before_scoring_allowed": False,
        "raw_images_committed_to_git": False,
    }.items():
        if amendment["inherited_frozen_rules"][key] is not expected:
            raise ValueError(f"Inherited frozen rule changed: {key}")


def output_paths(amendment: dict[str, Any], project_root: Path) -> tuple[Path, ...]:
    return tuple(project_root / value for value in amendment["outputs"].values())


def validate_outputs_absent(amendment: dict[str, Any], project_root: Path) -> None:
    present = [path for path in output_paths(amendment, project_root) if path.exists()]
    if present:
        raise ValueError(
            "Feature/evaluation output existed before amendment lock: "
            + ", ".join(str(path) for path in present)
        )


def observed_raw_duplicate_state(
    amendment: dict[str, Any], project_root: Path
) -> dict[str, Any]:
    raw_root = project_root / amendment["known_prefeature_state"]["raw_root"]
    class_dirs = {"real": raw_root / "real", "ai_generated": raw_root / "ai_generated"}
    hashes_by_class: dict[str, dict[str, list[str]]] = {}
    for class_name, directory in class_dirs.items():
        if not directory.is_dir():
            raise ValueError(f"Expected acquired class directory is missing: {directory}")
        grouped: dict[str, list[str]] = defaultdict(list)
        for path in sorted(item for item in directory.iterdir() if item.is_file()):
            grouped[sha256_file(path)].append(path.relative_to(project_root).as_posix())
        hashes_by_class[class_name] = dict(grouped)
    real = hashes_by_class["real"]
    ai = hashes_by_class["ai_generated"]
    duplicate_groups = {
        digest: paths for digest, paths in ai.items() if len(paths) > 1
    }
    return {
        "real_files": sum(len(paths) for paths in real.values()),
        "ai_files": sum(len(paths) for paths in ai.values()),
        "unique_real": len(real),
        "unique_ai": len(ai),
        "cross_class_duplicate_groups": len(set(real) & set(ai)),
        "ai_duplicate_groups": duplicate_groups,
    }


def validate_known_raw_state(amendment: dict[str, Any], project_root: Path) -> dict[str, Any]:
    state = observed_raw_duplicate_state(amendment, project_root)
    expected = amendment["known_prefeature_state"]
    values = {
        "real_files": expected["downloaded_real_files"],
        "ai_files": expected["downloaded_ai_files"],
        "unique_real": expected["unique_real_files"],
        "unique_ai": expected["unique_ai_files"],
        "cross_class_duplicate_groups": 0,
    }
    for key, expected_value in values.items():
        if state[key] != expected_value:
            raise ValueError(f"Observed pre-feature raw state changed: {key}")
    audit = json.loads(
        (project_root / amendment["duplicate_audit"]["path"]).read_text(encoding="utf-8")
    )
    expected_groups = {
        group["sha256"]: sorted(group["files"])
        for group in audit["content_integrity"]["duplicate_groups"]
    }
    observed_groups = {
        digest: sorted(paths) for digest, paths in state["ai_duplicate_groups"].items()
    }
    if observed_groups != expected_groups:
        raise ValueError("Observed duplicate groups changed after their audit")
    return state
