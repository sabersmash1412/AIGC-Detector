"""Validation helpers for the E6 dataset and evaluation registry."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ALLOWED_ROLES = {
    "development_train",
    "development_validation",
    "consumed_test",
    "locked_test",
    "prohibited",
}
REQUIRED_EXISTING_ROLES = {
    "cifake_internal_test": "consumed_test",
    "sid_set_validation_flux_audit": "consumed_test",
    "aigibench_midjourney_v6": "consumed_test",
    "organiser_validation_coco_dalle": "prohibited",
}


def load_evaluation_registry(path: Path) -> dict[str, Any]:
    """Load the registry as JSON."""

    return json.loads(path.read_text(encoding="utf-8"))


def _require_boolean(unit: dict[str, Any], key: str) -> bool:
    value = unit.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{unit.get('id', '<unknown>')} {key} must be boolean")
    return value


def _safe_project_path(project_root: Path, relative_path: str) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"Registry artifact path must stay inside the project: {relative_path}")
    return project_root / candidate


def _validate_artifact(record: dict[str, Any], project_root: Path, owner: str) -> None:
    relative_path = record.get("path")
    expected_sha256 = record.get("sha256")
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError(f"{owner} artifact path is missing")
    if not isinstance(expected_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_sha256
    ):
        raise ValueError(f"{owner} artifact SHA-256 is invalid")

    path = _safe_project_path(project_root, relative_path)
    if not path.is_file():
        raise ValueError(f"{owner} registry artifact is missing: {relative_path}")
    actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{owner} registry artifact hash changed: {relative_path}; "
            f"expected={expected_sha256}, actual={actual_sha256}"
        )


def validate_evaluation_registry(
    registry: dict[str, Any], project_root: Path, *, verify_artifacts: bool = True
) -> Counter[str]:
    """Validate role separation and the exact frozen E1-E5 artifact identities."""

    if registry.get("schema_version") != 1:
        raise ValueError("Evaluation registry schema_version must be 1")
    if registry.get("status") != "retrospective_e1_e5_baseline_and_forward_e6_policy":
        raise ValueError("Evaluation registry must disclose its retrospective E1-E5 status")
    policies = registry.get("policies")
    if not isinstance(policies, dict):
        raise ValueError("Evaluation registry policies are missing")
    if policies.get("organiser_validation_subset_used") is not False:
        raise ValueError("Organiser validation subset must remain unused")

    units = registry.get("dataset_units")
    if not isinstance(units, list) or not units:
        raise ValueError("Evaluation registry must contain dataset_units")

    seen_ids: set[str] = set()
    roles: Counter[str] = Counter()
    by_id: dict[str, dict[str, Any]] = {}
    locked_ids: list[str] = []

    for unit in units:
        if not isinstance(unit, dict):
            raise ValueError("Every dataset unit must be an object")
        unit_id = unit.get("id")
        if not isinstance(unit_id, str) or not unit_id:
            raise ValueError("Every dataset unit needs a non-empty id")
        if unit_id in seen_ids:
            raise ValueError(f"Duplicate dataset unit id: {unit_id}")
        seen_ids.add(unit_id)
        by_id[unit_id] = unit

        role = unit.get("role")
        if role not in ALLOWED_ROLES:
            raise ValueError(f"{unit_id} has unsupported role: {role}")
        roles[role] += 1

        upstream_real_sources = unit.get("upstream_real_sources")
        generator_families = unit.get("generator_families")
        if not isinstance(upstream_real_sources, list) or not all(
            isinstance(source, str) and source for source in upstream_real_sources
        ):
            raise ValueError(f"{unit_id} upstream_real_sources must be a list of names")
        if not isinstance(generator_families, list) or not all(
            isinstance(generator, str) and generator for generator in generator_families
        ):
            raise ValueError(f"{unit_id} generator_families must be a list of names")

        results_viewed = _require_boolean(unit, "results_viewed")
        influences_design = _require_boolean(unit, "influences_future_design")
        allowed_training = _require_boolean(unit, "allowed_for_training")
        allowed_validation = _require_boolean(unit, "allowed_for_validation")
        allowed_threshold = _require_boolean(unit, "allowed_for_threshold_selection")
        fresh_claim = _require_boolean(unit, "eligible_for_fresh_external_claim")
        _require_boolean(unit, "eligible_for_regression")

        if role == "development_train":
            if not allowed_training or allowed_validation or allowed_threshold or fresh_claim:
                raise ValueError(f"{unit_id} violates development-train role separation")
        elif role == "development_validation":
            if allowed_training or not allowed_validation or not allowed_threshold or fresh_claim:
                raise ValueError(f"{unit_id} violates development-validation role separation")
        elif role == "consumed_test":
            if not results_viewed or not influences_design:
                raise ValueError(f"{unit_id} consumed test must disclose viewed influential results")
            if allowed_training or allowed_validation or allowed_threshold or fresh_claim:
                raise ValueError(f"{unit_id} consumed test cannot be reused as fresh development data")
            if not isinstance(unit.get("consumed_reason"), str) or not unit["consumed_reason"]:
                raise ValueError(f"{unit_id} consumed test needs a consumed_reason")
        elif role == "locked_test":
            locked_ids.append(unit_id)
            if results_viewed or influences_design:
                raise ValueError(f"{unit_id} locked test cannot have viewed or influential results")
            if allowed_training or allowed_validation or allowed_threshold or not fresh_claim:
                raise ValueError(f"{unit_id} violates locked-test isolation")
            if unit.get("preregistered_before_access") is not True:
                raise ValueError(f"{unit_id} locked test must be preregistered before access")
        elif role == "prohibited":
            if unit.get("used") is not False:
                raise ValueError(f"{unit_id} prohibited data must remain unused")
            if any((allowed_training, allowed_validation, allowed_threshold, fresh_claim)):
                raise ValueError(f"{unit_id} prohibited data cannot be assigned a usable role")

        if verify_artifacts:
            for key in ("artifact", "evidence"):
                record = unit.get(key)
                if record is not None:
                    if not isinstance(record, dict):
                        raise ValueError(f"{unit_id} {key} must be an object")
                    _validate_artifact(record, project_root, f"{unit_id} {key}")

    for unit_id, expected_role in REQUIRED_EXISTING_ROLES.items():
        if unit_id not in by_id:
            raise ValueError(f"Required historical dataset unit is missing: {unit_id}")
        if by_id[unit_id]["role"] != expected_role:
            raise ValueError(
                f"{unit_id} must remain {expected_role}, got {by_id[unit_id]['role']}"
            )

    development_units = [
        unit
        for unit in units
        if unit["role"] in {"development_train", "development_validation"}
    ]
    development_real_sources = {
        source
        for unit in development_units
        for source in unit["upstream_real_sources"]
    }
    development_generators = {
        generator
        for unit in development_units
        for generator in unit["generator_families"]
    }
    for unit in units:
        if unit["role"] not in {"consumed_test", "locked_test"}:
            continue
        unit_id = unit["id"]
        real_sources = set(unit["upstream_real_sources"])
        generators = set(unit["generator_families"])
        expected_real_novelty = bool(real_sources) and real_sources.isdisjoint(
            development_real_sources
        )
        expected_generator_novelty = bool(generators) and generators.isdisjoint(
            development_generators
        )
        if unit.get("real_source_novelty_vs_development") is not expected_real_novelty:
            overlap = sorted(real_sources.intersection(development_real_sources))
            raise ValueError(
                f"{unit_id} real-source novelty is incorrect; development overlap={overlap}"
            )
        if unit.get("generator_novelty_vs_development") is not expected_generator_novelty:
            overlap = sorted(generators.intersection(development_generators))
            raise ValueError(
                f"{unit_id} generator novelty is incorrect; development overlap={overlap}"
            )

    recorded_locked_ids = registry.get("current_locked_test_ids")
    if recorded_locked_ids != locked_ids:
        raise ValueError(
            "current_locked_test_ids must exactly match dataset units with role locked_test"
        )

    candidates = registry.get("future_lockbox_candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Future lockbox candidates are missing")
    candidate_ids: set[str] = set()
    for candidate in candidates:
        candidate_id = candidate.get("id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("Every future lockbox candidate needs an id")
        if candidate_id in candidate_ids or candidate_id in seen_ids:
            raise ValueError(f"Duplicate registry or candidate id: {candidate_id}")
        candidate_ids.add(candidate_id)
        if candidate.get("status") != "candidate_only_not_locked":
            raise ValueError(f"{candidate_id} must remain candidate-only until preregistration")
        if candidate.get("access_state") != "not_acquired_or_scored_for_e6":
            raise ValueError(f"{candidate_id} access state changed; freeze it before access")
        if candidate.get("license_review_required") is not True:
            raise ValueError(f"{candidate_id} requires a licence review before locking")

    return roles
