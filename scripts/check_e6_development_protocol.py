#!/usr/bin/env python3
"""Validate and record the E6 pre-download development-data lock."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.extract_clip_features import sha256_file
from src.e6_protocol import (
    assignment_sha256,
    artifact_paths,
    build_pair_assignment,
    load_e6_protocol,
    validate_artifacts_absent,
    validate_e6_development_protocol,
)
from src.evaluation_registry import (
    load_evaluation_registry,
    validate_evaluation_registry,
)


DEFAULT_PROTOCOL = Path("configs/e6_development_protocol.json")
DEFAULT_REGISTRY = Path("configs/evaluation_registry.json")
DEFAULT_OUTPUT = Path("reports/e6_development_protocol_lock.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze-check E6 development roles before image payload access."
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _project_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = build_parser().parse_args()
    project_root = Path.cwd()
    protocol_path = _project_path(project_root, args.protocol)
    registry_path = _project_path(project_root, args.registry)
    output_path = _project_path(project_root, args.output)
    try:
        protocol = load_e6_protocol(protocol_path)
        validate_e6_development_protocol(protocol)
        validate_artifacts_absent(protocol, project_root)

        registry = load_evaluation_registry(registry_path)
        roles = validate_evaluation_registry(registry, project_root)
        units = {item["id"]: item for item in registry["dataset_units"]}
        for consumed_id in protocol["isolation"]["consumed_test_ids_excluded"]:
            if units.get(consumed_id, {}).get("role") != "consumed_test":
                raise ValueError(f"E6 consumed-test registry role changed: {consumed_id}")
        organiser_id = protocol["isolation"]["organiser_validation_id_excluded"]
        if units.get(organiser_id, {}).get("role") != "prohibited":
            raise ValueError("E6 organiser validation data is not prohibited in registry")
        if registry.get("current_locked_test_ids") != []:
            raise ValueError("A future lockbox was unexpectedly opened before E6D")

        assignments = build_pair_assignment(protocol)
        report = {
            "experiment": "e6_development_protocol_lock",
            "status": "PASS",
            "frozen_at_utc": protocol["frozen_at_utc"],
            "protocol": {
                "path": args.protocol.as_posix(),
                "sha256": sha256_file(protocol_path),
                "schema_version": protocol["schema_version"],
            },
            "registry": {
                "path": args.registry.as_posix(),
                "sha256": sha256_file(registry_path),
                "role_counts": dict(sorted(roles.items())),
            },
            "development_source": {
                "dataset_repository": protocol["development_source"][
                    "dataset_repository"
                ],
                "repository_revision": protocol["development_source"][
                    "repository_revision"
                ],
                "selected_generator": protocol["development_source"][
                    "selected_generator"
                ],
                "generator_architecture_family": protocol["development_source"][
                    "generator_architecture_family"
                ],
                "selected_pairs": protocol["selection"]["selected_pairs"],
                "assignment_sha256": assignment_sha256(assignments),
            },
            "role_pair_counts": protocol["selection"]["pairs_per_role"],
            "shortcut_gate": {
                "required_before_training": True,
                "maximum_metadata_only_validation_roc_auc": protocol[
                    "shortcut_gate"
                ]["maximum_acceptable_auc"],
                "training_allowed_before_pass": False,
            },
            "prefreeze_access": protocol["metadata_access_preflight"],
            "image_payload_present_at_freeze": False,
            "checked_absent_paths": [
                path.relative_to(project_root).as_posix()
                for path in artifact_paths(protocol, project_root)
            ],
            "guardrails": {
                "organiser_validation_subset_used": False,
                "consumed_tests_used_for_e6_fitting": False,
                "future_lockbox_payload_accessed": False,
                "manual_image_selection_allowed": False,
            },
        }
        _atomic_write_json(output_path, report)
        print(
            "PASS E6 development lock: source=Tiny-GenImage/BigGAN, "
            "pairs=1400, roles=800/200/200/200, image_payloads=0"
        )
        print(
            "Shortcut gate: no E6 training until metadata-only AUC <= 0.65 "
            "or a separately frozen normalization policy passes the gate"
        )
        print(f"Lock report: {args.output}")
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"E6 development lock failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
