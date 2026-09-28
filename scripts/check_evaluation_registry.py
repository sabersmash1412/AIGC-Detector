"""Check the E6 dataset-role registry and its frozen artifact hashes."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation_registry import (
    load_evaluation_registry,
    validate_evaluation_registry,
)


DEFAULT_REGISTRY = Path("configs/evaluation_registry.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate dataset roles before starting E6 work."
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--skip-artifact-hashes",
        action="store_true",
        help="Validate roles only. Normal project checks should verify hashes.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path.cwd()
    registry_path = args.registry
    if not registry_path.is_absolute():
        registry_path = project_root / registry_path

    registry = load_evaluation_registry(registry_path)
    roles = validate_evaluation_registry(
        registry,
        project_root,
        verify_artifacts=not args.skip_artifact_hashes,
    )
    ordered_roles = (
        "development_train",
        "development_validation",
        "consumed_test",
        "locked_test",
        "prohibited",
    )
    summary = ", ".join(f"{role}={roles[role]}" for role in ordered_roles)
    print(f"PASS evaluation registry: dataset_units={sum(roles.values())}; {summary}")
    print(
        "AIGIBench Midjourney V6 is consumed regression evidence, not a fresh E6 test."
    )
    print("Current locked E6 tests: 0. Candidate datasets remain unopened and unlocked.")


if __name__ == "__main__":
    main()
