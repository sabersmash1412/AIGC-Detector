"""Validate and record the pre-download E5 fresh-external evaluation lock."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.e5_external_protocol import (
    external_artifact_paths,
    validate_e5_external_protocol,
    validate_external_artifacts_absent,
    validate_frozen_e5_external_inputs,
)
from src.robust_linear_training import sha256_file


DEFAULT_PROTOCOL = Path("configs/e5_fresh_external_evaluation.json")
DEFAULT_OUTPUT = Path("reports/e5_external_evaluation_lock.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze-check E5 before any Synthbuster/RAISE data access."
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    project_root = Path.cwd()
    try:
        protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
        validate_e5_external_protocol(protocol)
        validate_frozen_e5_external_inputs(protocol, project_root)
        validate_external_artifacts_absent(protocol, project_root)

        frozen = protocol["frozen_e5"]
        report = {
            "experiment": "e5_fresh_external_evaluation_lock",
            "status": "PASS",
            "frozen_at_utc": protocol["frozen_at_utc"],
            "protocol": {
                "path": args.protocol.as_posix(),
                "sha256": sha256_file(args.protocol),
                "version": protocol["protocol_version"],
            },
            "frozen_e5": {
                "development_protocol_sha256": frozen[
                    "development_protocol_sha256"
                ],
                "training_report_sha256": frozen["training_report_sha256"],
                "checkpoint_sha256": frozen["checkpoint_sha256"],
                "anchor_weight": frozen["selected_anchor_weight"],
                "epoch": frozen["selected_epoch"],
                "real_threshold": frozen["real_threshold"],
                "ai_threshold": frozen["ai_threshold"],
            },
            "external_selection": {
                "benchmark": protocol["external_dataset"]["benchmark_name"],
                "real_source": protocol["external_dataset"]["real"]["source_name"],
                "generator": protocol["external_dataset"]["synthetic"][
                    "selected_generator"
                ],
                "images_per_class": protocol["external_dataset"]["real"][
                    "expected_images"
                ],
                "organiser_validation_subset_used": False,
            },
            "external_artifacts_present_at_freeze": False,
            "checked_absent_paths": [
                path.relative_to(project_root).as_posix()
                for path in external_artifact_paths(protocol, project_root)
            ],
            "guardrails": {
                "single_use": True,
                "retraining_after_results_allowed": False,
                "threshold_changes_after_results_allowed": False,
                "model_reselection_after_results_allowed": False,
                "manual_image_inspection_before_scoring_allowed": False,
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(
            "PASS E5 external lock: "
            f"checkpoint={frozen['checkpoint_sha256'][:12]}..., "
            "real=RAISE-1k, generator=Adobe Firefly, samples=1000+1000"
        )
        print(f"Lock report: {args.output}")
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"E5 external lock failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
