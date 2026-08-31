#!/usr/bin/env python3
"""Freeze the AIGIBench content-deduplication amendment before continuation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.e5_aigibench_amendment import (
    output_paths,
    validate_amendment,
    validate_known_raw_state,
    validate_outputs_absent,
)
from src.robust_linear_training import sha256_file


DEFAULT_AMENDMENT = Path("configs/e5_aigibench_deduplication_amendment.json")
DEFAULT_OUTPUT = Path("reports/e5_aigibench_deduplication_amendment_lock.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amendment", type=Path, default=DEFAULT_AMENDMENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = Path.cwd()
    try:
        amendment = json.loads(args.amendment.read_text(encoding="utf-8"))
        validate_amendment(amendment, root)
        validate_outputs_absent(amendment, root)
        state = validate_known_raw_state(amendment, root)
        report = {
            "experiment": "e5_aigibench_deduplication_amendment_lock",
            "status": "PASS",
            "frozen_at_utc": amendment["frozen_at_utc"],
            "amendment": {
                "path": args.amendment.as_posix(),
                "sha256": sha256_file(args.amendment),
                "version": amendment["protocol_version"],
            },
            "base_protocol": amendment["base_protocol"],
            "duplicate_audit": amendment["duplicate_audit"],
            "observed_prefeature_state": {
                "real_files": state["real_files"],
                "ai_files": state["ai_files"],
                "unique_real": state["unique_real"],
                "unique_ai": state["unique_ai"],
                "cross_class_duplicate_groups": state[
                    "cross_class_duplicate_groups"
                ],
                "within_ai_duplicate_groups": len(state["ai_duplicate_groups"]),
                "features_predictions_or_metrics_present": False,
            },
            "checked_absent_paths": [
                path.relative_to(root).as_posix()
                for path in output_paths(amendment, root)
            ],
            "frozen_selection": amendment["deduplication_selection"],
            "guardrails": amendment["inherited_frozen_rules"],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(
            "PASS E5 AIGIBench amendment lock: "
            "real_unique=1000, ai_unique=996, duplicate_groups=4, "
            "features/predictions=0"
        )
        print(f"Amendment lock: {args.output}")
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"E5 AIGIBench amendment lock failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
