"""Inspect one image with the frozen E6 forensic feature extractor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.forensic_features import (
    DEFAULT_SPEC_PATH,
    inspect_forensic_file,
    write_forensic_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract the frozen E6 high-pass/frequency descriptor from one image. "
            "This command returns features only; it does not classify the image."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Image to inspect")
    parser.add_argument(
        "--spec",
        type=Path,
        default=DEFAULT_SPEC_PATH,
        help="Versioned forensic extractor specification",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON report path; JSON is printed when omitted",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = inspect_forensic_file(args.input, spec_path=args.spec)
        if args.output is None:
            print(json.dumps(report, indent=2))
        else:
            write_forensic_report(report, args.output, source_path=args.input)
            extractor = report["extractor"]
            print(
                f"Wrote {args.output}: extractor={extractor['id']}, "
                f"features={extractor['dimension']}, prediction=none"
            )
        return 0
    except Exception as exc:
        print(f"Forensic feature inspection failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
