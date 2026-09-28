"""Inspect one image for trustworthy C2PA provenance evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.provenance import inspect_provenance_file, write_provenance_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect an image for C2PA provenance. A missing manifest is reported "
            "as no signal, not as proof that the image is real."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Image to inspect")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON report path; JSON is printed when omitted",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = inspect_provenance_file(args.input)
        if args.output is None:
            print(json.dumps(report, indent=2))
        else:
            write_provenance_report(report, args.output, source_path=args.input)
            provenance = report["provenance"]
            print(
                f"Wrote {args.output}: status={provenance['manifest_status']}, "
                f"decision={provenance['decision']}"
            )
        return 0
    except Exception as exc:
        print(f"Provenance inspection failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
