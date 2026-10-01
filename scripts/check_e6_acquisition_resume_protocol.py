#!/usr/bin/env python3
"""Freeze the offline E6 resume policy and the intact 340-pair cache prefix."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import tempfile
from pathlib import Path

from src.e6_acquisition_resume_protocol import (
    DEFAULT_RESUME_LOCK,
    DEFAULT_RESUME_PROTOCOL,
    build_resume_lock_receipt,
    load_resume_protocol,
    write_initial_rejection_journal,
)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("E6 resume lock output may not be a symlink")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_RESUME_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESUME_LOCK)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    project_root = Path.cwd().resolve()
    protocol_path = args.protocol if args.protocol.is_absolute() else project_root / args.protocol
    output_path = args.output if args.output.is_absolute() else project_root / args.output
    try:
        protocol = load_resume_protocol(protocol_path)
        raw_root = project_root / protocol["frozen_partial_cache"]["raw_root"]
        raw_root.mkdir(parents=True, exist_ok=True)
        preparation_lock_path = raw_root / ".prepare.lock"
        with preparation_lock_path.open("a+b") as preparation_lock:
            try:
                fcntl.flock(
                    preparation_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                )
            except BlockingIOError as exc:
                raise ValueError("E6 preparation is already running") from exc
            receipt = build_resume_lock_receipt(
                protocol, project_root, args.protocol
            )
            write_initial_rejection_journal(
                protocol, project_root, protocol_path=args.protocol
            )
            _atomic_json(output_path, receipt)
            fcntl.flock(preparation_lock.fileno(), fcntl.LOCK_UN)
        cache = receipt["frozen_partial_cache"]
        print(
            "PASS E6 resume lock: "
            f"pairs={cache['prefix_slots']}, assets={cache['asset_files']}, "
            "reserves_used=0"
        )
        print(
            "Policy: same-label pHash-only historical match=next frozen reserve; "
            "exact or conflicting-label match=hard abort"
        )
        print(f"Lock report: {args.output}")
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"E6 resume lock failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
