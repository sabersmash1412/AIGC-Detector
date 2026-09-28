#!/usr/bin/env python3
"""Emit the offline E6 acquisition/security lock before any image download."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from src.e6_acquisition_protocol import (
    build_acquisition_lock_receipt,
    load_e6_acquisition_protocol,
)


DEFAULT_PROTOCOL = Path("configs/e6_acquisition_protocol.json")
DEFAULT_OUTPUT = Path("reports/e6_acquisition_protocol_lock.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the frozen E6 source, security and duplicate-handling "
            "contract without network or image access."
        )
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _project_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _assert_no_symlink_components(path: Path, stop: Path) -> None:
    current = path
    while current != stop:
        if current.is_symlink():
            raise ValueError(f"Refusing symlink in lock output path: {current}")
        if current.parent == current:
            raise ValueError("Lock output path is outside the project")
        current = current.parent


def _atomic_write_json(path: Path, payload: dict, project_root: Path) -> None:
    try:
        path.relative_to(project_root)
    except ValueError as exc:
        raise ValueError("Lock report must stay inside the project") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_components(path, project_root)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise ValueError(f"Stale or unsafe lock temporary file exists: {temporary}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
        raise


def main() -> int:
    args = build_parser().parse_args()
    project_root = Path.cwd().resolve()
    protocol_path = _project_path(project_root, args.protocol)
    output_path = _project_path(project_root, args.output)
    try:
        protocol = load_e6_acquisition_protocol(protocol_path)
        receipt = build_acquisition_lock_receipt(
            protocol,
            project_root,
            args.protocol,
        )
        _atomic_write_json(output_path, receipt, project_root)
        print(
            "PASS E6 acquisition lock: source=Tiny-GenImage/BigGAN, "
            "pairs=1400+600 reserve, image_payloads=0"
        )
        print(
            "Security: pinned revision and cached-asset host/path; redirects=off; "
            "signed URLs are never persisted"
        )
        print(
            "Overlap: existing-data or conflicting-label match=hard abort; "
            "only corrupt or within-E6 same-label duplicates may use reserve pairs"
        )
        print(f"Lock report: {args.output}")
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"E6 acquisition lock failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
