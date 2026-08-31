#!/usr/bin/env python3
"""Prepare the frozen Synthbuster Firefly + RAISE-1k external test set."""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import sys
import zlib
from collections import OrderedDict
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO
from zipfile import BadZipFile, ZipFile, ZipInfo

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm

from scripts.extract_clip_features import atomic_json_write, sha256_file
from src.e5_external_protocol import (
    validate_e5_external_protocol,
    validate_frozen_e5_external_inputs,
)


DEFAULT_PROTOCOL = Path("configs/e5_fresh_external_evaluation.json")
DEFAULT_LOCK_REPORT = Path("reports/e5_external_evaluation_lock.json")
MANIFEST_FIELDS = ("image_path", "label", "class_name", "source", "split")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MAX_IMAGE_BYTES = 64 * 1024 * 1024
DEFAULT_BLOCK_SIZE = 4 * 1024 * 1024
DEFAULT_CACHE_BLOCKS = 16


class HTTPRangeReader(io.RawIOBase):
    """Seekable read-only HTTP file backed by strict byte-range requests."""

    def __init__(
        self,
        session: requests.Session,
        url: str,
        *,
        block_size: int = DEFAULT_BLOCK_SIZE,
        cache_blocks: int = DEFAULT_CACHE_BLOCKS,
    ) -> None:
        if block_size <= 0 or cache_blocks <= 0:
            raise ValueError("HTTP range cache settings must be positive")
        self._session = session
        self._url = url
        self._block_size = block_size
        self._cache_blocks = cache_blocks
        self._position = 0
        self._closed = False
        self._cache: OrderedDict[int, bytes] = OrderedDict()
        self._size = self._probe_size()

    @property
    def size(self) -> int:
        return self._size

    def _request_range(self, start: int, end: int) -> bytes:
        response = self._session.get(
            self._url,
            headers={"Range": f"bytes={start}-{end}"},
            stream=True,
            timeout=(30, 180),
        )
        try:
            if response.status_code != 206:
                raise ValueError(
                    f"Remote archive does not honour byte ranges: HTTP {response.status_code}"
                )
            content_range = response.headers.get("Content-Range", "")
            expected_prefix = f"bytes {start}-{end}/"
            if not content_range.startswith(expected_prefix):
                raise ValueError(
                    f"Unexpected Content-Range {content_range!r}; expected {expected_prefix!r}"
                )
            payload = b"".join(response.iter_content(chunk_size=1024 * 1024))
        finally:
            response.close()
        expected_length = end - start + 1
        if len(payload) != expected_length:
            raise ValueError(
                f"Truncated HTTP range {start}-{end}: received {len(payload)} bytes"
            )
        return payload

    def _probe_size(self) -> int:
        response = self._session.get(
            self._url,
            headers={"Range": "bytes=0-0"},
            stream=True,
            timeout=(30, 180),
        )
        try:
            if response.status_code != 206:
                raise ValueError(
                    f"Remote archive does not support selective byte ranges: "
                    f"HTTP {response.status_code}"
                )
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith("bytes 0-0/"):
                raise ValueError(f"Invalid range probe response: {content_range!r}")
            total = int(content_range.rsplit("/", maxsplit=1)[1])
            payload = b"".join(response.iter_content(chunk_size=2))
        finally:
            response.close()
        if total <= 0 or payload == b"":
            raise ValueError("Remote archive has an invalid or empty size")
        return total

    def _block(self, index: int) -> bytes:
        cached = self._cache.pop(index, None)
        if cached is not None:
            self._cache[index] = cached
            return cached
        start = index * self._block_size
        if start >= self._size:
            return b""
        end = min(self._size - 1, start + self._block_size - 1)
        payload = self._request_range(start, end)
        self._cache[index] = payload
        while len(self._cache) > self._cache_blocks:
            self._cache.popitem(last=False)
        return payload

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if self._closed:
            raise ValueError("I/O operation on closed HTTP range reader")
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self._position + offset
        elif whence == io.SEEK_END:
            position = self._size + offset
        else:
            raise ValueError(f"Unsupported seek mode: {whence}")
        if position < 0:
            raise ValueError("Negative seek position")
        self._position = position
        return position

    def read(self, size: int = -1) -> bytes:
        if self._closed:
            raise ValueError("I/O operation on closed HTTP range reader")
        remaining = max(0, self._size - self._position)
        if size is None or size < 0:
            size = remaining
        size = min(size, remaining)
        if size == 0:
            return b""
        output = bytearray()
        while len(output) < size:
            block_index = self._position // self._block_size
            block_offset = self._position % self._block_size
            block = self._block(block_index)
            take = min(size - len(output), len(block) - block_offset)
            if take <= 0:
                break
            output.extend(block[block_offset : block_offset + take])
            self._position += take
        return bytes(output)

    def readinto(self, buffer: bytearray | memoryview) -> int:
        payload = self.read(len(buffer))
        buffer[: len(payload)] = payload
        return len(payload)

    def close(self) -> None:
        self._cache.clear()
        self._closed = True
        super().close()


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {"User-Agent": "AIGC-detector-E5-external-evaluation/1.0"}
    )
    return session


def select_members(
    archive: ZipFile, *, prefix: str, expected_images: int
) -> list[ZipInfo]:
    """Select an exact frozen image directory without fuzzy fallback."""

    if not prefix.endswith("/") or prefix.startswith("/") or ".." in PurePosixPath(prefix).parts:
        raise ValueError(f"Unsafe or non-directory ZIP prefix: {prefix!r}")
    members: list[ZipInfo] = []
    seen_names: set[str] = set()
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        if member.is_dir() or not member.filename.startswith(prefix):
            continue
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.name in seen_names:
            raise ValueError(f"Duplicate basename under ZIP prefix {prefix}: {path.name}")
        if member.file_size <= 0 or member.file_size > MAX_IMAGE_BYTES:
            raise ValueError(
                f"Unsafe image size in ZIP member {member.filename}: {member.file_size}"
            )
        seen_names.add(path.name)
        members.append(member)
    members.sort(key=lambda member: member.filename)
    if len(members) != expected_images:
        raise ValueError(
            f"ZIP prefix {prefix!r} contains {len(members)} images; "
            f"the frozen protocol requires exactly {expected_images}"
        )
    return members


def _verify_image(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        width, height = image.size
        image_format = str(image.format or "unknown")
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    if width <= 0 or height <= 0 or rgb.size == 0 or not np.isfinite(rgb).all():
        raise ValueError(f"Invalid decoded image: {path}")
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "width": width,
        "height": height,
        "format": image_format,
    }


def _existing_member_is_valid(member: ZipInfo, destination: Path) -> bool:
    if not destination.is_file() or destination.stat().st_size != member.file_size:
        return False
    checksum = 0
    with destination.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum = zlib.crc32(chunk, checksum)
    if checksum & 0xFFFFFFFF != member.CRC:
        return False
    try:
        _verify_image(destination)
    except (OSError, ValueError):
        return False
    return True


def extract_member(archive: ZipFile, member: ZipInfo, destination: Path) -> dict[str, Any]:
    """CRC-check, decode-check, and atomically publish one ZIP image."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if not _existing_member_is_valid(member, destination):
        temporary = destination.with_suffix(destination.suffix + ".part")
        with archive.open(member, "r") as source, temporary.open("wb") as target:
            copied = 0
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                copied += len(chunk)
                if copied > MAX_IMAGE_BYTES:
                    raise ValueError(f"ZIP member exceeded image safety limit: {member.filename}")
                target.write(chunk)
        if copied != member.file_size:
            raise ValueError(
                f"ZIP member size mismatch for {member.filename}: {copied} != {member.file_size}"
            )
        _verify_image(temporary)
        temporary.replace(destination)
    verification = _verify_image(destination)
    return {
        "archive_member": member.filename,
        "zip_crc32": f"{member.CRC:08x}",
        "zip_compressed_bytes": member.compress_size,
        **verification,
    }


def _project_relative(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("E5 external outputs must remain inside the project") from exc


def _atomic_csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def validate_lock_receipt(
    protocol_path: Path, lock_report_path: Path
) -> dict[str, Any]:
    receipt = json.loads(lock_report_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "PASS":
        raise ValueError("E5 external lock receipt did not pass")
    frozen = receipt.get("protocol", {})
    if frozen.get("path") != protocol_path.as_posix():
        raise ValueError("E5 external lock receipt references a different protocol")
    if frozen.get("sha256") != sha256_file(protocol_path):
        raise ValueError("E5 external protocol changed after the pre-download lock")
    if receipt.get("external_artifacts_present_at_freeze") is not False:
        raise ValueError("External artifacts were present when the lock was recorded")
    return receipt


def load_development_content_hashes(project_root: Path) -> set[str]:
    """Hash every image used in E5 development or the prior SID audit."""

    manifest_paths = (
        Path("data/processed/train.csv"),
        Path("data/processed/val.csv"),
        Path("data/processed/e4_sid_real/train.csv"),
        Path("data/processed/e4_sid_real/val.csv"),
        Path("data/processed/e5_sid_flux/train.csv"),
        Path("data/processed/e5_sid_flux/val.csv"),
        Path("data/processed/sid_set_flux_heldout.csv"),
    )
    image_paths: set[Path] = set()
    for relative_manifest in manifest_paths:
        manifest = project_root / relative_manifest
        if not manifest.is_file():
            raise ValueError(f"Required exclusion manifest is missing: {manifest}")
        with manifest.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                image_paths.add(project_root / row["image_path"])
    hashes: set[str] = set()
    for path in tqdm(sorted(image_paths), desc="E5 exclusion hashes", unit="image"):
        if not path.is_file():
            raise ValueError(f"Development/audit image is missing: {path}")
        digest = sha256_file(path)
        if digest in hashes:
            continue
        hashes.add(digest)
    return hashes


def _open_remote_zip(
    session: requests.Session, source: dict[str, Any]
) -> tuple[HTTPRangeReader, ZipFile]:
    reader = HTTPRangeReader(session, str(source["source_url"]))
    try:
        archive = ZipFile(reader, "r")
    except Exception:
        reader.close()
        raise
    return reader, archive


def _prepare_source(
    *,
    session: requests.Session,
    source: dict[str, Any],
    destination_root: Path,
    label: int,
    class_name: str,
    project_root: Path,
    progress: tqdm,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    reader, archive = _open_remote_zip(session, source)
    try:
        members = select_members(
            archive,
            prefix=str(source["expected_archive_prefix"]),
            expected_images=int(source["expected_images"]),
        )
        manifest_rows: list[dict[str, Any]] = []
        provenance_rows: list[dict[str, Any]] = []
        for member in members:
            destination = destination_root / PurePosixPath(member.filename).name
            verification = extract_member(archive, member, destination)
            relative = _project_relative(destination, project_root)
            manifest_rows.append(
                {
                    "image_path": relative,
                    "label": label,
                    "class_name": class_name,
                    "source": "synthbuster_firefly_raise_external",
                    "split": "external_test",
                }
            )
            provenance_rows.append(
                {
                    "image_path": relative,
                    "label": label,
                    "class_name": class_name,
                    **verification,
                }
            )
            progress.update(1)
        archive_summary = {
            "source_archive": source["source_archive"],
            "published_md5": source["source_archive_md5"],
            "remote_bytes": reader.size,
            "selected_prefix": source["expected_archive_prefix"],
            "selected_images": len(members),
            "integrity": "ZIP central-directory structure plus per-entry CRC32; per-file SHA-256 recorded",
            "full_archive_md5_verified": False,
            "full_archive_md5_note": "Not applicable to selective HTTP range extraction; the complete archive was not downloaded.",
        }
        return manifest_rows, provenance_rows, archive_summary
    finally:
        archive.close()
        reader.close()


def prepare_external_set(
    protocol_path: Path,
    lock_report_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_e5_external_protocol(protocol)
    validate_frozen_e5_external_inputs(protocol, project_root)
    lock_receipt = validate_lock_receipt(protocol_path, lock_report_path)
    integrity = protocol["acquisition_and_integrity"]
    dataset = protocol["external_dataset"]
    raw_root = project_root / integrity["raw_root"]
    manifest_path = project_root / integrity["manifest"]
    provenance_path = project_root / integrity["provenance"]
    real_root = raw_root / "real"
    ai_root = raw_root / "ai_generated"

    session = _session()
    progress = tqdm(total=2000, desc="E5 fresh external images", unit="image")
    try:
        real_manifest, real_provenance, real_archive = _prepare_source(
            session=session,
            source=dataset["real"],
            destination_root=real_root,
            label=0,
            class_name="real",
            project_root=project_root,
            progress=progress,
        )
        ai_manifest, ai_provenance, ai_archive = _prepare_source(
            session=session,
            source=dataset["synthetic"],
            destination_root=ai_root,
            label=1,
            class_name="ai_generated",
            project_root=project_root,
            progress=progress,
        )
    finally:
        progress.close()
        session.close()

    provenance_rows = real_provenance + ai_provenance
    external_hashes = [str(row["sha256"]) for row in provenance_rows]
    if len(set(external_hashes)) != 2000:
        raise ValueError("The fresh external set contains byte-identical images")
    development_hashes = load_development_content_hashes(project_root)
    overlap = set(external_hashes) & development_hashes
    if overlap:
        raise ValueError(
            f"Fresh external content overlaps E5 development/audit data: {len(overlap)} files"
        )

    manifest_rows = real_manifest + ai_manifest
    _atomic_csv_write(manifest_path, manifest_rows)
    provenance = {
        "experiment": "e5_fresh_synthbuster_firefly_preparation",
        "protocol": {
            "path": protocol_path.as_posix(),
            "sha256": sha256_file(protocol_path),
        },
        "lock_receipt": {
            "path": lock_report_path.as_posix(),
            "sha256": sha256_file(lock_report_path),
            "frozen_at_utc": lock_receipt["frozen_at_utc"],
        },
        "dataset": {
            "benchmark": dataset["benchmark_name"],
            "doi": dataset["benchmark_doi"],
            "real_source": dataset["real"]["source_name"],
            "synthetic_generator": dataset["synthetic"]["selected_generator"],
            "organiser_validation_subset_used": False,
            "manual_image_inspection_before_scoring": False,
        },
        "archives": {"real": real_archive, "synthetic": ai_archive},
        "images": provenance_rows,
        "counts": {
            "total": len(provenance_rows),
            "real": len(real_provenance),
            "ai_generated": len(ai_provenance),
            "content_unique": len(set(external_hashes)),
            "development_or_prior_audit_overlap": len(overlap),
            "development_and_audit_exclusion_hashes": len(development_hashes),
        },
        "manifest": {
            "path": _project_relative(manifest_path, project_root),
            "sha256": None,
        },
        "provenance": {
            "path": _project_relative(provenance_path, project_root),
        },
        "publication": {
            "raw_images_committed_to_git": False,
            "individual_images_allowed_in_public_demo": False,
            "aggregate_outputs_only": True,
        },
    }
    # The manifest must exist before its digest can be frozen into provenance.
    provenance["manifest"]["sha256"] = sha256_file(manifest_path)
    atomic_json_write(provenance_path, provenance)
    return provenance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Selectively prepare the frozen RAISE-1k + Firefly external set."
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--lock-report", type=Path, default=DEFAULT_LOCK_REPORT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        free_gib = shutil.disk_usage(Path.cwd()).free / (1024**3)
        if free_gib < 4.0:
            raise ValueError(
                f"At least 4 GiB free space is required before acquisition; found {free_gib:.2f} GiB"
            )
        result = prepare_external_set(args.protocol, args.lock_report, Path.cwd())
        counts = result["counts"]
        print(
            "PASS E5 fresh external preparation: "
            f"real={counts['real']}, firefly={counts['ai_generated']}, "
            f"overlap={counts['development_or_prior_audit_overlap']}"
        )
        print(
            "Manifest="
            f"{result['manifest']['path']}; "
            f"provenance={result['provenance']['path']}"
        )
        return 0
    except (
        BadZipFile,
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
        requests.RequestException,
    ) as exc:
        print(f"E5 fresh external preparation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
