from __future__ import annotations

import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from PIL import Image

from scripts.prepare_e5_external_synthbuster import (
    HTTPRangeReader,
    extract_member,
    select_members,
    validate_lock_receipt,
)
from scripts.extract_clip_features import sha256_file


class FakeResponse:
    def __init__(self, status_code: int, payload: bytes, content_range: str | None):
        self.status_code = status_code
        self._payload = payload
        self.headers = {}
        if content_range is not None:
            self.headers["Content-Range"] = content_range

    def iter_content(self, chunk_size: int):
        for start in range(0, len(self._payload), chunk_size):
            yield self._payload[start : start + chunk_size]

    def close(self) -> None:
        return None


class FakeRangeSession:
    def __init__(self, payload: bytes, *, honour_ranges: bool = True):
        self.payload = payload
        self.honour_ranges = honour_ranges
        self.requests: list[tuple[int, int]] = []

    def get(self, url: str, *, headers: dict, stream: bool, timeout: tuple[int, int]):
        requested = headers["Range"].removeprefix("bytes=")
        start_text, end_text = requested.split("-", maxsplit=1)
        start, end = int(start_text), int(end_text)
        self.requests.append((start, end))
        if not self.honour_ranges:
            return FakeResponse(200, self.payload, None)
        chunk = self.payload[start : end + 1]
        return FakeResponse(
            206,
            chunk,
            f"bytes {start}-{end}/{len(self.payload)}",
        )


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 6), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _zip_bytes() -> bytes:
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("synthbuster/firefly/a.png", _png_bytes((1, 2, 3)))
        archive.writestr("synthbuster/firefly/b.png", _png_bytes((4, 5, 6)))
        archive.writestr("synthbuster/dalle3/not-selected.png", _png_bytes((7, 8, 9)))
        archive.writestr("synthbuster/firefly/prompts.csv", "prompt")
    return buffer.getvalue()


def test_http_range_reader_supports_zipfile_and_selective_extraction(tmp_path: Path) -> None:
    payload = _zip_bytes()
    session = FakeRangeSession(payload)
    reader = HTTPRangeReader(session, "https://example.test/archive.zip", block_size=64)
    with ZipFile(reader, "r") as archive:
        members = select_members(
            archive, prefix="synthbuster/firefly/", expected_images=2
        )
        output = tmp_path / "a.png"
        verification = extract_member(archive, members[0], output)
    assert verification["width"] == 8
    assert verification["height"] == 6
    assert verification["sha256"] == sha256_file(output)
    assert len(session.requests) < 20


def test_http_range_reader_rejects_server_that_ignores_ranges() -> None:
    with pytest.raises(ValueError, match="does not support selective byte ranges"):
        HTTPRangeReader(
            FakeRangeSession(_zip_bytes(), honour_ranges=False),
            "https://example.test/archive.zip",
        )


def test_select_members_requires_exact_frozen_count() -> None:
    with ZipFile(io.BytesIO(_zip_bytes()), "r") as archive:
        with pytest.raises(ValueError, match="requires exactly 1000"):
            select_members(
                archive, prefix="synthbuster/firefly/", expected_images=1000
            )


def test_select_members_rejects_unsafe_prefix() -> None:
    with ZipFile(io.BytesIO(_zip_bytes()), "r") as archive:
        with pytest.raises(ValueError, match="Unsafe"):
            select_members(archive, prefix="../firefly/", expected_images=2)


def test_extract_member_repairs_invalid_existing_file(tmp_path: Path) -> None:
    destination = tmp_path / "a.png"
    destination.write_bytes(b"corrupt")
    with ZipFile(io.BytesIO(_zip_bytes()), "r") as archive:
        member = archive.getinfo("synthbuster/firefly/a.png")
        verification = extract_member(archive, member, destination)
    assert verification["sha256"] == sha256_file(destination)
    with Image.open(destination) as image:
        assert image.size == (8, 6)


def test_validate_lock_receipt_rejects_protocol_drift(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.json"
    protocol.write_text('{"version": 1}\n', encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "PASS",
                "protocol": {
                    "path": protocol.as_posix(),
                    "sha256": sha256_file(protocol),
                },
                "external_artifacts_present_at_freeze": False,
            }
        ),
        encoding="utf-8",
    )
    validate_lock_receipt(protocol, receipt)
    protocol.write_text('{"version": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="changed after"):
        validate_lock_receipt(protocol, receipt)
