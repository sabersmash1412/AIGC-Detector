from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from PIL import Image

from scripts.prepare_e5_external_aigibench import (
    canonical_member_list_sha256,
    select_class_members,
    validate_lock_receipt,
)
from scripts.extract_clip_features import sha256_file


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 6), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _archive_bytes() -> bytes:
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for index in range(4):
            archive.writestr(
                f"Midjourney/0_real/r{index}.png", _png_bytes((index, 2, 3))
            )
            archive.writestr(
                f"Midjourney/1_fake/f{index}.png", _png_bytes((4, index, 6))
            )
        archive.writestr("Midjourney/notes.txt", "not an image")
    return buffer.getvalue()


def test_hash_ranked_selection_is_deterministic_and_class_exact() -> None:
    with ZipFile(io.BytesIO(_archive_bytes()), "r") as archive:
        first = select_class_members(
            archive,
            prefix="Midjourney/0_real/",
            available_images=4,
            selected_images=2,
            seed=42,
        )
        second = select_class_members(
            archive,
            prefix="Midjourney/0_real/",
            available_images=4,
            selected_images=2,
            seed=42,
        )
    assert [member.filename for member in first] == [member.filename for member in second]
    expected = sorted(
        [f"Midjourney/0_real/r{index}.png" for index in range(4)],
        key=lambda name: (hashlib.sha256(f"42:{name}".encode()).digest(), name),
    )[:2]
    assert [member.filename for member in first] == expected


def test_selection_requires_frozen_available_count() -> None:
    with ZipFile(io.BytesIO(_archive_bytes()), "r") as archive:
        with pytest.raises(ValueError, match="requires 3000 available"):
            select_class_members(
                archive,
                prefix="Midjourney/1_fake/",
                available_images=3000,
                selected_images=1000,
                seed=42,
            )


def test_canonical_member_digest_includes_class_and_order() -> None:
    with ZipFile(io.BytesIO(_archive_bytes()), "r") as archive:
        real = select_class_members(
            archive,
            prefix="Midjourney/0_real/",
            available_images=4,
            selected_images=2,
            seed=42,
        )
        fake = select_class_members(
            archive,
            prefix="Midjourney/1_fake/",
            available_images=4,
            selected_images=2,
            seed=42,
        )
        digest = canonical_member_list_sha256([real, fake])
        reversed_digest = canonical_member_list_sha256([fake, real])
    assert len(digest) == 64
    assert digest != reversed_digest


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
