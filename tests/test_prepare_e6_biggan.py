from __future__ import annotations

import copy
import io
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import requests
from PIL import Image

from scripts.extract_clip_features import sha256_file
from scripts.prepare_e6_biggan import (
    CONNECT_TIMEOUT_SECONDS,
    MAX_IMAGE_BYTES,
    NEAR_DUPLICATE_MAX_HAMMING,
    READ_TIMEOUT_SECONDS,
    FingerprintIndex,
    FingerprintOwner,
    HammingBKTree,
    ImageIntegrityError,
    ImageVerification,
    PreparedImage,
    SourceRow,
    TransientAcquisitionError,
    _cache_pending_receipt_path,
    _cache_receipt_path,
    _clear_stale_partial_downloads,
    _atomic_csv_write,
    _atomic_json_write_fsync,
    _manifest_row,
    _recover_or_reuse_publication,
    _reject_symlink_path,
    _write_cache_receipt,
    _validate_asset_url,
    decoded_pixel_sha256,
    download_source_image,
    inspect_image,
    pair_conflicts,
    parse_rows_page,
    perceptual_hash64,
    ranked_cycles,
    validate_acquisition_lock_receipt,
    validate_lock_receipt,
    validate_pair_rows,
    validate_transport_preflight_audit,
)
from src.e6_protocol import build_pair_assignment, load_e6_protocol


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/e6_development_protocol.json"
LOCK_PATH = ROOT / "reports/e6_development_protocol_lock.json"
ACQUISITION_PROTOCOL_PATH = ROOT / "configs/e6_acquisition_protocol.json"
ACQUISITION_LOCK_PATH = ROOT / "reports/e6_acquisition_protocol_lock.json"
TRANSPORT_PREFLIGHT_AUDIT_PATH = ROOT / "reports/e6_transport_preflight_audit.json"
REVISION = "89c4fe9efd0ebc7ce5c7641ef57d578ccd639c69"
REPOSITORY = "TheKernel01/Tiny-GenImage"


def _protocol() -> dict:
    return load_e6_protocol(PROTOCOL_PATH)


def _asset_url(row_idx: int) -> str:
    return (
        "https://datasets-server.huggingface.co/cached-assets/"
        f"TheKernel01/Tiny-GenImage/--/{REVISION}/--/default/train/"
        f"{row_idx}/image/source.png?Expires=123&Signature=redacted"
    )


def _row_envelope(
    row_idx: int,
    *,
    label: int,
    generator: int,
    width: int = 16,
    height: int = 12,
) -> dict:
    return {
        "row_idx": row_idx,
        "row": {
            "image": {
                "src": _asset_url(row_idx),
                "width": width,
                "height": height,
            },
            "label": label,
            "generator": generator,
        },
        "truncated_cells": [],
    }


def _pair_payload() -> dict:
    return {
        "features": [],
        "partial": False,
        "num_rows_total": 28000,
        "num_rows_per_page": 2,
        "rows": [
            _row_envelope(2, label=0, generator=0),
            _row_envelope(3, label=1, generator=2, width=128, height=128),
        ],
    }


def _verification(
    token: int,
    *,
    byte_hash: str | None = None,
    pixel_hash: str | None = None,
    phash: int | None = None,
) -> ImageVerification:
    return ImageVerification(
        file_bytes=100 + token,
        byte_sha256=byte_hash or f"{token:064x}",
        decoded_pixel_sha256=pixel_hash or f"{token + 100:064x}",
        perceptual_hash=f"{(token if phash is None else phash):016x}",
        original_width=16,
        original_height=12,
        aspect_ratio=16 / 12,
        decoded_pixel_count=192,
        display_width=16,
        display_height=12,
        file_format="PNG",
    )


def _prepared(row_idx: int, label: int, verification: ImageVerification) -> PreparedImage:
    return PreparedImage(
        row=SourceRow(
            row_idx=row_idx,
            label=label,
            generator_id=0 if label == 0 else 2,
            generator_name="Real" if label == 0 else "BigGAN",
            image_url=_asset_url(row_idx),
            source_width=16,
            source_height=12,
        ),
        image_path=f"data/raw/e6/{row_idx}.asset",
        verification=verification,
    )


def _png_bytes(size: tuple[int, int] = (16, 12), color=(20, 40, 60)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_checked_in_lock_binds_protocol_assignment_and_registry() -> None:
    protocol = _protocol()

    receipt = validate_lock_receipt(
        PROTOCOL_PATH, LOCK_PATH, protocol, ROOT
    )

    assert receipt["status"] == "PASS"
    assert receipt["development_source"]["assignment_sha256"] == (
        "c20aac52c0db04dfa2bf3ccda9a04ddd7858f82544442bc34cecd5b3161f4d68"
    )


def test_checked_in_acquisition_lock_binds_the_downloader_contract() -> None:
    acquisition = json.loads(ACQUISITION_PROTOCOL_PATH.read_text(encoding="utf-8"))

    receipt, development = validate_acquisition_lock_receipt(
        ACQUISITION_PROTOCOL_PATH,
        ACQUISITION_LOCK_PATH,
        acquisition,
        ROOT,
    )

    assert receipt["status"] == "PASS"
    assert development["development_source"]["selected_generator"] == "BigGAN"


def test_checked_in_transport_probe_is_explicitly_disclosed() -> None:
    audit = validate_transport_preflight_audit(
        TRANSPORT_PREFLIGHT_AUDIT_PATH,
        ACQUISITION_PROTOCOL_PATH,
        ACQUISITION_LOCK_PATH,
    )

    assert audit["probe"]["response_body_bytes_read"] == 64
    assert audit["probe"]["image_decoded"] is False
    assert audit["probe"]["image_body_saved"] is False
    assert (
        audit["chronology"][
            "acquisition_protocol_and_pass_receipt_git_committed_before_probe"
        ]
        is False
    )


def test_lock_rejects_protocol_or_registry_drift(tmp_path: Path) -> None:
    protocol = _protocol()
    (tmp_path / "configs").mkdir()
    (tmp_path / "reports").mkdir()
    protocol_copy = tmp_path / "configs/e6_development_protocol.json"
    registry_copy = tmp_path / "configs/evaluation_registry.json"
    protocol_copy.write_bytes(PROTOCOL_PATH.read_bytes())
    registry_copy.write_bytes((ROOT / "configs/evaluation_registry.json").read_bytes())
    receipt = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    receipt["registry"]["sha256"] = sha256_file(registry_copy)
    lock_copy = tmp_path / "reports/e6_development_protocol_lock.json"
    lock_copy.write_text(json.dumps(receipt), encoding="utf-8")

    validate_lock_receipt(protocol_copy, lock_copy, protocol, tmp_path)
    protocol_copy.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed after"):
        validate_lock_receipt(protocol_copy, lock_copy, protocol, tmp_path)

    protocol_copy.write_bytes(PROTOCOL_PATH.read_bytes())
    registry_copy.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="registry changed"):
        validate_lock_receipt(protocol_copy, lock_copy, protocol, tmp_path)


def test_frozen_cycle_ranking_matches_assignment_and_has_reserve() -> None:
    protocol = _protocol()
    ranked = ranked_cycles(protocol)
    assigned = tuple(item["pair_index"] for item in build_pair_assignment(protocol))

    assert len(ranked) == 2000
    assert len(set(ranked)) == 2000
    assert ranked[:1400] == assigned
    assert len(ranked[1400:]) == 600


def test_row_parser_uses_actual_image_label_generator_schema() -> None:
    rows = parse_rows_page(
        _pair_payload(),
        observed_revision=REVISION,
        expected_revision=REVISION,
        expected_total_rows=28000,
        repository=REPOSITORY,
        split="train",
    )
    real, ai = validate_pair_rows(_protocol(), 0, rows)

    assert (real.label, real.generator_id, real.generator_name) == (0, 0, "Real")
    assert (ai.label, ai.generator_id, ai.generator_name) == (1, 2, "BigGAN")
    assert ai.source_width == ai.source_height == 128


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(partial=True), "partial"),
        (lambda payload: payload.update(num_rows_total=1), "row count"),
        (
            lambda payload: payload["rows"][0]["row"].update(generator=2),
            "row contract drift",
        ),
        (
            lambda payload: payload["rows"][0]["row"]["image"].update(
                src="https://example.test/real.png"
            ),
            "Unsafe datasets-server",
        ),
    ],
)
def test_row_parser_or_pair_contract_rejects_source_drift(mutation, message: str) -> None:
    payload = _pair_payload()
    mutation(payload)
    if message == "row contract drift":
        rows = parse_rows_page(
            payload,
            observed_revision=REVISION,
            expected_revision=REVISION,
            expected_total_rows=28000,
            repository=REPOSITORY,
            split="train",
        )
        with pytest.raises(ValueError, match=message):
            validate_pair_rows(_protocol(), 0, rows)
    else:
        with pytest.raises(ValueError, match=message):
            parse_rows_page(
                payload,
                observed_revision=REVISION,
                expected_revision=REVISION,
                expected_total_rows=28000,
                repository=REPOSITORY,
                split="train",
            )


def test_row_parser_rejects_wrong_revision_missing_or_extra_indices() -> None:
    with pytest.raises(ValueError, match="revision drift"):
        parse_rows_page(
            _pair_payload(),
            observed_revision="0" * 40,
            expected_revision=REVISION,
            expected_total_rows=28000,
            repository=REPOSITORY,
            split="train",
        )

    with pytest.raises(ValueError, match="omitted, repeated, or added"):
        parse_rows_page(
            _pair_payload(),
            observed_revision=REVISION,
            expected_revision=REVISION,
            expected_total_rows=28000,
            repository=REPOSITORY,
            split="train",
            expected_offset=2,
            expected_length=3,
        )


@pytest.mark.parametrize(
    "url",
    [
        _asset_url(3).replace("https://", "http://"),
        _asset_url(3).replace("datasets-server.huggingface.co", "localhost"),
        _asset_url(3).replace("/3/image/", "/4/image/"),
        _asset_url(3).replace("/image/source.png", "/image/../source.png"),
        _asset_url(3) + "#fragment",
    ],
)
def test_asset_url_validator_rejects_unpinned_or_unsafe_urls(url: str) -> None:
    with pytest.raises(ValueError):
        _validate_asset_url(
            url,
            repository=REPOSITORY,
            revision=REVISION,
            split="train",
            row_idx=3,
        )


def test_canonical_pixel_hash_matches_lossless_encodings_and_binds_dimensions(
    tmp_path: Path,
) -> None:
    png = tmp_path / "same.png"
    second_png = tmp_path / "same-second.png"
    reshaped = tmp_path / "reshaped.png"
    pixels = [(index, index * 2 % 256, index * 3 % 256) for index in range(48)]
    first = Image.new("RGB", (8, 6))
    first.putdata(pixels)
    first.save(png, compress_level=0)
    first.save(second_png, compress_level=9)
    second = Image.new("RGB", (6, 8))
    second.putdata(pixels)
    second.save(reshaped)

    png_check = inspect_image(png)
    second_check = inspect_image(second_png)
    reshaped_check = inspect_image(reshaped)

    assert png_check.byte_sha256 != second_check.byte_sha256
    assert png_check.decoded_pixel_sha256 == second_check.decoded_pixel_sha256
    assert png_check.decoded_pixel_sha256 != reshaped_check.decoded_pixel_sha256
    with Image.open(png) as image:
        assert decoded_pixel_sha256(image) == png_check.decoded_pixel_sha256


def test_rgba_identity_keeps_alpha_while_phash_white_composites_it() -> None:
    first = Image.new("RGBA", (8, 8), (10, 20, 30, 0))
    second = Image.new("RGBA", (8, 8), (200, 150, 100, 0))

    assert decoded_pixel_sha256(first) != decoded_pixel_sha256(second)
    assert perceptual_hash64(first) == perceptual_hash64(second)


def test_decompression_bomb_warning_is_wrapped_as_image_integrity_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "warning.png"
    image_path.write_bytes(_png_bytes())
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)

    with pytest.raises(ImageIntegrityError) as caught:
        inspect_image(image_path)

    assert "decode verification" in str(caught.value)


def test_hamming_tree_enforces_radius_boundary() -> None:
    tree = HammingBKTree()
    owner = FingerprintOwner("first", "development_train", 0)
    tree.add(0, owner)

    assert tree.query((1 << NEAR_DUPLICATE_MAX_HAMMING) - 1, 4) == [owner]
    assert tree.query((1 << (NEAR_DUPLICATE_MAX_HAMMING + 1)) - 1, 4) == []


def test_perceptual_hash_has_a_pinned_nontrivial_golden_value() -> None:
    image = Image.new("RGB", (17, 13))
    image.putdata(
        [
            (
                (x * 17 + y * 3) % 256,
                (x * 5 + y * 29) % 256,
                (x * x + y * 11) % 256,
            )
            for y in range(13)
            for x in range(17)
        ]
    )

    assert f"{perceptual_hash64(image):016x}" == "14252f79cb2671c9"


def test_pair_conflicts_rejects_cross_role_near_duplicate() -> None:
    existing = FingerprintIndex()
    accepted = FingerprintIndex()
    accepted.add(
        _verification(1, phash=0),
        FingerprintOwner("prior", "development_train", 0),
    )
    real = _prepared(2, 0, _verification(2, phash=1))
    ai = _prepared(3, 1, _verification(3, phash=(1 << 64) - 1))

    reasons = pair_conflicts(
        (real, ai),
        role="development_model_selection",
        existing=existing,
        accepted=accepted,
    )

    assert any(reason["kind"] == "near_duplicate" for reason in reasons)


def test_pair_conflicts_aborts_conflicting_exact_labels() -> None:
    existing = FingerprintIndex()
    accepted = FingerprintIndex()
    same_byte = "a" * 64
    real = _prepared(2, 0, _verification(1, byte_hash=same_byte, phash=0))
    ai = _prepared(3, 1, _verification(2, byte_hash=same_byte, phash=(1 << 63)))

    with pytest.raises(ValueError, match="conflicting labels"):
        pair_conflicts(
            (real, ai),
            role="development_train",
            existing=existing,
            accepted=accepted,
        )


class FakeResponse:
    def __init__(
        self,
        payload: bytes,
        *,
        status_code: int = 200,
        length: int | None = None,
        content_type: str = "image/png",
    ):
        self.payload = payload
        self.status_code = status_code
        self.headers = {
            "Content-Type": content_type,
            "Content-Length": str(len(payload) if length is None else length),
        }

    def iter_content(self, chunk_size: int):
        for start in range(0, len(self.payload), chunk_size):
            yield self.payload[start : start + chunk_size]

    def close(self) -> None:
        return None


class FakeSession:
    def __init__(self, response: FakeResponse):
        self.response = response
        self.calls: list[dict] = []

    def get(self, url: str, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return self.response


class FailingSession:
    def get(self, url: str, **kwargs):
        raise requests.ConnectionError(f"failed signed URL {url}&Signature=SECRET")


class FailingStreamResponse(FakeResponse):
    def iter_content(self, chunk_size: int):
        raise requests.ConnectionError(
            "stream failed at https://example.test/?Signature=STREAM_SECRET"
        )


class SequenceSession:
    def __init__(self, responses: list[FakeResponse]):
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(self, url: str, **kwargs):
        self.calls.append(url)
        return self.responses.pop(0)


def test_downloader_is_atomic_disables_redirects_and_revalidates_cache(
    tmp_path: Path,
) -> None:
    payload = _png_bytes()
    session = FakeSession(FakeResponse(payload))
    row = SourceRow(2, 0, 0, "Real", _asset_url(2), 16, 12)
    destination = tmp_path / "data/raw/e6/images/row-00002.png"

    first = download_source_image(
        session, row, destination, project_root=tmp_path
    )
    second = download_source_image(
        session, row, destination, project_root=tmp_path
    )

    assert first == second
    assert len(session.calls) == 1
    assert session.calls[0]["allow_redirects"] is False
    assert session.calls[0]["timeout"] == (
        CONNECT_TIMEOUT_SECONDS,
        READ_TIMEOUT_SECONDS,
    )
    assert destination.read_bytes() == payload
    assert not list(destination.parent.glob("*.part"))


def test_downloader_redacts_signed_url_from_request_failure(tmp_path: Path) -> None:
    row = SourceRow(2, 0, 0, "Real", _asset_url(2), 16, 12)
    destination = tmp_path / "data/raw/e6/images/row-00002.png"

    with pytest.raises(TransientAcquisitionError) as caught:
        download_source_image(
            FailingSession(), row, destination, project_root=tmp_path
        )

    message = str(caught.value)
    assert "Signature" not in message
    assert "SECRET" not in message
    assert "row 2" in message


def test_downloader_redacts_signed_url_from_stream_failure(tmp_path: Path) -> None:
    row = SourceRow(2, 0, 0, "Real", _asset_url(2), 16, 12)
    destination = tmp_path / "data/raw/e6/images/row-00002.png"

    with pytest.raises(TransientAcquisitionError) as caught:
        download_source_image(
            FakeSession(FailingStreamResponse(_png_bytes())),
            row,
            destination,
            project_root=tmp_path,
        )

    message = str(caught.value)
    assert "Signature" not in message
    assert "STREAM_SECRET" not in message
    assert "row 2" in message


def test_downloader_refreshes_expired_url_without_changing_row_identity(
    tmp_path: Path,
) -> None:
    payload = _png_bytes()
    session = SequenceSession(
        [FakeResponse(b"", status_code=403), FakeResponse(payload)]
    )
    row = SourceRow(2, 0, 0, "Real", _asset_url(2), 16, 12)
    refreshed = SourceRow(
        2,
        0,
        0,
        "Real",
        _asset_url(2).replace("Signature=redacted", "Signature=fresh"),
        16,
        12,
    )
    destination = tmp_path / "data/raw/e6/images/row-00002.png"

    verification = download_source_image(
        session,
        row,
        destination,
        project_root=tmp_path,
        refresh_row=lambda _: refreshed,
    )

    assert verification.file_format == "PNG"
    assert len(session.calls) == 2
    assert "Signature=fresh" in session.calls[1]


def test_downloader_requires_content_length_to_avoid_transient_replacement(
    tmp_path: Path,
) -> None:
    response = FakeResponse(_png_bytes())
    response.headers.pop("Content-Length")
    row = SourceRow(2, 0, 0, "Real", _asset_url(2), 16, 12)
    destination = tmp_path / "data/raw/e6/images/row-00002.png"

    with pytest.raises(TransientAcquisitionError, match="omitted Content-Length"):
        download_source_image(
            FakeSession(response), row, destination, project_root=tmp_path
        )

    assert not destination.exists()
    assert not list(destination.parent.glob("*.part"))


def test_downloader_recovers_asset_published_before_receipt_commit(
    tmp_path: Path,
) -> None:
    payload = _png_bytes()
    session = FakeSession(FakeResponse(payload))
    row = SourceRow(2, 0, 0, "Real", _asset_url(2), 16, 12)
    destination = tmp_path / "data/raw/e6/images/row-00002.png"
    first = download_source_image(session, row, destination, project_root=tmp_path)
    receipt = _cache_receipt_path(destination)
    pending = _cache_pending_receipt_path(destination)
    receipt.replace(pending)

    second = download_source_image(session, row, destination, project_root=tmp_path)

    assert second == first
    assert len(session.calls) == 1
    assert receipt.is_file()
    assert not pending.exists()


def test_reject_symlink_path_detects_in_project_parent_symlink(
    tmp_path: Path,
) -> None:
    real_directory = tmp_path / "real"
    real_directory.mkdir()
    linked_directory = tmp_path / "linked"
    linked_directory.symlink_to(real_directory, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        _reject_symlink_path(linked_directory / "asset.png", tmp_path)


def test_stale_partial_download_is_removed_for_safe_rerun(tmp_path: Path) -> None:
    images = tmp_path / "data/raw/e6/images"
    images.mkdir(parents=True)
    stale = images / ".row-00002.png.abc12345.part"
    stale.write_bytes(b"partial")

    removed = _clear_stale_partial_downloads(
        tmp_path / "data/raw/e6", tmp_path
    )

    assert removed == 1
    assert not stale.exists()


def test_downloader_accepts_generic_binary_only_after_png_magic_validation(
    tmp_path: Path,
) -> None:
    payload = _png_bytes()
    session = FakeSession(
        FakeResponse(payload, content_type="application/octet-stream")
    )
    row = SourceRow(2, 0, 0, "Real", _asset_url(2), 16, 12)
    destination = tmp_path / "data/raw/e6/images/row-00002.png"

    verification = download_source_image(
        session, row, destination, project_root=tmp_path
    )

    assert verification.file_format == "PNG"
    assert destination.read_bytes() == payload


def test_downloader_rejects_declared_oversize_without_final_file(tmp_path: Path) -> None:
    session = FakeSession(FakeResponse(_png_bytes(), length=MAX_IMAGE_BYTES + 1))
    row = SourceRow(2, 0, 0, "Real", _asset_url(2), 16, 12)
    destination = tmp_path / "data/raw/e6/images/row-00002.png"

    with pytest.raises(ValueError, match="unsafe byte size"):
        download_source_image(session, row, destination, project_root=tmp_path)

    assert not destination.exists()
    assert not list(destination.parent.glob("*.part"))


def test_fingerprint_index_aborts_exact_content_with_conflicting_labels() -> None:
    index = FingerprintIndex()
    check = _verification(1)
    index.add(check, FingerprintOwner("real", "development_train", 0))

    with pytest.raises(ValueError, match="Conflicting labels"):
        index.add(check, FingerprintOwner("fake", "development_train", 1))


def _publication_fixture(tmp_path: Path) -> dict:
    protocol_path = tmp_path / "configs/protocol.json"
    lock_path = tmp_path / "reports/development-lock.json"
    acquisition_path = tmp_path / "configs/acquisition.json"
    acquisition_lock_path = tmp_path / "reports/acquisition-lock.json"
    for path in (protocol_path, lock_path, acquisition_path, acquisition_lock_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    transport_audit_path = tmp_path / "reports/e6_transport_preflight_audit.json"
    transport_audit_path.write_text("{}\n", encoding="utf-8")

    image_path = tmp_path / "data/raw/e6/images/row-00002.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(_png_bytes())
    verification = inspect_image(image_path)
    source_row = SourceRow(2, 0, 0, "Real", _asset_url(2), 16, 12)
    _write_cache_receipt(
        _cache_receipt_path(image_path), source_row, verification
    )
    prepared = PreparedImage(
        row=source_row,
        image_path=image_path.relative_to(tmp_path).as_posix(),
        verification=verification,
    )
    role = "development_train"
    manifest_row = _manifest_row(
        prepared,
        role,
        pair_key="pair-0",
        assigned_cycle=0,
        accepted_cycle=0,
    )
    manifest_root = tmp_path / "data/processed/manifests"
    manifest_root.mkdir(parents=True)
    manifest_path = manifest_root / f"{role}.csv"
    _atomic_csv_write(manifest_path, [manifest_row])
    provenance_path = tmp_path / "data/processed/provenance.json"
    staging_root = tmp_path / "data/raw/.e6-staging"
    staging_root.mkdir()
    staged_provenance = staging_root / "provenance.json"
    outputs = {
        "provenance": provenance_path.relative_to(tmp_path).as_posix(),
        "role_manifests": {
            role: manifest_path.relative_to(tmp_path).as_posix(),
        },
    }
    image_record = {
        "row_idx": 2,
        "source_label": 0,
        "project_label": 0,
        "generator_id": 0,
        "generator_name": "Real",
        "source_width": 16,
        "source_height": 12,
        "role": role,
        "image_path": prepared.image_path,
        **asdict(verification),
    }
    provenance = {
        "status": "PASS",
        "protocol": {"sha256": sha256_file(protocol_path)},
        "lock_receipt": {"sha256": sha256_file(lock_path)},
        "acquisition_protocol": {
            "sha256": sha256_file(acquisition_path),
            "lock_sha256": sha256_file(acquisition_lock_path),
        },
        "transport_preflight_audit": {
            "path": "reports/e6_transport_preflight_audit.json",
            "sha256": sha256_file(transport_audit_path),
        },
        "source": {
            "repository": REPOSITORY,
            "repository_revision": REVISION,
            "split": "train",
        },
        "images": [image_record],
        "manifests": {
            role: {
                "path": outputs["role_manifests"][role],
                "sha256": sha256_file(manifest_path),
                "rows": 1,
                "class_counts": {"real_0": 1, "ai_generated_1": 0},
            }
        },
        "provenance": {"path": outputs["provenance"]},
    }
    _atomic_json_write_fsync(staged_provenance, provenance)
    return {
        "protocol_path": protocol_path,
        "lock_path": lock_path,
        "acquisition_path": acquisition_path,
        "acquisition_lock_path": acquisition_lock_path,
        "image_path": image_path,
        "manifest_root": manifest_root,
        "provenance_path": provenance_path,
        "staging_root": staging_root,
        "outputs": outputs,
    }


def _recover_fixture(fixture: dict, tmp_path: Path) -> dict:
    return _recover_or_reuse_publication(
        manifest_root=fixture["manifest_root"],
        provenance_path=fixture["provenance_path"],
        staging_root=fixture["staging_root"],
        outputs=fixture["outputs"],
        protocol_path=fixture["protocol_path"],
        lock_report_path=fixture["lock_path"],
        acquisition_protocol_path=fixture["acquisition_path"],
        acquisition_lock_path=fixture["acquisition_lock_path"],
        project_root=tmp_path,
    )


def test_publication_recovers_manifest_then_provenance_crash_window(
    tmp_path: Path,
) -> None:
    fixture = _publication_fixture(tmp_path)

    recovered = _recover_fixture(fixture, tmp_path)

    assert recovered["status"] == "PASS"
    assert fixture["provenance_path"].is_file()
    assert not fixture["staging_root"].exists()
    assert _recover_fixture(fixture, tmp_path)["status"] == "PASS"


def test_completed_publication_reuse_revalidates_raw_images(tmp_path: Path) -> None:
    fixture = _publication_fixture(tmp_path)
    _recover_fixture(fixture, tmp_path)
    fixture["image_path"].unlink()

    with pytest.raises(ValueError, match="raw image is missing"):
        _recover_fixture(fixture, tmp_path)


def test_publication_rejects_extra_manifest_directory_entry(tmp_path: Path) -> None:
    fixture = _publication_fixture(tmp_path)
    (fixture["manifest_root"] / "unexpected").mkdir()

    with pytest.raises(ValueError, match="manifest files are incomplete"):
        _recover_fixture(fixture, tmp_path)
