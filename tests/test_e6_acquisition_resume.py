from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.prepare_e6_biggan import (
    NEAR_DUPLICATE_MAX_HAMMING,
    FingerprintIndex,
    FingerprintOwner,
    ImageVerification,
    SourceRow,
    _cache_receipt_path,
    _write_cache_receipt,
    classify_resume_conflicts,
    inspect_image,
    load_cached_pair,
)
from src.e6_acquisition_resume_protocol import (
    append_rejection_event,
    build_initial_rejection_journal,
    find_replay_event,
    load_resume_protocol,
    validate_rejection_journal,
    validate_resume_lock_receipt,
    validate_resume_protocol,
    validate_resume_upstream,
)
from src.e6_protocol import pair_rows


ROOT = Path(__file__).resolve().parents[1]
RESUME_PROTOCOL_PATH = ROOT / "configs/e6_acquisition_resume_protocol.json"
RESUME_LOCK_PATH = ROOT / "reports/e6_acquisition_resume_lock.json"
REJECTION_JOURNAL_PATH = (
    ROOT / "data/raw/e6_tiny_genimage_biggan/rejection_journal.json"
)


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
        perceptual_hash=f"{token if phash is None else phash:016x}",
        original_width=16,
        original_height=12,
        aspect_ratio=16 / 12,
        decoded_pixel_count=192,
        display_width=16,
        display_height=12,
        file_format="PNG",
    )


def _asset_url(row_idx: int) -> str:
    return (
        "https://datasets-server.huggingface.co/cached-assets/"
        "TheKernel01/Tiny-GenImage/--/"
        "89c4fe9efd0ebc7ce5c7641ef57d578ccd639c69/--/default/train/"
        f"{row_idx}/image/source.png?Expires=123&Signature=must-not-persist"
    )


def _png_bytes(
    size: tuple[int, int], color: tuple[int, int, int]
) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def _cache_pair_fixture(
    tmp_path: Path,
    protocol: dict,
    cycle: int,
) -> tuple[
    Path,
    tuple[SourceRow, SourceRow],
    tuple[ImageVerification, ImageVerification],
    tuple[Path, Path],
]:
    """Create a complete cache pair without opening a session or remote page."""

    real_idx, ai_idx = pair_rows(protocol, cycle)
    rows = (
        SourceRow(real_idx, 0, 0, "Real", _asset_url(real_idx), 16, 12),
        SourceRow(ai_idx, 1, 2, "BigGAN", _asset_url(ai_idx), 18, 14),
    )
    raw_root = tmp_path / "data/raw/e6_tiny_genimage_biggan"
    images = raw_root / "images"
    images.mkdir(parents=True)
    paths = (
        images / f"row-{real_idx:05d}.png",
        images / f"row-{ai_idx:05d}.png",
    )
    paths[0].write_bytes(_png_bytes((16, 12), (10, 20, 30)))
    paths[1].write_bytes(_png_bytes((18, 14), (40, 50, 60)))
    verifications = (inspect_image(paths[0]), inspect_image(paths[1]))
    for row, path, verification in zip(rows, paths, verifications, strict=True):
        _write_cache_receipt(_cache_receipt_path(path), row, verification)
    return raw_root, rows, verifications, paths


def _near_reason(
    *,
    row_idx: int = 1851,
    candidate_label: int = 1,
    owner_label: int = 1,
) -> dict:
    return {
        "kind": "near_duplicate",
        "scope": "existing_data",
        "row_idx": row_idx,
        "candidate_label": candidate_label,
        "candidate_byte_sha256": "1" * 64,
        "candidate_decoded_pixel_sha256": "2" * 64,
        "candidate_perceptual_hash": "000000000000000f",
        "owner": "historical-owner",
        "owner_role": "consumed_test",
        "owner_label": owner_label,
        "owner_perceptual_hash": "0000000000000000",
        "hamming_distance": 4,
    }


def _journal_event_fields(seed_event: dict) -> dict:
    generated = {"event_index", "previous_event_sha256", "event_sha256"}
    return {
        key: copy.deepcopy(value)
        for key, value in seed_event.items()
        if key not in generated
    }


def _second_rejection_fields(seed_event: dict) -> dict:
    fields = _journal_event_fields(seed_event)
    fields.update(
        {
            "slot": 340,
            "assigned_cycle": 132,
            "attempted_cycle": 247,
            "attempted_cycles": [132, 247],
            "real_row_index": 3460,
            "ai_row_index": 3461,
            "next_reserve_cycle": 1228,
        }
    )
    fields["reasons"][0]["row_idx"] = 3461
    return fields


def _rehash_last_event(journal: dict) -> None:
    event = journal["events"][-1]
    body = {key: value for key, value in event.items() if key != "event_sha256"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    event["event_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_checked_in_resume_protocol_and_upstream_evidence_validate() -> None:
    protocol = load_resume_protocol(RESUME_PROTOCOL_PATH)

    validate_resume_protocol(protocol)
    validate_resume_upstream(protocol, ROOT)

    assert protocol["policy_delta"]["historical_same_label_phash_only_action"] == (
        "reject_whole_pair_and_replace"
    )
    assert protocol["policy_delta"]["historical_byte_duplicate_action"] == (
        "hard_abort_without_replacement"
    )
    assert protocol["chronology"][
        "consumed_test_identity_metadata_used_for_integrity_filter"
    ] is True
    assert protocol["chronology"][
        "consumed_test_metrics_or_visual_content_used_for_replacement"
    ] is False


def test_checked_in_resume_lock_and_seeded_journal_validate_together() -> None:
    protocol = load_resume_protocol(RESUME_PROTOCOL_PATH)

    receipt, inventory = validate_resume_lock_receipt(
        protocol, RESUME_LOCK_PATH, ROOT
    )
    journal = json.loads(REJECTION_JOURNAL_PATH.read_text(encoding="utf-8"))
    validate_rejection_journal(
        journal, protocol, ROOT, protocol_path=RESUME_PROTOCOL_PATH
    )

    assert receipt["status"] == "PASS"
    assert inventory == receipt["frozen_partial_cache"]
    assert receipt["initial_rejection_journal"]["event_count"] == len(
        journal["events"]
    )
    assert receipt["initial_rejection_journal"]["head_event_sha256"] == (
        journal["events"][-1]["event_sha256"]
    )
    assert receipt["initial_rejection_journal"]["sha256"] == hashlib.sha256(
        REJECTION_JOURNAL_PATH.read_bytes()
    ).hexdigest()


def test_resume_classifier_only_replaces_same_label_historical_phash_matches() -> None:
    reasons = [_near_reason()]

    assert classify_resume_conflicts(reasons, {1850: 0, 1851: 1}) == (
        "reject_whole_pair_and_replace"
    )


@pytest.mark.parametrize("kind", ["byte_duplicate", "decoded_pixel_duplicate"])
def test_resume_classifier_never_replaces_exact_historical_content(kind: str) -> None:
    reason = _near_reason()
    reason["kind"] = kind
    reason.pop("owner_perceptual_hash")
    reason.pop("hamming_distance")

    assert classify_resume_conflicts([reason], {1850: 0, 1851: 1}) == (
        "hard_abort_without_replacement"
    )


def test_resume_classifier_never_replaces_a_conflicting_label_match() -> None:
    reason = _near_reason(owner_label=0)

    assert classify_resume_conflicts([reason], {1850: 0, 1851: 1}) == (
        "hard_abort_without_replacement"
    )


@pytest.mark.parametrize("opposite_first", [False, True])
def test_any_opposite_label_near_owner_is_fatal_regardless_of_order(
    opposite_first: bool,
) -> None:
    same = _near_reason()
    same["owner"] = "same-label-owner"
    opposite = _near_reason(owner_label=0)
    opposite["owner"] = "opposite-label-owner"
    reasons = [opposite, same] if opposite_first else [same, opposite]

    assert classify_resume_conflicts(reasons, {1850: 0, 1851: 1}) == (
        "hard_abort_without_replacement"
    )


@pytest.mark.parametrize(
    "kind", ["byte_duplicate", "decoded_pixel_duplicate", "near_duplicate"]
)
def test_same_label_accepted_e6_conflict_keeps_v1_replacement_policy(kind: str) -> None:
    reason = _near_reason()
    reason["kind"] = kind
    reason["scope"] = "accepted_e6"
    if kind != "near_duplicate":
        reason.pop("owner_perceptual_hash")
        reason.pop("hamming_distance")

    assert classify_resume_conflicts([reason], {1850: 0, 1851: 1}) == (
        "reject_whole_pair_and_replace"
    )


def test_resume_classifier_uses_fatal_precedence_for_mixed_conflicts() -> None:
    exact = _near_reason()
    exact["kind"] = "byte_duplicate"
    exact.pop("owner_perceptual_hash")
    exact.pop("hamming_distance")

    assert classify_resume_conflicts(
        [_near_reason(), exact], {1850: 0, 1851: 1}
    ) == "hard_abort_without_replacement"
    assert classify_resume_conflicts([], {1850: 0, 1851: 1}) == "accept"


def test_fingerprint_conflicts_enumerate_every_near_owner_with_distance() -> None:
    index = FingerprintIndex()
    index.add(
        _verification(10, phash=0),
        FingerprintOwner("owner-a", "consumed_test", 1),
    )
    index.add(
        _verification(11, phash=3),
        FingerprintOwner("owner-b", "consumed_test", 1),
    )
    candidate = _verification(12, phash=1)

    reasons = index.conflicts(
        candidate,
        role="development_train",
        label=1,
        all_near_duplicates=True,
    )
    near = [reason for reason in reasons if reason["kind"] == "near_duplicate"]

    assert len(near) == 2
    assert {reason["owner"] for reason in near} == {"owner-a", "owner-b"}
    assert {reason["owner_perceptual_hash"] for reason in near} == {
        "0000000000000000",
        "0000000000000003",
    }
    assert {reason["hamming_distance"] for reason in near} == {1}
    assert all(
        reason["hamming_distance"] <= NEAR_DUPLICATE_MAX_HAMMING
        for reason in near
    )


def test_initial_journal_validates_and_replays_the_recorded_v1_abort() -> None:
    protocol = load_resume_protocol(RESUME_PROTOCOL_PATH)
    journal = build_initial_rejection_journal(
        protocol, ROOT, protocol_path=RESUME_PROTOCOL_PATH
    )

    validate_rejection_journal(
        journal, protocol, ROOT, protocol_path=RESUME_PROTOCOL_PATH
    )
    event = find_replay_event(journal, slot=340, attempted_cycle=132)

    assert event is not None
    assert event["action"] == "reject_whole_pair_and_replace"
    assert event["real_row_index"] == 1850
    assert event["ai_row_index"] == 1851
    assert event["next_reserve_cycle"] == 247
    assert event["reasons"][0]["kind"] == "near_duplicate"
    assert find_replay_event(journal, slot=339, attempted_cycle=132) is None


def test_journal_append_is_atomic_idempotent_and_hash_chained(tmp_path: Path) -> None:
    protocol = load_resume_protocol(RESUME_PROTOCOL_PATH)
    journal = build_initial_rejection_journal(
        protocol, ROOT, protocol_path=RESUME_PROTOCOL_PATH
    )
    path = tmp_path / "rejection_journal.json"
    path.write_text(json.dumps(journal), encoding="utf-8")
    original_bytes = path.read_bytes()

    first_fields = _journal_event_fields(journal["events"][0])
    unchanged = append_rejection_event(path, journal, first_fields)

    assert unchanged == journal
    assert path.read_bytes() == original_bytes

    second_fields = _second_rejection_fields(journal["events"][0])
    updated = append_rejection_event(path, unchanged, second_fields)

    assert len(updated["events"]) == 2
    assert updated["events"][1]["previous_event_sha256"] == (
        updated["events"][0]["event_sha256"]
    )
    assert json.loads(path.read_text(encoding="utf-8")) == updated
    validate_rejection_journal(
        updated, protocol, ROOT, protocol_path=RESUME_PROTOCOL_PATH
    )

    repeated = append_rejection_event(path, updated, second_fields)
    assert repeated == updated
    assert len(repeated["events"]) == 2


@pytest.mark.parametrize(
    ("field", "invalid"),
    [("kind", "unknown_duplicate"), ("scope", "unknown_scope")],
)
def test_journal_rejects_unknown_overlap_kind_or_scope(
    tmp_path: Path, field: str, invalid: str
) -> None:
    protocol = load_resume_protocol(RESUME_PROTOCOL_PATH)
    journal = build_initial_rejection_journal(
        protocol, ROOT, protocol_path=RESUME_PROTOCOL_PATH
    )
    path = tmp_path / "rejection_journal.json"
    path.write_text(json.dumps(journal), encoding="utf-8")
    updated = append_rejection_event(
        path, journal, _second_rejection_fields(journal["events"][0])
    )
    updated["events"][-1]["reasons"][0][field] = invalid
    _rehash_last_event(updated)

    with pytest.raises(ValueError, match=f"{field}|reason|journal"):
        validate_rejection_journal(
            updated, protocol, ROOT, protocol_path=RESUME_PROTOCOL_PATH
        )


@pytest.mark.parametrize(
    ("field", "invalid"),
    [("row_idx", 9999), ("candidate_label", 0)],
)
def test_journal_rejects_reason_not_matching_event_rows_or_labels(
    tmp_path: Path, field: str, invalid: int
) -> None:
    protocol = load_resume_protocol(RESUME_PROTOCOL_PATH)
    journal = build_initial_rejection_journal(
        protocol, ROOT, protocol_path=RESUME_PROTOCOL_PATH
    )
    path = tmp_path / "rejection_journal.json"
    path.write_text(json.dumps(journal), encoding="utf-8")
    updated = append_rejection_event(
        path, journal, _second_rejection_fields(journal["events"][0])
    )
    updated["events"][-1]["reasons"][0][field] = invalid
    _rehash_last_event(updated)

    with pytest.raises(ValueError, match="row|label|reason|journal"):
        validate_rejection_journal(
            updated, protocol, ROOT, protocol_path=RESUME_PROTOCOL_PATH
        )


def test_append_validation_failure_happens_before_journal_write(
    tmp_path: Path,
) -> None:
    protocol = load_resume_protocol(RESUME_PROTOCOL_PATH)
    journal = build_initial_rejection_journal(
        protocol, ROOT, protocol_path=RESUME_PROTOCOL_PATH
    )
    path = tmp_path / "rejection_journal.json"
    path.write_text(json.dumps(journal), encoding="utf-8")
    original_bytes = path.read_bytes()
    invalid_fields = _second_rejection_fields(journal["events"][0])
    invalid_fields["reasons"][0]["scope"] = "unknown_scope"
    validation_calls: list[dict] = []

    def validate_before_write(candidate: dict) -> None:
        validation_calls.append(candidate)
        validate_rejection_journal(
            candidate, protocol, ROOT, protocol_path=RESUME_PROTOCOL_PATH
        )

    with pytest.raises(ValueError, match="scope|reason|journal"):
        append_rejection_event(
            path,
            journal,
            invalid_fields,
            validate_before_write=validate_before_write,
        )

    assert len(validation_calls) == 1
    assert path.read_bytes() == original_bytes
    assert json.loads(path.read_text(encoding="utf-8")) == journal


def test_journal_validation_rejects_event_tampering() -> None:
    protocol = load_resume_protocol(RESUME_PROTOCOL_PATH)
    journal = build_initial_rejection_journal(
        protocol, ROOT, protocol_path=RESUME_PROTOCOL_PATH
    )
    tampered = copy.deepcopy(journal)
    tampered["events"][0]["next_reserve_cycle"] += 1

    with pytest.raises(ValueError, match="hash|event|journal"):
        validate_rejection_journal(
            tampered, protocol, ROOT, protocol_path=RESUME_PROTOCOL_PATH
        )


def test_complete_pair_is_reconstructed_from_cache_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = load_resume_protocol(RESUME_PROTOCOL_PATH)
    development = json.loads(
        (ROOT / "configs/e6_development_protocol.json").read_text(encoding="utf-8")
    )
    cycle = 0
    raw_root, expected_rows, expected_verifications, paths = _cache_pair_fixture(
        tmp_path, development, cycle
    )

    def forbidden_network(*args, **kwargs):
        raise AssertionError("cache reconstruction attempted a network call")

    monkeypatch.setattr("requests.sessions.Session.get", forbidden_network)
    monkeypatch.setattr(
        "scripts.prepare_e6_biggan.RowsClient.pair", forbidden_network
    )
    cached = load_cached_pair(
        raw_root, development, cycle, project_root=tmp_path
    )

    assert cached is not None
    assert tuple(item.verification for item in cached) == expected_verifications
    for item, expected, path in zip(cached, expected_rows, paths, strict=True):
        assert (
            item.row.row_idx,
            item.row.label,
            item.row.generator_id,
            item.row.generator_name,
            item.row.source_width,
            item.row.source_height,
        ) == (
            expected.row_idx,
            expected.label,
            expected.generator_id,
            expected.generator_name,
            expected.source_width,
            expected.source_height,
        )
        assert "?" not in item.row.image_url
        assert item.image_path == path.relative_to(tmp_path).as_posix()

    # The protocol is loaded above so this test also proves the cache path does
    # not depend on a signed query from a datasets-server response.
    assert protocol["rejection_journal"]["signed_url_or_query_allowed"] is False


def test_finalized_cache_pair_does_not_decode_images_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A receipt-bound asset needs only its cheap file hash on later runs."""

    development = json.loads(
        (ROOT / "configs/e6_development_protocol.json").read_text(encoding="utf-8")
    )
    raw_root, _, expected_verifications, _ = _cache_pair_fixture(
        tmp_path, development, 0
    )

    def forbidden_decode(*args, **kwargs):
        raise AssertionError("finalized cache unexpectedly decoded an image")

    monkeypatch.setattr(
        "scripts.prepare_e6_biggan.inspect_image", forbidden_decode
    )

    cached = load_cached_pair(raw_root, development, 0, project_root=tmp_path)

    assert cached is not None
    assert tuple(item.verification for item in cached) == expected_verifications


def test_finalized_cache_pair_rejects_changed_asset_bytes_without_decoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    development = json.loads(
        (ROOT / "configs/e6_development_protocol.json").read_text(encoding="utf-8")
    )
    raw_root, _, _, paths = _cache_pair_fixture(tmp_path, development, 0)
    paths[0].write_bytes(paths[0].read_bytes() + b"tampered")

    def forbidden_decode(*args, **kwargs):
        raise AssertionError("tampered finalized cache should fail before decoding")

    monkeypatch.setattr(
        "scripts.prepare_e6_biggan.inspect_image", forbidden_decode
    )

    with pytest.raises(ValueError):
        load_cached_pair(raw_root, development, 0, project_root=tmp_path)


@pytest.mark.parametrize(
    ("field", "tampered_value"),
    [
        pytest.param("file_bytes", 0, id="unsafe-file-size"),
        pytest.param("byte_sha256", "0" * 64, id="file-hash"),
        pytest.param("decoded_pixel_sha256", "not-a-sha256", id="decoded-hash"),
        pytest.param("perceptual_hash", "not-a-phash", id="perceptual-hash"),
        pytest.param("original_width", 0, id="original-width"),
        pytest.param("original_height", 0, id="original-height"),
        pytest.param("display_width", 0, id="display-width"),
        pytest.param("display_height", 0, id="display-height"),
        pytest.param("aspect_ratio", 999.0, id="aspect-ratio"),
        pytest.param("decoded_pixel_count", 0, id="decoded-pixel-count"),
        pytest.param("file_format", "BMP", id="decoded-format"),
    ],
)
def test_finalized_cache_pair_rejects_tampered_verification_record_without_decoding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    tampered_value: object,
) -> None:
    development = json.loads(
        (ROOT / "configs/e6_development_protocol.json").read_text(encoding="utf-8")
    )
    raw_root, _, _, paths = _cache_pair_fixture(tmp_path, development, 0)
    receipt_path = _cache_receipt_path(paths[0])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["verification"][field] = tampered_value
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    def forbidden_decode(*args, **kwargs):
        raise AssertionError("invalid finalized receipt should fail without decoding")

    monkeypatch.setattr(
        "scripts.prepare_e6_biggan.inspect_image", forbidden_decode
    )

    with pytest.raises(ValueError):
        load_cached_pair(raw_root, development, 0, project_root=tmp_path)


@pytest.mark.parametrize(
    ("section", "field", "boolean_value"),
    [
        pytest.param("receipt", "schema_version", True, id="schema"),
        pytest.param("receipt", "row_idx", False, id="row"),
        pytest.param("receipt", "label", False, id="label"),
        pytest.param("receipt", "generator_id", False, id="generator"),
        pytest.param("receipt", "source_width", True, id="source-width"),
        pytest.param("receipt", "source_height", True, id="source-height"),
        pytest.param("verification", "file_bytes", True, id="file-size"),
        pytest.param("verification", "original_width", True, id="original-width"),
        pytest.param("verification", "original_height", True, id="original-height"),
        pytest.param(
            "verification", "decoded_pixel_count", True, id="decoded-pixel-count"
        ),
        pytest.param("verification", "display_width", True, id="display-width"),
        pytest.param("verification", "display_height", True, id="display-height"),
    ],
)
def test_finalized_cache_pair_rejects_booleans_for_json_integer_fields(
    tmp_path: Path,
    section: str,
    field: str,
    boolean_value: bool,
) -> None:
    development = json.loads(
        (ROOT / "configs/e6_development_protocol.json").read_text(encoding="utf-8")
    )
    raw_root, _, _, paths = _cache_pair_fixture(tmp_path, development, 0)
    receipt_path = _cache_receipt_path(paths[0])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    target = receipt if section == "receipt" else receipt["verification"]
    target[field] = boolean_value
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="JSON integer"):
        load_cached_pair(raw_root, development, 0, project_root=tmp_path)


@pytest.mark.parametrize(
    "aspect_ratio",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive-infinity"),
        pytest.param(float("-inf"), id="negative-infinity"),
    ],
)
def test_finalized_cache_pair_rejects_non_finite_aspect_ratio(
    tmp_path: Path, aspect_ratio: float
) -> None:
    development = json.loads(
        (ROOT / "configs/e6_development_protocol.json").read_text(encoding="utf-8")
    )
    raw_root, _, _, paths = _cache_pair_fixture(tmp_path, development, 0)
    receipt_path = _cache_receipt_path(paths[0])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["verification"]["aspect_ratio"] = aspect_ratio
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="dimensions|size"):
        load_cached_pair(raw_root, development, 0, project_root=tmp_path)


def test_finalized_cache_pair_rejects_same_product_but_impossible_display_size(
    tmp_path: Path,
) -> None:
    development = json.loads(
        (ROOT / "configs/e6_development_protocol.json").read_text(encoding="utf-8")
    )
    raw_root, _, _, paths = _cache_pair_fixture(tmp_path, development, 0)
    receipt_path = _cache_receipt_path(paths[0])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    # 24 * 8 == 16 * 12, but it is neither the original nor EXIF-swapped size.
    receipt["verification"]["display_width"] = 24
    receipt["verification"]["display_height"] = 8
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="dimensions|size"):
        load_cached_pair(raw_root, development, 0, project_root=tmp_path)


@pytest.mark.parametrize(
    ("field", "tampered_value"),
    [
        pytest.param("byte_sha256", "A" * 64, id="uppercase-byte-sha"),
        pytest.param("byte_sha256", "0" * 63, id="short-byte-sha"),
        pytest.param("decoded_pixel_sha256", "A" * 64, id="uppercase-pixel-sha"),
        pytest.param("decoded_pixel_sha256", "0" * 65, id="long-pixel-sha"),
        pytest.param("perceptual_hash", "A" * 16, id="uppercase-phash"),
        pytest.param("perceptual_hash", "0" * 15, id="short-phash"),
    ],
)
def test_finalized_cache_pair_rejects_noncanonical_hashes(
    tmp_path: Path, field: str, tampered_value: str
) -> None:
    development = json.loads(
        (ROOT / "configs/e6_development_protocol.json").read_text(encoding="utf-8")
    )
    raw_root, _, _, paths = _cache_pair_fixture(tmp_path, development, 0)
    receipt_path = _cache_receipt_path(paths[0])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["verification"][field] = tampered_value
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="identity"):
        load_cached_pair(raw_root, development, 0, project_root=tmp_path)


@pytest.mark.parametrize("drift", ["query", "revision", "row"])
def test_finalized_cache_pair_rejects_canonical_asset_path_drift(
    tmp_path: Path, drift: str
) -> None:
    development = json.loads(
        (ROOT / "configs/e6_development_protocol.json").read_text(encoding="utf-8")
    )
    raw_root, rows, _, paths = _cache_pair_fixture(tmp_path, development, 0)
    receipt_path = _cache_receipt_path(paths[0])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    canonical_path = receipt["canonical_asset_path"]
    if drift == "query":
        canonical_path += "?Signature=must-not-be-accepted"
    elif drift == "revision":
        canonical_path = canonical_path.replace(
            development["development_source"]["repository_revision"], "0" * 40
        )
    else:
        canonical_path = canonical_path.replace(
            f"/train/{rows[0].row_idx}/", f"/train/{rows[0].row_idx + 2}/"
        )
    receipt["canonical_asset_path"] = canonical_path
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="path|row|asset"):
        load_cached_pair(raw_root, development, 0, project_root=tmp_path)


def test_finalized_cache_pair_rejects_format_suffix_mismatch(tmp_path: Path) -> None:
    development = json.loads(
        (ROOT / "configs/e6_development_protocol.json").read_text(encoding="utf-8")
    )
    raw_root, _, _, paths = _cache_pair_fixture(tmp_path, development, 0)
    receipt_path = _cache_receipt_path(paths[0])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert paths[0].suffix == ".png"
    receipt["verification"]["file_format"] = "JPEG"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="extension|magic"):
        load_cached_pair(raw_root, development, 0, project_root=tmp_path)


def test_cache_pair_miss_does_not_reuse_only_one_complete_row(tmp_path: Path) -> None:
    development = json.loads(
        (ROOT / "configs/e6_development_protocol.json").read_text(encoding="utf-8")
    )
    cycle = 0
    raw_root, _, _, paths = _cache_pair_fixture(tmp_path, development, cycle)
    paths[1].unlink()
    _cache_receipt_path(paths[1]).unlink()

    assert load_cached_pair(
        raw_root, development, cycle, project_root=tmp_path
    ) is None


def test_cache_pair_miss_when_both_rows_are_absent(tmp_path: Path) -> None:
    development = json.loads(
        (ROOT / "configs/e6_development_protocol.json").read_text(encoding="utf-8")
    )

    assert load_cached_pair(
        tmp_path / "data/raw/e6_tiny_genimage_biggan",
        development,
        0,
        project_root=tmp_path,
    ) is None


def test_pending_receipt_cache_state_defers_to_normal_recovery(tmp_path: Path) -> None:
    development = json.loads(
        (ROOT / "configs/e6_development_protocol.json").read_text(encoding="utf-8")
    )
    raw_root, _, _, paths = _cache_pair_fixture(tmp_path, development, 0)
    receipt = _cache_receipt_path(paths[0])
    receipt.replace(paths[0].with_name(f"{paths[0].name}.receipt.pending.json"))

    assert load_cached_pair(
        raw_root, development, 0, project_root=tmp_path
    ) is None


@pytest.mark.parametrize(
    "malformation", ["asset_only", "receipt_only", "duplicate_suffix"]
)
def test_malformed_cache_pair_fails_closed(
    tmp_path: Path, malformation: str
) -> None:
    development = json.loads(
        (ROOT / "configs/e6_development_protocol.json").read_text(encoding="utf-8")
    )
    raw_root, _, _, paths = _cache_pair_fixture(tmp_path, development, 0)
    if malformation == "asset_only":
        _cache_receipt_path(paths[0]).unlink()
    elif malformation == "receipt_only":
        paths[0].unlink()
    else:
        paths[0].with_suffix(".jpg").write_bytes(paths[0].read_bytes())

    with pytest.raises(ValueError, match="cache|asset|receipt|suffix|row"):
        load_cached_pair(raw_root, development, 0, project_root=tmp_path)


@pytest.mark.parametrize("tampering", ["label", "row_idx", "verification", "bytes"])
def test_tampered_cache_pair_fails_closed(tmp_path: Path, tampering: str) -> None:
    development = json.loads(
        (ROOT / "configs/e6_development_protocol.json").read_text(encoding="utf-8")
    )
    raw_root, _, _, paths = _cache_pair_fixture(tmp_path, development, 0)
    receipt_path = _cache_receipt_path(paths[0])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if tampering == "label":
        receipt["label"] = 1
    elif tampering == "row_idx":
        receipt["row_idx"] += 2
    elif tampering == "verification":
        receipt["verification"]["byte_sha256"] = "0" * 64
    else:
        paths[0].write_bytes(_png_bytes((16, 12), (70, 80, 90)))
    if tampering != "bytes":
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="cache|bytes|identity|receipt|row"):
        load_cached_pair(raw_root, development, 0, project_root=tmp_path)


def test_cache_receipts_never_persist_signed_query(tmp_path: Path) -> None:
    development = json.loads(
        (ROOT / "configs/e6_development_protocol.json").read_text(encoding="utf-8")
    )
    raw_root, _, _, paths = _cache_pair_fixture(tmp_path, development, 0)

    for path in paths:
        text = _cache_receipt_path(path).read_text(encoding="utf-8")
        payload = json.loads(text)
        assert "Signature" not in text
        assert "Expires" not in text
        assert "?" not in payload["canonical_asset_path"]
        assert payload["signed_query_persisted"] is False

    cached = load_cached_pair(raw_root, development, 0, project_root=tmp_path)
    assert cached is not None
    assert all("?" not in item.row.image_url for item in cached)


def test_seeded_replay_identifies_rows_without_cache_or_metadata_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resume = load_resume_protocol(RESUME_PROTOCOL_PATH)
    development = json.loads(
        (ROOT / "configs/e6_development_protocol.json").read_text(encoding="utf-8")
    )
    journal = build_initial_rejection_journal(
        resume, ROOT, protocol_path=RESUME_PROTOCOL_PATH
    )
    event = find_replay_event(journal, slot=340, attempted_cycle=132)

    def forbidden_cache(*args, **kwargs):
        raise AssertionError("journal replay should precede cache reconstruction")

    monkeypatch.setattr(
        "scripts.prepare_e6_biggan.load_cached_pair", forbidden_cache
    )
    assert event is not None
    assert pair_rows(development, event["attempted_cycle"]) == (1850, 1851)
    assert event["next_reserve_cycle"] == 247
