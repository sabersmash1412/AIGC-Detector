import copy
import json
from pathlib import Path

import pytest

from scripts.prepare_e4_sid_real import (
    allocate_manifest_rows,
    candidate_rank,
    deterministic_page_order,
    load_audit_exclusions,
    parse_real_page,
    validate_protocol,
)
from scripts.prepare_sid_set_heldout import Candidate


PROTOCOL_PATH = Path("configs/e4_domain_adaptation.json")


def _protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _candidate(index: int, image_id: str | None = None) -> Candidate:
    return Candidate(
        row_idx=index,
        img_id=image_id or f"real-{index}",
        source_label=0,
        project_label=0,
        class_name="real",
        image_url=f"https://example.test/{index}.jpg",
        source_width=100,
        source_height=80,
    )


def test_checked_in_e4_protocol_is_frozen_and_real_only() -> None:
    protocol = _protocol()

    validate_protocol(protocol)

    sid = protocol["development_data"]["sid_set_real"]
    assert sid["source_split"] == "train"
    assert sid["source_rows"] == 210000
    assert sid["allowed_source_label"] == 0
    assert set(sid["forbidden_source_labels"]) == {"1", "2"}
    assert protocol["frozen_section4_audit"]["allowed_for_training"] is False
    assert protocol["organiser_validation_subset"]["used"] is False


@pytest.mark.parametrize(
    ("path", "unsafe_value"),
    [
        (("development_data", "sid_set_real", "source_split"), "validation"),
        (("development_data", "sid_set_real", "allowed_source_label"), 1),
        (("frozen_section4_audit", "allowed_for_training"), True),
        (("frozen_section4_audit", "allowed_for_epoch_or_threshold_selection"), True),
        (("organiser_validation_subset", "used"), True),
        (("research_positioning", "E4_is_post_hoc_follow_up"), False),
        (("privacy_and_publication", "individual_sid_images_allowed_in_public_demo"), True),
    ],
)
def test_protocol_rejects_leakage_or_scope_drift(
    path: tuple[str, ...], unsafe_value: object
) -> None:
    protocol = copy.deepcopy(_protocol())
    target = protocol
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = unsafe_value

    with pytest.raises(ValueError):
        validate_protocol(protocol)


def test_page_order_is_deterministic_and_complete() -> None:
    first = deterministic_page_order(20, 404)
    second = deterministic_page_order(20, 404)

    assert first == second
    assert sorted(first) == list(range(20))
    assert first != deterministic_page_order(20, 405)


def test_candidate_rank_is_stable_and_uses_identity() -> None:
    candidate = _candidate(7)

    assert candidate_rank(candidate, 404) == candidate_rank(candidate, 404)
    assert candidate_rank(candidate, 404) != candidate_rank(_candidate(8), 404)
    assert candidate_rank(candidate, 404) != candidate_rank(candidate, 405)


def test_parse_real_page_retains_only_label_zero() -> None:
    revision = "a" * 40
    payload = {
        "partial": False,
        "num_rows_total": 210000,
        "rows": [
            {
                "row_idx": 10,
                "row": {
                    "img_id": "real-id",
                    "image": {"src": "https://example.test/real.jpg"},
                    "width": 100,
                    "height": 80,
                    "label": 0,
                },
            },
            {
                "row_idx": 11,
                "row": {
                    "img_id": "full_synthetic-id",
                    "image": {"src": "https://example.test/fake.jpg"},
                    "width": 100,
                    "height": 100,
                    "label": 1,
                },
            },
            {
                "row_idx": 12,
                "row": {
                    "img_id": "tampered-id",
                    "image": {"src": "https://example.test/tampered.jpg"},
                    "width": 100,
                    "height": 100,
                    "label": 2,
                },
            },
        ],
    }

    result = parse_real_page(
        payload,
        expected_rows=210000,
        expected_revision=revision,
        observed_revision=revision,
    )

    assert len(result) == 1
    assert result[0].row_idx == 10
    assert result[0].source_label == 0
    assert result[0].project_label == 0


def test_parse_real_page_rejects_revision_or_row_count_drift() -> None:
    payload = {"partial": False, "num_rows_total": 210000, "rows": []}

    with pytest.raises(ValueError, match="revision drift"):
        parse_real_page(
            payload,
            expected_rows=210000,
            expected_revision="a" * 40,
            observed_revision="b" * 40,
        )
    with pytest.raises(ValueError, match="row count changed"):
        parse_real_page(
            {**payload, "num_rows_total": 1},
            expected_rows=210000,
            expected_revision="a" * 40,
            observed_revision="a" * 40,
        )


def test_allocate_manifest_rows_creates_disjoint_real_only_splits(
    tmp_path: Path,
) -> None:
    destinations = [tmp_path / "data" / f"{index}.jpg" for index in range(4)]
    accepted = [
        (
            _candidate(index),
            destination,
            {
                "bytes": 10,
                "sha256": f"{index:064x}",
                "width": 100,
                "height": 80,
                "format": "JPEG",
            },
        )
        for index, destination in enumerate(destinations)
    ]

    manifests, provenance = allocate_manifest_rows(
        accepted,
        train_count=3,
        validation_count=1,
        project_root=tmp_path,
    )

    assert len(manifests["train"]) == 3
    assert len(manifests["val"]) == 1
    assert {row["label"] for rows in manifests.values() for row in rows} == {0}
    assert {row["split"] for row in manifests["train"]} == {"train"}
    assert {row["split"] for row in manifests["val"]} == {"val"}
    assert {row["e4_split"] for row in provenance} == {"train", "val"}
    assert len({row["image_path"] for row in provenance}) == 4


def test_frozen_audit_exclusions_match_checked_in_provenance() -> None:
    exclusions = load_audit_exclusions(_protocol(), Path.cwd())

    assert len(exclusions.image_ids) == 2000
    assert len(exclusions.content_sha256) == 2000
