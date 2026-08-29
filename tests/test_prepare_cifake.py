import csv
import json
from pathlib import Path

from scripts.prepare_cifake import build_manifests


def _make_fake_dataset(root: Path, count: int = 8) -> None:
    for split in ("train", "test"):
        for class_name in ("REAL", "FAKE"):
            directory = root / split / class_name
            directory.mkdir(parents=True)
            for index in range(count):
                (directory / f"{index:03d}.jpg").write_bytes(b"test-placeholder")


def test_build_manifests_is_balanced_disjoint_and_reproducible(tmp_path: Path) -> None:
    raw_root = tmp_path / "data" / "raw" / "cifake"
    output_dir = tmp_path / "data" / "processed"
    _make_fake_dataset(raw_root)

    summary = build_manifests(
        raw_root=raw_root,
        output_dir=output_dir,
        project_root=tmp_path,
        train_per_class=3,
        val_per_class=2,
        test_per_class=2,
        seed=42,
    )

    assert summary["label_mapping"] == {"real": 0, "ai_generated": 1}
    assert summary["total_images"] == 14
    assert summary["duplicate_paths_across_splits"] == 0

    observed_paths: set[str] = set()
    first_run_contents: dict[str, str] = {}
    for split, expected_per_class in (("train", 3), ("val", 2), ("test", 2)):
        manifest_path = output_dir / f"{split}.csv"
        first_run_contents[split] = manifest_path.read_text(encoding="utf-8")
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        assert len(rows) == expected_per_class * 2
        assert {int(row["label"]) for row in rows} == {0, 1}
        assert {row["split"] for row in rows} == {split}
        assert sum(int(row["label"]) == 0 for row in rows) == expected_per_class
        assert sum(int(row["label"]) == 1 for row in rows) == expected_per_class

        paths = {row["image_path"] for row in rows}
        assert observed_paths.isdisjoint(paths)
        observed_paths.update(paths)

    build_manifests(
        raw_root=raw_root,
        output_dir=output_dir,
        project_root=tmp_path,
        train_per_class=3,
        val_per_class=2,
        test_per_class=2,
        seed=42,
    )

    for split in ("train", "val", "test"):
        assert (output_dir / f"{split}.csv").read_text(encoding="utf-8") == first_run_contents[
            split
        ]

    saved_summary = json.loads((output_dir / "dataset_summary.json").read_text())
    assert saved_summary == summary
