import json
from pathlib import Path

import pytest
from PIL import Image

from src.model import CIFAKESmokeCNN, save_checkpoint
from src.predict import discover_images, predict_directory


def _save_image(path: Path, colour: tuple[int, int, int]) -> None:
    Image.new("RGB", (32, 32), colour).save(path)


def test_predict_directory_writes_required_json_and_skips_corrupt_images(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    _save_image(input_dir / "b.png", (255, 255, 255))
    _save_image(input_dir / "a.jpg", (0, 0, 0))
    (input_dir / "corrupt.jpg").write_text("not an image", encoding="utf-8")
    (input_dir / "ignored.txt").write_text("not supported", encoding="utf-8")

    checkpoint_path = tmp_path / "model.pt"
    output_path = tmp_path / "predictions.json"
    save_checkpoint(checkpoint_path, CIFAKESmokeCNN())

    predictions, skipped, device = predict_directory(
        input_dir,
        checkpoint_path,
        output_path,
        batch_size=2,
        device_name="cpu",
    )

    assert device.type == "cpu"
    assert [Path(row["image_path"]).name for row in predictions] == ["a.jpg", "b.png"]
    assert all(set(row) == {"image_path", "pred"} for row in predictions)
    assert all(0.0 <= row["pred"] <= 1.0 for row in predictions)
    assert len(skipped) == 1
    assert "corrupt.jpg" in skipped[0]
    assert json.loads(output_path.read_text(encoding="utf-8")) == predictions


def test_discover_images_is_non_recursive_by_default(tmp_path: Path) -> None:
    _save_image(tmp_path / "top.png", (0, 0, 0))
    nested = tmp_path / "nested"
    nested.mkdir()
    _save_image(nested / "nested.png", (0, 0, 0))

    assert discover_images(tmp_path) == [tmp_path / "top.png"]
    assert discover_images(tmp_path, recursive=True) == [
        nested / "nested.png",
        tmp_path / "top.png",
    ]


def test_predict_directory_rejects_empty_directory(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model.pt"
    save_checkpoint(checkpoint_path, CIFAKESmokeCNN())

    with pytest.raises(ValueError, match="No supported image files"):
        predict_directory(
            tmp_path,
            checkpoint_path,
            tmp_path / "predictions.json",
            device_name="cpu",
        )
