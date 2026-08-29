import csv
from pathlib import Path

import pytest
import torch
from PIL import Image

from src.data import ManifestImageDataset
from src.metrics import binary_classification_metrics


def test_manifest_dataset_loads_relative_rgb_images(tmp_path: Path) -> None:
    image_path = tmp_path / "data" / "raw" / "example.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("L", (16, 24), 128).save(image_path)
    manifest_path = tmp_path / "data" / "processed" / "train.csv"
    manifest_path.parent.mkdir(parents=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["image_path", "label", "class_name", "source", "split"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "image_path": "data/raw/example.jpg",
                "label": 0,
                "class_name": "real",
                "source": "test",
                "split": "train",
            }
        )

    dataset = ManifestImageDataset(manifest_path, project_root=tmp_path)
    tensor, label, relative_path = dataset[0]
    assert tensor.shape == (3, 32, 32)
    assert tensor.dtype == torch.float32
    assert label.item() == 0.0
    assert relative_path == "data/raw/example.jpg"


def test_binary_metrics_are_correct_for_perfect_predictions() -> None:
    metrics = binary_classification_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]
