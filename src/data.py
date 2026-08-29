"""Shared image preprocessing for the CIFAKE smoke-test pipeline."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


DEFAULT_IMAGE_SIZE = 32
DEFAULT_MEAN = (0.5, 0.5, 0.5)
DEFAULT_STD = (0.5, 0.5, 0.5)
REQUIRED_MANIFEST_COLUMNS = {"image_path", "label", "class_name", "source", "split"}


def build_image_transform(
    image_size: int = DEFAULT_IMAGE_SIZE,
    mean: Sequence[float] = DEFAULT_MEAN,
    std: Sequence[float] = DEFAULT_STD,
) -> transforms.Compose:
    """Build deterministic RGB preprocessing shared by training and inference."""

    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if len(mean) != 3 or len(std) != 3:
        raise ValueError("mean and std must each contain three RGB values")
    if any(value <= 0 for value in std):
        raise ValueError("every standard-deviation value must be positive")

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=tuple(mean), std=tuple(std)),
        ]
    )


def preprocessing_config(
    image_size: int = DEFAULT_IMAGE_SIZE,
    mean: Sequence[float] = DEFAULT_MEAN,
    std: Sequence[float] = DEFAULT_STD,
) -> dict[str, object]:
    """Return JSON/checkpoint-safe preprocessing metadata."""

    return {
        "image_size": int(image_size),
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
        "colour_mode": "RGB",
    }


class ManifestImageDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    """Load labelled images from a project-relative CSV manifest."""

    def __init__(
        self,
        manifest_path: Path,
        transform: transforms.Compose | None = None,
        project_root: Path | None = None,
    ) -> None:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        self.project_root = project_root or Path(__file__).resolve().parents[1]
        self.transform = transform or build_image_transform()

        with manifest_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or [])
            missing = REQUIRED_MANIFEST_COLUMNS.difference(columns)
            if missing:
                raise ValueError(
                    f"Manifest {manifest_path} is missing columns: {sorted(missing)}"
                )
            self.records = list(reader)

        if not self.records:
            raise ValueError(f"Manifest contains no rows: {manifest_path}")

        invalid_labels = sorted(
            {row["label"] for row in self.records if row["label"] not in {"0", "1"}}
        )
        if invalid_labels:
            raise ValueError(f"Manifest contains invalid labels: {invalid_labels}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.records[index]
        image_path = self.project_root / row["image_path"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Manifest image not found: {image_path}")

        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return tensor, label, row["image_path"]
