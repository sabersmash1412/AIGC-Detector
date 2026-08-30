"""Paired transformed-image loading shared by caching and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.image_transforms import apply_evaluation_transform


class TransformedPathDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    """Load reference paths and deterministically transform each image."""

    def __init__(
        self,
        image_paths: np.ndarray,
        labels: np.ndarray,
        condition: str,
        clip_preprocess: Any,
        seed: int,
        project_root: Path,
    ) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.condition = condition
        self.clip_preprocess = clip_preprocess
        self.seed = seed
        self.project_root = project_root

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        relative_path = str(self.image_paths[index])
        absolute_path = self.project_root / relative_path
        if not absolute_path.is_file():
            raise FileNotFoundError(f"Reference image not found: {absolute_path}")
        with Image.open(absolute_path) as image:
            transformed = apply_evaluation_transform(
                image,
                self.condition,
                image_path=relative_path,
                seed=self.seed,
            )
            tensor = self.clip_preprocess(transformed)
        label = torch.tensor(int(self.labels[index]), dtype=torch.int64)
        return tensor, label, relative_path
