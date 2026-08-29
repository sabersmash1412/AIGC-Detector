"""Small CNN and checkpoint contract for the CIFAKE smoke test."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.data import preprocessing_config


ARCHITECTURE_NAME = "cifake_smoke_cnn_v1"
LABEL_MAPPING = {"real": 0, "ai_generated": 1}


class ConvBlock(nn.Sequential):
    """Convolution, normalisation, activation, and spatial downsampling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )


class CIFAKESmokeCNN(nn.Module):
    """Compact binary classifier used only to validate the pipeline."""

    def __init__(self, base_channels: int = 32, dropout: float = 0.2) -> None:
        super().__init__()
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.base_channels = base_channels
        self.dropout_probability = dropout
        self.features = nn.Sequential(
            ConvBlock(3, base_channels),
            ConvBlock(base_channels, base_channels * 2),
            ConvBlock(base_channels * 2, base_channels * 4),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(base_channels * 4, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return one unnormalised AIGC logit for every input image."""

        features = self.features(images)
        return self.classifier(self.pool(features)).squeeze(1)

    def config(self) -> dict[str, Any]:
        return {
            "architecture": ARCHITECTURE_NAME,
            "base_channels": self.base_channels,
            "dropout": self.dropout_probability,
        }


def create_model(config: dict[str, Any] | None = None) -> CIFAKESmokeCNN:
    """Create a model from checkpoint-safe configuration metadata."""

    config = config or {"architecture": ARCHITECTURE_NAME}
    architecture = config.get("architecture")
    if architecture != ARCHITECTURE_NAME:
        raise ValueError(
            f"Unsupported architecture {architecture!r}; expected {ARCHITECTURE_NAME!r}"
        )
    return CIFAKESmokeCNN(
        base_channels=int(config.get("base_channels", 32)),
        dropout=float(config.get("dropout", 0.2)),
    )


def checkpoint_payload(
    model: CIFAKESmokeCNN,
    *,
    epoch: int | None = None,
    metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build the versioned checkpoint format used by training and inference."""

    return {
        "checkpoint_version": 1,
        "model_config": model.config(),
        "model_state_dict": model.state_dict(),
        "preprocessing": preprocessing_config(),
        "label_mapping": LABEL_MAPPING,
        "epoch": epoch,
        "metrics": metrics or {},
    }


def save_checkpoint(
    path: Path,
    model: CIFAKESmokeCNN,
    *,
    epoch: int | None = None,
    metrics: dict[str, float] | None = None,
) -> None:
    """Save a self-describing model checkpoint."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model, epoch=epoch, metrics=metrics), path)


def load_checkpoint(path: Path) -> tuple[CIFAKESmokeCNN, dict[str, Any]]:
    """Load and validate a checkpoint without executing arbitrary pickled code."""

    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint must contain a dictionary payload")

    required = {
        "checkpoint_version",
        "model_config",
        "model_state_dict",
        "preprocessing",
        "label_mapping",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Checkpoint is missing required fields: {sorted(missing)}")
    if payload["checkpoint_version"] != 1:
        raise ValueError(f"Unsupported checkpoint version: {payload['checkpoint_version']}")
    if payload["label_mapping"] != LABEL_MAPPING:
        raise ValueError(
            "Checkpoint label mapping does not match project convention "
            f"{LABEL_MAPPING}: {payload['label_mapping']}"
        )

    model = create_model(payload["model_config"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, payload
