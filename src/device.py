"""Compute-device selection shared by training, evaluation, and inference."""

from __future__ import annotations

import torch


def choose_device(requested: str = "auto") -> torch.device:
    """Select CUDA, Apple MPS, or CPU, with an explicit override."""

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return torch.device(requested)
