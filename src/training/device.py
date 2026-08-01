"""Device selection helpers (MPS-first on Apple Silicon)."""

from __future__ import annotations

import os

import torch


def get_device() -> torch.device:
    """Return MPS if available, else CPU. Never assumes CUDA."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
