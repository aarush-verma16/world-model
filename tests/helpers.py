"""Tiny models/batches shared by ML unit tests.

Keep these well below training size so the suite stays CPU-fast. CUDA tests
reuse the same shapes; they only move the tensors onto the GPU.
"""

from __future__ import annotations

from typing import Any

import torch

from models.world_model import WorldModel


def tiny_world_model(*, action_dim: int = 5, **kwargs: Any) -> WorldModel:
    """A Dreamer graph small enough for CPU shape / gradient tests."""
    dims = dict(
        embed_dim=64,
        encoder_channels=(16, 32, 64, 64),
        action_dim=action_dim,
        deter_dim=32,
        stoch=4,
        classes=4,
        hidden=32,
        decoder_channels=(64, 32, 16, 8),
        head_hidden=32,
        head_layers=1,
        encoder_blocks=1,
        decoder_blocks=0,
    )
    dims.update(kwargs)
    return WorldModel.from_config_dims(**dims)


def tiny_batch(
    *,
    batch: int = 2,
    seq: int = 8,
    action_dim: int = 5,
    height: int = 64,
    width: int = 64,
) -> dict[str, torch.Tensor]:
    """Synthetic replay window: uint8 NHWC obs, int64 actions, float reward/cont."""
    return {
        "obs": torch.randint(0, 256, (batch, seq, height, width, 3), dtype=torch.uint8),
        "actions": torch.randint(0, action_dim, (batch, seq), dtype=torch.int64),
        "rewards": torch.zeros(batch, seq),
        "cont": torch.ones(batch, seq),
    }


def wm_train_cfg() -> dict[str, float | None]:
    """Minimal `train:` mapping for `world_model_step` (M5 defaults)."""
    return {
        "dyn_scale": 1.0,
        "rep_scale": 0.5,
        "free_nats": 1.0,
        "free_nats_dyn": 0.0,
        "recon_scale": 1.0,
        "reward_scale": 1.0,
        "continue_scale": 1.0,
        "kl_scale": 1.0,
    }
