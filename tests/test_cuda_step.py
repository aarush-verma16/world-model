"""Optional CUDA checks. Skipped on CPU-only machines; on this workstation
they confirm bf16 AMP + a tiny world-model step actually run on the GPU.
"""

from __future__ import annotations

import math

import pytest
import torch

from tests.helpers import tiny_batch, tiny_world_model, wm_train_cfg
from training.device import autocast_context, make_grad_scaler, parse_amp
from training.wm_step import world_model_step

pytestmark = pytest.mark.cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tiny_world_model_step_on_cuda_is_finite() -> None:
    device = torch.device("cuda")
    wm = tiny_world_model().to(device)
    before = next(wm.parameters()).detach().clone()
    optim = torch.optim.Adam(wm.parameters(), lr=1e-3)
    amp = parse_amp("bf16", device)
    scaler = make_grad_scaler(device, amp)
    loss, metrics = world_model_step(
        wm,
        optim,
        tiny_batch(batch=2, seq=4),
        device=device,
        train_cfg=wm_train_cfg(),
        amp_dtype=amp,
        scaler=scaler,
    )
    assert amp is torch.bfloat16
    assert torch.isfinite(loss.total)
    assert all(math.isfinite(metrics[k]) for k in ("recon", "kl", "reward", "continue"))
    after = next(wm.parameters()).detach()
    assert after.device.type == "cuda"
    assert not torch.equal(before, after)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bf16_autocast_forward_is_finite() -> None:
    device = torch.device("cuda")
    wm = tiny_world_model().to(device)
    batch = tiny_batch(batch=1, seq=4)
    obs = batch["obs"].to(device)
    actions = batch["actions"].to(device)
    with autocast_context(device, torch.bfloat16):
        out = wm(obs, actions)
    assert out.recon.device.type == "cuda"
    assert torch.isfinite(out.recon.float()).all()
    assert torch.isfinite(out.rssm.h.float()).all()
