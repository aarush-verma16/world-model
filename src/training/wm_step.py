"""One world-model optimizer step, shared by the CLI and the training notebook.

Keeps AMP / scaler / grad-clip in one place so the notebook cannot silently
drift back to a full-fp32 loop after a CUDA config change.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from models.preprocess import nhwc_uint8_to_nchw_unit
from models.world_model import WorldModel
from training.device import autocast_context, to_device
from training.losses import WorldModelLossBreakdown, world_model_loss


def loss_to_metrics(loss: WorldModelLossBreakdown) -> dict[str, float]:
    """Detach per-term losses to plain floats for logging."""
    return {
        "total": float(loss.total.detach()),
        "recon": float(loss.recon.detach()),
        "recon_l1": float(loss.recon_l1.detach()),
        "reward": float(loss.reward.detach()),
        "reward_mae": float(loss.reward_mae.detach()),
        "continue": float(loss.continue_loss.detach()),
        "kl": float(loss.kl.detach()),
        "kl_dyn": float(loss.kl_dyn.detach()),
        "kl_rep": float(loss.kl_rep.detach()),
        "kl_dyn_raw": float(loss.kl_dyn_raw.detach()),
        "kl_rep_raw": float(loss.kl_rep_raw.detach()),
    }


def world_model_step(
    model: WorldModel,
    optim: torch.optim.Optimizer,
    batch: dict[str, Tensor],
    *,
    device: torch.device,
    train_cfg: dict[str, Any],
    amp_dtype: torch.dtype | None,
    scaler: torch.amp.GradScaler,
    max_grad_norm: float = 1000.0,
) -> tuple[WorldModelLossBreakdown, dict[str, float]]:
    """Forward + backward + optimizer step on one replay window.

    Args:
        batch: CPU tensors from `ReplayBuffer.sample` (`obs` `[B,T,H,W,C]`).
        train_cfg: the `train:` mapping from the YAML config.
        amp_dtype: autocast dtype, or None for fp32.
        scaler: from `make_grad_scaler` (enabled only for fp16).
        max_grad_norm: DreamerV3's default grad-clip norm is 1000 (a high
            ceiling that only catches genuine blowups, not a routine clamp).

    Returns:
        `(loss_breakdown, metrics_dict)` with the same keys the logger uses.
    """
    batch = to_device(batch, device)
    obs = batch["obs"]
    optim.zero_grad(set_to_none=True)
    with autocast_context(device, amp_dtype):
        out = model(obs, batch["actions"])
        batch_n, time_n = obs.shape[:2]
        obs_f = nhwc_uint8_to_nchw_unit(obs.reshape(batch_n * time_n, *obs.shape[2:])).view(
            batch_n, time_n, 3, 64, 64
        )
        loss = world_model_loss(
            obs=obs_f,
            recon=out.recon,
            reward=batch["rewards"],
            reward_pred=out.reward_pred,
            reward_bins=model.reward_head.bins,
            cont=batch["cont"],
            cont_logit=out.cont_logit,
            post_logits=out.rssm.posterior_logits,
            prior_logits=out.rssm.prior_logits,
            unimix=model.rssm.unimix,
            dyn_scale=float(train_cfg["dyn_scale"]),
            rep_scale=float(train_cfg["rep_scale"]),
            free_nats=float(train_cfg["free_nats"]),
            free_nats_dyn=(
                None
                if train_cfg.get("free_nats_dyn") is None
                else float(train_cfg["free_nats_dyn"])
            ),
            recon_scale=float(train_cfg["recon_scale"]),
            reward_scale=float(train_cfg["reward_scale"]),
            continue_scale=float(train_cfg["continue_scale"]),
            kl_scale=float(train_cfg["kl_scale"]),
        )
        total = loss.total

    if scaler.is_enabled():
        scaler.scale(total).backward()
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optim)
        scaler.update()
    else:
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optim.step()

    return loss, loss_to_metrics(loss)
