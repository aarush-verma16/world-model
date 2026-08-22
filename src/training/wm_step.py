"""One world-model optimizer step, shared by the CLI and the training notebook.

Keeps AMP / scaler / grad-clip in one place so the notebook cannot silently
drift back to a full-fp32 loop after a CUDA config change.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from models.preprocess import nhwc_uint8_to_nchw_float
from models.world_model import WorldModel
from training.device import autocast_context, to_device
from training.losses import WorldModelLossBreakdown, world_model_loss


def loss_to_metrics(loss: WorldModelLossBreakdown) -> dict[str, float]:
    """Detach per-term losses to plain floats for logging."""
    return {
        "total": float(loss.total.detach()),
        "recon": float(loss.recon.detach()),
        "recon_embed": float(loss.recon_embed.detach()),
        "recon_bottleneck": float(loss.recon_bottleneck.detach()),
        "recon_l1": float(loss.recon_l1.detach()),
        "recon_embed_l1": float(loss.recon_embed_l1.detach()),
        "recon_bottleneck_l1": float(loss.recon_bottleneck_l1.detach()),
        "recon_map": float(loss.recon_map.detach()),
        "recon_blob": float(loss.recon_blob.detach()),
        "recon_embed_blob": float(loss.recon_embed_blob.detach()),
        "grad": float(loss.grad.detach()),
        "reward": float(loss.reward.detach()),
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
    max_grad_norm: float = 100.0,
) -> tuple[WorldModelLossBreakdown, dict[str, float]]:
    """Forward + backward + optimizer step on one replay window.

    Args:
        batch: CPU tensors from `ReplayBuffer.sample` (`obs` `[B,T,H,W,C]`).
        train_cfg: the `train:` mapping from the YAML config.
        amp_dtype: autocast dtype, or None for fp32.
        scaler: from `make_grad_scaler` (enabled only for fp16).

    Returns:
        `(loss_breakdown, metrics_dict)` with the same keys the logger uses.
    """
    batch = to_device(batch, device)
    obs = batch["obs"]
    optim.zero_grad(set_to_none=True)
    with autocast_context(device, amp_dtype):
        out = model(obs, batch["actions"])
        batch_n, time_n = obs.shape[:2]
        obs_f = nhwc_uint8_to_nchw_float(obs.reshape(batch_n * time_n, *obs.shape[2:])).view(
            batch_n, time_n, 3, 64, 64
        )
        loss = world_model_loss(
            obs=obs_f,
            recon=out.recon,
            recon_embed=out.recon_embed,
            recon_bottleneck=out.recon_bottleneck,
            reward=batch["rewards"],
            reward_pred=out.reward_pred,
            cont=batch["cont"],
            cont_logit=out.cont_logit,
            post_logits=out.rssm.posterior_logits,
            prior_logits=out.rssm.prior_logits,
            unimix=model.rssm.unimix,
            dyn_scale=float(train_cfg["dyn_scale"]),
            rep_scale=float(train_cfg["rep_scale"]),
            free_nats=float(train_cfg["free_nats"]),
            recon_scale=float(train_cfg["recon_scale"]),
            recon_embed_scale=float(train_cfg.get("recon_embed_scale", 1.0)),
            recon_bottleneck_scale=float(train_cfg.get("recon_bottleneck_scale", 0.0)),
            reward_scale=float(train_cfg["reward_scale"]),
            continue_scale=float(train_cfg["continue_scale"]),
            kl_scale=float(train_cfg["kl_scale"]),
            grad_scale=float(train_cfg.get("grad_scale", 0.0)),
            recon_loss_type=str(train_cfg.get("recon_loss", "l1")),
            edge_weight=float(train_cfg.get("edge_weight", 0.0)),
            hz_map=out.hz_map,
            embed_map=out.embed_map,
            recon_map_scale=float(train_cfg.get("recon_map_scale", 0.0)),
            recon_blob_scale=float(train_cfg.get("recon_blob_scale", 0.0)),
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
