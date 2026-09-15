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
from training.losses import WorldModelLossBreakdown, inventory_head_loss, world_model_loss


def loss_to_metrics(
    loss: WorldModelLossBreakdown,
    *,
    total_override: Tensor | None = None,
    inventory: Tensor | None = None,
) -> dict[str, float]:
    """Detach per-term losses to plain floats for logging.

    `total_override` / `inventory`: when the optional inventory head
    (finding 40, m18 only) is active, `total` here is the actual backward
    tensor (DreamerV3 total + `inventory_scale * inventory_loss`), not just
    `loss.total`, so the logged total matches what was backpropagated.
    """
    out = {
        "total": float((total_override if total_override is not None else loss.total).detach()),
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
    if inventory is not None:
        out["inventory"] = float(inventory.detach())
    return out


def _train_float(train_cfg: dict[str, Any], *keys: str) -> float:
    for key in keys:
        if key in train_cfg and train_cfg[key] is not None:
            return float(train_cfg[key])
    raise KeyError(keys[0])


def _train_optional_float(train_cfg: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in train_cfg:
            return None if train_cfg[key] is None else float(train_cfg[key])
    return None


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
        out = model(obs, batch["actions"], is_first=batch.get("is_first"))
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
            dyn_scale=_train_float(train_cfg, "dyn_scale", "dyn_scale"),
            rep_scale=_train_float(train_cfg, "rep_scale", "rep_scale"),
            free_nats=_train_float(train_cfg, "free_nats", "free_nats"),
            free_nats_dyn=_train_optional_float(train_cfg, "free_nats_dyn", "free_nats_dyn"),
            recon_scale=_train_float(train_cfg, "recon_scale", "recon_scale"),
            reward_scale=_train_float(train_cfg, "reward_scale", "reward_scale"),
            continue_scale=_train_float(train_cfg, "continue_scale", "continue_scale"),
            kl_scale=_train_float(train_cfg, "kl_scale", "kl_scale"),
        )
        total = loss.total

        # Optional (finding 40, m18 only): `model.inventory_head` is `None`
        # for every m6/m17-style config, so this branch is dead weight for
        # the faithful-Dreamer recipe, not a silently-added extra term on it.
        inv_loss: Tensor | None = None
        if model.inventory_head is not None and "inventory" in batch:
            inv_logits = model.inventory_head(out.feat)
            inv_loss = inventory_head_loss(
                inv_logits,
                batch["inventory"],
                batch.get("has_inventory", torch.zeros_like(batch["rewards"])),
            )
            inventory_scale = _train_optional_float(train_cfg, "inventory_scale")
            inventory_scale = 1.0 if inventory_scale is None else inventory_scale
            total = total + inventory_scale * inv_loss

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

    return loss, loss_to_metrics(loss, total_override=total, inventory=inv_loss)
