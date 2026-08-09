"""World-model loss terms, including Dreamer-style KL balancing.

Loss = recon_[h,z] + recon_embed + grad (edge-aware) + reward + continue + KL,
logged separately so a healthy sum cannot hide a collapsed/exploding KL (M3
exit criterion).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class WorldModelLossBreakdown:
    """Per-term losses (already reduced to scalars) plus the weighted total."""

    total: Tensor
    recon: Tensor
    recon_embed: Tensor
    grad: Tensor
    reward: Tensor
    continue_loss: Tensor
    kl: Tensor
    kl_dyn: Tensor
    kl_rep: Tensor
    kl_dyn_raw: Tensor
    kl_rep_raw: Tensor


def categorical_kl(
    post_logits: Tensor,
    prior_logits: Tensor,
    unimix: float = 0.01,
) -> Tensor:
    """KL(post || prior) per categorical variable, after unimix on both sides.

    Args:
        post_logits: `[..., stoch, classes]` raw posterior logits
        prior_logits: `[..., stoch, classes]` raw prior logits
        unimix: same floor used at sample time (must match RSSM.unimix)

    Returns:
        KL `[..., stoch]` in nats (summed over classes).
    """
    from models.rssm import unimix_probs

    post = unimix_probs(post_logits, unimix)
    prior = unimix_probs(prior_logits, unimix)
    return (post * (post.clamp_min(1e-8).log() - prior.clamp_min(1e-8).log())).sum(dim=-1)


def kl_balance(
    post_logits: Tensor,
    prior_logits: Tensor,
    *,
    unimix: float = 0.01,
    dyn_scale: float = 0.5,
    rep_scale: float = 0.1,
    free_nats: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """DreamerV2/V3 KL balancing with a free-nats floor on the *whole latent*.

    The `stoch` categorical variables are independent, so the joint KL of the
    full latent at one timestep is the **sum** (not average) of the per-variable
    KLs. Dreamer's `max(free_nats, KL)` floor applies to that per-timestep sum.
    Clamping each of the ~16-32 categorical variables individually (i.e. before
    summing) makes the floor far too strict — it would require *every single*
    variable to individually carry a full free nat, instead of the latent as a
    whole carrying one. That bug kept the KL loss pinned at the floor almost
    indefinitely even once the latent was already informative.

    Returns:
        `(kl_loss, kl_dyn_mean, kl_rep_mean, kl_dyn_raw_mean, kl_rep_raw_mean)`
        where the `_raw` values are the mean per-timestep *total* KL (summed
        over `stoch`, averaged over batch/time) before the free-nats floor.
    """
    kl_dyn_raw = categorical_kl(post_logits.detach(), prior_logits, unimix=unimix).sum(dim=-1)
    kl_rep_raw = categorical_kl(post_logits, prior_logits.detach(), unimix=unimix).sum(dim=-1)
    kl_dyn_raw_mean = kl_dyn_raw.mean()
    kl_rep_raw_mean = kl_rep_raw.mean()
    kl_dyn = kl_dyn_raw.clamp_min(free_nats) if free_nats > 0.0 else kl_dyn_raw
    kl_rep = kl_rep_raw.clamp_min(free_nats) if free_nats > 0.0 else kl_rep_raw
    kl_dyn_mean = kl_dyn.mean()
    kl_rep_mean = kl_rep.mean()
    kl_loss = dyn_scale * kl_dyn_mean + rep_scale * kl_rep_mean
    return kl_loss, kl_dyn_mean, kl_rep_mean, kl_dyn_raw_mean, kl_rep_raw_mean


def _pixel_loss(pred: Tensor, target: Tensor, kind: str) -> Tensor:
    if kind == "l1":
        return F.l1_loss(pred, target)
    if kind == "mse":
        return F.mse_loss(pred, target)
    raise ValueError(f"unknown recon_loss_type {kind!r}")


def gradient_l1_loss(pred: Tensor, target: Tensor) -> Tensor:
    """L1 loss between image finite-difference gradients (edge-aware term).

    Plain per-pixel L1/MSE rewards matching a region's *average* color, so a
    1-2px sprite outline or HUD icon/number stroke (a tiny fraction of a
    region's pixels) barely moves that average -- the optimizer has little
    incentive to sharpen it even as pixel loss keeps dropping. Comparing
    horizontal/vertical finite differences instead explicitly rewards
    matching *where* the image changes, which is what "sharp edges" actually
    means. This directly targets blurry backgrounds and illegible HUD bars
    that persist despite falling pixel loss.

    Args:
        pred / target: `[..., C, H, W]`, any leading dims.
    """
    pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
    pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def world_model_loss(
    *,
    obs: Tensor,
    recon: Tensor,
    recon_embed: Tensor,
    reward: Tensor,
    reward_pred: Tensor,
    cont: Tensor,
    cont_logit: Tensor,
    post_logits: Tensor,
    prior_logits: Tensor,
    unimix: float = 0.01,
    dyn_scale: float = 0.5,
    rep_scale: float = 0.1,
    free_nats: float = 1.0,
    recon_scale: float = 1.0,
    recon_embed_scale: float = 1.0,
    reward_scale: float = 1.0,
    continue_scale: float = 1.0,
    kl_scale: float = 1.0,
    grad_scale: float = 0.0,
    recon_loss_type: str = "l1",
) -> WorldModelLossBreakdown:
    """Assemble world-model terms.

    Args:
        obs / recon / recon_embed: float images `[B, T, 3, H, W]` in `[-1, 1]`
        reward: `[B, T]`
        reward_pred / cont_logit: `[B, T, 1]` or `[B, T]`
        cont: `[B, T]` in `{0, 1}`
        post_logits / prior_logits: `[B, T, stoch, classes]`
        grad_scale: weight on the edge-aware gradient term (see
            `gradient_l1_loss`); `0.0` disables it entirely (default, for
            backward compat with earlier checkpoints/configs).
    """
    recon_loss = _pixel_loss(recon, obs, recon_loss_type)
    recon_embed_loss = _pixel_loss(recon_embed, obs, recon_loss_type)
    grad_loss = gradient_l1_loss(recon, obs) + gradient_l1_loss(recon_embed, obs)

    reward_pred = reward_pred.squeeze(-1)
    reward_loss = F.mse_loss(reward_pred, reward)

    cont_logit = cont_logit.squeeze(-1)
    continue_loss = F.binary_cross_entropy_with_logits(cont_logit, cont.float())

    kl_loss, kl_dyn, kl_rep, kl_dyn_raw, kl_rep_raw = kl_balance(
        post_logits,
        prior_logits,
        unimix=unimix,
        dyn_scale=dyn_scale,
        rep_scale=rep_scale,
        free_nats=free_nats,
    )

    total = (
        recon_scale * recon_loss
        + recon_embed_scale * recon_embed_loss
        + reward_scale * reward_loss
        + continue_scale * continue_loss
        + kl_scale * kl_loss
        + grad_scale * grad_loss
    )
    return WorldModelLossBreakdown(
        total=total,
        recon=recon_loss,
        recon_embed=recon_embed_loss,
        grad=grad_loss,
        reward=reward_loss,
        continue_loss=continue_loss,
        kl=kl_loss,
        kl_dyn=kl_dyn,
        kl_rep=kl_rep,
        kl_dyn_raw=kl_dyn_raw,
        kl_rep_raw=kl_rep_raw,
    )
