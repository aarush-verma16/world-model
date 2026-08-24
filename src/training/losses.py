"""World-model loss terms: DreamerV3's four-term recipe, nothing else.

`total = recon_scale*recon + reward_scale*reward + continue_scale*continue
       + kl_scale*(dyn_scale*kl_dyn + rep_scale*kl_rep)`

Logged per term so a healthy sum cannot hide a collapsed/exploded KL (M3
exit criterion). `recon_l1` / `reward_mae` are unweighted, human-readable
metrics — never multiplied into `total` — kept alongside the trained losses
so "is it actually improving" doesn't require decoding the loss scale by
hand.

This module previously carried ~10 extra terms (embed-bypass recon, a frozen
`[h,z]` decoder, tile/avatar/HUD crop losses, an edge-aware reweighting).
None of that is in DreamerV3, and it is why M3 spent weeks not converging —
see `docs/experiments.md` ("DreamerV3 M3 reset") for the postmortem. Do not
re-add per-region loss terms; if reconstruction quality is the problem, it is
almost always a graph/scale bug (recon bypassing the RSSM, or recon too weak
relative to KL), not a missing loss term.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from models.symlog import symlog_twohot_loss, symlog_twohot_mean


@dataclass
class WorldModelLossBreakdown:
    """Per-term losses (already reduced to scalars) plus the weighted total.

    `recon` / `reward` / `continue_loss` / `kl` are the values actually
    multiplied into `total`. `recon_l1` / `reward_mae` are unweighted,
    log-only metrics for humans.
    """

    total: Tensor
    recon: Tensor
    recon_l1: Tensor
    reward: Tensor
    reward_mae: Tensor
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
    free_nats_dyn: float | None = None,
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

    The two terms do very different jobs, so they do not want the same floor:

    - `kl_rep` trains the **posterior** (prior detached). This is the actual
      information-rate constraint: `kl_rep_raw` in nats *is* how much the
      latent tells the decoder that the prior did not already predict. A floor
      here is what "up to `free_nats` of information is free" means.
    - `kl_dyn` trains the **prior** (posterior detached). It cannot restrict
      information at all — it only makes the dynamics model better at
      predicting `z`. DreamerV3's default (`free_nats_dyn=None` → reuse
      `free_nats`) floors this too; only pass `free_nats_dyn=0.0` if a run
      shows `kl_dyn_raw` welded exactly to `free_nats` for thousands of steps
      *and* open-loop video prediction is still broken after everything else
      in this recipe is paper-faithful (recon scale/reduction, live decoder).

    Args:
        free_nats: floor on the rep term (the rate budget).
        free_nats_dyn: floor on the dyn term; `None` reuses `free_nats` (the
            DreamerV3 default).

    Returns:
        `(kl_loss, kl_dyn_mean, kl_rep_mean, kl_dyn_raw_mean, kl_rep_raw_mean)`
        where the `_raw` values are the mean per-timestep *total* KL (summed
        over `stoch`, averaged over batch/time) before the free-nats floor.
    """
    if free_nats_dyn is None:
        free_nats_dyn = free_nats
    kl_dyn_raw = categorical_kl(post_logits.detach(), prior_logits, unimix=unimix).sum(dim=-1)
    kl_rep_raw = categorical_kl(post_logits, prior_logits.detach(), unimix=unimix).sum(dim=-1)
    kl_dyn_raw_mean = kl_dyn_raw.mean()
    kl_rep_raw_mean = kl_rep_raw.mean()
    kl_dyn = kl_dyn_raw.clamp_min(free_nats_dyn) if free_nats_dyn > 0.0 else kl_dyn_raw
    kl_rep = kl_rep_raw.clamp_min(free_nats) if free_nats > 0.0 else kl_rep_raw
    kl_dyn_mean = kl_dyn.mean()
    kl_rep_mean = kl_rep.mean()
    kl_loss = dyn_scale * kl_dyn_mean + rep_scale * kl_rep_mean
    return kl_loss, kl_dyn_mean, kl_rep_mean, kl_dyn_raw_mean, kl_rep_raw_mean


def image_mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Squared-error image loss, DreamerV3 reduction: sum over pixels, mean over B/T.

    This is what makes reconstruction the dominant term against a KL that
    free-bits pins near 1 nat — a per-pixel *mean* (as an unweighted L1/MSE
    metric would give) is ~4 orders of magnitude smaller than the paper's
    per-timestep sum and lets KL dominate the gradient instead, which is what
    silently starved reconstruction in the pre-reset loss (see
    `docs/experiments.md`).

    Args:
        pred / target: `[..., C, H, W]`, any leading dims.

    Returns:
        scalar mean over the leading dims of the per-step summed squared error.
    """
    se = (pred - target).pow(2)
    return se.flatten(start_dim=pred.ndim - 3).sum(dim=-1).mean()


def world_model_loss(
    *,
    obs: Tensor,
    recon: Tensor,
    reward: Tensor,
    reward_pred: Tensor,
    reward_bins: Tensor,
    cont: Tensor,
    cont_logit: Tensor,
    post_logits: Tensor,
    prior_logits: Tensor,
    unimix: float = 0.01,
    dyn_scale: float = 0.5,
    rep_scale: float = 0.1,
    free_nats: float = 1.0,
    free_nats_dyn: float | None = None,
    recon_scale: float = 1.0,
    reward_scale: float = 1.0,
    continue_scale: float = 1.0,
    kl_scale: float = 1.0,
) -> WorldModelLossBreakdown:
    """Assemble the four DreamerV3 world-model loss terms.

    Args:
        obs / recon: float images `[B, T, 3, H, W]`, same range (`[0, 1]` for
            the M3 `output_activation="linear"` decoder).
        reward: `[B, T]` real reward.
        reward_pred: `[B, T, num_bins]` logits from `RewardHead`.
        reward_bins: `RewardHead.bins`, `[num_bins]`.
        cont_logit: `[B, T, 1]` or `[B, T]`.
        cont: `[B, T]` in `{0, 1}`.
        post_logits / prior_logits: `[B, T, stoch, classes]`.
        free_nats: floor on the rep KL (the information-rate budget).
        free_nats_dyn: floor on the dyn KL; `None` reuses `free_nats`
            (DreamerV3 default). See `kl_balance`.
    """
    recon_loss = image_mse_loss(recon, obs)
    recon_l1 = F.l1_loss(recon, obs)

    reward_nll = symlog_twohot_loss(reward_pred, reward_bins, reward)
    reward_loss = reward_nll.mean()
    with torch.no_grad():
        reward_decoded = symlog_twohot_mean(reward_pred, reward_bins)
        reward_mae = F.l1_loss(reward_decoded, reward)

    cont_logit = cont_logit.squeeze(-1)
    continue_loss = F.binary_cross_entropy_with_logits(cont_logit, cont.float())

    kl_loss, kl_dyn, kl_rep, kl_dyn_raw, kl_rep_raw = kl_balance(
        post_logits,
        prior_logits,
        unimix=unimix,
        dyn_scale=dyn_scale,
        rep_scale=rep_scale,
        free_nats=free_nats,
        free_nats_dyn=free_nats_dyn,
    )

    total = (
        recon_scale * recon_loss
        + reward_scale * reward_loss
        + continue_scale * continue_loss
        + kl_scale * kl_loss
    )
    return WorldModelLossBreakdown(
        total=total,
        recon=recon_loss,
        recon_l1=recon_l1,
        reward=reward_loss,
        reward_mae=reward_mae,
        continue_loss=continue_loss,
        kl=kl_loss,
        kl_dyn=kl_dyn,
        kl_rep=kl_rep,
        kl_dyn_raw=kl_dyn_raw,
        kl_rep_raw=kl_rep_raw,
    )
