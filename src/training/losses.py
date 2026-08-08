"""World-model loss terms, including Dreamer-style KL balancing.

Loss = recon + reward + continue + KL, logged separately so a healthy sum
cannot hide a collapsed/exploding KL (M3 exit criterion).
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
    reward: Tensor
    continue_loss: Tensor
    kl: Tensor
    kl_dyn: Tensor
    kl_rep: Tensor


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
    # Local import avoids a circular dependency through models.__init__.
    from models.rssm import unimix_probs

    post = unimix_probs(post_logits, unimix)
    prior = unimix_probs(prior_logits, unimix)
    # KL(p||q) = sum p * (log p - log q)
    return (post * (post.clamp_min(1e-8).log() - prior.clamp_min(1e-8).log())).sum(dim=-1)


def kl_balance(
    post_logits: Tensor,
    prior_logits: Tensor,
    *,
    unimix: float = 0.01,
    dyn_scale: float = 0.5,
    rep_scale: float = 0.1,
    free_nats: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """DreamerV2/V3 KL balancing with free-nats floor.

    - `kl_dyn`: train prior toward posterior (`post` stopgrad) — dynamics learning
    - `kl_rep`: train posterior toward prior (`prior` stopgrad) — representation
    - Each side is clamped below by `free_nats` (per categorical variable) so
      the model is not forced to crush tiny informative KLs to zero.

    Args:
        post_logits / prior_logits: `[B, T, stoch, classes]`
        dyn_scale / rep_scale: asymmetric weights (Dreamer defaults ≈ 0.5 / 0.1)
        free_nats: minimum KL credited per categorical variable

    Returns:
        `(kl_loss, kl_dyn_mean, kl_rep_mean)` — `kl_loss` is the scalar used in
        the total loss; the means are for TensorBoard.
    """
    kl_dyn = categorical_kl(post_logits.detach(), prior_logits, unimix=unimix)
    kl_rep = categorical_kl(post_logits, prior_logits.detach(), unimix=unimix)
    if free_nats > 0.0:
        kl_dyn = kl_dyn.clamp_min(free_nats)
        kl_rep = kl_rep.clamp_min(free_nats)
    # Mean over batch, time, and categorical variables.
    kl_dyn_mean = kl_dyn.mean()
    kl_rep_mean = kl_rep.mean()
    kl_loss = dyn_scale * kl_dyn_mean + rep_scale * kl_rep_mean
    return kl_loss, kl_dyn_mean, kl_rep_mean


def world_model_loss(
    *,
    obs: Tensor,
    recon: Tensor,
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
    reward_scale: float = 1.0,
    continue_scale: float = 1.0,
    kl_scale: float = 1.0,
) -> WorldModelLossBreakdown:
    """Assemble the four world-model terms.

    Args:
        obs / recon: float images `[B, T, 3, H, W]` in `[-1, 1]`
        reward: `[B, T]` target rewards
        reward_pred: `[B, T, 1]` or `[B, T]` predictions
        cont: `[B, T]` continue targets in `{0, 1}`
        cont_logit: `[B, T, 1]` or `[B, T]` continue logits
        post_logits / prior_logits: `[B, T, stoch, classes]`

    Returns:
        `WorldModelLossBreakdown` with scalar tensors (keep graph on `total`).
    """
    recon_loss = F.mse_loss(recon, obs)

    reward_pred = reward_pred.squeeze(-1)
    reward_loss = F.mse_loss(reward_pred, reward)

    cont_logit = cont_logit.squeeze(-1)
    continue_loss = F.binary_cross_entropy_with_logits(cont_logit, cont.float())

    kl_loss, kl_dyn, kl_rep = kl_balance(
        post_logits,
        prior_logits,
        unimix=unimix,
        dyn_scale=dyn_scale,
        rep_scale=rep_scale,
        free_nats=free_nats,
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
        reward=reward_loss,
        continue_loss=continue_loss,
        kl=kl_loss,
        kl_dyn=kl_dyn,
        kl_rep=kl_rep,
    )
