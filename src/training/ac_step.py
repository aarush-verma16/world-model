"""One frozen-world-model actor-critic step (shared by CLI and notebook)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from agents.actor_critic import Actor, Critic, SlowCritic
from models.symlog import symlog_twohot_loss, symlog_twohot_mean
from models.world_model import WorldModel
from training.device import autocast_context, to_device
from training.imagine import freeze_world_model, imagine_ahead
from training.returns import PercentileReturnNorm, imagined_targets


@dataclass
class ActorCriticLoss:
    """Scalar losses for logging. `total` is the tensor that was backward'd."""

    total: Tensor
    actor: Tensor
    critic: Tensor
    entropy: Tensor
    reinforce: Tensor
    backprop: Tensor


def loss_to_metrics(loss: ActorCriticLoss, extra: dict[str, float] | None = None) -> dict[str, float]:
    out = {
        "total": float(loss.total.detach()),
        "actor": float(loss.actor.detach()),
        "critic": float(loss.critic.detach()),
        "entropy": float(loss.entropy.detach()),
        "reinforce": float(loss.reinforce.detach()),
        "backprop": float(loss.backprop.detach()),
    }
    if extra:
        out.update(extra)
    return out


def actor_critic_step(
    world_model: WorldModel,
    actor: Actor,
    critic: Critic,
    optim: torch.optim.Optimizer,
    batch: dict[str, Tensor],
    *,
    device: torch.device,
    retnorm: PercentileReturnNorm,
    horizon: int,
    start_mode: str = "all",
    lam: float = 0.95,
    discount: float = 0.997,
    entropy_scale: float = 3.0e-4,
    imag_gradient: str = "reinforce",
    imag_gradient_mix: float = 0.0,
    slow_critic: SlowCritic | None = None,
    amp_dtype: torch.dtype | None,
    scaler: torch.amp.GradScaler,
    max_grad_norm: float = 100.0,
) -> tuple[ActorCriticLoss, dict[str, float], Any]:
    """Imagine `horizon` steps and update actor + critic. World model frozen.

    Args:
        batch: CPU replay window (`obs` `[B,T,H,W,C]` uint8, `actions` `[B,T]`).
        retnorm: percentile EMA; mutated in place and should be checkpointed.
        slow_critic: EMA critic copy used as a second regression target. Passing
            `None` drops DreamerV3's slow-target term.

    Returns:
        `(loss, metrics, rollout)` — rollout is for optional visualization.

    Alignment (DreamerV3 `ImagBehavior._compute_target`): the rollout is
    state-indexed over `s_0 .. s_H`. `returns[:, i]` is the λ-return target for
    `V(s_i)`, so its baseline is `value[:, i]` and the critic head trained on it
    is the one evaluated at `s_i` — not at `s_{i+1}`.
    """
    freeze_world_model(world_model)
    actor.train()
    critic.train()
    if slow_critic is not None:
        slow_critic.update(critic)
    batch = to_device(batch, device)
    optim.zero_grad(set_to_none=True)

    with autocast_context(device, amp_dtype):
        mode = str(imag_gradient).lower()
        mix = float(imag_gradient_mix)
        if mode == "both" and not 0.0 <= mix <= 1.0:
            raise ValueError(f"imag_gradient_mix must be in [0, 1], got {mix}")
        # Crafter is reinforce (or both with mix 0). Do not allocate the
        # straight-through RSSM graph if it cannot receive gradient.
        dynamics_graph = mode == "dynamics" or (mode == "both" and mix > 0.0)
        rollout = imagine_ahead(
            world_model,
            actor,
            critic,
            batch["obs"],
            batch["actions"],
            horizon=horizon,
            start_mode=start_mode,
            discount=discount,
            dynamics_graph=dynamics_graph,
            is_first=batch.get("is_first"),
        )
        # Value is detached so the critic target does not move with the critic;
        # reward/cont keep their graph for the `dynamics` path.
        returns, base, weights = imagined_targets(
            rollout.reward,
            rollout.cont,
            rollout.value.detach(),
            lam=lam,
        )
        scale = retnorm.update(returns)
        adv = ((returns - base) / scale).detach()
        reinforce = -(weights * adv * rollout.log_prob).mean()
        entropy = rollout.entropy.mean()
        entropy_loss = -entropy_scale * entropy
        backprop = -(weights * returns / scale).mean()
        # DreamerV3-torch Crafter uses imag_gradient=reinforce (discrete);
        # `dynamics` is the straight-through path through the frozen RSSM.
        # `both` is a *convex mix*, not a sum: the reference default
        # imag_gradient_mix=0.0 weights the dynamics term at zero, because the
        # straight-through gradient through a one-hot action is biased and
        # nothing normalizes it against the reinforce term.
        if mode == "reinforce":
            actor_loss = reinforce + entropy_loss
            backprop = backprop.detach()
        elif mode == "dynamics":
            actor_loss = backprop + entropy_loss
            reinforce = reinforce.detach()
        elif mode == "both":
            actor_loss = mix * backprop + (1.0 - mix) * reinforce + entropy_loss
        else:
            raise ValueError(
                f"imag_gradient must be 'reinforce', 'dynamics', or 'both', got {imag_gradient!r}"
            )
        n_bins = critic.bins.shape[0]
        value_logits = rollout.value_logits[:, :-1]
        critic_nll = symlog_twohot_loss(
            value_logits.reshape(-1, n_bins),
            critic.bins,
            returns.detach().reshape(-1),
        ).view_as(weights)
        slow_value = float("nan")
        if slow_critic is not None:
            with torch.no_grad():
                slow_target = symlog_twohot_mean(
                    slow_critic(rollout.feat[:, :-1].detach()), slow_critic.bins
                )
            slow_value = float(slow_target.mean())
            critic_nll = critic_nll + symlog_twohot_loss(
                value_logits.reshape(-1, n_bins),
                critic.bins,
                slow_target.reshape(-1),
            ).view_as(weights)
        critic_loss = (weights * critic_nll).mean()
        total = actor_loss + critic_loss

    # Clip the actor and the critic separately: DreamerV3 gives each its own
    # optimizer and its own clip, so a large critic gradient must not consume
    # the actor's budget.
    def _clip() -> None:
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)

    if scaler.is_enabled():
        scaler.scale(total).backward()
        scaler.unscale_(optim)
        _clip()
        scaler.step(optim)
        scaler.update()
    else:
        total.backward()
        _clip()
        optim.step()

    loss = ActorCriticLoss(
        total=total,
        actor=actor_loss,
        critic=critic_loss,
        entropy=entropy,
        reinforce=reinforce,
        backprop=backprop,
    )
    extra = {
        "return": float(returns.detach().mean()),
        "return_std": float(returns.detach().std(unbiased=False)),
        "value": float(base.mean()),
        "reward": float(rollout.reward.detach().mean()),
        "adv": float(adv.mean()),
        "adv_std": float(adv.std(unbiased=False)),
        "retnorm_scale": float(scale.detach()),
        "entropy": float(entropy.detach()),
        "weight": float(weights.mean()),
        "slow_value": slow_value,
    }
    return loss, loss_to_metrics(loss, extra), rollout
