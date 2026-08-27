"""One frozen-world-model actor-critic step (shared by CLI and notebook)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from agents.actor_critic import Actor, Critic
from models.symlog import symlog_twohot_loss
from models.world_model import WorldModel
from training.device import autocast_context, to_device
from training.imagine import freeze_world_model, imagine_ahead
from training.returns import PercentileReturnNorm, lambda_returns


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
    imag_gradient: str = "both",
    amp_dtype: torch.dtype | None,
    scaler: torch.amp.GradScaler,
    max_grad_norm: float = 100.0,
) -> tuple[ActorCriticLoss, dict[str, float], Any]:
    """Imagine `horizon` steps and update actor + critic. World model frozen.

    Args:
        batch: CPU replay window (`obs` `[B,T,H,W,C]` uint8, `actions` `[B,T]`).
        retnorm: percentile EMA; mutated in place and should be checkpointed.

    Returns:
        `(loss, metrics, rollout)` — rollout is for optional visualization.
    """
    freeze_world_model(world_model)
    actor.train()
    critic.train()
    batch = to_device(batch, device)
    optim.zero_grad(set_to_none=True)

    with autocast_context(device, amp_dtype):
        rollout = imagine_ahead(
            world_model,
            actor,
            critic,
            batch["obs"],
            batch["actions"],
            horizon=horizon,
            start_mode=start_mode,
            discount=discount,
        )
        # Detach value inside λ-returns so the critic target is not a moving
        # function of the current V; reward/cont still depend on STE actions.
        returns = lambda_returns(
            rollout.reward,
            rollout.cont,
            rollout.value.detach(),
            lam=lam,
        )
        scale = retnorm.update(returns)
        adv = ((returns - rollout.value) / scale).detach()
        reinforce = -(adv * rollout.log_prob).mean()
        entropy = rollout.entropy.mean()
        entropy_loss = -entropy_scale * entropy
        backprop = -(returns / scale).mean()
        # DreamerV3-torch Crafter uses imag_gradient=reinforce (discrete).
        # `dynamics` is the STE path through the frozen RSSM; `both` is M4–M6.
        mode = str(imag_gradient).lower()
        if mode == "reinforce":
            actor_loss = reinforce + entropy_loss
            backprop = backprop.detach()
        elif mode == "dynamics":
            actor_loss = backprop + entropy_loss
            reinforce = reinforce.detach()
        elif mode == "both":
            actor_loss = reinforce + entropy_loss + backprop
        else:
            raise ValueError(
                f"imag_gradient must be 'reinforce', 'dynamics', or 'both', got {imag_gradient!r}"
            )
        n_bins = critic.bins.shape[0]
        critic_loss = symlog_twohot_loss(
            rollout.value_logits.reshape(-1, n_bins),
            critic.bins,
            returns.detach().reshape(-1),
        ).mean()
        total = actor_loss + critic_loss

    ac_params = list(actor.parameters()) + list(critic.parameters())
    if scaler.is_enabled():
        scaler.scale(total).backward()
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(ac_params, max_grad_norm)
        scaler.step(optim)
        scaler.update()
    else:
        total.backward()
        torch.nn.utils.clip_grad_norm_(ac_params, max_grad_norm)
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
        "value": float(rollout.value.detach().mean()),
        "reward": float(rollout.reward.detach().mean()),
        "adv": float(adv.mean()),
        "retnorm_scale": float(scale.detach()),
        "entropy": float(entropy.detach()),
    }
    return loss, loss_to_metrics(loss, extra), rollout
