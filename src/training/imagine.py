"""Imagine H steps with the actor choosing STE actions (`z_prior` only).

The seed is a real posterior `(h, z_posterior)` from `RSSM.observe` on a
replay window, then detached so actor-critic gradients cannot leak into the
encoder. Every subsequent latent is `z_prior` from `RSSM.img_step`.
World-model *parameters* must be frozen (`requires_grad=False`); the
forward graph stays open so straight-through actions reach the actor.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from agents.actor_critic import Actor, Critic
from models.heads import rssm_features
from models.rssm import one_hot_action
from models.symlog import symlog_twohot_mean
from models.world_model import WorldModel


@dataclass
class ImaginedRollout:
    """One imagination batch. Leading dim `N` is start states (`B` or `B*T`).

    All time-indexed tensors are `[N, H, ...]`. `z_prior` is the latent after
    each `img_step` — never a posterior.
    """

    h: Tensor
    z_prior: Tensor
    feat: Tensor
    action: Tensor
    log_prob: Tensor
    entropy: Tensor
    reward: Tensor
    cont: Tensor
    value_logits: Tensor
    value: Tensor


def freeze_world_model(model: WorldModel) -> None:
    """Eval + freeze every world-model parameter. Actor-critic stays trainable."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)


def unfreeze_world_model(model: WorldModel) -> None:
    """Train + unfreeze every world-model parameter after an actor-critic phase.

    `actor_critic_step` calls `freeze_world_model` every time. The outer loop
    must call this before `world_model_step` or the WM silently stops learning.
    """
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)


def _start_states(
    world_model: WorldModel,
    obs_u8: Tensor,
    actions: Tensor,
    start_mode: str,
) -> tuple[Tensor, Tensor]:
    """Observe a replay window; return detached `(h, z_posterior)` starts.

    Args:
        obs_u8: `[B, T, 64, 64, 3]` uint8.
        actions: `[B, T]` int64 (real actions used only to condition the
            posterior; imagination then ignores them).
        start_mode: `"all"` flattens every posterior in the window (`N=B*T`);
            `"last"` keeps the final timestep (`N=B`).
    """
    embeds = world_model.encode(obs_u8)
    act = one_hot_action(actions, world_model.rssm.action_dim)
    rssm_out = world_model.rssm.observe(embeds, act)
    h = rssm_out.h
    z_posterior = rssm_out.z_posterior
    if start_mode == "all":
        batch, time = h.shape[:2]
        h = h.reshape(batch * time, h.shape[-1])
        z_posterior = z_posterior.reshape(batch * time, *z_posterior.shape[-2:])
    elif start_mode == "last":
        h = h[:, -1]
        z_posterior = z_posterior[:, -1]
    else:
        raise ValueError(f"start_mode must be 'all' or 'last', got {start_mode!r}")
    return h.detach(), z_posterior.detach()


def imagine_ahead(
    world_model: WorldModel,
    actor: Actor,
    critic: Critic,
    obs_u8: Tensor,
    actions: Tensor,
    *,
    horizon: int,
    start_mode: str = "all",
    discount: float = 0.997,
) -> ImaginedRollout:
    """Roll `horizon` `z_prior` steps with actor-chosen STE actions.

    Args:
        obs_u8: `[B, T, 64, 64, 3]` uint8 replay window.
        actions: `[B, T]` int64 actions that produced the window (observe only).
        horizon: imagination length `H` (DreamerV3 default 15).
        start_mode: `"all"` or `"last"`.
        discount: extra multiplier on predicted continue (DreamerV3 0.997).

    Returns:
        `ImaginedRollout` with time dim `H`. Reward / continue / value are
        decoded from the *next* state after each `img_step`.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    h, z_prior = _start_states(world_model, obs_u8, actions, start_mode)
    # After detach the seed is a posterior sample; every img_step overwrites
    # z_prior with a prior sample. The name marks the imagination path.
    hs: list[Tensor] = []
    zs: list[Tensor] = []
    feats: list[Tensor] = []
    acts: list[Tensor] = []
    logps: list[Tensor] = []
    ents: list[Tensor] = []
    rewards: list[Tensor] = []
    conts: list[Tensor] = []
    v_logits: list[Tensor] = []
    values: list[Tensor] = []

    for _ in range(horizon):
        feat_in = rssm_features(h, z_prior)
        action, log_prob, entropy, _probs = actor.policy(feat_in)
        h, z_prior, _prior_logits = world_model.rssm.img_step(h, z_prior, action)
        feat = rssm_features(h, z_prior)
        reward_logits = world_model.reward_head(feat)
        cont_logit = world_model.continue_head(feat).squeeze(-1)
        value_logits = critic(feat)
        reward = symlog_twohot_mean(reward_logits, world_model.reward_head.bins)
        value = symlog_twohot_mean(value_logits, critic.bins)
        cont = torch.sigmoid(cont_logit) * discount

        hs.append(h)
        zs.append(z_prior)
        feats.append(feat)
        acts.append(action)
        logps.append(log_prob)
        ents.append(entropy)
        rewards.append(reward)
        conts.append(cont)
        v_logits.append(value_logits)
        values.append(value)

    return ImaginedRollout(
        h=torch.stack(hs, dim=1),
        z_prior=torch.stack(zs, dim=1),
        feat=torch.stack(feats, dim=1),
        action=torch.stack(acts, dim=1),
        log_prob=torch.stack(logps, dim=1),
        entropy=torch.stack(ents, dim=1),
        reward=torch.stack(rewards, dim=1),
        cont=torch.stack(conts, dim=1),
        value_logits=torch.stack(v_logits, dim=1),
        value=torch.stack(values, dim=1),
    )


@torch.no_grad()
def decode_imagination(
    world_model: WorldModel,
    feat: Tensor,
    max_starts: int = 1,
) -> Tensor:
    """Decode imagined features for visualization only. `feat` `[N, H, feat_dim]`.

    Returns uint8-ready float images `[min(N,max_starts), H, 3, 64, 64]` in `[0, 1]`.
    """
    n = min(int(max_starts), feat.shape[0])
    return world_model.decode(feat[:n]).clamp(0.0, 1.0)
