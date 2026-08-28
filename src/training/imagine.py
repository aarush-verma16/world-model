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

    Indexing follows DreamerV3: **state-indexed** quantities cover the `H + 1`
    states `s_0 .. s_H` (`s_0` is the replay posterior, `s_i` is after `i`
    `img_step`s), while **action-indexed** quantities cover the `H` actions
    taken *at* `s_0 .. s_{H-1}`.

    State-indexed `[N, H+1, ...]`: `h`, `z_prior`, `feat`, `reward`, `cont`,
    `value`, `value_logits`.
    Action-indexed `[N, H, ...]`: `action`, `log_prob`, `entropy`.

    So `value[:, i]` is `V(s_i)` — the baseline for `log_prob[:, i]` — and
    `reward[:, i]` is the reward predicted *at* `s_i`. Mixing the two is how
    the advantage silently becomes `reward + (γ-1)·V` instead of `Q - V`.
    `z_prior` is a prior sample at every index except the detached seed.
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
    is_first: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Observe a replay window; return detached `(h, z_posterior)` starts.

    Encode/observe run under `no_grad`: the starts are detached anyway, so the
    XL encoder backward was allocated and thrown away (16 GiB thrash on this
    box). Imagination never trains the encoder.

    Args:
        obs_u8: `[B, T, 64, 64, 3]` uint8.
        actions: `[B, T]` int64 (real actions used only to condition the
            posterior; imagination then ignores them).
        start_mode: `"all"` flattens every posterior in the window (`N=B*T`);
            `"last"` keeps the final timestep (`N=B`).
        is_first: optional `[B, T]` episode-start flags for `RSSM.observe`.
    """
    with torch.no_grad():
        embeds = world_model.encode(obs_u8)
        act = one_hot_action(actions, world_model.rssm.action_dim)
        rssm_out = world_model.rssm.observe(embeds, act, is_first=is_first)
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
    dynamics_graph: bool = True,
    is_first: Tensor | None = None,
) -> ImaginedRollout:
    """Roll `horizon` `z_prior` steps with actor-chosen STE actions.

    Args:
        obs_u8: `[B, T, 64, 64, 3]` uint8 replay window.
        actions: `[B, T]` int64 actions that produced the window (observe only).
        horizon: imagination length `H` (DreamerV3 default 15).
        start_mode: `"all"` or `"last"`.
        discount: extra multiplier on predicted continue (DreamerV3 0.997).
        dynamics_graph: keep the straight-through RSSM graph (`dynamics` /
            `both` with mix > 0). Crafter uses `imag_gradient=reinforce`, which
            never backprops through `img_step`; building that graph anyway is
            how XL hits 16 GiB and pages.
        is_first: optional `[B, T]` flags forwarded to `RSSM.observe`.

    Returns:
        `ImaginedRollout`. State-indexed fields have time dim `H + 1` (the seed
        state plus one per `img_step`); action-indexed fields have time dim `H`.

    The actor sees a **detached** feature at every step, matching DreamerV3's
    `inp = feat.detach()`. When `dynamics_graph` is True the straight-through
    action still carries gradient into `img_step`. The critic always reads
    detached features so the critic loss cannot leak into the actor.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    h, z_prior = _start_states(
        world_model, obs_u8, actions, start_mode, is_first=is_first
    )
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

    for step in range(horizon + 1):
        if dynamics_graph:
            feat = rssm_features(h, z_prior)
            reward_logits = world_model.reward_head(feat)
            cont_logit = world_model.continue_head(feat).squeeze(-1)
        else:
            with torch.no_grad():
                feat = rssm_features(h, z_prior)
                reward_logits = world_model.reward_head(feat)
                cont_logit = world_model.continue_head(feat).squeeze(-1)
            feat = feat.detach()
        value_logits = critic(feat.detach())

        hs.append(h)
        zs.append(z_prior)
        feats.append(feat)
        rewards.append(symlog_twohot_mean(reward_logits, world_model.reward_head.bins))
        conts.append(torch.sigmoid(cont_logit) * discount)
        v_logits.append(value_logits)
        values.append(symlog_twohot_mean(value_logits, critic.bins))

        if step == horizon:
            break
        action, log_prob, entropy, _probs = actor.policy(feat.detach())
        acts.append(action)
        logps.append(log_prob)
        ents.append(entropy)
        if dynamics_graph:
            h, z_prior, _prior_logits = world_model.rssm.img_step(h, z_prior, action)
        else:
            with torch.no_grad():
                h, z_prior, _prior_logits = world_model.rssm.img_step(h, z_prior, action)

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
