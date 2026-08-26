"""Policy-driven collect into the episode replay buffer (M5).

The actor samples STE one-hot actions; `env.step` gets the integer
`argmax` of that sample (stochastic, not greedy logits). RSSM state is
carried with `obs_step` across real observations. Alignment matches
`collect_random_episodes`: store `(obs_t, action_t, reward_t, cont_t)`
where `action_t` is taken after observing `t`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from agents.actor_critic import Actor
from models.heads import rssm_features
from models.rssm import RSSMState
from models.world_model import WorldModel
from training.crafter_score import achievement_counts_from_info
from training.device import autocast_context
from training.replay_buffer import ReplayBuffer


def count_unlocked_achievements(info: dict[str, Any] | None) -> int:
    """How many Crafter achievements have count > 0. Diagnostic, not the M6 score."""
    if not info or "achievements" not in info:
        return 0
    ach = info["achievements"]
    if not isinstance(ach, dict):
        return 0
    return sum(1 for v in ach.values() if float(v) > 0)


def ste_action_to_int(action_oh: Tensor) -> int:
    """STE one-hot `[..., action_dim]` → Discrete env index (the sampled class)."""
    if action_oh.numel() == 0:
        raise ValueError("empty action tensor")
    flat = action_oh.reshape(-1, action_oh.shape[-1])
    return int(flat[0].argmax(dim=-1).item())


@torch.no_grad()
def rssm_policy_step(
    world_model: WorldModel,
    actor: Actor,
    obs_u8: np.ndarray,
    state: RSSMState,
    prev_action: Tensor,
    *,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> tuple[RSSMState, Tensor, int, float]:
    """One real observe + stochastic actor sample.

    Args:
        obs_u8: `[H, W, C]` uint8 current frame.
        state: previous `h` / `z_posterior` (`rssm.initial` at episode start).
        prev_action: one-hot `[1, action_dim]` (zeros at t=0).

    Returns:
        `(new_state, action_oh, action_int, entropy)` — `action_oh` is `[1, A]`.
    """
    world_model.eval()
    actor.eval()
    obs_t = torch.from_numpy(np.ascontiguousarray(obs_u8, dtype=np.uint8)).unsqueeze(0)
    obs_t = obs_t.to(device, non_blocking=True)
    with autocast_context(device, amp_dtype):
        embed = world_model.encode(obs_t)
        new_state, _z_prior, _prior_logits, _post_logits = world_model.rssm.obs_step(
            state, prev_action, embed
        )
        feat = rssm_features(new_state.h, new_state.z_posterior)
        action_oh, _log_prob, entropy, _probs = actor.policy(feat)
    action_i = ste_action_to_int(action_oh)
    return new_state, action_oh.float(), action_i, float(entropy.reshape(-1)[0].item())


class Collector:
    """One Crafter env, RSSM-conditioned actor, appends finished episodes to replay."""

    def __init__(
        self,
        env: Any,
        world_model: WorldModel,
        actor: Actor,
        buffer: ReplayBuffer,
        *,
        device: torch.device,
        max_episode_steps: int = 400,
        amp_dtype: torch.dtype | None = None,
        seed: int = 0,
    ) -> None:
        self.env = env
        self.world_model = world_model
        self.actor = actor
        self.buffer = buffer
        self.device = device
        self.max_episode_steps = int(max_episode_steps)
        self.amp_dtype = amp_dtype
        self.next_seed = int(seed)
        self._obs: np.ndarray | None = None
        self._state: RSSMState | None = None
        self._prev_action: Tensor | None = None
        self._obs_buf: list[np.ndarray] = []
        self._act_buf: list[int] = []
        self._rew_buf: list[float] = []
        self._cont_buf: list[float] = []
        self._ep_steps = 0
        self._need_reset = True

    def reset(self, seed: int | None = None) -> None:
        """Start a new episode. Increments `next_seed` when `seed` is omitted."""
        if seed is None:
            seed = self.next_seed
            self.next_seed += 1
        obs, _info = self.env.reset(seed=int(seed))
        self._obs = np.asarray(obs, dtype=np.uint8)
        rssm = self.world_model.rssm
        self._state = rssm.initial(1, device=self.device)
        self._prev_action = torch.zeros(1, rssm.action_dim, device=self.device)
        self._obs_buf = []
        self._act_buf = []
        self._rew_buf = []
        self._cont_buf = []
        self._ep_steps = 0
        self._need_reset = False

    def step(self) -> dict[str, Any]:
        """One env step. Finished episodes are added to the buffer."""
        if self._need_reset or self._obs is None:
            self.reset()
        assert self._obs is not None and self._state is not None
        assert self._prev_action is not None

        new_state, action_oh, action_i, entropy = rssm_policy_step(
            self.world_model,
            self.actor,
            self._obs,
            self._state,
            self._prev_action,
            device=self.device,
            amp_dtype=self.amp_dtype,
        )
        next_obs, reward, terminated, truncated, info = self.env.step(action_i)
        terminated = bool(terminated)
        truncated = bool(truncated)
        self._ep_steps += 1
        hit_cap = self._ep_steps >= self.max_episode_steps
        done = terminated or truncated or hit_cap
        # Truncation / cap still continue=1 (bootstrap); only true death is 0.
        cont = 0.0 if terminated else 1.0

        self._obs_buf.append(np.asarray(self._obs, dtype=np.uint8))
        self._act_buf.append(action_i)
        self._rew_buf.append(float(reward))
        self._cont_buf.append(cont)

        self._state = new_state
        self._prev_action = action_oh
        self._obs = np.asarray(next_obs, dtype=np.uint8)

        finished: dict[str, Any] | None = None
        if done:
            ep_len = len(self._obs_buf)
            ep_ret = float(sum(self._rew_buf))
            if ep_len > 0:
                self.buffer.add_episode(
                    self._obs_buf, self._act_buf, self._rew_buf, self._cont_buf
                )
            ach_counts = achievement_counts_from_info(
                info if isinstance(info, dict) else None
            )
            raw = {}
            if isinstance(info, dict) and isinstance(info.get("achievements"), dict):
                raw = {str(k): int(float(v)) for k, v in info["achievements"].items()}
            finished = {
                "return": ep_ret,
                "length": ep_len,
                "achievements": count_unlocked_achievements(
                    info if isinstance(info, dict) else None
                ),
                "achievement_counts": ach_counts,
                "achievement_counts_raw": raw,
            }
            self._need_reset = True
            self._obs = None

        dim = self.world_model.rssm.action_dim
        if not (0 <= action_i < dim):
            raise RuntimeError(f"action {action_i} outside [0, {dim})")

        return {
            "reward": float(reward),
            "action": action_i,
            "entropy": entropy,
            "done": done,
            "episode": finished,
        }

    def collect(self, n_steps: int) -> dict[str, Any]:
        """Take `n_steps` env steps. Returns reward mean and completed episodes."""
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        rewards: list[float] = []
        episodes: list[dict[str, Any]] = []
        last_entropy = 0.0
        for _ in range(int(n_steps)):
            out = self.step()
            rewards.append(float(out["reward"]))
            last_entropy = float(out["entropy"])
            if out["episode"] is not None:
                episodes.append(out["episode"])
        return {
            "steps": int(n_steps),
            "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
            "entropy": last_entropy,
            "episodes": episodes,
        }
