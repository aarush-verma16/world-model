"""Sequential replay buffer for RSSM world-model training.

Returns contiguous length-L chunks (not i.i.d. frames). The RSSM needs temporal
continuity; shuffling independent timesteps breaks the recurrence signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass
class EpisodeBatch:
    """One stored episode (variable length)."""

    obs: Tensor  # uint8 [T, 64, 64, 3]
    actions: Tensor  # int64 [T]
    rewards: Tensor  # float32 [T]
    cont: Tensor  # float32 [T]  — 1 if episode continues after this step


class ReplayBuffer:
    """Stores full episodes and samples fixed-length contiguous windows."""

    def __init__(self, seed: int = 0) -> None:
        self._episodes: list[EpisodeBatch] = []
        self._rng = np.random.default_rng(seed)
        self._total_steps = 0

    def __len__(self) -> int:
        return len(self._episodes)

    @property
    def num_steps(self) -> int:
        return self._total_steps

    def add_episode(
        self,
        obs: Tensor | np.ndarray,
        actions: Tensor | np.ndarray,
        rewards: Tensor | np.ndarray,
        cont: Tensor | np.ndarray,
    ) -> None:
        """Append one episode. All arrays length `T` along dim 0."""
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.uint8), dtype=torch.uint8)
        act_t = torch.as_tensor(np.asarray(actions), dtype=torch.int64)
        rew_t = torch.as_tensor(np.asarray(rewards), dtype=torch.float32)
        cont_t = torch.as_tensor(np.asarray(cont), dtype=torch.float32)
        if obs_t.ndim != 4 or obs_t.shape[-1] != 3:
            raise ValueError(f"obs must be [T,H,W,3] uint8, got {tuple(obs_t.shape)}")
        t = obs_t.shape[0]
        if act_t.shape != (t,) or rew_t.shape != (t,) or cont_t.shape != (t,):
            raise ValueError(
                f"length mismatch: obs {t}, actions {tuple(act_t.shape)}, "
                f"rewards {tuple(rew_t.shape)}, cont {tuple(cont_t.shape)}"
            )
        self._episodes.append(
            EpisodeBatch(obs=obs_t, actions=act_t, rewards=rew_t, cont=cont_t)
        )
        self._total_steps += t

    def sample(self, batch_size: int, seq_len: int) -> dict[str, Tensor]:
        """Sample contiguous windows.

        Returns dict with:
            obs `[B, L, 64, 64, 3]` uint8
            actions `[B, L]` int64
            rewards `[B, L]` float32
            cont `[B, L]` float32
        """
        if not self._episodes:
            raise RuntimeError("replay buffer is empty")
        # Episodes long enough to hold a window.
        eligible = [ep for ep in self._episodes if ep.obs.shape[0] >= seq_len]
        if not eligible:
            raise RuntimeError(
                f"no episode with length >= seq_len={seq_len} "
                f"(have {len(self._episodes)} episodes, max_len="
                f"{max(ep.obs.shape[0] for ep in self._episodes)})"
            )

        obs_list: list[Tensor] = []
        act_list: list[Tensor] = []
        rew_list: list[Tensor] = []
        cont_list: list[Tensor] = []
        for _ in range(batch_size):
            ep = eligible[int(self._rng.integers(0, len(eligible)))]
            t = ep.obs.shape[0]
            start = int(self._rng.integers(0, t - seq_len + 1))
            end = start + seq_len
            obs_list.append(ep.obs[start:end])
            act_list.append(ep.actions[start:end])
            rew_list.append(ep.rewards[start:end])
            cont_list.append(ep.cont[start:end])

        return {
            "obs": torch.stack(obs_list, dim=0),
            "actions": torch.stack(act_list, dim=0),
            "rewards": torch.stack(rew_list, dim=0),
            "cont": torch.stack(cont_list, dim=0),
        }

    def state_dict(self) -> dict:
        return {
            "episodes": [
                {
                    "obs": ep.obs,
                    "actions": ep.actions,
                    "rewards": ep.rewards,
                    "cont": ep.cont,
                }
                for ep in self._episodes
            ],
            "total_steps": self._total_steps,
        }

    def load_state_dict(self, state: dict) -> None:
        self._episodes = [
            EpisodeBatch(
                obs=torch.as_tensor(item["obs"], dtype=torch.uint8),
                actions=torch.as_tensor(item["actions"], dtype=torch.int64),
                rewards=torch.as_tensor(item["rewards"], dtype=torch.float32),
                cont=torch.as_tensor(item["cont"], dtype=torch.float32),
            )
            for item in state["episodes"]
        ]
        self._total_steps = int(state.get("total_steps", sum(ep.obs.shape[0] for ep in self._episodes)))


def collect_random_episodes(
    *,
    env_id: str,
    num_episodes: int,
    max_episode_steps: int,
    action_dim: int,
    seed: int = 0,
) -> ReplayBuffer:
    """Fill a buffer with random-policy Crafter episodes (obs/action/reward/cont)."""
    import gymnasium as gym

    from envs.crafter_env import register_crafter_envs

    register_crafter_envs()
    env = gym.make(env_id)
    if int(env.action_space.n) != action_dim:
        n = int(env.action_space.n)
        env.close()
        raise ValueError(f"config action_dim={action_dim} != env.action_space.n={n}")

    buffer = ReplayBuffer(seed=seed)
    try:
        for ep in range(num_episodes):
            obs, _ = env.reset(seed=seed + ep)
            obs_buf: list = []
            act_buf: list[int] = []
            rew_buf: list[float] = []
            cont_buf: list[float] = []
            for _ in range(max_episode_steps):
                action = int(env.action_space.sample())
                next_obs, reward, terminated, truncated, _info = env.step(action)
                done = bool(terminated or truncated)
                obs_buf.append(np.asarray(obs, dtype=np.uint8))
                act_buf.append(action)
                rew_buf.append(float(reward))
                # Continue = not terminated. Truncation still counts as continue=1
                # for bootstrap semantics; we still break the episode storage on either.
                cont_buf.append(0.0 if terminated else 1.0)
                obs = next_obs
                if done:
                    break
            if len(obs_buf) == 0:
                continue
            buffer.add_episode(obs_buf, act_buf, rew_buf, cont_buf)
    finally:
        env.close()
    return buffer
