"""Sequential replay buffer for RSSM world-model training.

Returns contiguous length-L chunks (not i.i.d. frames). The RSSM needs temporal
continuity; shuffling independent timesteps breaks the recurrence signal.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

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
    """Stores full episodes and samples fixed-length contiguous windows.

    Windows never cross episode boundaries (no `is_first`). Optional FIFO
    `max_steps` drops the oldest episodes so host RAM cannot grow forever.
    """

    def __init__(self, seed: int = 0, max_steps: int | None = None) -> None:
        self._episodes: list[EpisodeBatch] = []
        self._rng = np.random.default_rng(seed)
        self._total_steps = 0
        self.max_steps = None if max_steps is None else int(max_steps)

    def __len__(self) -> int:
        return len(self._episodes)

    @property
    def num_steps(self) -> int:
        return self._total_steps

    def _evict(self) -> None:
        """Drop oldest episodes until `num_steps <= max_steps`.

        A single episode longer than `max_steps` is kept (we do not split).
        """
        if self.max_steps is None:
            return
        while len(self._episodes) > 1 and self._total_steps > self.max_steps:
            old = self._episodes.pop(0)
            self._total_steps -= int(old.obs.shape[0])

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
        self._evict()

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
        self._total_steps = int(
            state.get("total_steps", sum(ep.obs.shape[0] for ep in self._episodes))
        )
        self._evict()


def _fmt_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, sec = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def collect_random_episodes(
    *,
    env_id: str,
    num_episodes: int,
    max_episode_steps: int,
    action_dim: int,
    seed: int = 0,
    progress: bool = True,
) -> ReplayBuffer:
    """Fill a buffer with random-policy Crafter episodes (obs/action/reward/cont).

    When `progress=True` (default), prints one flushed line per episode so a
    notebook/CLI run does not look hung during a long collect.
    """
    import gymnasium as gym

    from envs.crafter_env import register_crafter_envs

    def log(msg: str) -> None:
        if progress:
            print(msg, flush=True)

    register_crafter_envs()
    env = gym.make(env_id)
    if int(env.action_space.n) != action_dim:
        n = int(env.action_space.n)
        env.close()
        raise ValueError(f"config action_dim={action_dim} != env.action_space.n={n}")

    buffer = ReplayBuffer(seed=seed)
    lengths: list[int] = []
    returns: list[float] = []
    nonzero_reward_steps = 0
    t0 = time.perf_counter()
    log(
        f"collecting {num_episodes} random episodes from {env_id} "
        f"(max {max_episode_steps} steps/ep, seed={seed})"
    )
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
            ep_len = len(obs_buf)
            ep_ret = float(sum(rew_buf))
            lengths.append(ep_len)
            returns.append(ep_ret)
            nonzero_reward_steps += sum(1 for r in rew_buf if r != 0.0)
            done_n = len(buffer)
            elapsed = time.perf_counter() - t0
            rate = done_n / max(elapsed, 1e-6)
            remaining = (num_episodes - done_n) / max(rate, 1e-6)
            pct = 100.0 * done_n / num_episodes
            log(
                f"  [{done_n:4d}/{num_episodes}] {pct:5.1f}%  "
                f"len={ep_len:3d}  ret={ep_ret:+6.2f}  "
                f"steps={buffer.num_steps:<7d}  "
                f"{rate:.2f} ep/s  elapsed {_fmt_duration(elapsed)}  "
                f"eta {_fmt_duration(remaining)}"
            )
            if done_n % 25 == 0 or done_n == num_episodes:
                mean_len = sum(lengths) / len(lengths)
                mean_ret = sum(returns) / len(returns)
                log(
                    f"  -- {done_n}/{num_episodes} checkpoint: "
                    f"mean_len={mean_len:.1f}  mean_ret={mean_ret:+.3f}  "
                    f"nonzero_reward_steps={nonzero_reward_steps}  "
                    f"total_steps={buffer.num_steps}"
                )
    finally:
        env.close()
    elapsed = time.perf_counter() - t0
    log(
        f"done: {len(buffer)} episodes, {buffer.num_steps} steps "
        f"in {_fmt_duration(elapsed)} ({len(buffer) / max(elapsed, 1e-6):.2f} ep/s)"
    )
    return buffer
