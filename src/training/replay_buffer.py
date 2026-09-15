"""Sequential replay buffer for RSSM world-model training.

Returns contiguous length-L chunks (not i.i.d. frames). The RSSM needs temporal
continuity; shuffling independent timesteps breaks the recurrence signal.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor

from training.crafter_rules import ACTION_DIM as CRAFTER_ACTION_DIM
from training.crafter_rules import ITEM_NAMES, inventory_vector, legal_action_mask

N_ITEMS = len(ITEM_NAMES)


@dataclass
class EpisodeBatch:
    """One stored episode (variable length)."""

    obs: Tensor  # uint8 [T, 64, 64, 3]
    actions: Tensor  # int64 [T]
    rewards: Tensor  # float32 [T]
    cont: Tensor  # float32 [T]  — 1 if episode continues after this step
    is_first: Tensor | None = None  # float32 [T] — 1 at the episode start
    inventory: Tensor | None = None  # int64 [T, N_ITEMS], fixed order ITEM_NAMES
    has_inventory: Tensor | None = None  # float32 [T] — 1 if `inventory` is real


def _episode_is_first(ep: EpisodeBatch) -> Tensor:
    """Per-step first flags; missing field means only index 0 is first."""
    t = int(ep.obs.shape[0])
    if ep.is_first is not None:
        return ep.is_first.to(dtype=torch.float32)
    flags = torch.zeros(t, dtype=torch.float32)
    if t:
        flags[0] = 1.0
    return flags


def _episode_inventory(ep: EpisodeBatch) -> tuple[Tensor, Tensor]:
    """`(inventory [T, N_ITEMS] int64, has_inventory [T] float32)`.

    Episodes loaded from a replay dump written before finding 40 (inventory
    head) have neither field — they get zeros / `has_inventory=0` so the
    inventory-head loss (Phase 3) skips them instead of training on a fake
    all-zero inventory as if it were ground truth.
    """
    t = int(ep.obs.shape[0])
    if ep.inventory is not None:
        inv = ep.inventory.to(dtype=torch.int64)
    else:
        inv = torch.zeros(t, N_ITEMS, dtype=torch.int64)
    if ep.has_inventory is not None:
        has = ep.has_inventory.to(dtype=torch.float32)
    else:
        has = torch.zeros(t, dtype=torch.float32)
    return inv, has


class ReplayBuffer:
    """Stores episodes and samples fixed-length contiguous windows.

    DreamerV3 samples subsequences across episode boundaries and resets the
    GRU where `is_first` is set. Unfinished lives are already in the buffer
    (`add_step`) so training does not wait for death. Optional FIFO
    `max_steps` drops the oldest finished episodes so host RAM cannot grow
    forever.
    """

    def __init__(self, seed: int = 0, max_steps: int | None = None) -> None:
        self._episodes: list[EpisodeBatch] = []
        self._rng = np.random.default_rng(seed)
        self._total_steps = 0
        self.max_steps = None if max_steps is None else int(max_steps)
        self._live_obs: list[np.ndarray] = []
        self._live_act: list[int] = []
        self._live_rew: list[float] = []
        self._live_cont: list[float] = []
        self._live_first: list[float] = []
        self._live_inventory: list[np.ndarray] = []
        self._live_has_inventory: list[float] = []

    def __len__(self) -> int:
        return len(self._episodes) + (1 if self._live_obs else 0)

    @property
    def num_steps(self) -> int:
        return self._total_steps + len(self._live_obs)

    def can_sample(self, seq_len: int) -> bool:
        """True when a length-`seq_len` window exists anywhere in the stream."""
        return self.num_steps >= int(seq_len)

    def _live_batch(self) -> EpisodeBatch | None:
        n = len(self._live_obs)
        if n == 0:
            return None
        return EpisodeBatch(
            obs=torch.as_tensor(np.stack(self._live_obs, axis=0), dtype=torch.uint8),
            actions=torch.as_tensor(self._live_act, dtype=torch.int64),
            rewards=torch.as_tensor(self._live_rew, dtype=torch.float32),
            cont=torch.as_tensor(self._live_cont, dtype=torch.float32),
            is_first=torch.as_tensor(self._live_first, dtype=torch.float32),
            inventory=torch.as_tensor(
                np.stack(self._live_inventory, axis=0), dtype=torch.int64
            ),
            has_inventory=torch.as_tensor(self._live_has_inventory, dtype=torch.float32),
        )

    def _parts(self) -> list[EpisodeBatch]:
        parts = list(self._episodes)
        live = self._live_batch()
        if live is not None:
            parts.append(live)
        return parts

    def _clear_live(self) -> None:
        self._live_obs = []
        self._live_act = []
        self._live_rew = []
        self._live_cont = []
        self._live_first = []
        self._live_inventory = []
        self._live_has_inventory = []

    def close_episode(self) -> None:
        """Freeze the in-progress life so FIFO eviction can drop it later."""
        live = self._live_batch()
        if live is None:
            return
        self._episodes.append(live)
        self._total_steps += int(live.obs.shape[0])
        self._clear_live()
        self._evict()

    def _evict(self) -> None:
        """Drop oldest finished episodes until `num_steps <= max_steps`.

        A single episode longer than `max_steps` is kept (we do not split).
        The live (unfinished) life is never dropped from the front.
        """
        if self.max_steps is None:
            return
        live_n = len(self._live_obs)
        while len(self._episodes) > 1 and self._total_steps + live_n > self.max_steps:
            old = self._episodes.pop(0)
            self._total_steps -= int(old.obs.shape[0])

    def add_step(
        self,
        obs: Tensor | np.ndarray,
        action: int | Tensor,
        reward: float | Tensor,
        cont: float | Tensor,
        is_first: bool,
        inventory: np.ndarray | None = None,
    ) -> None:
        """Append one transition. `is_first` starts a new life in the stream.

        `inventory` (finding 40): fixed-order `[N_ITEMS]` counts from
        `training.crafter_rules.inventory_vector`. Omit it (non-Crafter env,
        or a caller that predates this) to store zeros with
        `has_inventory=0`, which the Phase 3 inventory-head loss skips.
        """
        if bool(is_first) and self._live_obs:
            self.close_episode()
        frame = np.asarray(obs, dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(f"obs step must be [H,W,3] uint8, got {tuple(frame.shape)}")
        self._live_obs.append(frame)
        self._live_act.append(int(action))
        self._live_rew.append(float(reward))
        self._live_cont.append(float(cont))
        self._live_first.append(1.0 if bool(is_first) or not self._live_obs[:-1] else 0.0)
        if inventory is None:
            self._live_inventory.append(np.zeros(N_ITEMS, dtype=np.int64))
            self._live_has_inventory.append(0.0)
        else:
            vec = np.asarray(inventory, dtype=np.int64)
            if vec.shape != (N_ITEMS,):
                raise ValueError(f"inventory must be [{N_ITEMS}], got {tuple(vec.shape)}")
            self._live_inventory.append(vec)
            self._live_has_inventory.append(1.0)
        self._evict()

    def add_episode(
        self,
        obs: Tensor | np.ndarray,
        actions: Tensor | np.ndarray,
        rewards: Tensor | np.ndarray,
        cont: Tensor | np.ndarray,
        inventory: Tensor | np.ndarray | None = None,
    ) -> None:
        """Append one finished episode. All arrays length `T` along dim 0.

        `inventory` `[T, N_ITEMS]`: omit for `has_inventory=0` (zeros).
        """
        if self._live_obs:
            self.close_episode()
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
        first = torch.zeros(t, dtype=torch.float32)
        if t:
            first[0] = 1.0
        if inventory is None:
            inv_t = torch.zeros(t, N_ITEMS, dtype=torch.int64)
            has_inv_t = torch.zeros(t, dtype=torch.float32)
        else:
            inv_t = torch.as_tensor(np.asarray(inventory), dtype=torch.int64)
            if inv_t.shape != (t, N_ITEMS):
                raise ValueError(f"inventory must be [{t},{N_ITEMS}], got {tuple(inv_t.shape)}")
            has_inv_t = torch.ones(t, dtype=torch.float32)
        self._episodes.append(
            EpisodeBatch(
                obs=obs_t,
                actions=act_t,
                rewards=rew_t,
                cont=cont_t,
                is_first=first,
                inventory=inv_t,
                has_inventory=has_inv_t,
            )
        )
        self._total_steps += t
        self._evict()

    def _gather(
        self, parts: list[EpisodeBatch], start: int, seq_len: int
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        lengths = [int(ep.obs.shape[0]) for ep in parts]
        idx = 0
        offset = start
        while offset >= lengths[idx]:
            offset -= lengths[idx]
            idx += 1
        obs_c: list[Tensor] = []
        act_c: list[Tensor] = []
        rew_c: list[Tensor] = []
        cont_c: list[Tensor] = []
        first_c: list[Tensor] = []
        inv_c: list[Tensor] = []
        has_inv_c: list[Tensor] = []
        remaining = seq_len
        while remaining > 0:
            ep = parts[idx]
            take = min(remaining, lengths[idx] - offset)
            end = offset + take
            obs_c.append(ep.obs[offset:end])
            act_c.append(ep.actions[offset:end])
            rew_c.append(ep.rewards[offset:end])
            cont_c.append(ep.cont[offset:end])
            first_c.append(_episode_is_first(ep)[offset:end])
            inv, has_inv = _episode_inventory(ep)
            inv_c.append(inv[offset:end])
            has_inv_c.append(has_inv[offset:end])
            remaining -= take
            idx += 1
            offset = 0
        return (
            torch.cat(obs_c, dim=0),
            torch.cat(act_c, dim=0),
            torch.cat(rew_c, dim=0),
            torch.cat(cont_c, dim=0),
            torch.cat(first_c, dim=0),
            torch.cat(inv_c, dim=0),
            torch.cat(has_inv_c, dim=0),
        )

    def sample(self, batch_size: int, seq_len: int) -> dict[str, Tensor]:
        """Sample contiguous windows, including across episode boundaries.

        Returns dict with:
            obs `[B, L, 64, 64, 3]` uint8
            actions `[B, L]` int64
            rewards `[B, L]` float32
            cont `[B, L]` float32
            is_first `[B, L]` float32
            inventory `[B, L, N_ITEMS]` int64 (finding 40; zeros where stale)
            has_inventory `[B, L]` float32 — 1 where `inventory` is real
        """
        if not self.can_sample(seq_len):
            raise RuntimeError(
                f"need {seq_len} stream steps to sample, have {self.num_steps} "
                f"({len(self._episodes)} finished episodes)"
            )
        parts = self._parts()
        total = self.num_steps
        n_starts = total - int(seq_len) + 1
        obs_list: list[Tensor] = []
        act_list: list[Tensor] = []
        rew_list: list[Tensor] = []
        cont_list: list[Tensor] = []
        first_list: list[Tensor] = []
        inv_list: list[Tensor] = []
        has_inv_list: list[Tensor] = []
        for _ in range(batch_size):
            start = int(self._rng.integers(0, n_starts))
            obs, act, rew, cont, first, inv, has_inv = self._gather(parts, start, int(seq_len))
            obs_list.append(obs)
            act_list.append(act)
            rew_list.append(rew)
            cont_list.append(cont)
            first_list.append(first)
            inv_list.append(inv)
            has_inv_list.append(has_inv)
        return {
            "obs": torch.stack(obs_list, dim=0),
            "actions": torch.stack(act_list, dim=0),
            "rewards": torch.stack(rew_list, dim=0),
            "cont": torch.stack(cont_list, dim=0),
            "is_first": torch.stack(first_list, dim=0),
            "inventory": torch.stack(inv_list, dim=0),
            "has_inventory": torch.stack(has_inv_list, dim=0),
        }

    def state_dict(self) -> dict:
        if self._live_obs:
            self.close_episode()
        rows = []
        for ep in self._episodes:
            inv, has_inv = _episode_inventory(ep)
            rows.append(
                {
                    "obs": ep.obs,
                    "actions": ep.actions,
                    "rewards": ep.rewards,
                    "cont": ep.cont,
                    "is_first": _episode_is_first(ep),
                    "inventory": inv,
                    "has_inventory": has_inv,
                }
            )
        return {"episodes": rows, "total_steps": self._total_steps}

    def load_state_dict(self, state: dict) -> None:
        """Backward-compatible: dumps written before finding 40 have no
        `inventory` / `has_inventory` keys. Those episodes load with zeros /
        `has_inventory=0` rather than raising, so old replay (e.g. the M17
        500k dump) keeps loading — the inventory head just skips them.
        """
        self._clear_live()
        loaded: list[EpisodeBatch] = []
        for item in state["episodes"]:
            obs = torch.as_tensor(item["obs"], dtype=torch.uint8)
            t = int(obs.shape[0])
            first = item.get("is_first")
            if first is None:
                flags = torch.zeros(t, dtype=torch.float32)
                if t:
                    flags[0] = 1.0
            else:
                flags = torch.as_tensor(first, dtype=torch.float32)
            inv = item.get("inventory")
            has_inv = item.get("has_inventory")
            inv_t = (
                torch.as_tensor(inv, dtype=torch.int64)
                if inv is not None
                else torch.zeros(t, N_ITEMS, dtype=torch.int64)
            )
            has_inv_t = (
                torch.as_tensor(has_inv, dtype=torch.float32)
                if has_inv is not None
                else torch.zeros(t, dtype=torch.float32)
            )
            loaded.append(
                EpisodeBatch(
                    obs=obs,
                    actions=torch.as_tensor(item["actions"], dtype=torch.int64),
                    rewards=torch.as_tensor(item["rewards"], dtype=torch.float32),
                    cont=torch.as_tensor(item["cont"], dtype=torch.float32),
                    is_first=flags,
                    inventory=inv_t,
                    has_inventory=has_inv_t,
                )
            )
        self._episodes = loaded
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
            obs, info = env.reset(seed=seed + ep)
            obs_buf: list = []
            act_buf: list[int] = []
            rew_buf: list[float] = []
            cont_buf: list[float] = []
            inv_buf: list[np.ndarray] = []
            for _ in range(max_episode_steps):
                # Strictly uniform random (M3 baseline, frozen across
                # m6/m16/m17) — no legality mask here. See
                # `prefill_random_steps(mask_illegal=...)` for the masked
                # variant used by the m18 deviation.
                action = int(env.action_space.sample())
                obs_buf.append(np.asarray(obs, dtype=np.uint8))
                act_buf.append(action)
                inv_buf.append(
                    inventory_vector(info.get("inventory") if isinstance(info, dict) else None)
                )
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                rew_buf.append(float(reward))
                # Continue = not terminated. Truncation still counts as continue=1
                # for bootstrap semantics; we still break the episode storage on either.
                cont_buf.append(0.0 if terminated else 1.0)
                obs = next_obs
                if done:
                    break
            if len(obs_buf) == 0:
                continue
            buffer.add_episode(obs_buf, act_buf, rew_buf, cont_buf, inventory=inv_buf)
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


def _sample_action(env: Any, info: dict[str, Any] | None, mask_illegal: bool) -> int:
    space = env.action_space
    n = int(space.n)
    if mask_illegal and n == CRAFTER_ACTION_DIM:
        legal = np.flatnonzero(legal_action_mask(info))
        if legal.size > 0:
            return int(np.random.choice(legal))
    if hasattr(space, "sample"):
        return int(space.sample())
    return int(np.random.randint(0, n))


def prefill_random_steps(
    env: Any,
    buffer: ReplayBuffer,
    *,
    steps: int,
    max_episode_steps: int,
    seq_len: int,
    seed: int = 0,
    mask_illegal: bool = False,
) -> int:
    """Uniform-random actions until the buffer holds `steps` transitions.

    DreamerV3-torch `prefill: 2500` (defaults). Stops on step count, not on
    the first episode longer than `seq_len`. Flushes a trailing partial life
    if it is at least `seq_len` frames. Returns env steps taken this call.

    `mask_illegal=False` (default) is the faithful-Dreamer path used by
    m6/m16/m17 — pure uniform random, unchanged. `mask_illegal=True` is the
    m18 deviation (finding 39/40): sample uniformly only over
    `crafter_rules.legal_action_mask(info)` on the real 17-action Crafter
    task, so prefill never wastes a step pressing an illegal `place_*`/
    `make_*`. It does not make movement purposeful — see finding 40 for why
    that was deliberately left out of scope.
    """
    target = int(steps)
    seq_len = int(seq_len)
    cap = int(max_episode_steps)
    if target <= 0:
        return 0
    if buffer.can_sample(seq_len) and buffer.num_steps >= target:
        print(
            f"prefill skip: replay already {buffer.num_steps} steps "
            f"(need {target}, seq_len={seq_len})",
            flush=True,
        )
        return 0

    got = 0
    ep_i = 0
    while buffer.num_steps < target or not buffer.can_sample(seq_len):
        obs, info = env.reset(seed=int(seed) + ep_i)
        ep_i += 1
        obs_buf: list = []
        act_buf: list[int] = []
        rew_buf: list[float] = []
        cont_buf: list[float] = []
        inv_buf: list[np.ndarray] = []
        for _ in range(cap):
            action = _sample_action(env, info if isinstance(info, dict) else None, mask_illegal)
            obs_buf.append(np.asarray(obs, dtype=np.uint8))
            act_buf.append(action)
            inv_buf.append(
                inventory_vector(info.get("inventory") if isinstance(info, dict) else None)
            )
            next_obs, reward, terminated, truncated, info = env.step(action)
            terminated = bool(terminated)
            truncated = bool(truncated)
            rew_buf.append(float(reward))
            cont_buf.append(0.0 if terminated else 1.0)
            obs = next_obs
            got += 1
            if terminated or truncated or len(obs_buf) >= cap:
                break
            if buffer.num_steps + len(obs_buf) >= target and buffer.num_steps + len(
                obs_buf
            ) >= seq_len:
                break
        if len(obs_buf) == 0:
            continue
        buffer.add_episode(obs_buf, act_buf, rew_buf, cont_buf, inventory=inv_buf)
        if got >= target * 4:
            break

    if not buffer.can_sample(seq_len):
        raise RuntimeError(
            f"prefill {got} env steps produced only {buffer.num_steps} stream "
            f"steps (need seq_len={seq_len})"
        )
    print(
        f"prefill {got} random env steps  episodes={len(buffer)} "
        f"steps={buffer.num_steps} (target {target})",
        flush=True,
    )
    return got
