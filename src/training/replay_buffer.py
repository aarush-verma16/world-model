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
from training.crafter_rules import (
    ITEM_NAMES,
    LOCAL_CELLS,
    MATERIAL_NAMES,
    inventory_vector,
    legal_action_mask,
    local_map_from_info,
    spatial_from_info,
)

N_ITEMS = len(ITEM_NAMES)
N_MATERIALS = len(MATERIAL_NAMES)
LocalMap = tuple[np.ndarray, np.ndarray]


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
    facing: Tensor | None = None  # int64 [T] material id
    facing_object: Tensor | None = None  # float32 [T] — 1 if the faced tile is occupied
    nearby: Tensor | None = None  # float32 [T, N_MATERIALS] multi-hot
    has_spatial: Tensor | None = None  # float32 [T] — 1 if facing/nearby are real
    local_mat: Tensor | None = None  # uint8 [T, LOCAL_CELLS] material codes (finding 47)
    local_obj: Tensor | None = None  # uint8 [T, LOCAL_CELLS] object codes
    has_local: Tensor | None = None  # float32 [T] — 1 if the local map is real
    teacher: bool = False  # map-teacher episode (finding 46); excluded from the score


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


def _episode_spatial(ep: EpisodeBatch) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """`(facing [T], facing_object [T], nearby [T, N_MATERIALS], has_spatial [T])`.

    Replay written before finding 45 has none of these fields. Those steps
    get `has_spatial=0` so `spatial_head_loss` skips them.
    """
    t = int(ep.obs.shape[0])
    facing = (
        ep.facing.to(dtype=torch.int64)
        if ep.facing is not None
        else torch.zeros(t, dtype=torch.int64)
    )
    facing_object = (
        ep.facing_object.to(dtype=torch.float32)
        if ep.facing_object is not None
        else torch.zeros(t, dtype=torch.float32)
    )
    nearby = (
        ep.nearby.to(dtype=torch.float32)
        if ep.nearby is not None
        else torch.zeros(t, N_MATERIALS, dtype=torch.float32)
    )
    has_spatial = (
        ep.has_spatial.to(dtype=torch.float32)
        if ep.has_spatial is not None
        else torch.zeros(t, dtype=torch.float32)
    )
    return facing, facing_object, nearby, has_spatial


def _episode_local(ep: EpisodeBatch) -> tuple[Tensor, Tensor, Tensor]:
    """`(local_mat [T, C] uint8, local_obj [T, C] uint8, has_local [T])`.

    Replay written before finding 47 has no local map; those steps get
    `has_local=0` so `local_map_head_loss` skips them.
    """
    t = int(ep.obs.shape[0])
    mat = (
        ep.local_mat.to(dtype=torch.uint8)
        if ep.local_mat is not None
        else torch.zeros(t, LOCAL_CELLS, dtype=torch.uint8)
    )
    obj = (
        ep.local_obj.to(dtype=torch.uint8)
        if ep.local_obj is not None
        else torch.zeros(t, LOCAL_CELLS, dtype=torch.uint8)
    )
    has = (
        ep.has_local.to(dtype=torch.float32)
        if ep.has_local is not None
        else torch.zeros(t, dtype=torch.float32)
    )
    return mat, obj, has


def _teacher_flag(value: Any) -> bool:
    """`state_dict` stores a bool; tolerate a per-step tensor too."""
    if isinstance(value, Tensor):
        return bool(value.numel() and float(value.reshape(-1)[0]) > 0.5)
    return bool(value)


def _pack_local(rows: list[LocalMap | None]) -> tuple[np.ndarray, np.ndarray] | None:
    """Stack per-step local maps. `None` if any step lacked one."""
    if not rows or any(row is None for row in rows):
        return None
    mats = np.stack([row[0] for row in rows], axis=0).astype(np.uint8)
    objs = np.stack([row[1] for row in rows], axis=0).astype(np.uint8)
    return mats, objs


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
        self._live_facing: list[int] = []
        self._live_facing_object: list[float] = []
        self._live_nearby: list[np.ndarray] = []
        self._live_has_spatial: list[float] = []
        self._live_local_mat: list[np.ndarray] = []
        self._live_local_obj: list[np.ndarray] = []
        self._live_has_local: list[float] = []
        self._teacher_starts_key: tuple[int, int, int, int] | None = None
        self._teacher_starts: np.ndarray | None = None

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
            facing=torch.as_tensor(self._live_facing, dtype=torch.int64),
            facing_object=torch.as_tensor(self._live_facing_object, dtype=torch.float32),
            nearby=torch.as_tensor(np.stack(self._live_nearby, axis=0), dtype=torch.float32),
            has_spatial=torch.as_tensor(self._live_has_spatial, dtype=torch.float32),
            local_mat=torch.as_tensor(np.stack(self._live_local_mat, axis=0), dtype=torch.uint8),
            local_obj=torch.as_tensor(np.stack(self._live_local_obj, axis=0), dtype=torch.uint8),
            has_local=torch.as_tensor(self._live_has_local, dtype=torch.float32),
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
        self._live_facing = []
        self._live_facing_object = []
        self._live_nearby = []
        self._live_has_spatial = []
        self._live_local_mat = []
        self._live_local_obj = []
        self._live_has_local = []

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
            # Teacher episodes stay. They are the only stone transitions until
            # the actor mines on its own, and a 1e6 FIFO would delete them.
            idx = next((i for i, ep in enumerate(self._episodes) if not ep.teacher), None)
            if idx is None:
                break
            old = self._episodes.pop(idx)
            self._total_steps -= int(old.obs.shape[0])

    def add_step(
        self,
        obs: Tensor | np.ndarray,
        action: int | Tensor,
        reward: float | Tensor,
        cont: float | Tensor,
        is_first: bool,
        inventory: np.ndarray | None = None,
        spatial: tuple[int, float, np.ndarray] | None = None,
        local: LocalMap | None = None,
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
        self._append_spatial(spatial)
        self._append_local(local)
        self._evict()

    def _append_local(self, local: LocalMap | None) -> None:
        if local is None:
            self._live_local_mat.append(np.zeros(LOCAL_CELLS, dtype=np.uint8))
            self._live_local_obj.append(np.zeros(LOCAL_CELLS, dtype=np.uint8))
            self._live_has_local.append(0.0)
            return
        mats = np.asarray(local[0], dtype=np.uint8).reshape(-1)
        objs = np.asarray(local[1], dtype=np.uint8).reshape(-1)
        if mats.shape != (LOCAL_CELLS,) or objs.shape != (LOCAL_CELLS,):
            raise ValueError(f"local map must be [{LOCAL_CELLS}] x2, got {mats.shape} {objs.shape}")
        self._live_local_mat.append(mats)
        self._live_local_obj.append(objs)
        self._live_has_local.append(1.0)

    def _append_spatial(self, spatial: tuple[int, float, np.ndarray] | None) -> None:
        if spatial is None:
            self._live_facing.append(0)
            self._live_facing_object.append(0.0)
            self._live_nearby.append(np.zeros(N_MATERIALS, dtype=np.float32))
            self._live_has_spatial.append(0.0)
            return
        facing_id, occupied, nearby = spatial
        near = np.asarray(nearby, dtype=np.float32)
        if near.shape != (N_MATERIALS,):
            raise ValueError(f"nearby must be [{N_MATERIALS}], got {tuple(near.shape)}")
        self._live_facing.append(int(facing_id))
        self._live_facing_object.append(float(occupied))
        self._live_nearby.append(near)
        self._live_has_spatial.append(1.0)

    def add_episode(
        self,
        obs: Tensor | np.ndarray,
        actions: Tensor | np.ndarray,
        rewards: Tensor | np.ndarray,
        cont: Tensor | np.ndarray,
        inventory: Tensor | np.ndarray | None = None,
        spatial: tuple[Tensor | np.ndarray, Tensor | np.ndarray, Tensor | np.ndarray] | None = None,
        teacher: bool = False,
        local: tuple[Tensor | np.ndarray, Tensor | np.ndarray] | None = None,
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
        if spatial is None:
            facing_t = torch.zeros(t, dtype=torch.int64)
            obj_t = torch.zeros(t, dtype=torch.float32)
            near_t = torch.zeros(t, N_MATERIALS, dtype=torch.float32)
            has_sp_t = torch.zeros(t, dtype=torch.float32)
        else:
            facing_t = torch.as_tensor(np.asarray(spatial[0]), dtype=torch.int64)
            obj_t = torch.as_tensor(np.asarray(spatial[1]), dtype=torch.float32)
            near_t = torch.as_tensor(np.asarray(spatial[2]), dtype=torch.float32)
            if facing_t.shape != (t,) or obj_t.shape != (t,) or near_t.shape != (t, N_MATERIALS):
                raise ValueError(
                    f"spatial must be [{t}], [{t}], [{t},{N_MATERIALS}], "
                    f"got {tuple(facing_t.shape)} {tuple(obj_t.shape)} {tuple(near_t.shape)}"
                )
            has_sp_t = torch.ones(t, dtype=torch.float32)
        if local is None:
            lmat_t = torch.zeros(t, LOCAL_CELLS, dtype=torch.uint8)
            lobj_t = torch.zeros(t, LOCAL_CELLS, dtype=torch.uint8)
            has_l_t = torch.zeros(t, dtype=torch.float32)
        else:
            lmat_t = torch.as_tensor(np.asarray(local[0]), dtype=torch.uint8)
            lobj_t = torch.as_tensor(np.asarray(local[1]), dtype=torch.uint8)
            if lmat_t.shape != (t, LOCAL_CELLS) or lobj_t.shape != (t, LOCAL_CELLS):
                raise ValueError(
                    f"local map must be [{t},{LOCAL_CELLS}] x2, "
                    f"got {tuple(lmat_t.shape)} {tuple(lobj_t.shape)}"
                )
            has_l_t = torch.ones(t, dtype=torch.float32)
        self._episodes.append(
            EpisodeBatch(
                obs=obs_t,
                actions=act_t,
                rewards=rew_t,
                cont=cont_t,
                is_first=first,
                inventory=inv_t,
                has_inventory=has_inv_t,
                facing=facing_t,
                facing_object=obj_t,
                nearby=near_t,
                has_spatial=has_sp_t,
                local_mat=lmat_t,
                local_obj=lobj_t,
                has_local=has_l_t,
                teacher=bool(teacher),
            )
        )
        self._total_steps += t
        self._evict()

    @staticmethod
    def _episode_fields(ep: EpisodeBatch) -> dict[str, Tensor]:
        """Every per-step replay field of one episode, keyed like `sample`."""
        inv, has_inv = _episode_inventory(ep)
        facing, facing_object, nearby, has_spatial = _episode_spatial(ep)
        local_mat, local_obj, has_local = _episode_local(ep)
        t = int(ep.obs.shape[0])
        return {
            "obs": ep.obs,
            "actions": ep.actions,
            "rewards": ep.rewards,
            "cont": ep.cont,
            "is_first": _episode_is_first(ep),
            "inventory": inv,
            "has_inventory": has_inv,
            "facing": facing,
            "facing_object": facing_object,
            "nearby": nearby,
            "has_spatial": has_spatial,
            "local_mat": local_mat,
            "local_obj": local_obj,
            "has_local": has_local,
            "teacher": torch.full((t,), 1.0 if ep.teacher else 0.0, dtype=torch.float32),
        }

    def _gather(self, parts: list[EpisodeBatch], start: int, seq_len: int) -> dict[str, Tensor]:
        """One length-`seq_len` window starting at global stream index `start`."""
        lengths = [int(ep.obs.shape[0]) for ep in parts]
        idx = 0
        offset = start
        while offset >= lengths[idx]:
            offset -= lengths[idx]
            idx += 1
        chunks: dict[str, list[Tensor]] = {}
        remaining = seq_len
        while remaining > 0:
            ep = parts[idx]
            take = min(remaining, lengths[idx] - offset)
            end = offset + take
            for key, value in self._episode_fields(ep).items():
                chunks.setdefault(key, []).append(value[offset:end])
            remaining -= take
            idx += 1
            offset = 0
        return {key: torch.cat(values, dim=0) for key, values in chunks.items()}

    def _teacher_overlap_starts(self, seq_len: int) -> np.ndarray:
        """Window starts whose span overlaps a teacher episode.

        Teacher lives are short (~30 steps) and `seq_len` is 64, so the
        window is allowed to spill into the neighboring agent episode.
        The spill is what keeps the stone transition inside a legal sample.
        """
        seq_len = int(seq_len)
        key = (seq_len, len(self._episodes), int(self._total_steps), len(self._live_obs))
        if self._teacher_starts_key == key and self._teacher_starts is not None:
            return self._teacher_starts
        n_starts = self.num_steps - seq_len + 1
        if n_starts <= 0:
            empty = np.zeros(0, dtype=np.int64)
            self._teacher_starts_key = key
            self._teacher_starts = empty
            return empty
        cursor = 0
        starts: list[int] = []
        for ep in self._parts():
            length = int(ep.obs.shape[0])
            if ep.teacher:
                lo = max(0, cursor - seq_len + 1)
                hi = min(n_starts, cursor + length)
                starts.extend(range(lo, hi))
            cursor += length
        if not starts:
            result = np.zeros(0, dtype=np.int64)
        else:
            result = np.unique(np.asarray(starts, dtype=np.int64))
        self._teacher_starts_key = key
        self._teacher_starts = result
        return result

    def sample(
        self,
        batch_size: int,
        seq_len: int,
        *,
        teacher_fraction: float = 0.0,
    ) -> dict[str, Tensor]:
        """Sample contiguous windows, including across episode boundaries.

        Returns dict with:
            obs `[B, L, 64, 64, 3]` uint8
            actions `[B, L]` int64
            rewards `[B, L]` float32
            cont `[B, L]` float32
            is_first `[B, L]` float32
            inventory `[B, L, N_ITEMS]` int64 (finding 40; zeros where stale)
            has_inventory `[B, L]` float32 — 1 where `inventory` is real
            facing `[B, L]` int64, facing_object `[B, L]` float32,
            nearby `[B, L, N_MATERIALS]` float32, has_spatial `[B, L]` float32
            (finding 45; zeros / 0 where the replay predates the spatial head)
            local_mat / local_obj `[B, L, LOCAL_CELLS]` uint8, has_local `[B, L]`
            (finding 47)
            teacher `[B, L]` float32 — 1 on steps from a map-teacher episode
        """
        if not self.can_sample(seq_len):
            raise RuntimeError(
                f"need {seq_len} stream steps to sample, have {self.num_steps} "
                f"({len(self._episodes)} finished episodes)"
            )
        parts = self._parts()
        total = self.num_steps
        n_starts = total - int(seq_len) + 1
        frac = float(teacher_fraction)
        overlap = self._teacher_overlap_starts(int(seq_len)) if frac > 0.0 else None
        rows: list[dict[str, Tensor]] = []
        for _ in range(batch_size):
            if overlap is not None and overlap.size and float(self._rng.random()) < frac:
                start = int(self._rng.choice(overlap))
            else:
                start = int(self._rng.integers(0, n_starts))
            rows.append(self._gather(parts, start, int(seq_len)))
        return {key: torch.stack([row[key] for row in rows], dim=0) for key in rows[0]}

    def state_dict(self) -> dict:
        if self._live_obs:
            self.close_episode()
        rows = []
        for ep in self._episodes:
            row = self._episode_fields(ep)
            row["teacher"] = bool(ep.teacher)
            rows.append(row)
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
            facing = item.get("facing")
            facing_object = item.get("facing_object")
            nearby = item.get("nearby")
            has_spatial = item.get("has_spatial")
            loaded.append(
                EpisodeBatch(
                    obs=obs,
                    actions=torch.as_tensor(item["actions"], dtype=torch.int64),
                    rewards=torch.as_tensor(item["rewards"], dtype=torch.float32),
                    cont=torch.as_tensor(item["cont"], dtype=torch.float32),
                    is_first=flags,
                    inventory=inv_t,
                    has_inventory=has_inv_t,
                    facing=(
                        torch.as_tensor(facing, dtype=torch.int64)
                        if facing is not None
                        else torch.zeros(t, dtype=torch.int64)
                    ),
                    facing_object=(
                        torch.as_tensor(facing_object, dtype=torch.float32)
                        if facing_object is not None
                        else torch.zeros(t, dtype=torch.float32)
                    ),
                    nearby=(
                        torch.as_tensor(nearby, dtype=torch.float32)
                        if nearby is not None
                        else torch.zeros(t, N_MATERIALS, dtype=torch.float32)
                    ),
                    has_spatial=(
                        torch.as_tensor(has_spatial, dtype=torch.float32)
                        if has_spatial is not None
                        else torch.zeros(t, dtype=torch.float32)
                    ),
                    local_mat=(
                        torch.as_tensor(item["local_mat"], dtype=torch.uint8)
                        if item.get("local_mat") is not None
                        else torch.zeros(t, LOCAL_CELLS, dtype=torch.uint8)
                    ),
                    local_obj=(
                        torch.as_tensor(item["local_obj"], dtype=torch.uint8)
                        if item.get("local_obj") is not None
                        else torch.zeros(t, LOCAL_CELLS, dtype=torch.uint8)
                    ),
                    has_local=(
                        torch.as_tensor(item["has_local"], dtype=torch.float32)
                        if item.get("has_local") is not None
                        else torch.zeros(t, dtype=torch.float32)
                    ),
                    teacher=_teacher_flag(item.get("teacher", False)),
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


def _pack_spatial(
    rows: list[tuple[int, float, np.ndarray] | None],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Stack per-step spatial tuples. `None` if any step lacked ground truth."""
    if not rows or any(row is None for row in rows):
        return None
    facing = np.asarray([row[0] for row in rows], dtype=np.int64)
    occupied = np.asarray([row[1] for row in rows], dtype=np.float32)
    nearby = np.stack([row[2] for row in rows], axis=0).astype(np.float32)
    return facing, occupied, nearby


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
            spatial_buf: list[tuple[int, float, np.ndarray] | None] = []
            local_buf: list[LocalMap | None] = []
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
                spatial_buf.append(
                    spatial_from_info(info if isinstance(info, dict) else None)
                )
                local_buf.append(local_map_from_info(info if isinstance(info, dict) else None))
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
            buffer.add_episode(
                obs_buf,
                act_buf,
                rew_buf,
                cont_buf,
                inventory=inv_buf,
                spatial=_pack_spatial(spatial_buf),
                local=_pack_local(local_buf),
            )
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
        spatial_buf: list[tuple[int, float, np.ndarray] | None] = []
        local_buf: list[LocalMap | None] = []
        for _ in range(cap):
            action = _sample_action(env, info if isinstance(info, dict) else None, mask_illegal)
            obs_buf.append(np.asarray(obs, dtype=np.uint8))
            act_buf.append(action)
            inv_buf.append(
                inventory_vector(info.get("inventory") if isinstance(info, dict) else None)
            )
            spatial_buf.append(spatial_from_info(info if isinstance(info, dict) else None))
            local_buf.append(local_map_from_info(info if isinstance(info, dict) else None))
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
        buffer.add_episode(
            obs_buf,
            act_buf,
            rew_buf,
            cont_buf,
            inventory=inv_buf,
            spatial=_pack_spatial(spatial_buf),
            local=_pack_local(local_buf),
        )
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
