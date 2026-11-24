"""Replay for the room. Windows stay inside one episode.

The Crafter buffer is 63 cells and 17 actions. This one is 16 cells and
anchors on `use` (action 4) when asked. Crafter sampling is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


@dataclass
class _Episode:
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    cont: np.ndarray
    local_mat: np.ndarray
    local_obj: np.ndarray
    teacher: bool


class RoomReplay:
    """In-memory episodes. `sample` returns the batch `world_model_step` reads."""

    def __init__(self, *, cells: int, craft_actions: tuple[int, ...], seed: int = 0) -> None:
        self.cells = int(cells)
        self.craft_actions = frozenset(int(a) for a in craft_actions)
        self._rng = np.random.default_rng(int(seed))
        self._episodes: list[_Episode] = []

    def __len__(self) -> int:
        return len(self._episodes)

    def add_episode(
        self,
        obs: list[np.ndarray],
        actions: list[int],
        rewards: list[float],
        cont: list[float],
        local_mat: list[np.ndarray],
        local_obj: list[np.ndarray],
        *,
        teacher: bool,
    ) -> None:
        if not obs:
            return
        episode = _Episode(
            obs=np.stack(obs).astype(np.uint8),
            actions=np.asarray(actions, dtype=np.int64),
            rewards=np.asarray(rewards, dtype=np.float32),
            cont=np.asarray(cont, dtype=np.float32),
            local_mat=np.stack(local_mat).astype(np.uint8),
            local_obj=np.stack(local_obj).astype(np.uint8),
            teacher=bool(teacher),
        )
        if episode.local_mat.shape[1] != self.cells:
            raise ValueError(
                f"local map has {episode.local_mat.shape[1]} cells, buffer expects {self.cells}"
            )
        self._episodes.append(episode)

    def can_sample(self, seq_len: int) -> bool:
        return any(int(ep.obs.shape[0]) >= int(seq_len) for ep in self._episodes)

    def sample(
        self,
        batch_size: int,
        seq_len: int,
        *,
        teacher_fraction: float = 0.0,
        craft_window_fraction: float = 0.0,
    ) -> dict[str, Tensor]:
        """`obs` `[B, L, 64, 64, 3]` uint8, `actions` `[B, L]` int64, and the field."""
        seq_len = int(seq_len)
        if not self.can_sample(seq_len):
            raise RuntimeError(f"need an episode of length {seq_len}")
        teacher_pool = self._starts(seq_len, teacher_only=True, craft=False)
        craft_pool = self._starts(seq_len, teacher_only=True, craft=True)
        any_pool = self._starts(seq_len, teacher_only=False, craft=False)
        rows = [
            self._gather(*self._pick(teacher_pool, craft_pool, any_pool, teacher_fraction, craft_window_fraction), seq_len)
            for _ in range(int(batch_size))
        ]
        keys = rows[0].keys()
        return {key: torch.stack([row[key] for row in rows], dim=0) for key in keys}

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            [
                {
                    "obs": ep.obs,
                    "actions": ep.actions,
                    "rewards": ep.rewards,
                    "cont": ep.cont,
                    "local_mat": ep.local_mat,
                    "local_obj": ep.local_obj,
                    "teacher": ep.teacher,
                }
                for ep in self._episodes
            ],
            path,
        )

    def load(self, path) -> None:
        rows = torch.load(path, weights_only=False, map_location="cpu")
        self._episodes = [
            _Episode(
                obs=row["obs"],
                actions=row["actions"],
                rewards=row["rewards"],
                cont=row["cont"],
                local_mat=row["local_mat"],
                local_obj=row["local_obj"],
                teacher=bool(row["teacher"]),
            )
            for row in rows
        ]

    def _pick(
        self,
        teacher_pool: list[tuple[int, int]],
        craft_pool: list[tuple[int, int]],
        any_pool: list[tuple[int, int]],
        teacher_fraction: float,
        craft_window_fraction: float,
    ) -> tuple[int, int]:
        use_teacher = bool(teacher_pool) and float(self._rng.random()) < float(teacher_fraction)
        if use_teacher:
            use_craft = bool(craft_pool) and float(self._rng.random()) < float(craft_window_fraction)
            pool = craft_pool if use_craft else teacher_pool
        else:
            pool = any_pool
        choice = int(self._rng.integers(0, len(pool)))
        return pool[choice]

    def _starts(self, seq_len: int, *, teacher_only: bool, craft: bool) -> list[tuple[int, int]]:
        starts: list[tuple[int, int]] = []
        for index, episode in enumerate(self._episodes):
            if teacher_only and not episode.teacher:
                continue
            length = int(episode.actions.shape[0])
            if length < seq_len:
                continue
            if craft:
                for t, action in enumerate(episode.actions.tolist()):
                    if int(action) not in self.craft_actions:
                        continue
                    start = t - seq_len + 1
                    if start < 0:
                        start = 0
                    if start + seq_len <= length and start <= t <= start + seq_len - 1:
                        starts.append((index, start))
            else:
                starts.extend((index, start) for start in range(length - seq_len + 1))
        return starts

    def _gather(self, index: int, start: int, seq_len: int) -> dict[str, Tensor]:
        episode = self._episodes[index]
        stop = start + seq_len
        first = np.zeros(seq_len, dtype=np.float32)
        if start == 0:
            first[0] = 1.0
        teacher = np.full(seq_len, 1.0 if episode.teacher else 0.0, dtype=np.float32)
        has_local = np.ones(seq_len, dtype=np.float32)
        return {
            "obs": torch.from_numpy(episode.obs[start:stop].copy()),
            "actions": torch.from_numpy(episode.actions[start:stop].copy()),
            "rewards": torch.from_numpy(episode.rewards[start:stop].copy()),
            "cont": torch.from_numpy(episode.cont[start:stop].copy()),
            "is_first": torch.from_numpy(first),
            "local_mat": torch.from_numpy(episode.local_mat[start:stop].copy()),
            "local_obj": torch.from_numpy(episode.local_obj[start:stop].copy()),
            "has_local": torch.from_numpy(has_local),
            "teacher": torch.from_numpy(teacher),
        }
