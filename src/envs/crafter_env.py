"""Gymnasium wrappers for Crafter environments.

Crafter registers `CrafterReward-v1` only with the legacy `gym` package. This
module wraps `crafter.Env` for Gymnasium and re-registers the same env IDs so
the rest of the codebase can use `gymnasium.make("CrafterReward-v1")`.
"""

from __future__ import annotations

from typing import Any, SupportsFloat

import crafter
import gymnasium as gym
import numpy as np
from gymnasium import spaces


def split_crafter_done(
    done: bool, info: dict[str, Any] | None
) -> tuple[bool, bool]:
    """Map crafter `done` to Gymnasium `(terminated, truncated)`.

    `crafter.Env` sets `done = dead or timeout` and `info['discount'] =
    1 - float(dead)`. Treating every `done` as `terminated` stores
    `continue=0` on a 10k timeout (finding 12). Death is terminated;
    time-limit is truncated so the collector can bootstrap.
    """
    if not info or "discount" not in info:
        return bool(done), False
    dead = float(info["discount"]) < 0.5
    terminated = bool(dead)
    truncated = bool(done) and not terminated
    return terminated, truncated


def _extra_legality_info(env: "crafter.Env") -> dict[str, Any]:
    """Ground-truth facing/nearby facts `training.crafter_rules` needs.

    Reads Crafter's own engine state directly (`_player`, `_world`) instead
    of decoding pixels, so this is exact, not learned. `player.facing` is the
    same `(dx, dy)` Crafter's `Player.update` adds to `pos` before every
    `do` / `place_*` / `_do_material` check; `world.nearby(pos, 1)` is the
    same call `Player._make` uses for the `nearby: [table]` requirement.
    """
    player = env._player
    world = env._world
    pos = (int(player.pos[0]), int(player.pos[1]))
    facing = (int(player.facing[0]), int(player.facing[1]))
    target = (pos[0] + facing[0], pos[1] + facing[1])
    facing_material, facing_obj = world[target]
    nearby_materials, _nearby_objs = world.nearby(pos, 1)
    return {
        "facing": facing,
        "facing_material": facing_material,
        "facing_object_present": facing_obj is not None,
        "nearby_materials": tuple(nearby_materials),
    }


class CrafterEnv(gym.Env):
    """Thin Gymnasium adapter around `crafter.Env`.

    Observation: uint8 image of shape (64, 64, 3).
    Actions: Discrete(17) matching Crafter's action set.

    `info` on both `reset` and `step` carries `inventory` (native Crafter)
    plus `facing` / `facing_material` / `facing_object_present` /
    `nearby_materials` (added here) so `training.crafter_rules` can compute
    an exact legality mask without decoding pixels (finding 39).
    """

    metadata = {"render_modes": []}

    def __init__(self, reward: bool = True, seed: int | None = None, **kwargs: Any):
        super().__init__()
        self._reward = reward
        self._env_kwargs = kwargs
        self._env = crafter.Env(reward=reward, seed=seed, **kwargs)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(64, 64, 3), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(int(self._env.action_space.n))

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            # Recreate so the underlying Crafter seed is applied.
            self._env = crafter.Env(
                reward=self._reward, seed=seed, **self._env_kwargs
            )
            self.action_space = spaces.Discrete(int(self._env.action_space.n))
        obs = self._env.reset()
        info: dict[str, Any] = {"inventory": dict(self._env._player.inventory)}
        info.update(_extra_legality_info(self._env))
        return np.asarray(obs, dtype=np.uint8), info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, done, info = self._env.step(action)
        terminated, truncated = split_crafter_done(bool(done), info if isinstance(info, dict) else None)
        if isinstance(info, dict):
            info.update(_extra_legality_info(self._env))
        return np.asarray(obs, dtype=np.uint8), float(reward), terminated, truncated, info

    def close(self) -> None:
        return None


def register_crafter_envs() -> None:
    """Register Crafter env IDs with Gymnasium (idempotent)."""
    specs = {
        "CrafterReward-v1": {"reward": True},
        "CrafterNoReward-v1": {"reward": False},
    }
    for env_id, kwargs in specs.items():
        if env_id in gym.envs.registry:
            continue
        gym.register(
            id=env_id,
            entry_point="envs.crafter_env:CrafterEnv",
            max_episode_steps=10000,
            kwargs=kwargs,
        )


register_crafter_envs()
