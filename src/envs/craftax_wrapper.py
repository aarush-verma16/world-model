"""Stateful Gymnasium-like wrapper around Craftax-Classic (Pixels).

Craftax's native API is functional/JAX-style:
    obs, state = env.reset(key, params)
    obs, state, reward, done, info = env.step(key, state, action, params)

This wrapper owns the PRNG key and env state so the rest of the codebase can
call reset()/step(action) the same way as the old Gymnasium Crafter wrapper.

Observations are resized from Craftax-Classic's native resolution to 64x64x3
uint8 so encoder/decoder/RSSM code can stay unchanged.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from craftax.craftax_env import make_craftax_env_from_name
from PIL import Image


ENV_NAME = "Craftax-Classic-Pixels-v1"
TARGET_OBS_SHAPE = (64, 64, 3)


class CraftaxClassicPixelsEnv:
    """Thin stateful adapter around Craftax-Classic-Pixels-v1 (JAX, CPU)."""

    def __init__(self, seed: int = 0):
        # CPU-only JAX is intentional: see PROJECT_BRIEF (no jax-metal).
        self._env = make_craftax_env_from_name(ENV_NAME, auto_reset=True)
        self._params = self._env.default_params
        self._key = jax.random.PRNGKey(int(seed))
        self._state = None
        self._printed_raw_shape = False
        self._action_space_n = int(self._env.action_space(self._params).n)

    @property
    def action_space_n(self) -> int:
        """Number of discrete actions (Craftax-Classic matches Crafter: 17)."""
        return self._action_space_n

    def _split_key(self) -> jax.Array:
        self._key, subkey = jax.random.split(self._key)
        return subkey

    def _to_numpy_obs(self, obs: jax.Array) -> np.ndarray:
        """Convert JAX obs to numpy uint8 (64, 64, 3), printing raw shape once."""
        raw = np.asarray(obs)
        if not self._printed_raw_shape:
            print(f"Craftax-Classic raw observation shape (pre-resize): {raw.shape}")
            self._printed_raw_shape = True

        # Native Classic-Pixels obs is float32 in [0, 1].
        if np.issubdtype(raw.dtype, np.floating):
            frame = np.clip(raw, 0.0, 1.0)
            frame_u8 = (frame * 255.0).astype(np.uint8)
        else:
            frame_u8 = raw.astype(np.uint8)

        if frame_u8.shape != TARGET_OBS_SHAPE:
            image = Image.fromarray(frame_u8, mode="RGB")
            image = image.resize((TARGET_OBS_SHAPE[1], TARGET_OBS_SHAPE[0]), Image.Resampling.BILINEAR)
            frame_u8 = np.asarray(image, dtype=np.uint8)

        return frame_u8

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the env. Returns (obs, info) like Gymnasium."""
        if seed is not None:
            self._key = jax.random.PRNGKey(int(seed))
        obs, self._state = self._env.reset(self._split_key(), self._params)
        return self._to_numpy_obs(obs), {}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Step the env. Returns (obs, reward, terminated, truncated, info)."""
        if self._state is None:
            raise RuntimeError("Call reset() before step().")

        action_jax = jnp.asarray(int(action), dtype=jnp.int32)
        obs, self._state, reward, done, info = self._env.step(
            self._split_key(), self._state, action_jax, self._params
        )
        terminated = bool(np.asarray(done))
        truncated = False
        # Craftax info values may be JAX arrays; convert lightly for logging.
        info_np = {
            k: (np.asarray(v) if hasattr(v, "__array__") else v) for k, v in dict(info).items()
        }
        return (
            self._to_numpy_obs(obs),
            float(np.asarray(reward)),
            terminated,
            truncated,
            info_np,
        )

    def close(self) -> None:
        return None
