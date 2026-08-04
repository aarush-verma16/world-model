"""Shared helpers for collecting short Crafter rollouts and encoding them.

Used by M2 verification/diagnostic scripts (RSSM forward-pass + visual checks).
Not a replay buffer — M3 builds the real sequential replay buffer.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from envs.crafter_env import register_crafter_envs
from models.encoder import Encoder
from models.preprocess import nhwc_uint8_to_nchw_float

import gymnasium as gym  # noqa: E402  (after register_crafter_envs is defined above)


def collect_sequences(
    env_id: str,
    num_episodes: int,
    seq_len: int,
    max_episode_steps: int,
    action_dim: int,
) -> tuple[Tensor, Tensor]:
    """Return uint8 obs `[E, T, 64, 64, 3]` and int actions `[E, T]` from a
    random policy. Each episode contributes one fixed-length sequence."""
    register_crafter_envs()
    env = gym.make(env_id)
    if int(env.action_space.n) != action_dim:
        n = int(env.action_space.n)
        env.close()
        raise ValueError(f"config action_dim={action_dim} != env.action_space.n={n}")
    all_obs: list[np.ndarray] = []
    all_act: list[np.ndarray] = []
    try:
        for ep in range(num_episodes):
            obs, _ = env.reset(seed=ep)
            obs_buf: list[np.ndarray] = []
            act_buf: list[int] = []
            for _ in range(max_episode_steps):
                action = int(env.action_space.sample())
                next_obs, _reward, terminated, truncated, _info = env.step(action)
                obs_buf.append(np.asarray(obs, dtype=np.uint8))
                act_buf.append(action)
                obs = next_obs
                if terminated or truncated:
                    obs, _ = env.reset()
                if len(obs_buf) >= seq_len:
                    break
            if len(obs_buf) < seq_len:
                raise RuntimeError(
                    f"episode {ep} too short: got {len(obs_buf)} < {seq_len}"
                )
            all_obs.append(np.stack(obs_buf[:seq_len], axis=0))
            all_act.append(np.asarray(act_buf[:seq_len], dtype=np.int64))
    finally:
        env.close()
    obs_t = torch.from_numpy(np.stack(all_obs, axis=0))  # [E,T,H,W,C]
    act_t = torch.from_numpy(np.stack(all_act, axis=0))
    return obs_t, act_t


def encode_sequence(encoder: Encoder, obs_u8: Tensor, device: torch.device) -> Tensor:
    """obs `[B, T, H, W, C]` uint8 → embeds `[B, T, embed_dim]`."""
    batch, time = obs_u8.shape[:2]
    flat = obs_u8.reshape(batch * time, *obs_u8.shape[2:])
    obs = nhwc_uint8_to_nchw_float(flat.to(device))
    embeds = encoder(obs)
    return embeds.view(batch, time, -1)
