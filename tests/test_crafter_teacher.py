"""The map teacher must put stone in the replay, and sampling must keep it there."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from envs.crafter_env import register_crafter_envs
from training.crafter_teacher import seed_teacher_episodes
from training.replay_buffer import ReplayBuffer


def test_teacher_mines_stone_on_the_real_env() -> None:
    register_crafter_envs()
    env = gym.make("CrafterReward-v1")
    try:
        buf = ReplayBuffer()
        stats = seed_teacher_episodes(env, buf, episodes=2, max_episode_steps=400, seed=0)
    finally:
        env.close()
    assert stats["stone"] == 2
    assert stats["pickaxe"] == 2
    assert all(ep.teacher for ep in buf._episodes)
    assert buf.num_steps == stats["steps"]


def test_teacher_fraction_hits_the_marked_episode_and_eviction_keeps_it() -> None:
    buf = ReplayBuffer(seed=0, max_steps=20)
    buf.add_episode(
        torch.zeros(16, 64, 64, 3, dtype=torch.uint8),
        torch.zeros(16, dtype=torch.int64),
        torch.zeros(16),
        torch.ones(16),
    )
    buf.add_episode(
        torch.zeros(8, 64, 64, 3, dtype=torch.uint8),
        torch.full((8,), 7, dtype=torch.int64),
        torch.zeros(8),
        torch.ones(8),
        teacher=True,
    )
    buf.add_episode(
        torch.zeros(16, 64, 64, 3, dtype=torch.uint8),
        torch.ones(16, dtype=torch.int64),
        torch.zeros(16),
        torch.ones(16),
    )
    assert buf.num_steps <= 20 or any(ep.teacher for ep in buf._episodes)
    assert any(ep.teacher for ep in buf._episodes)
    batch = buf.sample(32, 4, teacher_fraction=1.0)
    assert int((batch["actions"] == 7).sum()) > 0
    # A later flood of agent data must not delete the teacher life.
    for _ in range(5):
        buf.add_episode(
            torch.zeros(16, 64, 64, 3, dtype=torch.uint8),
            torch.zeros(16, dtype=torch.int64),
            torch.zeros(16),
            torch.ones(16),
        )
    assert any(ep.teacher for ep in buf._episodes)
    assert buf.num_steps <= 20 + 8
