"""The map teacher must put stone in the replay, and sampling must keep it there."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from envs.crafter_env import register_crafter_envs
from training.crafter_rules import ACTION_INDEX, FIELD_CRAFT_ACTIONS
from training.crafter_teacher import craft_assist_active, seed_teacher_episodes
from training.losses import bc_step_weight
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


def test_teacher_v3_crafts_iron_pickaxe_where_v2_stockpiles() -> None:
    """Seed 2 dies holding coal=2 iron=2 under v2, and crafts under v3."""
    register_crafter_envs()
    env = gym.make("CrafterReward-v1")
    try:
        v2 = seed_teacher_episodes(
            env, ReplayBuffer(), episodes=1, max_episode_steps=800, seed=2, version=2
        )
        v3 = seed_teacher_episodes(
            env, ReplayBuffer(), episodes=1, max_episode_steps=800, seed=2, version=3
        )
    finally:
        env.close()
    assert v2["unlocks"]["collect_iron"] == 1
    assert v2["unlocks"]["make_iron_pickaxe"] == 0
    assert v3["unlocks"]["make_iron_pickaxe"] == 1


def test_teacher_v4_crafts_where_v3_holds_the_iron() -> None:
    """Seed 3 under PYTHONHASHSEED=0: v3 dies holding iron, v4 places and crafts.

    Crafter's map id is hash((seed, episode)), which changes across processes.
    The child process pins that hash. This is the first reset of a fresh env.
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    code = (
        "import gymnasium as gym\n"
        "from envs.crafter_env import register_crafter_envs\n"
        "from training.crafter_teacher import seed_teacher_episodes\n"
        "from training.replay_buffer import ReplayBuffer\n"
        "register_crafter_envs()\n"
        "env = gym.make('CrafterReward-v1')\n"
        "try:\n"
        "    v3 = seed_teacher_episodes(env, ReplayBuffer(), episodes=1, max_episode_steps=800, seed=3, version=3)\n"
        "finally:\n"
        "    env.close()\n"
        "env = gym.make('CrafterReward-v1')\n"
        "try:\n"
        "    v4 = seed_teacher_episodes(env, ReplayBuffer(), episodes=1, max_episode_steps=800, seed=3, version=4)\n"
        "finally:\n"
        "    env.close()\n"
        "print(v3['unlocks']['collect_iron'], v3['unlocks']['make_iron_pickaxe'], v4['unlocks']['make_iron_pickaxe'])\n"
    )
    child_env = os.environ.copy()
    child_env["PYTHONHASHSEED"] = "0"
    child_env["PYTHONPATH"] = os.pathsep.join(
        [str(root / "src"), str(root), child_env.get("PYTHONPATH", "")]
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=child_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    line = [ln for ln in proc.stdout.splitlines() if ln[:1].isdigit()][-1]
    iron, v3_px, v4_px = (int(part) for part in line.split())
    assert iron == 1
    assert v3_px == 0
    assert v4_px == 1


def test_teacher_v2_reaches_stone_pickaxe() -> None:
    register_crafter_envs()
    env = gym.make("CrafterReward-v1")
    try:
        buf = ReplayBuffer()
        stats = seed_teacher_episodes(
            env, buf, episodes=1, max_episode_steps=800, seed=0, version=2
        )
    finally:
        env.close()
    assert stats["unlocks"]["collect_stone"] == 1
    assert stats["unlocks"]["defeat_zombie"] == 1
    assert stats["mean_length"] > 150


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


def test_teacher_fraction_draws_the_iron_craft_not_a_random_slice() -> None:
    craft = int(ACTION_INDEX["make_iron_pickaxe"])
    actions = torch.zeros(200, dtype=torch.int64)
    actions[180] = craft
    buf = ReplayBuffer(seed=0)
    buf.add_episode(
        torch.zeros(200, 64, 64, 3, dtype=torch.uint8),
        actions,
        torch.zeros(200),
        torch.ones(200),
        teacher=True,
    )
    buf.add_episode(
        torch.zeros(200, 64, 64, 3, dtype=torch.uint8),
        torch.zeros(200, dtype=torch.int64),
        torch.zeros(200),
        torch.ones(200),
    )
    batch = buf.sample(16, 64, teacher_fraction=1.0)
    assert bool((batch["actions"] == craft).any(dim=1).all())
    # The craft is the last step of the window, not a random hit in the middle.
    assert bool((batch["actions"][:, -1] == craft).all())


def test_bc_weight_ups_the_craft_button() -> None:
    actions = torch.tensor([[0, ACTION_INDEX["make_iron_pickaxe"], 5]])
    teacher = torch.tensor([[1.0, 1.0, 0.0]])
    w = bc_step_weight(teacher, actions, craft_ids=tuple(sorted(FIELD_CRAFT_ACTIONS)))
    assert float(w[0, 0]) == 1.0
    assert float(w[0, 1]) == 8.0
    assert float(w[0, 2]) == 0.0


def test_craft_assist_turns_on_after_the_stone_pickaxe() -> None:
    assert craft_assist_active(None) is False
    assert craft_assist_active({"inventory": {"stone_pickaxe": 0}}) is False
    assert craft_assist_active({"inventory": {"stone_pickaxe": 1, "iron_pickaxe": 0}}) is True
    assert craft_assist_active({"inventory": {"stone_pickaxe": 1, "iron_pickaxe": 1}}) is False
