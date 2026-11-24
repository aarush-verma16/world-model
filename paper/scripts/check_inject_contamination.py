"""Does a teacher injection on the collect env leak into the actor's open life?

No model, no training: a random "actor" steps a CrafterEnv, the 25k-step
teacher injection runs on the same env object, then the open life continues.

    c:/Users/Aarus/miniforge3/envs/worldmodel/python.exe paper/scripts/check_inject_contamination.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from envs.crafter_env import CrafterEnv  # noqa: E402
from training.crafter_teacher import seed_teacher_episodes  # noqa: E402
from training.replay_buffer import ReplayBuffer  # noqa: E402


def unlocked(info: dict) -> list[str]:
    return sorted(k for k, v in info["achievements"].items() if v > 0)


def main() -> None:
    env = CrafterEnv(reward=True)
    rng = np.random.default_rng(0)
    _obs, info = env.reset(seed=123)
    for _ in range(40):
        _obs, _r, _te, _tr, info = env.step(int(rng.integers(0, 5)))
    print("actor life before injection:", unlocked(info), dict(info["inventory"]))

    buffer = ReplayBuffer()
    inj = seed_teacher_episodes(env, buffer, episodes=8, max_episode_steps=800, seed=20_000, version=4)
    print("injection iron pickaxes:", inj["unlocks"].get("make_iron_pickaxe", 0), "of", inj["episodes"])

    for step in range(1, 60):
        _obs, _r, terminated, truncated, info = env.step(0)
        if terminated or truncated:
            break
    print(f"actor life after injection ({step} more noop steps):", unlocked(info))
    print("inventory:", {k: v for k, v in info["inventory"].items() if v})


if __name__ == "__main__":
    main()
