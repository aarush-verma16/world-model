"""Play Crafter with a random policy and save a short visual GIF (M0 visual check).

Usage (worldmodel env active):
    python scripts/watch_crafter.py
    open results/m0_random_rollout.gif
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np

import envs  # noqa: F401  — registers CrafterReward-v1


ENV_ID = "CrafterReward-v1"
OUT_PATH = Path("results/m0_random_rollout.gif")
NUM_STEPS = 120
SEED = 0


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    env = gym.make(ENV_ID)
    try:
        obs, _ = env.reset(seed=SEED)
        frames = [np.asarray(obs, dtype=np.uint8)]
        for _ in range(NUM_STEPS):
            action = env.action_space.sample()
            obs, _reward, terminated, truncated, _info = env.step(action)
            frames.append(np.asarray(obs, dtype=np.uint8))
            if terminated or truncated:
                obs, _ = env.reset()
                frames.append(np.asarray(obs, dtype=np.uint8))
    finally:
        env.close()

    imageio.mimsave(OUT_PATH, frames, duration=0.08, loop=0)
    print(f"Saved {len(frames)} frames -> {OUT_PATH.resolve()}")
    print(f"Open with: open {OUT_PATH}")
    print("PASS: Crafter visual rollout written")


if __name__ == "__main__":
    main()
