"""Play Craftax-Classic with a random policy and save a short visual GIF (M0).

Usage (worldmodel env active):
    python scripts/watch_craftax.py
    open results/m0_random_rollout.gif
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from envs import CraftaxClassicPixelsEnv


OUT_PATH = Path("results/m0_random_rollout.gif")
NUM_STEPS = 120
SEED = 0


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    env = CraftaxClassicPixelsEnv(seed=SEED)
    try:
        obs, _ = env.reset(seed=SEED)
        frames = [np.asarray(obs, dtype=np.uint8)]
        for _ in range(NUM_STEPS):
            action = int(np.random.randint(0, env.action_space_n))
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
    print("PASS: Craftax visual rollout written")


if __name__ == "__main__":
    main()
