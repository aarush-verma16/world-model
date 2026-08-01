"""Collect random-policy Crafter frames for M1 autoencoder training.

Usage:
    conda activate worldmodel
    python scripts/collect_random_frames.py --config configs/m1_autoencoder.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import yaml

import envs  # noqa: F401


def collect_frames(
    num_frames: int,
    max_episode_steps: int,
    seed: int,
) -> torch.Tensor:
    env = gym.make("CrafterReward-v1")
    frames: list[np.ndarray] = []
    episode = 0
    try:
        obs, _ = env.reset(seed=seed)
        frames.append(np.asarray(obs, dtype=np.uint8))
        steps_in_ep = 0
        while len(frames) < num_frames:
            action = env.action_space.sample()
            obs, _reward, terminated, truncated, _info = env.step(action)
            frames.append(np.asarray(obs, dtype=np.uint8))
            steps_in_ep += 1
            if terminated or truncated or steps_in_ep >= max_episode_steps:
                episode += 1
                obs, _ = env.reset()
                frames.append(np.asarray(obs, dtype=np.uint8))
                steps_in_ep = 0
                if episode % 25 == 0:
                    print(f"collected {len(frames)}/{num_frames} frames ({episode} episodes)")
    finally:
        env.close()

    arr = np.stack(frames[:num_frames], axis=0)
    return torch.from_numpy(arr)  # uint8 [N, 64, 64, 3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/m1_autoencoder.yaml"))
    args = parser.parse_args()

    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    out_path = Path(cfg["collect"]["out_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Collecting {cfg['collect']['num_frames']} random-policy Crafter frames...")
    frames = collect_frames(
        num_frames=int(cfg["collect"]["num_frames"]),
        max_episode_steps=int(cfg["collect"]["max_episode_steps"]),
        seed=int(cfg["seed"]),
    )
    payload = {
        "frames": frames,
        "seed": int(cfg["seed"]),
        "env_id": "CrafterReward-v1",
    }
    torch.save(payload, out_path)
    print(f"Saved {tuple(frames.shape)} uint8 frames -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
