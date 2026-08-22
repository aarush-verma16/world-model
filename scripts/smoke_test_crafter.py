"""Smoke-test that CrafterReward-v1 resets/steps with the expected observation shape.

Run from the repo root with the worldmodel conda env active:
    python scripts/smoke_test_crafter.py
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

import envs  # noqa: F401  — registers CrafterReward-v1 with Gymnasium


ENV_ID = "CrafterReward-v1"
EXPECTED_OBS_SHAPE = (64, 64, 3)
NUM_STEPS = 5


def check_torch() -> None:
    print(f"torch {torch.__version__}")
    print(f"CUDA built: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        x = torch.ones(2, device="cuda")
        print(f"CUDA tensor device: {x.device}")
    else:
        print("WARN: CUDA not available — training will not use the 5080.")


def check_crafter() -> None:
    env = gym.make(ENV_ID)
    try:
        obs, info = env.reset(seed=0)
        if not isinstance(obs, np.ndarray):
            raise SystemExit(f"FAIL: observation type {type(obs)}, expected np.ndarray")
        if obs.shape != EXPECTED_OBS_SHAPE:
            raise SystemExit(
                f"FAIL: observation shape {obs.shape}, expected {EXPECTED_OBS_SHAPE}"
            )
        if obs.dtype != np.uint8:
            print(f"WARN: observation dtype is {obs.dtype}, expected uint8")

        action_space = env.action_space
        for step_i in range(NUM_STEPS):
            action = action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if obs.shape != EXPECTED_OBS_SHAPE:
                raise SystemExit(
                    f"FAIL: step {step_i} obs shape {obs.shape}, "
                    f"expected {EXPECTED_OBS_SHAPE}"
                )
            if terminated or truncated:
                obs, info = env.reset()

        print(f"Env: {ENV_ID}")
        print(f"Observation shape: {obs.shape}")
        print(f"Action space: {action_space}")
        print(f"Stepped {NUM_STEPS} times successfully")
    finally:
        env.close()


def main() -> int:
    check_torch()
    check_crafter()
    print("PASS: Crafter smoke test succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
