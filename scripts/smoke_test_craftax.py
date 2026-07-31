"""Smoke-test Craftax-Classic-Pixels via CraftaxClassicPixelsEnv (M0).

Run from the repo root with the worldmodel conda env active:
    python scripts/smoke_test_craftax.py
"""

from __future__ import annotations

import sys

import jax
import numpy as np
import torch

from envs import CraftaxClassicPixelsEnv


EXPECTED_OBS_SHAPE = (64, 64, 3)
NUM_STEPS = 5


def check_mps() -> None:
    available = torch.backends.mps.is_available()
    print(f"torch {torch.__version__}")
    print(f"MPS available: {available}")
    if not available:
        raise SystemExit("FAIL: torch.backends.mps.is_available() is False")


def check_jax_cpu() -> None:
    devices = jax.devices()
    backend = jax.default_backend()
    print(f"jax {jax.__version__}")
    print(f"jax.devices(): {devices}")
    print(f"jax.default_backend(): {backend}")
    if backend != "cpu":
        raise SystemExit(f"FAIL: expected JAX CPU backend, got {backend!r}")
    if not any(d.platform == "cpu" for d in devices):
        raise SystemExit(f"FAIL: no CPU device in jax.devices(): {devices}")
    if any(d.platform == "METAL" or "metal" in str(d).lower() for d in devices):
        raise SystemExit(f"FAIL: Metal JAX device present (not allowed): {devices}")


def check_craftax() -> None:
    env = CraftaxClassicPixelsEnv(seed=0)
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

        n = env.action_space_n
        print(f"Env: Craftax-Classic-Pixels-v1 (wrapped)")
        print(f"action_space_n: {n}")
        for step_i in range(NUM_STEPS):
            action = int(np.random.randint(0, n))
            obs, reward, terminated, truncated, info = env.step(action)
            if obs.shape != EXPECTED_OBS_SHAPE:
                raise SystemExit(
                    f"FAIL: step {step_i} obs shape {obs.shape}, "
                    f"expected {EXPECTED_OBS_SHAPE}"
                )
            if terminated or truncated:
                obs, info = env.reset()

        print(f"Observation shape: {obs.shape}")
        print(f"Stepped {NUM_STEPS} times successfully")
    finally:
        env.close()


def main() -> int:
    check_mps()
    check_jax_cpu()
    check_craftax()
    print("PASS: Craftax smoke test succeeded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
