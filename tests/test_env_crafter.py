"""Crafter env contract: 64x64x3 uint8, Discrete(17). Marked `env` so
`pytest -m "not env"` stays a pure tensor suite.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.env


def test_crafter_reward_obs_and_action_space() -> None:
    import gymnasium as gym

    import envs  # noqa: F401  — registers CrafterReward-v1

    env = gym.make("CrafterReward-v1")
    try:
        obs, info = env.reset(seed=0)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (64, 64, 3)
        assert obs.dtype == np.uint8
        assert int(env.action_space.n) == 17
        for _ in range(3):
            obs, reward, terminated, truncated, info = env.step(int(env.action_space.sample()))
            assert obs.shape == (64, 64, 3)
            assert obs.dtype == np.uint8
            assert np.isfinite(reward)
            assert isinstance(terminated, (bool, np.bool_))
            assert isinstance(truncated, (bool, np.bool_))
            assert "achievements" in info
            assert "discount" in info
            assert float(info["discount"]) in (0.0, 1.0)
            if terminated:
                assert float(info["discount"]) == 0.0
                assert not bool(truncated)
            if terminated or truncated:
                obs, info = env.reset()
    finally:
        env.close()


def test_register_crafter_envs_is_idempotent() -> None:
    import gymnasium as gym

    from envs.crafter_env import register_crafter_envs

    register_crafter_envs()
    register_crafter_envs()
    assert "CrafterReward-v1" in gym.envs.registry
    assert "CrafterNoReward-v1" in gym.envs.registry
