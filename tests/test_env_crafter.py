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


def test_crafter_env_info_carries_legality_facts() -> None:
    """finding 39 fix: `info` must carry what `training.crafter_rules` needs."""
    import gymnasium as gym

    import envs  # noqa: F401

    env = gym.make("CrafterReward-v1")
    try:
        _obs, info = env.reset(seed=0)
        for key in ("inventory", "facing", "facing_material", "facing_object_present", "nearby_materials"):
            assert key in info, key
        assert isinstance(info["inventory"], dict)
        assert len(info["facing"]) == 2
        assert isinstance(info["nearby_materials"], tuple)
        for _ in range(5):
            _obs, _reward, terminated, truncated, info = env.step(int(env.action_space.sample()))
            for key in ("facing", "facing_material", "facing_object_present", "nearby_materials"):
                assert key in info, key
            if terminated or truncated:
                break
    finally:
        env.close()


def test_crafter_rules_mask_matches_real_env_on_fresh_reset() -> None:
    """On a fresh reset (0 wood/stone/etc.) every place_*/make_* must be illegal."""
    import gymnasium as gym

    import envs  # noqa: F401
    from training.crafter_rules import ACTION_INDEX, legal_action_mask

    env = gym.make("CrafterReward-v1")
    try:
        _obs, info = env.reset(seed=0)
        mask = legal_action_mask(info)
        for name in (
            "place_stone", "place_table", "place_furnace", "place_plant",
            "make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe",
            "make_wood_sword", "make_stone_sword", "make_iron_sword",
        ):
            assert not bool(mask[ACTION_INDEX[name]]), name
        for name in ("noop", "do", "sleep", "move_left", "move_right", "move_up", "move_down"):
            assert bool(mask[ACTION_INDEX[name]]), name
    finally:
        env.close()


def test_collector_masked_real_steps_store_ground_truth_inventory() -> None:
    """End-to-end finding 39/40 plumbing: a real `Collector` against live
    Crafter runs (mask never blocks every action), never produces an
    out-of-range action, and every stored step carries real inventory
    (`has_inventory == 1`), not the legacy all-zero placeholder.
    """
    import gymnasium as gym
    import torch

    import envs  # noqa: F401
    from agents.actor_critic import Actor
    from tests.helpers import tiny_world_model
    from training.collect import Collector
    from training.replay_buffer import ReplayBuffer

    wm = tiny_world_model(action_dim=17)
    actor = Actor(wm.feat_dim, wm.rssm.action_dim, hidden=16, layers=1)
    buf = ReplayBuffer(seed=0)
    env = gym.make("CrafterReward-v1")
    collector = Collector(
        env, wm, actor, buf, device=torch.device("cpu"), max_episode_steps=40, seed=0
    )
    try:
        for _ in range(40):
            out = collector.step()
            assert 0 <= out["action"] < 17
    finally:
        env.close()
    assert buf.num_steps == 40
    episodes = list(buf._episodes)
    live = buf._live_batch()
    if live is not None:
        episodes.append(live)
    for ep in episodes:
        assert ep.has_inventory is not None
        assert bool((ep.has_inventory == 1.0).all())


def test_collector_with_inventory_head_runs_end_to_end_on_live_env() -> None:
    """finding 40 end-to-end on the real engine: an inventory-headed world
    model + extended actor/critic can drive `Collector` against live
    Crafter without a shape error (ground-truth concat in `rssm_policy_step`
    matches the actor's extended `feat_dim`).
    """
    import gymnasium as gym
    import torch

    import envs  # noqa: F401
    from agents.actor_critic import Actor
    from models.world_model import WorldModel
    from training.collect import Collector
    from training.crafter_rules import ITEM_NAMES
    from training.replay_buffer import ReplayBuffer

    n_items = len(ITEM_NAMES)
    wm = WorldModel.from_config_dims(
        embed_dim=32,
        encoder_channels=(8, 16, 16, 16),
        action_dim=17,
        deter_dim=16,
        stoch=4,
        classes=4,
        hidden=16,
        decoder_channels=(16, 8, 8, 8),
        head_hidden=16,
        head_layers=1,
        encoder_blocks=1,
        decoder_blocks=0,
        inventory_n_items=n_items,
        inventory_num_classes=10,
    )
    actor = Actor(wm.feat_dim + n_items, wm.rssm.action_dim, hidden=16, layers=1)
    buf = ReplayBuffer(seed=0)
    env = gym.make("CrafterReward-v1")
    collector = Collector(
        env, wm, actor, buf, device=torch.device("cpu"), max_episode_steps=20, seed=0
    )
    try:
        for _ in range(20):
            out = collector.step()
            assert 0 <= out["action"] < 17
    finally:
        env.close()
    assert buf.num_steps == 20


def test_register_crafter_envs_is_idempotent() -> None:
    import gymnasium as gym

    from envs.crafter_env import register_crafter_envs

    register_crafter_envs()
    register_crafter_envs()
    assert "CrafterReward-v1" in gym.envs.registry
    assert "CrafterNoReward-v1" in gym.envs.registry
