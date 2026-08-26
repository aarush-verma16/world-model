"""Actor / critic unit checks: unimix policy, STE one-hot, two-hot values."""

from __future__ import annotations

import torch

from agents.actor_critic import Actor, Critic
from models.heads import MLPHead, rssm_features
from models.rssm import unimix_probs
from tests.helpers import tiny_batch, tiny_world_model
from training.imagine import decode_imagination, freeze_world_model, imagine_ahead


def test_actor_policy_probs_sum_to_one_and_action_is_onehot() -> None:
    actor = Actor(feat_dim=16, action_dim=7, hidden=8, layers=1, unimix=0.01)
    feat = torch.randn(4, 16)
    action, log_prob, entropy, probs = actor.policy(feat)
    assert action.shape == (4, 7)
    assert log_prob.shape == (4,)
    assert entropy.shape == (4,)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)
    assert torch.allclose(action.sum(dim=-1), torch.ones(4), atol=1e-5)
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(entropy).all()
    # Unimix floor: every action keeps at least 0.01/7 mass.
    assert bool((probs >= (0.01 / 7) - 1e-6).all())


def test_actor_rejects_binary_action_space() -> None:
    try:
        Actor(feat_dim=8, action_dim=1, hidden=8, layers=1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for action_dim < 2")


def test_critic_twohot_logits_and_bins() -> None:
    critic = Critic(feat_dim=16, hidden=8, layers=1, num_bins=21, low=-5.0, high=5.0)
    feat = torch.randn(3, 5, 16)
    logits = critic(feat)
    assert logits.shape == (3, 5, 21)
    assert critic.bins.shape == (21,)
    assert float(critic.bins[0]) == -5.0
    assert float(critic.bins[-1]) == 5.0


def test_mlp_head_rejects_zero_layers() -> None:
    try:
        MLPHead(8, 4, hidden=8, layers=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for layers < 1")


def test_rssm_features_rejects_h_z_mismatch() -> None:
    h = torch.randn(2, 8, 32)
    z = torch.zeros(2, 7, 4, 4)
    try:
        rssm_features(h, z)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for h/z batch-time mismatch")


def test_imagine_ahead_rejects_bad_start_mode_and_horizon() -> None:
    wm = tiny_world_model()
    freeze_world_model(wm)
    actor = Actor(wm.feat_dim, wm.rssm.action_dim, hidden=32, layers=1)
    critic = Critic(wm.feat_dim, hidden=32, layers=1, num_bins=21)
    batch = tiny_batch()
    try:
        imagine_ahead(
            wm, actor, critic, batch["obs"], batch["actions"],
            horizon=3, start_mode="middle",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for bad start_mode")
    try:
        imagine_ahead(
            wm, actor, critic, batch["obs"], batch["actions"],
            horizon=0, start_mode="last",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for horizon < 1")


def test_decode_imagination_shapes() -> None:
    wm = tiny_world_model()
    freeze_world_model(wm)
    actor = Actor(wm.feat_dim, wm.rssm.action_dim, hidden=32, layers=1)
    critic = Critic(wm.feat_dim, hidden=32, layers=1, num_bins=21)
    batch = tiny_batch(batch=3, seq=4)
    rollout = imagine_ahead(
        wm, actor, critic, batch["obs"], batch["actions"],
        horizon=3, start_mode="last",
    )
    frames = decode_imagination(wm, rollout.feat, max_starts=2)
    assert frames.shape == (2, 3, 3, 64, 64)
    assert float(frames.min()) >= 0.0
    assert float(frames.max()) <= 1.0


def test_unimix_rejects_out_of_range() -> None:
    logits = torch.zeros(2, 4)
    try:
        unimix_probs(logits, unimix=1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unimix=1")
