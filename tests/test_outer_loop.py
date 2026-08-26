"""Milestone 5: outer loop collect / eval / freeze-unfreeze / FIFO / joint ckpt."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from agents.actor_critic import Actor, Critic
from models.world_model import WorldModel
from training.ac_step import actor_critic_step
from training.collect import Collector, ste_action_to_int
from training.device import make_grad_scaler, parse_amp
from training.evaluate import evaluate_policy
from training.imagine import freeze_world_model, unfreeze_world_model
from training.outer_loop import (
    crossed_interval,
    joint_payload,
    load_checkpoint,
    outer_cycle,
    save_checkpoint,
)
from training.replay_buffer import ReplayBuffer
from training.returns import PercentileReturnNorm
from training.wm_step import world_model_step


def test_crossed_interval_skips_non_multiples_of_collect() -> None:
    """collect_every=16 never lands on 5000, so modulo scheduling missed eval."""
    assert not crossed_interval(0, 16, 5000)
    assert not crossed_interval(16, 32, 5000)
    assert crossed_interval(4992, 5008, 5000)
    assert crossed_interval(9984, 10000, 10000)
    assert crossed_interval(9984, 10000, 5000)
    assert not crossed_interval(50000, 50016, 10000)
    assert crossed_interval(0, 16, 16)
    assert not crossed_interval(16, 32, 0)


def _tiny_wm(action_dim: int = 5) -> WorldModel:
    return WorldModel.from_config_dims(
        embed_dim=64,
        encoder_channels=(16, 32, 64, 64),
        action_dim=action_dim,
        deter_dim=32,
        stoch=4,
        classes=4,
        hidden=32,
        decoder_channels=(64, 32, 16, 8),
        head_hidden=32,
        head_layers=1,
        encoder_blocks=1,
        decoder_blocks=0,
    )


def _wm_train_cfg() -> dict:
    return {
        "dyn_scale": 1.0,
        "rep_scale": 0.5,
        "free_nats": 1.0,
        "free_nats_dyn": 0.0,
        "recon_scale": 1.0,
        "reward_scale": 1.0,
        "continue_scale": 1.0,
        "kl_scale": 1.0,
    }


def _seed_buffer(action_dim: int = 5, n_ep: int = 3, t: int = 16, seed: int = 0) -> ReplayBuffer:
    buf = ReplayBuffer(seed=seed, max_steps=10_000)
    for i in range(n_ep):
        obs = torch.randint(0, 256, (t, 64, 64, 3), dtype=torch.uint8)
        actions = torch.randint(0, action_dim, (t,), dtype=torch.int64)
        rewards = torch.zeros(t)
        cont = torch.ones(t)
        cont[-1] = 0.0
        buf.add_episode(obs, actions, rewards, cont)
    return buf


class FakeEnv:
    """Minimal Discrete env: 64×64 uint8, optional achievements in info."""

    def __init__(self, action_dim: int = 5, ep_len: int = 8) -> None:
        self.action_dim = int(action_dim)
        self.ep_len = int(ep_len)
        self.t = 0
        self.action_space = type("Space", (), {"n": self.action_dim})()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self.t = 0
        return np.zeros((64, 64, 3), dtype=np.uint8), {"achievements": {"wood": 0}}

    def step(self, action: int):
        action_i = int(action)
        assert 0 <= action_i < self.action_dim, action_i
        self.t += 1
        obs = np.full((64, 64, 3), self.t % 256, dtype=np.uint8)
        reward = 1.0 if self.t == 3 else 0.0
        terminated = self.t >= self.ep_len
        info = {"achievements": {"wood": 1 if self.t >= 3 else 0}}
        return obs, reward, terminated, False, info

    def close(self) -> None:
        return None


def test_ste_action_to_int_in_range() -> None:
    action_dim = 7
    onehot = torch.zeros(1, action_dim)
    onehot[0, 4] = 1.0
    assert ste_action_to_int(onehot) == 4
    wm = _tiny_wm(action_dim)
    actor = Actor(wm.feat_dim, action_dim, hidden=32, layers=1)
    feat = torch.randn(3, wm.feat_dim)
    for _ in range(20):
        action_oh, _, _, _ = actor.policy(feat)
        idx = ste_action_to_int(action_oh)
        assert 0 <= idx < action_dim


def test_replay_fifo_drops_oldest() -> None:
    buf = ReplayBuffer(seed=0, max_steps=25)
    for i in range(3):
        t = 10
        obs = torch.full((t, 64, 64, 3), i, dtype=torch.uint8)
        buf.add_episode(
            obs,
            torch.zeros(t, dtype=torch.int64),
            torch.zeros(t),
            torch.ones(t),
        )
    assert len(buf) == 2
    assert buf.num_steps == 20
    # Oldest (fill value 0) is gone; remaining starts at 1.
    assert int(buf._episodes[0].obs[0, 0, 0, 0]) == 1


def test_freeze_then_unfreeze_wm_weights_change() -> None:
    device = torch.device("cpu")
    wm = _tiny_wm()
    actor = Actor(wm.feat_dim, wm.rssm.action_dim, hidden=32, layers=1)
    critic = Critic(wm.feat_dim, hidden=32, layers=1, num_bins=21)
    freeze_world_model(wm)
    snap = {k: v.detach().clone() for k, v in wm.state_dict().items()}
    ac_optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-3)
    amp = parse_amp("off", device)
    scaler = make_grad_scaler(device, amp)
    batch = {
        "obs": torch.randint(0, 256, (2, 8, 64, 64, 3), dtype=torch.uint8),
        "actions": torch.randint(0, 5, (2, 8), dtype=torch.int64),
        "rewards": torch.zeros(2, 8),
        "cont": torch.ones(2, 8),
    }
    actor_critic_step(
        wm, actor, critic, ac_optim, batch,
        device=device, retnorm=PercentileReturnNorm(), horizon=3,
        start_mode="last", amp_dtype=amp, scaler=scaler,
    )
    for k, v in wm.state_dict().items():
        assert torch.equal(v.cpu(), snap[k].cpu()), k
    for p in wm.parameters():
        assert p.requires_grad is False

    unfreeze_world_model(wm)
    for p in wm.parameters():
        assert p.requires_grad is True
    wm_optim = torch.optim.Adam(wm.parameters(), lr=1e-3)
    before = next(wm.parameters()).detach().clone()
    world_model_step(
        wm, wm_optim, batch,
        device=device, train_cfg=_wm_train_cfg(), amp_dtype=amp, scaler=scaler,
    )
    after = next(wm.parameters()).detach()
    assert not torch.equal(before, after)


def test_collector_adds_episode() -> None:
    device = torch.device("cpu")
    wm = _tiny_wm()
    actor = Actor(wm.feat_dim, wm.rssm.action_dim, hidden=32, layers=1)
    buf = ReplayBuffer(seed=0)
    env = FakeEnv(action_dim=5, ep_len=8)
    collector = Collector(
        env, wm, actor, buf, device=device, max_episode_steps=8, amp_dtype=None, seed=0
    )
    out = collector.collect(8)
    assert out["steps"] == 8
    assert len(out["episodes"]) == 1
    assert buf.num_steps == 8
    assert len(buf) == 1
    assert out["episodes"][0]["length"] == 8
    assert out["episodes"][0]["return"] == 1.0
    assert "achievement_counts" in out["episodes"][0]
    env.close()


def test_evaluate_policy_finite_return() -> None:
    device = torch.device("cpu")
    wm = _tiny_wm()
    actor = Actor(wm.feat_dim, wm.rssm.action_dim, hidden=32, layers=1)
    env = FakeEnv(action_dim=5, ep_len=8)
    result = evaluate_policy(
        env, wm, actor, device=device, n_episodes=2, max_steps=6, amp_dtype=None, seed=0
    )
    assert len(result.returns) == 2
    assert np.isfinite(result.mean_return)
    assert result.mean_length == 6
    assert result.frames is not None
    assert result.frames.shape[1:] == (64, 64, 3)
    assert np.isfinite(result.crafter_score)
    env.close()


def test_joint_checkpoint_roundtrip(tmp_path: Path) -> None:
    device = torch.device("cpu")
    wm = _tiny_wm()
    actor = Actor(wm.feat_dim, wm.rssm.action_dim, hidden=32, layers=1)
    critic = Critic(wm.feat_dim, hidden=32, layers=1, num_bins=21)
    wm_optim = torch.optim.Adam(wm.parameters(), lr=1e-4)
    ac_optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-5)
    retnorm = PercentileReturnNorm()
    actor.net.net[-1].weight.data.add_(0.5)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(
        path,
        joint_payload(
            env_steps=48,
            wm_steps=3,
            ac_steps=3,
            world_model=wm,
            wm_optim=wm_optim,
            actor=actor,
            critic=critic,
            ac_optim=ac_optim,
            retnorm=retnorm,
            collect_seed=9,
        ),
    )
    wm2 = _tiny_wm()
    actor2 = Actor(wm2.feat_dim, wm2.rssm.action_dim, hidden=32, layers=1)
    critic2 = Critic(wm2.feat_dim, hidden=32, layers=1, num_bins=21)
    wm_optim2 = torch.optim.Adam(wm2.parameters(), lr=1e-4)
    ac_optim2 = torch.optim.Adam(list(actor2.parameters()) + list(critic2.parameters()), lr=3e-5)
    retnorm2 = PercentileReturnNorm()
    counters = load_checkpoint(
        path, wm2, wm_optim2, actor2, critic2, ac_optim2, retnorm2, device
    )
    assert counters["env_steps"] == 48
    assert counters["collect_seed"] == 9
    assert torch.equal(actor.net.net[-1].weight.detach(), actor2.net.net[-1].weight.detach())


def test_outer_cycle_collect_wm_ac() -> None:
    device = torch.device("cpu")
    wm = _tiny_wm()
    actor = Actor(wm.feat_dim, wm.rssm.action_dim, hidden=32, layers=1)
    critic = Critic(wm.feat_dim, hidden=32, layers=1, num_bins=21)
    buf = _seed_buffer()
    env = FakeEnv(action_dim=5, ep_len=20)
    collector = Collector(
        env, wm, actor, buf, device=device, max_episode_steps=20, amp_dtype=None, seed=1
    )
    wm_optim = torch.optim.Adam(wm.parameters(), lr=1e-3)
    ac_optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-3)
    amp = parse_amp("off", device)
    scaler = make_grad_scaler(device, amp)
    wm_before = next(wm.parameters()).detach().clone()
    actor_before = actor.net.net[-1].weight.detach().clone()
    result = outer_cycle(
        collector, wm, actor, critic, wm_optim, ac_optim, buf,
        device=device,
        wm_train_cfg=_wm_train_cfg(),
        collect_every=4,
        wm_updates=1,
        ac_updates=1,
        batch_size=2,
        seq_len=8,
        retnorm=PercentileReturnNorm(),
        horizon=3,
        start_mode="last",
        lam=0.95,
        discount=0.997,
        entropy_scale=3e-4,
        amp_dtype=amp,
        scaler=scaler,
    )
    assert result.collect["steps"] == 4
    assert result.wm_metrics is not None
    assert result.ac_metrics is not None
    assert np.isfinite(result.wm_metrics["total"])
    assert np.isfinite(result.ac_metrics["entropy"])
    assert not torch.equal(wm_before, next(wm.parameters()).detach())
    assert not torch.equal(actor_before, actor.net.net[-1].weight.detach())
    env.close()
