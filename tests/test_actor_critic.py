"""Milestone 4: actor-critic on frozen world-model imagination."""

from __future__ import annotations

import torch

from agents.actor_critic import Actor, Critic
from models.world_model import WorldModel
from training.ac_step import actor_critic_step
from training.device import make_grad_scaler, parse_amp
from training.imagine import freeze_world_model, imagine_ahead
from training.returns import PercentileReturnNorm, lambda_returns


def _tiny_wm() -> WorldModel:
    return WorldModel.from_config_dims(
        embed_dim=64,
        encoder_channels=(16, 32, 64, 64),
        action_dim=5,
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


def _batch(batch: int = 2, seq: int = 8, action_dim: int = 5) -> dict[str, torch.Tensor]:
    return {
        "obs": torch.randint(0, 256, (batch, seq, 64, 64, 3), dtype=torch.uint8),
        "actions": torch.randint(0, action_dim, (batch, seq), dtype=torch.int64),
        "rewards": torch.zeros(batch, seq),
        "cont": torch.ones(batch, seq),
    }


def test_lambda_returns_cont_zero_is_just_reward() -> None:
    reward = torch.tensor([[1.0, 2.0, 3.0]])
    cont = torch.zeros_like(reward)
    value = torch.ones_like(reward) * 99.0
    out = lambda_returns(reward, cont, value, lam=0.95)
    assert torch.allclose(out, reward)


def test_lambda_returns_lam_one_cont_one_is_value() -> None:
    value = torch.tensor([[4.0, 5.0, 6.0]])
    reward = torch.zeros_like(value)
    cont = torch.ones_like(value)
    out = lambda_returns(reward, cont, value, lam=1.0)
    assert torch.allclose(out, torch.full_like(value, 6.0))


def test_imagine_ahead_shapes_all_starts() -> None:
    wm = _tiny_wm()
    freeze_world_model(wm)
    actor = Actor(wm.feat_dim, wm.rssm.action_dim, hidden=32, layers=1)
    critic = Critic(wm.feat_dim, hidden=32, layers=1, num_bins=21)
    batch = _batch()
    horizon = 4
    rollout = imagine_ahead(
        wm, actor, critic, batch["obs"], batch["actions"],
        horizon=horizon, start_mode="all", discount=0.997,
    )
    n = 2 * 8
    assert rollout.h.shape == (n, horizon, 32)
    assert rollout.z_prior.shape == (n, horizon, 4, 4)
    assert rollout.feat.shape == (n, horizon, wm.feat_dim)
    assert rollout.action.shape == (n, horizon, 5)
    assert rollout.log_prob.shape == (n, horizon)
    assert rollout.reward.shape == (n, horizon)
    assert rollout.value_logits.shape[1:] == (horizon, 21)
    assert torch.isfinite(rollout.reward).all()
    assert torch.isfinite(rollout.value).all()


def test_imagine_ahead_last_start_n_equals_batch() -> None:
    wm = _tiny_wm()
    freeze_world_model(wm)
    actor = Actor(wm.feat_dim, wm.rssm.action_dim, hidden=32, layers=1)
    critic = Critic(wm.feat_dim, hidden=32, layers=1, num_bins=21)
    batch = _batch(batch=3, seq=6)
    rollout = imagine_ahead(
        wm, actor, critic, batch["obs"], batch["actions"],
        horizon=3, start_mode="last",
    )
    assert rollout.h.shape[0] == 3


def test_actor_critic_step_freezes_wm_and_updates_actor() -> None:
    device = torch.device("cpu")
    wm = _tiny_wm()
    actor = Actor(wm.feat_dim, wm.rssm.action_dim, hidden=32, layers=1)
    critic = Critic(wm.feat_dim, hidden=32, layers=1, num_bins=21)
    snap = {k: v.detach().clone() for k, v in wm.state_dict().items()}
    actor_before = actor.net.net[-1].weight.detach().clone()
    optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-3)
    retnorm = PercentileReturnNorm()
    amp = parse_amp("off", device)
    scaler = make_grad_scaler(device, amp)
    loss, metrics, rollout = actor_critic_step(
        wm, actor, critic, optim, _batch(),
        device=device,
        retnorm=retnorm,
        horizon=3,
        start_mode="last",
        amp_dtype=amp,
        scaler=scaler,
    )
    assert torch.isfinite(loss.total)
    assert "critic" in metrics
    for k, v in wm.state_dict().items():
        assert torch.equal(v.cpu(), snap[k].cpu()), k
    for p in wm.parameters():
        assert p.requires_grad is False
        assert p.grad is None
    assert actor.net.net[-1].weight.grad is not None
    assert torch.isfinite(actor.net.net[-1].weight.grad).all()
    assert not torch.equal(actor.net.net[-1].weight.detach(), actor_before)
    assert rollout.log_prob.requires_grad


def test_ste_action_reaches_actor() -> None:
    """Imagined return depends on STE actions, so actor logits get a dynamics grad."""
    wm = _tiny_wm()
    freeze_world_model(wm)
    actor = Actor(wm.feat_dim, wm.rssm.action_dim, hidden=32, layers=1)
    critic = Critic(wm.feat_dim, hidden=32, layers=1, num_bins=21)
    batch = _batch(batch=2, seq=4)
    rollout = imagine_ahead(
        wm, actor, critic, batch["obs"], batch["actions"],
        horizon=2, start_mode="last",
    )
    rollout.reward.mean().backward()
    assert actor.net.net[-1].weight.grad is not None
    assert torch.isfinite(actor.net.net[-1].weight.grad).all()
