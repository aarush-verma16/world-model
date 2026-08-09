"""Tests for M3 world-model heads, KL balancing, and sequential replay."""

from __future__ import annotations

import torch

from models.heads import ContinueHead, RewardHead, rssm_features
from models.world_model import WorldModel
from training.losses import categorical_kl, kl_balance, world_model_loss
from training.replay_buffer import ReplayBuffer
from models.preprocess import nhwc_uint8_to_nchw_float


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
    )


def test_rssm_features_shape() -> None:
    h = torch.randn(2, 8, 32)
    z = torch.zeros(2, 8, 4, 4)
    z[..., 0] = 1.0
    feat = rssm_features(h, z)
    assert feat.shape == (2, 8, 32 + 16)


def test_heads_forward_and_grad() -> None:
    feat = torch.randn(3, 48, requires_grad=True)
    reward = RewardHead(48, hidden=16, layers=1)
    cont = ContinueHead(48, hidden=16, layers=1)
    r = reward(feat)
    c = cont(feat)
    assert r.shape == (3, 1)
    assert c.shape == (3, 1)
    (r.sum() + c.sum()).backward()
    assert feat.grad is not None
    assert torch.isfinite(feat.grad).all()


def test_kl_balance_asymmetric_and_finite() -> None:
    post = torch.randn(2, 4, 3, 5, requires_grad=True)
    prior = torch.randn(2, 4, 3, 5, requires_grad=True)
    kl, dyn, rep, dyn_raw, rep_raw = kl_balance(
        post, prior, unimix=0.01, dyn_scale=0.5, rep_scale=0.1, free_nats=1.0
    )
    assert torch.isfinite(kl)
    assert dyn.item() >= 1.0 - 1e-5  # free-nats floor after mean ≥ 1 if all clamped
    assert dyn_raw.item() <= dyn.item() + 1e-5
    assert rep_raw.item() <= rep.item() + 1e-5
    kl.backward()
    assert prior.grad is not None and post.grad is not None
    # dyn path stopgrads post for that term; rep path stopgrads prior — both
    # still receive grad from the other term, so neither grad is all zeros.
    assert prior.grad.abs().sum() > 0
    assert post.grad.abs().sum() > 0


def test_categorical_kl_zero_when_identical() -> None:
    logits = torch.randn(2, 3, 4, 5)
    kl = categorical_kl(logits, logits.clone(), unimix=0.0)
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-5)


def test_world_model_forward_shapes_and_loss_backward() -> None:
    wm = _tiny_wm()
    b, t = 2, 6
    obs = torch.randint(0, 256, (b, t, 64, 64, 3), dtype=torch.uint8)
    actions = torch.randint(0, 5, (b, t), dtype=torch.int64)
    rewards = torch.randn(b, t)
    cont = torch.ones(b, t)

    out = wm(obs, actions)
    assert out.recon.shape == (b, t, 3, 64, 64)
    assert out.reward_pred.shape == (b, t, 1)
    assert out.cont_logit.shape == (b, t, 1)
    assert out.rssm.z_prior.shape == (b, t, 4, 4)
    assert out.rssm.z_posterior.shape == (b, t, 4, 4)

    obs_f = nhwc_uint8_to_nchw_float(obs.reshape(b * t, 64, 64, 3)).view(b, t, 3, 64, 64)
    loss = world_model_loss(
        obs=obs_f,
        recon=out.recon,
        reward=rewards,
        reward_pred=out.reward_pred,
        cont=cont,
        cont_logit=out.cont_logit,
        post_logits=out.rssm.posterior_logits,
        prior_logits=out.rssm.prior_logits,
        unimix=wm.rssm.unimix,
        recon_loss_type="l1",
    )
    assert torch.isfinite(loss.total)
    loss.total.backward()
    # Gradient should reach encoder, prior net, and reward head.
    assert wm.encoder.conv[0].weight.grad is not None
    assert wm.rssm.prior_net[0].weight.grad is not None
    assert wm.reward_head.net[0].weight.grad is not None


def test_replay_buffer_samples_contiguous_windows() -> None:
    buf = ReplayBuffer(seed=0)
    T = 20
    for i in range(3):
        obs = torch.arange(T * 64 * 64 * 3, dtype=torch.uint8).view(T, 64, 64, 3)
        # Distinct per-episode marker in first pixel so we can check contiguity.
        obs = obs.clone()
        obs[:, 0, 0, 0] = i
        actions = torch.arange(T, dtype=torch.int64)
        rewards = torch.arange(T, dtype=torch.float32)
        cont = torch.ones(T)
        cont[-1] = 0.0
        buf.add_episode(obs, actions, rewards, cont)

    batch = buf.sample(batch_size=4, seq_len=8)
    assert batch["obs"].shape == (4, 8, 64, 64, 3)
    assert batch["actions"].shape == (4, 8)
    # Contiguous actions: each window is an arithmetic sequence of step 1.
    for i in range(4):
        acts = batch["actions"][i]
        assert torch.equal(acts[1:] - acts[:-1], torch.ones(7, dtype=torch.int64))
