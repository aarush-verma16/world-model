"""Milestone 4: actor-critic on frozen world-model imagination."""

from __future__ import annotations

import torch

from agents.actor_critic import Actor, Critic, SlowCritic
from models.symlog import symlog_twohot_mean
from models.world_model import WorldModel
from training.ac_step import actor_critic_step
from training.device import make_grad_scaler, parse_amp
from training.imagine import freeze_world_model, imagine_ahead
from training.returns import PercentileReturnNorm, imagined_targets, lambda_returns


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


def test_imagined_targets_baseline_is_value_at_the_acting_state() -> None:
    """The regression that collapsed M7: baseline must be V(s_i), not V(s_i+1).

    With zero reward, cont=1 and lam=0 the target for `V(s_i)` is exactly
    `V(s_{i+1})`, so a correct advantage is the value *increment*. Using
    `V(s_{i+1})` as the baseline instead makes every advantage 0 here, and in
    general turns `Q - V` into `reward + (γ-1)·V` — a term with a sign that
    barely depends on the action, which saturates REINFORCE.
    """
    value = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    reward = torch.zeros_like(value)
    cont = torch.ones_like(value)
    returns, base, weights = imagined_targets(reward, cont, value, lam=0.0)
    assert returns.shape == (1, 3)
    assert torch.allclose(returns, value[:, 1:])
    assert torch.allclose(base, value[:, :-1])
    assert torch.allclose(returns - base, torch.ones(1, 3))
    assert torch.allclose(weights, torch.ones(1, 3))


def test_imagined_targets_weights_zero_after_predicted_death() -> None:
    reward = torch.zeros(1, 4)
    value = torch.zeros(1, 4)
    cont = torch.tensor([[1.0, 0.0, 1.0, 1.0]])
    _returns, _base, weights = imagined_targets(reward, cont, value, lam=0.95)
    # s_1 is reachable; cont(s_1)=0 kills every later imagined step.
    assert torch.allclose(weights, torch.tensor([[1.0, 1.0, 0.0]]))


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
    # State-indexed fields span s_0..s_H; action-indexed fields span the H
    # actions taken at s_0..s_{H-1}.
    assert rollout.h.shape == (n, horizon + 1, 32)
    assert rollout.z_prior.shape == (n, horizon + 1, 4, 4)
    assert rollout.feat.shape == (n, horizon + 1, wm.feat_dim)
    assert rollout.reward.shape == (n, horizon + 1)
    assert rollout.cont.shape == (n, horizon + 1)
    assert rollout.value.shape == (n, horizon + 1)
    assert rollout.value_logits.shape[1:] == (horizon + 1, 21)
    assert rollout.action.shape == (n, horizon, 5)
    assert rollout.log_prob.shape == (n, horizon)
    assert rollout.entropy.shape == (n, horizon)
    assert torch.isfinite(rollout.reward).all()
    assert torch.isfinite(rollout.value).all()


def test_imagine_ahead_seed_state_is_the_replay_posterior() -> None:
    """`feat[:, 0]` must be the observed start, so `value[:, 0]` is V(s_0)."""
    wm = _tiny_wm()
    freeze_world_model(wm)
    actor = Actor(wm.feat_dim, wm.rssm.action_dim, hidden=32, layers=1)
    critic = Critic(wm.feat_dim, hidden=32, layers=1, num_bins=21)
    batch = _batch(batch=2, seq=6)
    from models.heads import rssm_features
    from training.imagine import _start_states

    # z_posterior is sampled, so both calls need the same RNG state.
    torch.manual_seed(0)
    rollout = imagine_ahead(
        wm, actor, critic, batch["obs"], batch["actions"],
        horizon=3, start_mode="last",
    )
    torch.manual_seed(0)
    h0, z0 = _start_states(wm, batch["obs"], batch["actions"], "last")
    assert torch.allclose(rollout.feat[:, 0], rssm_features(h0, z0))
    # The seed is detached, so nothing about s_0 can carry an actor gradient.
    # (backward through the stack still allocates a grad buffer; it must be 0.)
    rollout.feat[:, 0].sum().backward()
    grad = actor.net.net[-1].weight.grad
    assert grad is not None and not grad.any()


def test_critic_starts_at_zero_value() -> None:
    """DreamerV3 critic outscale 0.0: the initial value must be exactly 0."""
    critic = Critic(16, hidden=8, layers=1, num_bins=21)
    logits = critic(torch.randn(4, 16))
    assert torch.allclose(logits, torch.zeros_like(logits))
    value = symlog_twohot_mean(logits, critic.bins)
    assert torch.allclose(value, torch.zeros_like(value), atol=1e-5)


def test_slow_critic_is_an_ema_and_holds_no_trainable_params() -> None:
    critic = Critic(16, hidden=8, layers=1, num_bins=21)
    slow = SlowCritic(critic, fraction=0.5)
    assert all(not p.requires_grad for p in slow.parameters())
    last = critic.net.net[-1]
    with torch.no_grad():
        last.weight.fill_(1.0)
    slow.update(critic)
    got = slow.critic.net.net[-1].weight
    assert torch.allclose(got, torch.full_like(got, 0.5))
    slow.update(critic)
    assert torch.allclose(got, torch.full_like(got, 0.75))


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


def test_reinforce_imag_gradient_omits_dynamics_term() -> None:
    device = torch.device("cpu")
    wm = _tiny_wm()
    actor = Actor(wm.feat_dim, wm.rssm.action_dim, hidden=32, layers=1)
    critic = Critic(wm.feat_dim, hidden=32, layers=1, num_bins=21)
    optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-3)
    amp = parse_amp("off", device)
    scaler = make_grad_scaler(device, amp)
    loss, _metrics, _rollout = actor_critic_step(
        wm, actor, critic, optim, _batch(),
        device=device,
        retnorm=PercentileReturnNorm(),
        horizon=3,
        start_mode="last",
        imag_gradient="reinforce",
        amp_dtype=amp,
        scaler=scaler,
    )
    assert torch.isfinite(loss.total)
    assert torch.isfinite(loss.reinforce)
    # Actor objective is reinforce + entropy; dynamics backprop is computed
    # for logs but must not be added when imag_gradient=reinforce.
    assert torch.allclose(loss.actor, loss.reinforce - 3.0e-4 * loss.entropy, atol=1e-5)


def test_critic_reads_detached_features() -> None:
    """Critic loss must not reach the actor through the imagined dynamics."""
    wm = _tiny_wm()
    freeze_world_model(wm)
    actor = Actor(wm.feat_dim, wm.rssm.action_dim, hidden=32, layers=1)
    critic = Critic(wm.feat_dim, hidden=32, layers=1, num_bins=21, out_scale=1.0)
    batch = _batch(batch=2, seq=4)
    rollout = imagine_ahead(
        wm, actor, critic, batch["obs"], batch["actions"],
        horizon=3, start_mode="last",
    )
    rollout.value_logits.mean().backward()
    assert critic.net.net[-1].weight.grad is not None
    assert actor.net.net[-1].weight.grad is None


def test_actor_critic_step_with_slow_critic_moves_the_ema() -> None:
    device = torch.device("cpu")
    wm = _tiny_wm()
    actor = Actor(wm.feat_dim, wm.rssm.action_dim, hidden=32, layers=1)
    critic = Critic(wm.feat_dim, hidden=32, layers=1, num_bins=21)
    slow = SlowCritic(critic, fraction=0.02)
    optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-2)
    amp = parse_amp("off", device)
    scaler = make_grad_scaler(device, amp)
    for _ in range(3):
        loss, metrics, _rollout = actor_critic_step(
            wm, actor, critic, optim, _batch(),
            device=device,
            retnorm=PercentileReturnNorm(),
            horizon=3,
            start_mode="last",
            imag_gradient="reinforce",
            slow_critic=slow,
            amp_dtype=amp,
            scaler=scaler,
        )
    assert torch.isfinite(loss.total)
    assert int(slow.updates) == 3
    assert not torch.equal(slow.critic.net.net[-1].weight, critic.net.net[-1].weight)
    assert "slow_value" in metrics
    assert "weight" in metrics


def test_both_mode_default_mix_is_pure_reinforce() -> None:
    """DreamerV3's `imag_gradient_mix` is 0.0, so `both` must not add dynamics.

    The pre-fix code summed reinforce and the straight-through dynamics term at
    full weight, and the notebook silently ran this mode.
    """
    device = torch.device("cpu")
    wm = _tiny_wm()
    actor = Actor(wm.feat_dim, wm.rssm.action_dim, hidden=32, layers=1)
    critic = Critic(wm.feat_dim, hidden=32, layers=1, num_bins=21)
    optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-3)
    amp = parse_amp("off", device)
    scaler = make_grad_scaler(device, amp)
    loss, _metrics, _rollout = actor_critic_step(
        wm, actor, critic, optim, _batch(),
        device=device,
        retnorm=PercentileReturnNorm(),
        horizon=3,
        start_mode="last",
        imag_gradient="both",
        amp_dtype=amp,
        scaler=scaler,
    )
    assert torch.allclose(loss.actor, loss.reinforce - 3.0e-4 * loss.entropy, atol=1e-5)


def test_both_mode_mix_one_is_pure_dynamics() -> None:
    device = torch.device("cpu")
    wm = _tiny_wm()
    actor = Actor(wm.feat_dim, wm.rssm.action_dim, hidden=32, layers=1)
    critic = Critic(wm.feat_dim, hidden=32, layers=1, num_bins=21)
    optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-3)
    amp = parse_amp("off", device)
    scaler = make_grad_scaler(device, amp)
    loss, _metrics, _rollout = actor_critic_step(
        wm, actor, critic, optim, _batch(),
        device=device,
        retnorm=PercentileReturnNorm(),
        horizon=3,
        start_mode="last",
        imag_gradient="both",
        imag_gradient_mix=1.0,
        amp_dtype=amp,
        scaler=scaler,
    )
    assert torch.allclose(loss.actor, loss.backprop - 3.0e-4 * loss.entropy, atol=1e-5)


def test_reinforce_imagination_does_not_keep_rssm_graph() -> None:
    """Crafter is reinforce: img_step must not store a dynamics backward."""
    wm = _tiny_wm()
    freeze_world_model(wm)
    actor = Actor(wm.feat_dim, wm.rssm.action_dim, hidden=32, layers=1)
    critic = Critic(wm.feat_dim, hidden=32, layers=1, num_bins=21)
    batch = _batch(batch=2, seq=4)
    rollout = imagine_ahead(
        wm, actor, critic, batch["obs"], batch["actions"],
        horizon=3, start_mode="last", dynamics_graph=False,
    )
    assert not rollout.feat.requires_grad
    assert not rollout.reward.requires_grad
    assert rollout.log_prob.requires_grad
    actor.zero_grad(set_to_none=True)
    rollout.log_prob.mean().backward()
    assert actor.net.net[-1].weight.grad is not None


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
