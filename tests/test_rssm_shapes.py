"""Shape, naming, STE gradient, unimix, and long-horizon stability tests for
the RSSM (layer-normalized GRU + unimix categorical latents)."""

from __future__ import annotations

import torch

from models.rssm import (
    RSSM,
    GRUCellLayerNorm,
    get_activation,
    sample_onehot_ste,
    unimix_probs,
)


def test_unimix_probs_sums_to_one_and_has_no_dead_classes() -> None:
    # Extremely peaked logits: without unimix this would put ~0 probability
    # on every class but one.
    logits = torch.zeros(2, 4, 8)
    logits[..., 0] = 50.0
    probs = unimix_probs(logits, unimix=0.01)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2, 4), atol=1e-5)
    # Every class keeps at least the unimix floor mass.
    floor = 0.01 / 8
    assert bool((probs >= floor - 1e-6).all())


def test_sample_onehot_ste_shapes_and_grads() -> None:
    logits = torch.randn(3, 8, 16, requires_grad=True)
    probs = unimix_probs(logits, unimix=0.01)
    sample = sample_onehot_ste(probs)
    assert sample.shape == (3, 8, 16)
    assert torch.allclose(sample.sum(dim=-1), torch.ones(3, 8), atol=1e-5)
    sample.sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_gru_cell_layer_norm_shapes_and_finite() -> None:
    cell = GRUCellLayerNorm(input_size=10, hidden_size=6)
    x = torch.randn(4, 10)
    h = torch.zeros(4, 6)
    h2 = cell(x, h)
    assert h2.shape == (4, 6)
    assert torch.isfinite(h2).all()


def test_gru_cell_default_candidate_activation_is_silu() -> None:
    # DreamerV3's GRU activates the candidate gate with the network's default
    # act (SiLU), not textbook GRU's tanh — verify that's actually wired up.
    cell = GRUCellLayerNorm(input_size=4, hidden_size=4)
    assert cell.act is get_activation("silu")
    tanh_cell = GRUCellLayerNorm(input_size=4, hidden_size=4, act="tanh")
    assert tanh_cell.act is get_activation("tanh")


def test_get_activation_unknown_raises() -> None:
    try:
        get_activation("not-a-real-activation")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown activation")


def test_observe_shapes_and_naming() -> None:
    batch, time = 2, 12
    embed_dim, action_dim = 64, 17
    rssm = RSSM(
        embed_dim=embed_dim,
        action_dim=action_dim,
        deter_dim=32,
        stoch=8,
        classes=8,
        hidden=64,
    )
    embeds = torch.randn(batch, time, embed_dim)
    actions = torch.zeros(batch, time, action_dim)
    actions[:, :, 0] = 1.0
    out = rssm.observe(embeds, actions)

    assert out.h.shape == (batch, time, 32)
    assert out.z_prior.shape == (batch, time, 8, 8)
    assert out.z_posterior.shape == (batch, time, 8, 8)
    assert out.prior_logits.shape == (batch, time, 8, 8)
    assert out.posterior_logits.shape == (batch, time, 8, 8)
    # Prior and posterior must be distinct tensors (not aliased).
    assert out.z_prior.data_ptr() != out.z_posterior.data_ptr()
    assert out.prior_logits.data_ptr() != out.posterior_logits.data_ptr()


def test_observe_gradients_reach_logits() -> None:
    rssm = RSSM(embed_dim=32, action_dim=5, deter_dim=16, stoch=4, classes=4, hidden=32)
    embeds = torch.randn(2, 8, 32, requires_grad=True)
    actions = torch.zeros(2, 8, 5)
    actions[..., 1] = 1.0
    out = rssm.observe(embeds, actions)
    # Weighted sums so the STE soft path is non-constant (bare .sum() on one-hots is).
    w_prior = torch.randn_like(out.z_prior)
    w_post = torch.randn_like(out.z_posterior)
    loss = (
        (out.z_prior * w_prior).sum()
        + (out.z_posterior * w_post).sum()
        + out.h.pow(2).mean()
    )
    loss.backward()

    prior_w = rssm.prior_net[-1].weight
    post_w = rssm.posterior_net[-1].weight
    assert prior_w.grad is not None and prior_w.grad.abs().sum() > 0
    assert post_w.grad is not None and post_w.grad.abs().sum() > 0
    assert embeds.grad is not None and embeds.grad.abs().sum() > 0


def test_long_horizon_no_nan() -> None:
    rssm = RSSM(embed_dim=48, action_dim=17, deter_dim=64, stoch=8, classes=8, hidden=64)
    embeds = torch.randn(2, 128, 48)
    actions = torch.zeros(2, 128, 17)
    actions[..., 3] = 1.0
    out = rssm.observe(embeds, actions)
    for name in ("h", "z_prior", "z_posterior", "prior_logits", "posterior_logits"):
        tensor = getattr(out, name)
        assert torch.isfinite(tensor).all(), f"{name} has non-finite values"


def test_learned_initial_state_matches_zeros_at_init_but_gets_gradient() -> None:
    # DreamerV3's learned h_0 is init to zero (so it starts identical to the
    # "zeros" mode) but is a real trainable parameter that gets corrected once
    # a loss backprops through it.
    torch.manual_seed(0)
    rssm = RSSM(
        embed_dim=16, action_dim=4, deter_dim=8, stoch=4, classes=4, hidden=16,
        initial="learned",
    )
    state = rssm.initial(batch_size=5)
    assert state.h.shape == (5, 8)
    assert torch.allclose(state.h, torch.zeros(5, 8), atol=1e-6)
    assert rssm._initial_deter.requires_grad

    embeds = torch.randn(5, 6, 16)
    actions = torch.zeros(5, 6, 4)
    actions[..., 0] = 1.0
    out = rssm.observe(embeds, actions)
    out.h.pow(2).mean().backward()
    assert rssm._initial_deter.grad is not None
    assert float(rssm._initial_deter.grad.abs().sum()) > 0


def test_zeros_initial_state_has_no_learned_parameter() -> None:
    rssm = RSSM(
        embed_dim=8, action_dim=3, deter_dim=6, stoch=2, classes=4, hidden=8,
        initial="zeros",
    )
    assert not hasattr(rssm, "_initial_deter")
    state = rssm.initial(batch_size=3)
    assert torch.equal(state.h, torch.zeros(3, 6))


def test_invalid_initial_mode_raises() -> None:
    try:
        RSSM(embed_dim=8, action_dim=3, deter_dim=6, stoch=2, classes=4, initial="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for invalid initial mode")


def test_rec_depth_repeats_cell_and_changes_result() -> None:
    embed_dim, action_dim = 12, 4
    embeds = torch.randn(3, 5, embed_dim)
    actions = torch.zeros(3, 5, action_dim)
    actions[..., 1] = 1.0

    torch.manual_seed(42)
    rssm1 = RSSM(
        embed_dim=embed_dim, action_dim=action_dim, deter_dim=10, stoch=4,
        classes=4, hidden=16, initial="zeros", rec_depth=1,
    )
    torch.manual_seed(42)
    rssm3 = RSSM(
        embed_dim=embed_dim, action_dim=action_dim, deter_dim=10, stoch=4,
        classes=4, hidden=16, initial="zeros", rec_depth=3,
    )
    out1 = rssm1.observe(embeds, actions)
    out3 = rssm3.observe(embeds, actions)
    assert out1.h.shape == out3.h.shape
    assert torch.isfinite(out3.h).all()
    # Same weights (same seed), but 3x recurrent depth should move h further
    # from the zero starting state than a single application.
    assert not torch.allclose(out1.h, out3.h)


def test_rec_depth_must_be_positive() -> None:
    try:
        RSSM(embed_dim=8, action_dim=3, deter_dim=6, stoch=2, classes=4, rec_depth=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for rec_depth < 1")


def test_imagine_shapes() -> None:
    rssm = RSSM(embed_dim=32, action_dim=5, deter_dim=16, stoch=4, classes=4, hidden=32)
    state = rssm.initial(3)
    actions = torch.zeros(3, 10, 5)
    actions[..., 0] = 1.0
    h, z_prior, prior_logits = rssm.imagine(state.h, state.z_posterior, actions)
    assert h.shape == (3, 10, 16)
    assert z_prior.shape == (3, 10, 4, 4)
    assert prior_logits.shape == (3, 10, 4, 4)
    assert torch.isfinite(h).all()
