"""Lambda-returns and DreamerV3 percentile return normalization."""

from __future__ import annotations

import torch

from training.returns import PercentileReturnNorm, lambda_returns


def test_lambda_returns_rejects_shape_mismatch() -> None:
    reward = torch.ones(2, 4)
    try:
        lambda_returns(reward, torch.ones(2, 3), torch.ones(2, 4), lam=0.95)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for shape mismatch")


def test_lambda_returns_rejects_non_2d() -> None:
    x = torch.ones(2, 4, 1)
    try:
        lambda_returns(x, x, x, lam=0.95)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for ndim != 2")


def test_percentile_norm_scale_never_below_limit() -> None:
    norm = PercentileReturnNorm(decay=0.5, limit=1.0)
    tiny = torch.full((8, 4), 0.01)
    scale = norm.update(tiny)
    assert float(scale) >= 1.0


def test_percentile_norm_wide_returns_grow_scale() -> None:
    norm = PercentileReturnNorm(decay=0.0, limit=1.0)  # no EMA smoothing
    wide = torch.linspace(-10.0, 10.0, 200)
    scale = norm.update(wide)
    # 5th–95th of [-10, 10] is well above the floor of 1.
    assert float(scale) > 5.0


def test_percentile_norm_does_not_change_returns_graph() -> None:
    norm = PercentileReturnNorm()
    returns = torch.randn(4, 6, requires_grad=True)
    scaled, scale = norm.normalize(returns)
    assert scaled.shape == returns.shape
    assert not scale.requires_grad
    scaled.sum().backward()
    assert returns.grad is not None
    assert torch.isfinite(returns.grad).all()


def test_percentile_norm_state_dict_roundtrip() -> None:
    norm = PercentileReturnNorm(decay=0.8, limit=2.0)
    norm.update(torch.linspace(0.0, 20.0, 50))
    other = PercentileReturnNorm()
    other.load_state_dict(norm.state_dict())
    assert other.decay == 0.8
    assert other.limit == 2.0
    assert other._low is not None and other._high is not None
    assert abs(float(other._low) - float(norm._low)) < 1e-6
    assert abs(float(other._high) - float(norm._high)) < 1e-6
