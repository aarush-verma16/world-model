"""RSSM diagnostic math (entropy, KL, PCA, imagination drift). Plotting is
exercised with the Agg backend so the suite never opens a window.
"""

from __future__ import annotations

import numpy as np
import torch

from models.rssm import RSSM, unimix_probs
from training.rssm_diagnostics import (
    entropy,
    imagination_divergence,
    kl_categorical,
    pca_2d,
    plot_entropy_over_time,
    plot_h_trajectory,
    plot_imagination_divergence,
    plot_latent_occupancy,
)


def test_uniform_entropy_is_log_classes() -> None:
    classes = 8
    probs = torch.full((2, 5, 4, classes), 1.0 / classes)
    ent = entropy(probs)
    assert ent.shape == (2, 5, 4)
    assert torch.allclose(ent, torch.full_like(ent, float(np.log(classes))), atol=1e-5)


def test_kl_categorical_zero_iff_identical_and_nonnegative() -> None:
    p = unimix_probs(torch.randn(3, 6, 4), unimix=0.01)
    q = unimix_probs(torch.randn(3, 6, 4), unimix=0.01)
    assert torch.allclose(kl_categorical(p, p), torch.zeros(3, 6), atol=1e-5)
    kl = kl_categorical(p, q)
    assert bool((kl >= -1e-6).all())
    assert float(kl.mean()) > 0.0


def test_pca_2d_shape() -> None:
    x = np.random.randn(20, 7).astype(np.float64)
    coords = pca_2d(x)
    assert coords.shape == (20, 2)


def test_imagination_divergence_finite_curve() -> None:
    rssm = RSSM(embed_dim=16, action_dim=4, deter_dim=8, stoch=4, classes=4, hidden=16)
    batch, time = 2, 12
    embeds = torch.randn(batch, time, 16)
    actions = torch.zeros(batch, time, 4)
    actions[..., 0] = 1.0
    out = rssm.observe(embeds, actions)
    mean, std = imagination_divergence(rssm, out, actions, horizon=5, num_starts=3)
    assert mean.shape == (5,)
    assert std.shape == (5,)
    assert np.isfinite(mean).all()
    assert np.isfinite(std).all()


def test_diagnostic_plots_return_finite_arrays() -> None:
    logits = torch.randn(2, 6, 4, 8)
    prior = unimix_probs(logits, 0.01)
    post = unimix_probs(logits + 0.3, 0.01)
    fig, prior_ent, post_ent = plot_entropy_over_time(prior, post)
    assert prior_ent.shape == (6,)
    assert post_ent.shape == (6,)
    fig2, underused = plot_latent_occupancy(post, "posterior occupancy")
    assert 0.0 <= underused <= 1.0
    fig3 = plot_h_trajectory(torch.randn(2, 8, 16))
    fig4 = plot_imagination_divergence(np.linspace(0.1, 1.0, 5), np.full(5, 0.05))
    import matplotlib.pyplot as plt

    plt.close(fig)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)
