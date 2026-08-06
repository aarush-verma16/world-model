"""RSSM mechanism diagnostics — shared by scripts and notebooks.

These plots check whether the RSSM *mechanism* is sane (entropy budget,
latent occupancy, h motion, imagination drift). They are not a quality
score until the world-model loss (M3) is training the model.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from torch import Tensor

from models.rssm import RSSM, RSSMOutput


def entropy(probs: Tensor) -> Tensor:
    """Per-categorical entropy in nats. `probs`: `[..., classes]` → `[...]`."""
    return -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)


def kl_categorical(p: Tensor, q: Tensor) -> Tensor:
    """KL(p || q) per categorical variable. `p`, `q`: `[..., classes]` → `[...]`."""
    return (p * (p.clamp_min(1e-8).log() - q.clamp_min(1e-8).log())).sum(dim=-1)


def pca_2d(x: np.ndarray) -> np.ndarray:
    """2-component PCA via SVD (no sklearn dependency)."""
    mean = x.mean(axis=0, keepdims=True)
    centered = x - mean
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def _maybe_save(fig: Figure, path: Path | None) -> Figure:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=140)
    return fig


def plot_entropy_over_time(
    prior_probs: Tensor,
    posterior_probs: Tensor,
    path: Path | None = None,
) -> tuple[Figure, np.ndarray, np.ndarray]:
    """Mean categorical entropy per timestep for prior vs posterior.

    Returns:
        (fig, prior_ent, post_ent) where each entropy array is `[time]`.
    """
    prior_ent = entropy(prior_probs).mean(dim=(0, 2)).detach().cpu().numpy()
    post_ent = entropy(posterior_probs).mean(dim=(0, 2)).detach().cpu().numpy()
    max_ent = float(np.log(prior_probs.shape[-1]))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(prior_ent, label="z_prior entropy", color="tab:orange")
    ax.plot(post_ent, label="z_posterior entropy", color="tab:blue")
    ax.axhline(
        max_ent, color="gray", linestyle="--", linewidth=1, label="max entropy (uniform)"
    )
    ax.set_xlabel("timestep")
    ax.set_ylabel("mean categorical entropy (nats)")
    ax.set_title("Latent entropy over a real rollout")
    ax.legend()
    fig.tight_layout()
    return _maybe_save(fig, path), prior_ent, post_ent


def plot_latent_occupancy(
    probs: Tensor,
    title: str,
    path: Path | None = None,
) -> tuple[Figure, float]:
    """Heatmap of mean per-class probability, aggregated over batch+time.

    Returns:
        (fig, underused_frac) where underused_frac is the fraction of
        categorical variables whose peak class never rises above 2x uniform.
    """
    occ = probs.mean(dim=(0, 1)).detach().cpu().numpy()  # [stoch, classes]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(occ, aspect="auto", cmap="viridis", vmin=0.0)
    ax.set_xlabel("class index")
    ax.set_ylabel("categorical variable index")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="mean probability")
    fig.tight_layout()
    underused = float((occ.max(axis=1) < (2.0 / occ.shape[1])).mean())
    return _maybe_save(fig, path), underused


def plot_h_trajectory(h: Tensor, path: Path | None = None) -> Figure:
    """PCA scatter of the deterministic state over time, one line per sequence."""
    batch, time, _dim = h.shape
    flat = h.reshape(batch * time, -1).detach().cpu().numpy()
    coords = pca_2d(flat).reshape(batch, time, 2)

    fig, ax = plt.subplots(figsize=(6, 6))
    sc = None
    for i in range(batch):
        ax.plot(coords[i, :, 0], coords[i, :, 1], alpha=0.5, linewidth=1, color="gray")
        sc = ax.scatter(
            coords[i, :, 0], coords[i, :, 1], c=np.arange(time), cmap="plasma", s=12
        )
    ax.set_title("Deterministic state h over time (PCA)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("timestep")
    fig.tight_layout()
    return _maybe_save(fig, path)


@torch.no_grad()
def imagination_divergence(
    rssm: RSSM,
    out: RSSMOutput,
    actions: Tensor,
    horizon: int,
    num_starts: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll `img_step` from several starts; measure drift from grounded `h`.

    Returns:
        (mean_curve, std_curve), each `[horizon]`.
    """
    batch, time, _deter = out.h.shape
    max_h = min(horizon, time - 1)
    if max_h < 1:
        raise ValueError("sequence too short for imagination_divergence")
    starts = list(range(0, time - max_h, max(1, (time - max_h) // num_starts)))[:num_starts]
    if not starts:
        starts = [0]

    curves = []
    for t0 in starts:
        h = out.h[:, t0]
        z_prior = out.z_prior[:, t0]
        dists = []
        for step in range(max_h):
            action = actions[:, t0 + step]
            h, z_prior, _ = rssm.img_step(h, z_prior, action)
            real_h = out.h[:, t0 + step + 1]
            dist = (h - real_h).pow(2).sum(dim=-1).sqrt().mean()
            dists.append(float(dist))
        curves.append(dists)
    arr = np.array(curves)  # [num_starts, max_h]
    return arr.mean(axis=0), arr.std(axis=0)


def plot_imagination_divergence(
    mean_curve: np.ndarray,
    std_curve: np.ndarray,
    path: Path | None = None,
) -> Figure:
    xs = np.arange(1, len(mean_curve) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, mean_curve, color="tab:red", label="mean ||h_imagined - h_real||")
    ax.fill_between(
        xs, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2, color="tab:red"
    )
    ax.set_xlabel("imagination steps ahead")
    ax.set_ylabel("L2 distance")
    ax.set_title("Imagination drift vs horizon")
    ax.legend()
    fig.tight_layout()
    return _maybe_save(fig, path)
