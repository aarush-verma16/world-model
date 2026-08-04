"""M2 visual diagnostics for the RSSM — mechanism checks, not quality checks.

The RSSM has no training signal yet (that's M3's KL loss), so these plots are
NOT about whether it predicts the future well. They're about whether the
*mechanism* itself is sane on a fresh, randomly-initialized model:

  1. latent_entropy.png       — does z_prior / z_posterior actually use its
                                 categorical budget, or collapse toward one class?
  2. latent_occupancy_*.png   — per-categorical-variable class usage heatmap;
                                 dark columns = "dead" classes.
  3. h_trajectory_pca.png     — does the deterministic state move smoothly
                                 over a rollout, or jump/freeze?
  4. imagination_divergence.png — how fast does open-loop imagination (h from
                                 img_step only) drift from the grounded h that
                                 saw real observations, as a function of
                                 steps-ahead? A preview of the M4/ablation
                                 "imagination error growth" question.

Usage:
    conda activate worldmodel
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    python scripts/visualize_rssm.py
    open results/m2/*.png
    tensorboard --logdir runs   # runs/m2_rssm_diagnostics
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplcache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from models.encoder import Encoder
from models.rssm import RSSM, RSSMOutput, one_hot_action, unimix_probs
from training.device import get_device
from training.rollout import collect_sequences, encode_sequence


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def entropy(probs: Tensor) -> Tensor:
    """Per-categorical entropy in nats. `probs`: `[..., classes]` → `[...]`."""
    return -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)


def kl_categorical(p: Tensor, q: Tensor) -> Tensor:
    """KL(p || q) per categorical variable. `p`, `q`: `[..., classes]` → `[...]`.

    Diagnostic only (no gradient/loss use here — that's M3's KL-balanced loss).
    """
    return (p * (p.clamp_min(1e-8).log() - q.clamp_min(1e-8).log())).sum(dim=-1)


def plot_entropy_over_time(
    prior_probs: Tensor, posterior_probs: Tensor, path: Path
) -> tuple[np.ndarray, np.ndarray]:
    """Mean categorical entropy per timestep for prior vs posterior."""
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
    ax.set_title("Latent entropy over a real rollout (untrained RSSM)")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return prior_ent, post_ent


def plot_latent_occupancy(probs: Tensor, path: Path, title: str) -> float:
    """Heatmap of mean per-class probability, aggregated over batch+time.

    Returns the fraction of (categorical variable) rows whose peak class never
    rises above 2x uniform — a rough "dead / underused latent" indicator.
    """
    occ = probs.mean(dim=(0, 1)).detach().cpu().numpy()  # [stoch, classes]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(occ, aspect="auto", cmap="viridis", vmin=0.0)
    ax.set_xlabel("class index")
    ax.set_ylabel("categorical variable index")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="mean probability")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    underused = float((occ.max(axis=1) < (2.0 / occ.shape[1])).mean())
    return underused


def pca_2d(x: np.ndarray) -> np.ndarray:
    """2-component PCA via SVD (no sklearn dependency)."""
    mean = x.mean(axis=0, keepdims=True)
    centered = x - mean
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def plot_h_trajectory(h: Tensor, path: Path) -> None:
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
    ax.set_title("Deterministic state h over time (PCA, untrained RSSM)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("timestep")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


@torch.no_grad()
def imagination_divergence(
    rssm: RSSM,
    out: RSSMOutput,
    actions: Tensor,
    horizon: int,
    num_starts: int,
) -> tuple[np.ndarray, np.ndarray]:
    """From several start points on the real trajectory, roll `img_step`
    forward with the real actions and measure drift from the grounded `h`
    that continued to see real observations at each step.

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


def plot_imagination_divergence(mean_curve: np.ndarray, std_curve: np.ndarray, path: Path) -> None:
    xs = np.arange(1, len(mean_curve) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, mean_curve, color="tab:red", label="mean ||h_imagined - h_real||")
    ax.fill_between(
        xs, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2, color="tab:red"
    )
    ax.set_xlabel("imagination steps ahead")
    ax.set_ylabel("L2 distance")
    ax.set_title("Imagination drift vs horizon (untrained RSSM, mechanism check)")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/m2_rssm.yaml"))
    args = parser.parse_args()
    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg["seed"]))
    device = get_device()
    print(f"device: {device}")

    enc_cfg = cfg["encoder"]
    rssm_cfg = cfg["rssm"]
    env_cfg = cfg["env"]
    ver = cfg["verify"]
    diag = cfg.get("diagnostics", {})
    action_dim = int(env_cfg["action_dim"])

    encoder = Encoder(
        embed_dim=int(enc_cfg["embed_dim"]),
        channels=tuple(int(c) for c in enc_cfg["channels"]),
    ).to(device)
    rssm = RSSM(
        embed_dim=int(enc_cfg["embed_dim"]),
        action_dim=action_dim,
        deter_dim=int(rssm_cfg["deter_dim"]),
        stoch=int(rssm_cfg["stoch"]),
        classes=int(rssm_cfg["classes"]),
        hidden=int(rssm_cfg["hidden"]),
        unimix=float(rssm_cfg.get("unimix", 0.01)),
        act=str(rssm_cfg.get("act", "silu")),
        initial=str(rssm_cfg.get("initial", "learned")),
        rec_depth=int(rssm_cfg.get("rec_depth", 1)),
    ).to(device)
    encoder.eval()
    rssm.eval()
    print(
        f"RSSM: unimix={rssm.unimix} initial={rssm.initial_mode} "
        f"rec_depth={rssm.rec_depth} deter={rssm.deter_dim} z={rssm.stoch}x{rssm.classes}"
    )

    print(
        f"NOTE: model is randomly initialized (no training yet — that's M3).\n"
        f"These plots check the RSSM *mechanism*, not prediction quality."
    )

    print("collecting Crafter sequences...")
    obs_u8, actions_i = collect_sequences(
        env_id=str(env_cfg["id"]),
        num_episodes=int(ver["num_episodes"]),
        seq_len=int(ver["seq_len"]),
        max_episode_steps=int(ver["max_episode_steps"]),
        action_dim=action_dim,
    )
    batch = min(int(ver["batch_size"]), obs_u8.shape[0])
    obs_u8 = obs_u8[:batch]
    actions_i = actions_i[:batch]

    results_dir = Path(diag.get("results_dir", "results/m2"))
    log_dir = Path(diag.get("log_dir", "runs/m2_rssm_diagnostics"))
    results_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    with torch.no_grad():
        embeds = encode_sequence(encoder, obs_u8, device)
        actions = one_hot_action(actions_i.to(device), action_dim)
        out = rssm.observe(embeds, actions)
        prior_probs = unimix_probs(out.prior_logits, rssm.unimix)
        posterior_probs = unimix_probs(out.posterior_logits, rssm.unimix)

    print(f"rollout: batch={batch} time={embeds.shape[1]}")

    # 1) Entropy over time.
    prior_ent, post_ent = plot_entropy_over_time(
        prior_probs, posterior_probs, results_dir / "latent_entropy.png"
    )
    max_ent = float(np.log(rssm.classes))
    print(
        f"entropy: prior mean={prior_ent.mean():.3f}  posterior mean={post_ent.mean():.3f}  "
        f"(max possible with {rssm.classes} classes={max_ent:.3f})"
    )
    writer.add_scalar("m2/entropy_prior_mean", float(prior_ent.mean()), 0)
    writer.add_scalar("m2/entropy_posterior_mean", float(post_ent.mean()), 0)

    # 2) Latent occupancy (dead-code check).
    dead_prior = plot_latent_occupancy(
        prior_probs, results_dir / "latent_occupancy_prior.png", "z_prior class occupancy"
    )
    dead_post = plot_latent_occupancy(
        posterior_probs,
        results_dir / "latent_occupancy_posterior.png",
        "z_posterior class occupancy",
    )
    print(
        f"underused-class fraction: prior={dead_prior:.3f}  posterior={dead_post:.3f}  "
        f"(fraction of categorical variables with no class > 2x uniform mass. "
        f"On this untrained model ~1.0 is *expected*: weights are random, so "
        f"probabilities are near-uniform everywhere — nothing has collapsed. "
        f"After M3 training, watch this: it should drop as latents specialize, "
        f"and a value stuck near 1.0 post-training would flag posterior collapse.)"
    )
    writer.add_scalar("m2/underused_class_frac_prior", dead_prior, 0)
    writer.add_scalar("m2/underused_class_frac_posterior", dead_post, 0)

    # 3) h trajectory.
    plot_h_trajectory(out.h, results_dir / "h_trajectory_pca.png")

    # 4) Imagination divergence vs horizon.
    horizon = int(diag.get("imagination_horizon", min(30, embeds.shape[1] - 1)))
    num_starts = int(diag.get("imagination_num_starts", 6))
    mean_curve, std_curve = imagination_divergence(
        rssm, out, actions, horizon=horizon, num_starts=num_starts
    )
    plot_imagination_divergence(
        mean_curve, std_curve, results_dir / "imagination_divergence.png"
    )
    print(
        f"imagination drift: step1={mean_curve[0]:.3f}  "
        f"step{len(mean_curve)}={mean_curve[-1]:.3f}  "
        f"(untrained model — expect drift to grow with horizon; the *rate* of "
        f"growth is what M4's ablation will care about once trained)"
    )
    for i, v in enumerate(mean_curve, start=1):
        writer.add_scalar("m2/imagination_drift", float(v), i)

    # Log the PNGs as TensorBoard images too, for one-stop viewing.
    from PIL import Image
    from torchvision.transforms.functional import to_tensor

    for name in (
        "latent_entropy.png",
        "latent_occupancy_prior.png",
        "latent_occupancy_posterior.png",
        "h_trajectory_pca.png",
        "imagination_divergence.png",
    ):
        img = Image.open(results_dir / name).convert("RGB")
        writer.add_image(f"m2/{name}", to_tensor(img), 0)

    writer.flush()
    writer.close()

    print()
    print(f"Wrote diagnostics to {results_dir}/ and {log_dir}/")
    print("M2 visual diagnostics complete.")


if __name__ == "__main__":
    main()
