"""M2 visual diagnostics for the RSSM — mechanism checks, not quality checks.

The RSSM has no training signal yet (that's M3's KL loss), so these plots are
NOT about whether it predicts the future well. They're about whether the
*mechanism* itself is sane on a fresh, randomly-initialized model.

Plot helpers live in `training.rssm_diagnostics` so notebooks can render the
same graphs inline. This script writes PNGs + TensorBoard images.

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
from torch.utils.tensorboard import SummaryWriter

from models.encoder import Encoder
from models.rssm import RSSM, one_hot_action, unimix_probs
from training.device import get_device
from training.rollout import collect_sequences, encode_sequence
from training.rssm_diagnostics import (
    imagination_divergence,
    plot_entropy_over_time,
    plot_h_trajectory,
    plot_imagination_divergence,
    plot_latent_occupancy,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
        "NOTE: model is randomly initialized (no training yet — that's M3).\n"
        "These plots check the RSSM *mechanism*, not prediction quality."
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

    fig, prior_ent, post_ent = plot_entropy_over_time(
        prior_probs, posterior_probs, results_dir / "latent_entropy.png"
    )
    plt.close(fig)
    max_ent = float(np.log(rssm.classes))
    print(
        f"entropy: prior mean={prior_ent.mean():.3f}  posterior mean={post_ent.mean():.3f}  "
        f"(max possible with {rssm.classes} classes={max_ent:.3f})"
    )
    writer.add_scalar("m2/entropy_prior_mean", float(prior_ent.mean()), 0)
    writer.add_scalar("m2/entropy_posterior_mean", float(post_ent.mean()), 0)

    fig, dead_prior = plot_latent_occupancy(
        prior_probs,
        "z_prior class occupancy",
        results_dir / "latent_occupancy_prior.png",
    )
    plt.close(fig)
    fig, dead_post = plot_latent_occupancy(
        posterior_probs,
        "z_posterior class occupancy",
        results_dir / "latent_occupancy_posterior.png",
    )
    plt.close(fig)
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

    fig = plot_h_trajectory(out.h, results_dir / "h_trajectory_pca.png")
    plt.close(fig)

    horizon = int(diag.get("imagination_horizon", min(30, embeds.shape[1] - 1)))
    num_starts = int(diag.get("imagination_num_starts", 6))
    mean_curve, std_curve = imagination_divergence(
        rssm, out, actions, horizon=horizon, num_starts=num_starts
    )
    fig = plot_imagination_divergence(
        mean_curve, std_curve, results_dir / "imagination_divergence.png"
    )
    plt.close(fig)
    print(
        f"imagination drift: step1={mean_curve[0]:.3f}  "
        f"step{len(mean_curve)}={mean_curve[-1]:.3f}  "
        f"(untrained model — expect drift to grow with horizon; the *rate* of "
        f"growth is what M4's ablation will care about once trained)"
    )
    for i, v in enumerate(mean_curve, start=1):
        writer.add_scalar("m2/imagination_drift", float(v), i)

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
