"""M2 exit-criteria check: RSSM forward pass on real Crafter sequences.

Collects short random-policy rollouts, encodes frames, runs RSSM.observe over
batch×time, then checks:
  - shapes / finite values over long horizons (100+ steps)
  - gradients flow into prior and posterior categorical logits (STE)

Usage:
    conda activate worldmodel
    python scripts/verify_rssm_forward.py

For visual/mechanism diagnostics (entropy, latent occupancy, h trajectory,
imagination drift), see scripts/visualize_rssm.py instead.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from models.encoder import Encoder
from models.rssm import RSSM, one_hot_action
from training.device import configure_runtime, describe_device, get_device
from training.rollout import collect_sequences, encode_sequence


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
    configure_runtime(device)
    print(f"device: {describe_device(device)}")

    enc_cfg = cfg["encoder"]
    rssm_cfg = cfg["rssm"]
    env_cfg = cfg["env"]
    ver = cfg["verify"]
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

    print(
        f"Encoder embed={enc_cfg['embed_dim']} | "
        f"RSSM h={rssm_cfg['deter_dim']} z={rssm_cfg['stoch']}x{rssm_cfg['classes']} "
        f"unimix={rssm.unimix} act={rssm.cell.act.__name__ if hasattr(rssm.cell.act, '__name__') else rssm_cfg.get('act')} "
        f"initial={rssm.initial_mode} rec_depth={rssm.rec_depth} cell=GRUCellLayerNorm"
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
    print(f"obs {tuple(obs_u8.shape)}  actions {tuple(actions_i.shape)}")

    embeds = encode_sequence(encoder, obs_u8, device)
    actions = one_hot_action(actions_i.to(device), action_dim)
    print(f"embeds {tuple(embeds.shape)}  actions_oh {tuple(actions.shape)}")

    out = rssm.observe(embeds, actions)
    print(
        f"observe → h {tuple(out.h.shape)}  "
        f"z_prior {tuple(out.z_prior.shape)}  "
        f"z_posterior {tuple(out.z_posterior.shape)}"
    )
    for name in ("h", "z_prior", "z_posterior", "prior_logits", "posterior_logits"):
        t = getattr(out, name)
        if not torch.isfinite(t).all():
            raise SystemExit(f"FAIL: non-finite values in {name}")
    print("finite check (seq_len): PASS")

    # Long-horizon stress on random embeddings (catch broken recurrence early).
    long_t = int(ver["long_horizon"])
    long_embeds = torch.randn(batch, long_t, int(enc_cfg["embed_dim"]), device=device)
    long_actions = one_hot_action(
        torch.randint(0, action_dim, (batch, long_t), device=device), action_dim
    )
    long_out = rssm.observe(long_embeds, long_actions)
    for name in ("h", "z_prior", "z_posterior", "prior_logits", "posterior_logits"):
        t = getattr(long_out, name)
        if not torch.isfinite(t).all():
            raise SystemExit(f"FAIL: non-finite in long-horizon {name}")
    print(f"finite check (T={long_t}): PASS")

    # STE gradient check into categorical logit heads.
    rssm.zero_grad(set_to_none=True)
    encoder.zero_grad(set_to_none=True)
    embeds_g = encode_sequence(encoder, obs_u8, device)
    out_g = rssm.observe(embeds_g, actions)
    # Weighted one-hot sums → non-constant STE soft path into both logit heads.
    w_prior = torch.randn_like(out_g.z_prior)
    w_post = torch.randn_like(out_g.z_posterior)
    loss = (
        (out_g.z_prior * w_prior).sum()
        + (out_g.z_posterior * w_post).sum()
        + out_g.h.pow(2).mean()
    )
    loss.backward()

    prior_grad = rssm.prior_net[-1].weight.grad
    post_grad = rssm.posterior_net[-1].weight.grad
    if prior_grad is None or float(prior_grad.abs().sum()) == 0.0:
        raise SystemExit("FAIL: no gradient into prior logits (STE broken?)")
    if post_grad is None or float(post_grad.abs().sum()) == 0.0:
        raise SystemExit("FAIL: no gradient into posterior logits (STE broken?)")
    print(
        f"STE grads: prior={float(prior_grad.abs().sum()):.4f}  "
        f"posterior={float(post_grad.abs().sum()):.4f}  PASS"
    )

    # Naming sanity: ensure we never only expose a bare z.
    assert hasattr(out, "z_prior") and hasattr(out, "z_posterior")
    print()
    print("M2 RSSM forward-pass verification: ALL CHECKS PASSED")
    print("For visual diagnostics: python scripts/visualize_rssm.py")
    print("Next: Milestone 3 (full world-model loss + replay).")


if __name__ == "__main__":
    main()
