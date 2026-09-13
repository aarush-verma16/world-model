"""Read-only actor action mass on a frozen joint checkpoint + replay.

Does not train. Finding 38 kill criteria:

- Unimix kill: if `place_table` and `make_wood_pickaxe` are both < 2×
  unimix/17 on this sample, do not train (imagination cannot invent craft).
  M17 500k already fails this kill (table is ~100× floor).
- Claimed-fix kill: same `--seed` / `--batch` / `--windows` after a graph
  change. Do not start a 5-day run unless `P(make_wood_pickaxe)` moves ≥2×
  vs the frozen baseline printed below (or advantage `R2_state_only` drops
  ≥0.10). Then a 25k env probe; abort if last-50 table/pickaxe stay ~0.

    conda activate worldmodel
    python scripts/diag_policy_mass.py
    python scripts/diag_policy_mass.py --joint-ckpt checkpoints/m17_xl_paper/ckpt_step_500000.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import crafter.constants as crafter_constants

from models.heads import rssm_features
from models.rssm import one_hot_action
from training.device import autocast_context, configure_runtime, get_device, parse_amp, warn_if_not_cuda
from training.imagine import freeze_world_model
from training.replay_buffer import ReplayBuffer

from train_agent import load_yaml, make_actor_critic
from train_world_model import build_model

# Actions that leave the sleep/plant island (finding 37). Unimix floor is
# `unimix / action_dim` (~0.000588 at 0.01 / 17). Frozen M17 500k baseline
# (seed 0, batch 8, 32 windows) is in finding 38 / catalog.json policy_mass.
KEY_ACTIONS: tuple[str, ...] = (
    "sleep",
    "place_plant",
    "do",
    "place_table",
    "make_wood_pickaxe",
    "make_wood_sword",
    "place_stone",
)


def _names() -> tuple[str, ...]:
    return tuple(str(n) for n in crafter_constants.actions)


def _mean_entropy(probs: torch.Tensor) -> float:
    p = probs.clamp_min(1e-8)
    return float((-(p * p.log()).sum(dim=-1)).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/m17_xl_paper.yaml"))
    parser.add_argument(
        "--joint-ckpt",
        type=Path,
        default=Path("checkpoints/m17_xl_paper/ckpt_step_500000.pt"),
    )
    parser.add_argument(
        "--replay",
        type=Path,
        default=None,
        help="Replay dump. Default: train.replay_out from the config.",
    )
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--windows", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    train = cfg["train"]
    seq_len = int(train["seq_len"])
    replay_path = Path(args.replay) if args.replay is not None else Path(train["replay_out"])
    if not args.joint_ckpt.is_file():
        raise FileNotFoundError(f"joint checkpoint not found: {args.joint_ckpt}")
    if not replay_path.is_file():
        raise FileNotFoundError(f"replay not found: {replay_path}")

    device = get_device()
    configure_runtime(device)
    warn_if_not_cuda(device)
    amp_dtype = parse_amp(train.get("amp", "bf16"), device)

    import yaml

    wm_cfg = yaml.safe_load(Path(cfg["world_model_config"]).read_text(encoding="utf-8"))
    world_model = build_model(wm_cfg).to(device)
    actor, _critic = make_actor_critic(cfg, world_model, device)
    payload = torch.load(args.joint_ckpt, weights_only=False, map_location=device)
    world_model.load_state_dict(payload["world_model"], strict=True)
    actor.load_state_dict(payload["actor"], strict=True)
    freeze_world_model(world_model)
    actor.eval()
    env_steps = int(payload.get("env_steps", -1))
    unimix = float(actor.unimix)
    names = _names()
    action_dim = int(actor.action_dim)
    if len(names) != action_dim:
        raise RuntimeError(f"crafter actions {len(names)} != actor.action_dim {action_dim}")
    floor = unimix / float(action_dim)
    name_to_i = {n: i for i, n in enumerate(names)}

    buffer = ReplayBuffer(seed=int(args.seed), max_steps=None)
    buffer.load_state_dict(torch.load(replay_path, weights_only=False, map_location="cpu"))
    print(
        f"joint {args.joint_ckpt} env_steps={env_steps}  "
        f"replay {replay_path} episodes={len(buffer)} steps={buffer.num_steps}  "
        f"device={device}  seed={args.seed} batch={args.batch} windows={args.windows} seq={seq_len}"
    )
    print(f"unimix={unimix}  floor=unimix/A={floor:.6f}  entropy_floor~0.079 if greedy")

    prob_chunks: list[torch.Tensor] = []
    stored_chunks: list[torch.Tensor] = []
    n_need = int(args.windows)
    n_got = 0
    while n_got < n_need:
        take = min(int(args.batch), n_need - n_got)
        batch = buffer.sample(take, seq_len)
        obs = batch["obs"].to(device, non_blocking=True)
        act = batch["actions"].to(device, non_blocking=True)
        is_first = batch.get("is_first")
        if is_first is not None:
            is_first = is_first.to(device, non_blocking=True)
        with torch.no_grad(), autocast_context(device, amp_dtype):
            embeds = world_model.encode(obs)
            act_oh = one_hot_action(act, world_model.rssm.action_dim)
            rssm_out = world_model.rssm.observe(embeds, act_oh, is_first=is_first)
            feat = rssm_features(rssm_out.h, rssm_out.z_posterior)
            _action, _logp, _ent, probs = actor.policy(feat)
        prob_chunks.append(probs.float().reshape(-1, action_dim).cpu())
        stored_chunks.append(act.reshape(-1).cpu())
        n_got += take

    probs = torch.cat(prob_chunks, dim=0)
    stored = torch.cat(stored_chunks, dim=0)
    mean_p = probs.mean(dim=0).numpy()
    entropy = _mean_entropy(probs)
    stored_hist = np.bincount(stored.numpy(), minlength=action_dim).astype(np.float64)
    stored_hist /= stored_hist.sum()

    print(f"n_states={probs.shape[0]}  mean_entropy={entropy:.4f}")
    print(f"{'action':22s}  {'P_actor':>9s}  {'x_floor':>8s}  {'replay%':>8s}")
    order = list(KEY_ACTIONS) + [n for n in names if n not in KEY_ACTIONS]
    for name in order:
        i = name_to_i[name]
        p = float(mean_p[i])
        print(
            f"{name:22s}  {p:9.5f}  {p / floor:8.2f}  {100.0 * stored_hist[i]:8.2f}"
        )

    craft = ("place_table", "make_wood_pickaxe")
    craft_max = max(float(mean_p[name_to_i[n]]) for n in craft)
    p_pick = float(mean_p[name_to_i["make_wood_pickaxe"]])
    # Frozen M17 500k, seed 0 / batch 8 / 32 windows (finding 38).
    baseline_pick = 0.0035
    print(
        "kill: craft mass is unimix-floor if place_table and make_wood_pickaxe "
        f"are both < 2*floor ({2 * floor:.6f}); max(table,pick)={craft_max:.5f}"
    )
    if craft_max < 2.0 * floor:
        print("GATE FAIL: crafting actions at unimix floor on this replay sample.")
    else:
        print("GATE: crafting mass is above 2x floor on this sample (not a kill by itself).")
    print(
        f"claimed-fix kill (finding 38): P(make_wood_pickaxe)={p_pick:.5f}; "
        f"need ≥ {2.0 * baseline_pick:.5f} (2× frozen 500k baseline {baseline_pick}) "
        "on seed=0/batch=8/windows=32 before a 5-day run."
    )


if __name__ == "__main__":
    main()
