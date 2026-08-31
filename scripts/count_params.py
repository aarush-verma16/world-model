"""Print Dreamer size-S/M/L/XL parameter counts; optional one CUDA train step.

The paper's Crafter 14.5 number is size XL (~200M). This box is 16 GiB.
Weights at 200M bf16 are ~0.4 GiB; the risk is activation VRAM on
batch 16 × seq 32, especially imagination with start_mode=all.

    python scripts/count_params.py
    python scripts/count_params.py --smoke --size xl
    python scripts/count_params.py --smoke --size xl_b2
    python scripts/count_params.py --smoke --size xl --start-mode last --batch-size 8
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import yaml

from agents.actor_critic import Actor, Critic
from training.ac_step import actor_critic_step
from training.device import (
    configure_runtime,
    describe_device,
    get_device,
    make_grad_scaler,
    parse_amp,
    vram_peak_gb,
    warn_if_not_cuda,
)
from training.returns import PercentileReturnNorm
from training.wm_step import world_model_step
from train_world_model import build_model

SIZES = {
    "s": Path("configs/sizes/dreamer_s.yaml"),
    "m": Path("configs/sizes/dreamer_m.yaml"),
    "l": Path("configs/sizes/dreamer_l.yaml"),
    "xl": Path("configs/sizes/dreamer_xl.yaml"),
    "xl_b2": Path("configs/sizes/dreamer_xl_b2.yaml"),
}

ACTOR = {
    "s": (512, 2),
    "m": (640, 3),
    "l": (768, 4),
    "xl": (1024, 5),
    "xl_b2": (1024, 5),
}

PAPER_M = {"s": 18, "m": 37, "l": 77, "xl": 200, "xl_b2": 200}


def _n_m(module: torch.nn.Module) -> float:
    return sum(p.numel() for p in module.parameters()) / 1e6


def _load(size: str) -> dict:
    path = SIZES[size]
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def count_one(size: str) -> dict[str, float]:
    cfg = _load(size)
    wm = build_model(cfg)
    hidden, layers = ACTOR[size]
    actor = Actor(wm.feat_dim, wm.rssm.action_dim, hidden=hidden, layers=layers)
    critic = Critic(wm.feat_dim, hidden=hidden, layers=layers)
    wm_m = _n_m(wm)
    ac_m = _n_m(actor) + _n_m(critic)
    return {
        "wm": wm_m,
        "actor_critic": ac_m,
        "total": wm_m + ac_m,
        "paper_wm": float(PAPER_M[size]),
    }


def smoke(
    size: str,
    *,
    batch_size: int | None,
    seq_len: int | None,
    start_mode: str,
) -> None:
    cfg = _load(size)
    device = get_device()
    configure_runtime(device)
    warn_if_not_cuda(device)
    print(f"device: {describe_device(device)}")
    train = cfg["train"]
    b = int(batch_size or train["batch_size"])
    t = int(seq_len or train["seq_len"])
    amp = parse_amp(train.get("amp", "bf16"), device)
    hidden, layers = ACTOR[size]
    wm = build_model(cfg).to(device)
    actor = Actor(wm.feat_dim, wm.rssm.action_dim, hidden=hidden, layers=layers).to(device)
    critic = Critic(wm.feat_dim, hidden=hidden, layers=layers).to(device)
    print(
        f"size={size}  wm={_n_m(wm):.1f}M  ac={_n_m(actor)+_n_m(critic):.1f}M  "
        f"batch={b} seq={t} start_mode={start_mode}"
    )
    dummy = {
        "obs": torch.randint(0, 256, (b, t, 64, 64, 3), dtype=torch.uint8),
        "actions": torch.randint(0, 17, (b, t), dtype=torch.int64),
        "rewards": torch.zeros(b, t),
        "cont": torch.ones(b, t),
    }
    wm_optim = torch.optim.Adam(wm.parameters(), lr=1e-4)
    ac_optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-5)
    scaler = make_grad_scaler(device, amp)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    world_model_step(
        wm, wm_optim, dummy, device=device, train_cfg=train, amp_dtype=amp, scaler=scaler
    )
    actor_critic_step(
        wm,
        actor,
        critic,
        ac_optim,
        dummy,
        device=device,
        retnorm=PercentileReturnNorm(),
        horizon=15,
        start_mode=start_mode,
        amp_dtype=amp,
        scaler=scaler,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    vram = vram_peak_gb()
    vram_s = f"  peak {vram[0]:.2f}/{vram[1]:.2f} GiB" if vram else ""
    print(f"PASS  wm+ac step {dt:.2f}s{vram_s}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=list(SIZES), default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--start-mode", default="all")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    args = parser.parse_args()
    sizes = [args.size] if args.size else list(SIZES)
    print(f"{'size':<8} {'ours WM':>10} {'paper WM':>10} {'+actor/critic':>14} {'total':>10}")
    for size in sizes:
        n = count_one(size)
        print(
            f"{size:<8} {n['wm']:9.1f}M {n['paper_wm']:9.0f}M "
            f"{n['actor_critic']:13.1f}M {n['total']:9.1f}M"
        )
    if args.smoke:
        for size in sizes:
            print("---")
            try:
                smoke(
                    size,
                    batch_size=args.batch_size,
                    seq_len=args.seq_len,
                    start_mode=args.start_mode,
                )
            except torch.cuda.OutOfMemoryError:
                print(f"FAIL: CUDA OOM on size={size} start_mode={args.start_mode}")
                raise


if __name__ == "__main__":
    main()
