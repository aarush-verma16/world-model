"""Tiny CUDA throughput/VRAM smoke: one world-model train step, then exit.

Does NOT stand in for a real training run. Used after a config/device change
to confirm the XL CNN + AMP batch actually fits on the GPU.

Usage:
    conda activate worldmodel
    python scripts/smoke_cuda_step.py
    python scripts/smoke_cuda_step.py --batch-size 16 --seq-len 32
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import yaml

from training.device import (
    configure_runtime,
    describe_device,
    get_device,
    make_grad_scaler,
    parse_amp,
    vram_peak_gb,
    warn_if_not_cuda,
)
from training.wm_step import world_model_step

# `python scripts/smoke_cuda_step.py` puts this file's directory on sys.path,
# so the sibling script is importable directly.
from train_world_model import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/m3_dreamer_s.yaml"))
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--steps", type=int, default=3, help="warmup + timed steps")
    args = parser.parse_args()

    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    configure_runtime(device)
    warn_if_not_cuda(device)
    print(f"device: {describe_device(device)}")

    train = dict(cfg["train"])
    batch_size = int(args.batch_size or train["batch_size"])
    seq_len = int(args.seq_len or train["seq_len"])
    action_dim = int(cfg["env"]["action_dim"])
    amp_dtype = parse_amp(train.get("amp", "bf16"), device)
    print(f"amp={train.get('amp', 'bf16')}  batch={batch_size}  seq_len={seq_len}")

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"params: {n_params:.2f}M")

    optim = torch.optim.Adam(model.parameters(), lr=float(train["lr"]))
    scaler = make_grad_scaler(device, amp_dtype)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    dummy = {
        "obs": torch.randint(0, 256, (batch_size, seq_len, 64, 64, 3), dtype=torch.uint8),
        "actions": torch.randint(0, action_dim, (batch_size, seq_len), dtype=torch.int64),
        "rewards": torch.zeros(batch_size, seq_len, dtype=torch.float32),
        "cont": torch.ones(batch_size, seq_len, dtype=torch.float32),
    }

    model.train()
    times: list[float] = []
    try:
        for i in range(int(args.steps)):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _loss, metrics = world_model_step(
                model,
                optim,
                dummy,
                device=device,
                train_cfg=train,
                amp_dtype=amp_dtype,
                scaler=scaler,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if i > 0:
                times.append(dt)
            print(f"  step {i + 1}/{args.steps}  {dt:.3f}s  total={metrics['total']:.4f}")
    except torch.cuda.OutOfMemoryError:
        print(f"FAIL: CUDA OOM. Drop `train.batch_size` 16 -> 8 in {args.config}.")
        raise

    vram = vram_peak_gb()
    if times:
        mean = sum(times) / len(times)
        print(f"mean step (excl. warmup): {mean:.3f}s  ({1.0 / mean:.2f} steps/s)")
    if vram:
        print(f"peak VRAM: {vram[0]:.2f} GiB allocated / {vram[1]:.2f} GiB reserved")
    print("PASS: CUDA train-step smoke succeeded")


if __name__ == "__main__":
    main()
