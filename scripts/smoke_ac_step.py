"""CUDA VRAM/throughput smoke for one actor-critic imagination step.

Does NOT stand in for a real M4 run. Random size-S weights, dummy tensors —
no replay, no 700k checkpoint. If `start_mode=all` OOMs, retry `last`.

    conda activate worldmodel
    python scripts/smoke_ac_step.py
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
from training.imagine import freeze_world_model
from training.returns import PercentileReturnNorm
from train_world_model import build_model


def _dummy_batch(batch_size: int, seq_len: int, action_dim: int) -> dict[str, torch.Tensor]:
    return {
        "obs": torch.randint(0, 256, (batch_size, seq_len, 64, 64, 3), dtype=torch.uint8),
        "actions": torch.randint(0, action_dim, (batch_size, seq_len), dtype=torch.int64),
        "rewards": torch.zeros(batch_size, seq_len),
        "cont": torch.ones(batch_size, seq_len),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/m4_actor_critic.yaml"))
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--start-mode", default=None)
    args = parser.parse_args()

    with args.config.open() as f:
        m4 = yaml.safe_load(f)
    with Path(m4["world_model_config"]).open() as f:
        wm_cfg = yaml.safe_load(f)

    device = get_device()
    configure_runtime(device)
    warn_if_not_cuda(device)
    print(f"device: {describe_device(device)}")

    train = m4["train"]
    batch_size = int(train["batch_size"])
    seq_len = int(train["seq_len"])
    horizon = int(train["horizon"])
    start_mode = str(args.start_mode or train.get("start_mode", "all"))
    action_dim = int(wm_cfg["env"]["action_dim"])
    amp_dtype = parse_amp(train.get("amp", "bf16"), device)
    print(
        f"amp={train.get('amp', 'bf16')}  batch={batch_size}  seq={seq_len}  "
        f"horizon={horizon}  start_mode={start_mode}"
    )

    world_model = build_model(wm_cfg).to(device)
    freeze_world_model(world_model)
    actor = Actor(world_model.feat_dim, action_dim).to(device)
    critic = Critic(world_model.feat_dim).to(device)
    optim = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=float(train["actor_lr"]),
    )
    retnorm = PercentileReturnNorm()
    scaler = make_grad_scaler(device, amp_dtype)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    dummy = _dummy_batch(batch_size, seq_len, action_dim)

    def run_mode(mode: str) -> None:
        times: list[float] = []
        for i in range(int(args.steps)):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _loss, metrics, _rollout = actor_critic_step(
                world_model,
                actor,
                critic,
                optim,
                dummy,
                device=device,
                retnorm=retnorm,
                horizon=horizon,
                start_mode=mode,
                amp_dtype=amp_dtype,
                scaler=scaler,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            times.append(dt)
            print(
                f"  step {i + 1}/{args.steps}  {dt:.3f}s  total={metrics['total']:.3f}  "
                f"H={metrics['entropy']:.3f}"
            )
        vram = vram_peak_gb()
        if vram:
            print(f"peak vram {vram[0]:.2f}/{vram[1]:.2f} GiB  mean {sum(times)/len(times):.3f}s/step")
        else:
            print(f"mean {sum(times)/len(times):.3f}s/step")

    try:
        print(f"trying start_mode={start_mode}")
        run_mode(start_mode)
    except torch.cuda.OutOfMemoryError:
        if start_mode == "last":
            raise
        print("OOM on start_mode=all — retrying start_mode=last (do not shrink the latent)")
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        run_mode("last")
        print("SMOKE: use train.start_mode: last in configs/m4_actor_critic.yaml")
    else:
        print("SMOKE OK")


if __name__ == "__main__":
    main()
