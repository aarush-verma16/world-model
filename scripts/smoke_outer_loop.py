"""CUDA smoke for one outer-loop cycle (collect + WM + AC + eval + ckpt).

Dozens of env steps only — not a real training run. Loads M3/M4 checkpoints
when present; does not load the seed replay (collects one short episode).

    conda activate worldmodel
    python scripts/smoke_outer_loop.py
    python scripts/smoke_outer_loop.py --config configs/m6_baseline.yaml --env-steps 32
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import torch
import yaml

from training.collect import Collector
from training.device import (
    configure_runtime,
    describe_device,
    get_device,
    make_grad_scaler,
    parse_amp,
    vram_peak_gb,
    warn_if_not_cuda,
)
from training.evaluate import evaluate_policy
from training.outer_loop import (
    joint_payload,
    load_checkpoint,
    outer_cycle,
    save_checkpoint,
)
from training.replay_buffer import ReplayBuffer
from training.returns import PercentileReturnNorm

from train_agent import (
    load_seed_actor_critic,
    load_seed_world_model,
    make_actor_critic,
    make_envs,
    make_slow_critic,
)
from train_world_model import build_model, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/m5_outer_loop.yaml"))
    parser.add_argument("--env-steps", type=int, default=32)
    parser.add_argument("--eval-steps", type=int, default=8)
    parser.add_argument("--start-mode", default=None)
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with Path(cfg["world_model_config"]).open(encoding="utf-8") as f:
        wm_cfg = yaml.safe_load(f)

    set_seed(0)
    device = get_device()
    configure_runtime(device)
    warn_if_not_cuda(device)
    print(f"device: {describe_device(device)}")

    train = cfg["train"]
    start_mode = str(args.start_mode or train.get("start_mode", "all"))
    amp_dtype = parse_amp(train.get("amp", "bf16"), device)
    print(
        f"amp={train.get('amp', 'bf16')}  batch={train['batch_size']}  "
        f"seq={train['seq_len']}  start_mode={start_mode}  env_steps={args.env_steps}"
    )

    # From-scratch configs (M7/M8 XL) leave both ckpt keys null.
    raw_wm_ckpt = cfg.get("world_model_ckpt")
    wm_ckpt = Path(raw_wm_ckpt) if raw_wm_ckpt else None
    if wm_ckpt is not None and wm_ckpt.is_file():
        world_model, wm_cfg = load_seed_world_model(cfg, device)
    else:
        print(f"no {wm_ckpt} — random weights from {cfg['world_model_config']} (smoke only)")
        world_model = build_model(wm_cfg).to(device)

    actor, critic = make_actor_critic(cfg, world_model, device)
    wm_optim = torch.optim.Adam(world_model.parameters(), lr=float(train["wm_lr"]))
    ac_optim = torch.optim.Adam(
        [
            {"params": actor.parameters(), "lr": float(train["actor_lr"])},
            {"params": critic.parameters(), "lr": float(train["critic_lr"])},
        ]
    )
    retnorm = PercentileReturnNorm()
    slow_critic = (
        make_slow_critic(cfg, critic, device)
        if bool(cfg.get("critic", {}).get("slow_target", True))
        else None
    )
    raw_ac_ckpt = cfg.get("actor_critic_ckpt")
    if raw_ac_ckpt and Path(raw_ac_ckpt).is_file() and wm_ckpt is not None and wm_ckpt.is_file():
        load_seed_actor_critic(cfg, actor, critic, ac_optim, retnorm, device)
    scaler = make_grad_scaler(device, amp_dtype)

    buffer = ReplayBuffer(seed=0, max_steps=10_000)
    collect_env, eval_env = make_envs(cfg)
    seq_len = int(train["seq_len"])
    collector = Collector(
        collect_env,
        world_model,
        actor,
        buffer,
        device=device,
        max_episode_steps=max(int(args.env_steps), seq_len),
        amp_dtype=amp_dtype,
        seed=0,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def _has_seq() -> bool:
        return buffer.can_sample(seq_len)

    def run(mode: str) -> None:
        collected = 0
        while collected < 256 and not _has_seq():
            collector.collect(int(args.env_steps))
            collected += int(args.env_steps)
        if not _has_seq():
            raise RuntimeError(
                f"smoke collect produced < seq_len={seq_len} stream steps "
                f"after {collected} env steps (have {buffer.num_steps})"
            )
        t0 = time.perf_counter()
        cycle = outer_cycle(
            collector,
            world_model,
            actor,
            critic,
            wm_optim,
            ac_optim,
            buffer,
            device=device,
            wm_train_cfg=wm_cfg["train"],
            collect_every=int(args.env_steps),
            wm_updates=2,
            ac_updates=2,
            batch_size=int(train["batch_size"]),
            seq_len=seq_len,
            retnorm=retnorm,
            horizon=int(train["horizon"]),
            start_mode=mode,
            lam=float(train.get("lam", 0.95)),
            discount=float(train.get("discount", 0.997)),
            entropy_scale=float(train.get("entropy_scale", 3.0e-4)),
            imag_gradient=str(train.get("imag_gradient", "reinforce")),
            imag_gradient_mix=float(train.get("imag_gradient_mix", 0.0)),
            slow_critic=slow_critic,
            amp_dtype=amp_dtype,
            scaler=scaler,
            wm_max_grad_norm=float(train.get("wm_max_grad_norm", 1000.0)),
            ac_max_grad_norm=float(train.get("ac_max_grad_norm", 100.0)),
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(
            f"cycle {dt:.2f}s  episodes={len(cycle.collect['episodes'])}  "
            f"wm={cycle.wm_metrics['total'] if cycle.wm_metrics else None}  "
            f"ac_H={cycle.ac_metrics['entropy'] if cycle.ac_metrics else None}"
        )
        if cycle.ac_metrics:
            m = cycle.ac_metrics
            print(
                f"adv={m['adv']:+.4f}±{m['adv_std']:.4f}  value={m['value']:+.4f}  "
                f"slow_value={m['slow_value']:+.4f}  weight={m['weight']:.4f}  "
                f"retnorm={m['retnorm_scale']:.3f}"
            )
        ev = evaluate_policy(
            eval_env,
            world_model,
            actor,
            device=device,
            n_episodes=1,
            max_steps=int(args.eval_steps),
            amp_dtype=amp_dtype,
            seed=100_000,
        )
        print(
            f"eval return={ev.mean_return:.3f}  len={ev.mean_length:.1f}  "
            f"score={ev.crafter_score:.4f}"
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ckpt.pt"
            save_checkpoint(
                path,
                joint_payload(
                    env_steps=int(args.env_steps),
                    wm_steps=2,
                    ac_steps=2,
                    world_model=world_model,
                    wm_optim=wm_optim,
                    actor=actor,
                    critic=critic,
                    ac_optim=ac_optim,
                    retnorm=retnorm,
                    collect_seed=collector.next_seed,
                    slow_critic=slow_critic,
                ),
            )
            counters = load_checkpoint(
                path,
                world_model,
                wm_optim,
                actor,
                critic,
                ac_optim,
                retnorm,
                device,
                slow_critic=slow_critic,
            )
            assert counters["env_steps"] == int(args.env_steps)
        vram = vram_peak_gb()
        if vram:
            print(f"peak vram {vram[0]:.2f}/{vram[1]:.2f} GiB")
        rss = None
        try:
            import psutil

            rss = psutil.Process().memory_info().rss / (1024**3)
        except ImportError:
            pass
        if rss is not None:
            print(f"rss {rss:.2f} GiB")

    try:
        print(f"trying start_mode={start_mode}")
        run(start_mode)
    except torch.cuda.OutOfMemoryError:
        if start_mode == "last":
            raise
        print("OOM on start_mode=all — retrying start_mode=last (do not shrink the latent)")
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        run("last")
        print("SMOKE: use train.start_mode: last in configs/m5_outer_loop.yaml")
    else:
        print("SMOKE OK")
    finally:
        collect_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
