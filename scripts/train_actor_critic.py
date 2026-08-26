"""Train actor + critic on imagined rollouts from a frozen world model (M4).

No environment interaction. Load the M3 size-S checkpoint, freeze it, sample
replay windows, imagine `horizon` steps with the actor, update actor/critic.

Prefer `notebooks/07_train_actor_critic.ipynb` for a live run you can stop.
This CLI is the canonical loop the notebook imports (`actor_critic_step`).

    conda activate worldmodel
    python scripts/train_actor_critic.py --config configs/m4_actor_critic.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from agents.actor_critic import Actor, Critic
from models.world_model import WorldModel
from training.ac_step import actor_critic_step
from training.ckpt import resolve_resume
from training.device import (
    configure_runtime,
    describe_device,
    get_device,
    make_grad_scaler,
    parse_amp,
    vram_peak_gb,
    warn_if_not_cuda,
)
from training.imagine import decode_imagination, freeze_world_model
from training.replay_buffer import ReplayBuffer
from training.returns import PercentileReturnNorm

# Sibling script (same directory) — used when invoked as `python scripts/...`.
from train_world_model import build_model, set_seed


def _uint8_row(frames: torch.Tensor) -> np.ndarray:
    """`[T, 3, 64, 64]` float in `[0, 1]` → horizontal uint8 strip."""
    x = frames.permute(0, 2, 3, 1).clamp(0.0, 1.0).cpu()
    tiles = [(x[t] * 255.0).round().to(torch.uint8).numpy() for t in range(x.shape[0])]
    return np.concatenate(tiles, axis=1)


def save_imagination_strip(images: torch.Tensor, path: Path) -> None:
    """`images` `[N, H, 3, 64, 64]` → PNG of the first start state's horizon."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_uint8_row(images[0]), mode="RGB").save(path)


def save_imagination_gif(images: torch.Tensor, path: Path, duration_ms: int = 250) -> None:
    """`images` `[N, H, 3, 64, 64]` → GIF of the first start state."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    seq = images[0].permute(0, 2, 3, 1).clamp(0.0, 1.0).cpu()
    frames = [
        Image.fromarray((seq[t] * 255.0).round().to(torch.uint8).numpy(), mode="RGB")
        for t in range(seq.shape[0])
    ]
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )


def load_frozen_world_model(m4_cfg: dict, device: torch.device) -> tuple[WorldModel, dict]:
    """Build from the M3 yaml, load `world_model_ckpt`, freeze."""
    wm_path = Path(m4_cfg["world_model_config"])
    with wm_path.open() as f:
        wm_cfg = yaml.safe_load(f)
    model = build_model(wm_cfg).to(device)
    ckpt_path = Path(m4_cfg["world_model_ckpt"])
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"world-model checkpoint not found: {ckpt_path}. Train M3 first."
        )
    payload = torch.load(ckpt_path, weights_only=False, map_location=device)
    model.load_state_dict(payload["model"], strict=True)
    freeze_world_model(model)
    print(f"frozen world model from {ckpt_path} (wm step {payload.get('step', '?')})")
    return model, wm_cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/m4_actor_critic.yaml"))
    parser.add_argument(
        "--resume",
        default=None,
        help='Actor-critic checkpoint, or "auto" for ckpt_latest in checkpoint_dir.',
    )
    args = parser.parse_args()
    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg["seed"]))
    device = get_device()
    configure_runtime(device)
    warn_if_not_cuda(device)
    print(f"device: {describe_device(device)}")

    train = cfg["train"]
    world_model, wm_cfg = load_frozen_world_model(cfg, device)

    replay_path = Path(wm_cfg["collect"]["out_path"])
    if not replay_path.exists():
        raise SystemExit(f"Missing {replay_path}. Collect with the M3 config first.")
    buffer = ReplayBuffer(seed=int(cfg["seed"]))
    buffer.load_state_dict(torch.load(replay_path, weights_only=False))
    print(f"replay: episodes={len(buffer)} steps={buffer.num_steps}")

    actor_cfg = cfg.get("actor", {})
    critic_cfg = cfg.get("critic", {})
    actor = Actor(
        world_model.feat_dim,
        world_model.rssm.action_dim,
        hidden=int(actor_cfg.get("hidden", 512)),
        layers=int(actor_cfg.get("layers", 2)),
        unimix=float(actor_cfg.get("unimix", 0.01)),
    ).to(device)
    critic = Critic(
        world_model.feat_dim,
        hidden=int(critic_cfg.get("hidden", 512)),
        layers=int(critic_cfg.get("layers", 2)),
        num_bins=int(critic_cfg.get("num_bins", 255)),
        low=float(critic_cfg.get("low", -20.0)),
        high=float(critic_cfg.get("high", 20.0)),
    ).to(device)

    optim = torch.optim.Adam(
        [
            {"params": actor.parameters(), "lr": float(train["actor_lr"])},
            {"params": critic.parameters(), "lr": float(train["critic_lr"])},
        ]
    )
    retnorm = PercentileReturnNorm()
    amp_dtype = parse_amp(train.get("amp", "bf16"), device)
    scaler = make_grad_scaler(device, amp_dtype)

    ckpt_dir = Path(train["checkpoint_dir"])
    results_dir = Path(train["results_dir"])
    log_dir = Path(train["log_dir"])
    for p in (ckpt_dir, results_dir, log_dir):
        p.mkdir(parents=True, exist_ok=True)

    start_step = 0
    resume_path = resolve_resume(args.resume, ckpt_dir)
    if resume_path is not None:
        ac_ckpt = torch.load(resume_path, weights_only=False, map_location=device)
        actor.load_state_dict(ac_ckpt["actor"])
        critic.load_state_dict(ac_ckpt["critic"])
        if "optim" in ac_ckpt:
            optim.load_state_dict(ac_ckpt["optim"])
        if "retnorm" in ac_ckpt:
            retnorm.load_state_dict(ac_ckpt["retnorm"])
        start_step = int(ac_ckpt.get("step", 0))
        print(f"resumed actor-critic from {resume_path} at step {start_step}")

    writer = SummaryWriter(log_dir=str(log_dir))
    steps = int(train["steps"])
    batch_size = int(train["batch_size"])
    seq_len = int(train["seq_len"])
    horizon = int(train["horizon"])
    start_mode = str(train.get("start_mode", "all"))
    log_every = int(train["log_every"])
    image_every = int(train["image_every"])
    ckpt_every = int(train["checkpoint_every"])
    history: list[dict] = []
    metrics_path = results_dir / "train_metrics.json"
    if start_step > 0 and metrics_path.is_file():
        prev = json.loads(metrics_path.read_text(encoding="utf-8"))
        history = [h for h in prev if int(h.get("step", 0)) <= start_step]

    print(
        f"training actor-critic to step {steps}  horizon={horizon}  "
        f"start_mode={start_mode}  (start={start_step})"
    )
    last_log_time = time.time()
    last_log_step = start_step
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for step in range(start_step + 1, steps + 1):
        batch = buffer.sample(batch_size, seq_len)
        _loss, metrics, rollout = actor_critic_step(
            world_model,
            actor,
            critic,
            optim,
            batch,
            device=device,
            retnorm=retnorm,
            horizon=horizon,
            start_mode=start_mode,
            lam=float(train.get("lam", 0.95)),
            discount=float(train.get("discount", 0.997)),
            entropy_scale=float(train.get("entropy_scale", 3.0e-4)),
            amp_dtype=amp_dtype,
            scaler=scaler,
            max_grad_norm=float(train.get("max_grad_norm", 100.0)),
        )
        metrics["step"] = step
        if step % log_every == 0 or step == start_step + 1:
            now = time.time()
            dt = max(now - last_log_time, 1e-6)
            sps = (step - last_log_step) / dt
            last_log_time = now
            last_log_step = step
            metrics["steps_per_sec"] = sps
            history.append(metrics)
            for k, v in metrics.items():
                if k != "step":
                    writer.add_scalar(f"ac/{k}", v, step)
            vram = vram_peak_gb()
            vram_s = f"  vram {vram[0]:.1f}/{vram[1]:.1f} GiB" if vram else ""
            print(
                f"step {step}/{steps}  total={metrics['total']:.4f}  "
                f"actor={metrics['actor']:.4f} critic={metrics['critic']:.4f}  "
                f"H={metrics['entropy']:.3f}  ret={metrics['return']:.4f}  "
                f"({sps:.2f} steps/s){vram_s}",
                flush=True,
            )

        if step % image_every == 0 or step == start_step + 1:
            vis = decode_imagination(world_model, rollout.feat, max_starts=1)
            save_imagination_strip(vis, results_dir / f"imagine_step_{step:06d}.png")
            save_imagination_gif(vis, results_dir / f"imagine_step_{step:06d}.gif")

        if step % ckpt_every == 0:
            payload = {
                "step": step,
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "optim": optim.state_dict(),
                "retnorm": retnorm.state_dict(),
            }
            torch.save(payload, ckpt_dir / f"ckpt_step_{step}.pt")
            torch.save(payload, ckpt_dir / "ckpt_latest.pt")
            metrics_path.write_text(json.dumps(history), encoding="utf-8")
            print(f"wrote {ckpt_dir / f'ckpt_step_{step}.pt'}", flush=True)

    final = {
        "step": steps,
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "optim": optim.state_dict(),
        "retnorm": retnorm.state_dict(),
    }
    torch.save(final, ckpt_dir / "ckpt_final.pt")
    torch.save(final, ckpt_dir / "ckpt_latest.pt")
    metrics_path.write_text(json.dumps(history), encoding="utf-8")
    vis = decode_imagination(world_model, rollout.feat, max_starts=1)
    save_imagination_strip(vis, results_dir / "imagine_final.png")
    save_imagination_gif(vis, results_dir / "imagine_final.gif")
    writer.flush()
    writer.close()
    print("done", ckpt_dir / "ckpt_final.pt")


if __name__ == "__main__":
    main()
