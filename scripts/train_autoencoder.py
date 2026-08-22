"""Train M1 perception autoencoder toward near pixel-identical reconstructions.

Usage:
    conda activate worldmodel
    python scripts/collect_random_frames.py
    python scripts/train_autoencoder.py

Watch:
    tensorboard --logdir runs
    results/m1/recon_final.png
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from models.autoencoder import PerceptionAutoencoder
from models.preprocess import nchw_float_to_nhwc_uint8, nhwc_uint8_to_nchw_float
from training.device import configure_runtime, describe_device, get_device, warn_if_not_cuda


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_diverse_frames(frames: Tensor, n: int) -> Tensor:
    if n >= frames.shape[0]:
        return frames.clone()
    flat = frames.float().mean(dim=(1, 2))
    chosen = [int(frames.shape[0] // 5)]
    for _ in range(n - 1):
        refs = flat[chosen]
        d = torch.cdist(flat, refs).min(dim=1).values
        d[chosen] = -1.0
        chosen.append(int(torch.argmax(d).item()))
    return frames[chosen].clone()


def make_side_by_side(real_u8: Tensor, recon_u8: Tensor) -> Tensor:
    pair = torch.cat([real_u8, recon_u8], dim=2)
    return pair.permute(0, 3, 1, 2).float() / 255.0


def save_comparison_png(real_u8: Tensor, recon_u8: Tensor, path: Path) -> None:
    from PIL import Image

    strips = [
        np.concatenate([real_u8[i].cpu().numpy(), recon_u8[i].cpu().numpy()], axis=1)
        for i in range(real_u8.shape[0])
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(strips, axis=0), mode="RGB").save(path)


def uint8_fidelity(real_u8: Tensor, recon_u8: Tensor) -> dict[str, float]:
    diff = (real_u8.float() - recon_u8.float()).abs()
    channel_max = diff.amax(dim=-1)
    return {
        "uint8_mad": float(diff.mean().item()),
        "within1": float(channel_max.le(1).float().mean().item()),
        "within2": float(channel_max.le(2).float().mean().item()),
        "exact": float(channel_max.eq(0).float().mean().item()),
    }


@torch.no_grad()
def evaluate_batch(
    model: PerceptionAutoencoder, batch_u8: Tensor, device: torch.device
) -> tuple[dict[str, float], Tensor, Tensor]:
    obs = nhwc_uint8_to_nchw_float(batch_u8.to(device))
    recon, _ = model(obs)
    real_u8 = nchw_float_to_nhwc_uint8(obs.cpu())
    recon_u8 = nchw_float_to_nhwc_uint8(recon.cpu())
    metrics = {
        "recon_mse": float(F.mse_loss(recon, obs).item()),
        "recon_l1": float(F.l1_loss(recon, obs).item()),
        **uint8_fidelity(real_u8, recon_u8),
    }
    return metrics, real_u8, recon_u8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/m1_autoencoder.yaml"))
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Optional checkpoint to warm-start model (+ optimizer if present).",
    )
    args = parser.parse_args()
    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg["seed"]))
    device = get_device()
    configure_runtime(device)
    print(f"device: {describe_device(device)}")
    warn_if_not_cuda(device)

    frames_path = Path(cfg["collect"]["out_path"])
    if not frames_path.exists():
        raise SystemExit(
            f"Missing {frames_path}. Run: python scripts/collect_random_frames.py"
        )
    frames: Tensor = torch.load(frames_path, weights_only=False)["frames"]
    print(f"loaded frames: {tuple(frames.shape)}")

    train_cfg = cfg["train"]
    loader = DataLoader(
        TensorDataset(frames),
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )

    channels = tuple(int(c) for c in cfg.get("encoder_channels", [64, 128, 256, 512]))
    stem_channels = int(cfg.get("stem_channels", 64))
    model = PerceptionAutoencoder(
        embed_dim=int(cfg["embed_dim"]),
        channels=channels,
        stem_channels=stem_channels,
    ).to(device)
    lr = float(train_cfg["lr"])
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    if args.resume is not None:
        ckpt = torch.load(args.resume, weights_only=False, map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)
        if "optim" in ckpt:
            try:
                optim.load_state_dict(ckpt["optim"])
            except (ValueError, RuntimeError):
                print("optimizer state incompatible; keeping fresh Adam")
        print(f"resumed from {args.resume} (ckpt step={int(ckpt.get('step', 0))})")
    print(
        f"PerceptionAutoencoder embed={cfg['embed_dim']} "
        f"stem={stem_channels} channels={channels} lr={lr}"
    )

    log_dir = Path(train_cfg["log_dir"])
    ckpt_dir = Path(train_cfg["checkpoint_dir"])
    results_dir = Path(train_cfg["results_dir"])
    for p in (log_dir, ckpt_dir, results_dir):
        p.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(log_dir))
    vis_n = int(train_cfg["num_vis_samples"])
    vis_batch = (
        select_diverse_frames(frames, vis_n)
        if bool(train_cfg.get("diverse_vis", True))
        else frames[:vis_n].clone()
    )

    steps = int(train_cfg["steps"])
    log_every = int(train_cfg["log_every"])
    image_every = int(train_cfg["image_every"])
    ckpt_every = int(train_cfg["checkpoint_every"])
    target_within1 = float(train_cfg.get("target_within1", 0.98))
    target_mad = float(train_cfg.get("target_uint8_mad", 0.5))
    min_steps = int(train_cfg.get("min_steps_before_early_stop", 0))
    use_cosine = bool(train_cfg.get("cosine_lr", False))
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(steps, 1), eta_min=lr * 0.05)
        if use_cosine
        else None
    )
    data_iter = iter(loader)
    history: list[dict[str, float]] = []
    stopped_early = False

    model.train()
    for step in range(1, steps + 1):
        try:
            (batch_u8,) = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            (batch_u8,) = next(data_iter)

        obs = nhwc_uint8_to_nchw_float(batch_u8.to(device))
        recon, _embed = model(obs)
        l1 = F.l1_loss(recon, obs)
        mse = F.mse_loss(recon, obs)
        recon_u8 = ((recon + 1.0) * 127.5).clamp(0.0, 255.0)
        obs_u8 = batch_u8.to(device=device, dtype=recon.dtype).permute(0, 3, 1, 2)
        uint8_l1 = F.l1_loss(recon_u8, obs_u8) / 255.0
        loss = l1 + 0.25 * mse + float(train_cfg.get("uint8_loss_weight", 1.0)) * uint8_l1

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        if scheduler is not None:
            scheduler.step()

        if step % log_every == 0 or step == 1:
            writer.add_scalar("m1/recon_loss", float(loss.item()), step)
            writer.add_scalar("m1/recon_l1", float(l1.item()), step)
            writer.add_scalar("m1/recon_mse", float(mse.item()), step)
            writer.add_scalar("m1/uint8_l1", float(uint8_l1.item()), step)
            history.append(
                {
                    "step": float(step),
                    "recon_l1": float(l1.item()),
                    "recon_mse": float(mse.item()),
                    "uint8_l1": float(uint8_l1.item()),
                }
            )
            print(
                f"step {step:5d}/{steps}  l1={l1.item():.6f}  "
                f"mse={mse.item():.6f}  u8={uint8_l1.item():.6f}"
            )

        if step % image_every == 0 or step == 1:
            model.eval()
            metrics, real_u8, recon_u8_vis = evaluate_batch(model, vis_batch, device)
            writer.add_images(
                "m1/real_vs_recon",
                make_side_by_side(real_u8, recon_u8_vis),
                step,
                dataformats="NCHW",
            )
            save_comparison_png(
                real_u8, recon_u8_vis, results_dir / f"recon_step_{step:05d}.png"
            )
            for k, v in metrics.items():
                writer.add_scalar(f"m1/vis_{k}", v, step)
            print(
                f"  vis uint8_mad={metrics['uint8_mad']:.3f}  "
                f"within1={metrics['within1']:.3f}  exact={metrics['exact']:.3f}"
            )
            model.train()

            if (
                step >= min_steps
                and metrics["within1"] >= target_within1
                and metrics["uint8_mad"] <= target_mad
            ):
                print(
                    f"Early stop at step {step}: "
                    f"within1={metrics['within1']:.4f} mad={metrics['uint8_mad']:.4f}"
                )
                stopped_early = True
                torch.save(
                    {
                        "step": step,
                        "model": model.state_dict(),
                        "optim": optim.state_dict(),
                        "config": cfg,
                        "metrics": metrics,
                    },
                    ckpt_dir / f"ckpt_step_{step:05d}.pt",
                )
                break

        if step % ckpt_every == 0 or step == steps:
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "optim": optim.state_dict(),
                    "config": cfg,
                },
                ckpt_dir / f"ckpt_step_{step:05d}.pt",
            )

    model.eval()
    final_metrics, real_u8, recon_u8_vis = evaluate_batch(model, vis_batch, device)
    save_comparison_png(real_u8, recon_u8_vis, results_dir / "recon_final.png")
    save_comparison_png(real_u8, recon_u8_vis, results_dir / "recon_diverse_check.png")

    rand_idx = torch.randperm(frames.shape[0])[:256]
    holdout_metrics, _, _ = evaluate_batch(model, frames[rand_idx], device)

    (results_dir / "train_metrics.json").write_text(
        json.dumps(
            {
                "history": history,
                "final_vis": final_metrics,
                "holdout256": holdout_metrics,
                "stopped_early": stopped_early,
            },
            indent=2,
        )
    )
    writer.flush()
    writer.close()

    print()
    print(f"first l1: {history[0]['recon_l1']:.6f}")
    print(f"last  l1: {history[-1]['recon_l1']:.6f}")
    print(
        f"final vis  mad={final_metrics['uint8_mad']:.4f} "
        f"within1={final_metrics['within1']:.4f} exact={final_metrics['exact']:.4f}"
    )
    print(
        f"holdout256 mad={holdout_metrics['uint8_mad']:.4f} "
        f"within1={holdout_metrics['within1']:.4f} exact={holdout_metrics['exact']:.4f}"
    )
    print(f"Images: {results_dir / 'recon_final.png'}")
    if holdout_metrics["within1"] >= target_within1:
        print("PASS: near pixel-identical reconstructions")
    else:
        print("WARN: still below target_within1 — inspect PNGs / train longer")


if __name__ == "__main__":
    main()
