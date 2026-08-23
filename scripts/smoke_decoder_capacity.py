"""Can the decoder even *represent* a 7px sprite and a 4px inventory digit?

This is a capacity probe, not a training run. It overfits the skip-free
encode→decode path on a handful of real replay frames for a few hundred steps.
If sprites and inventory glyphs cannot appear when the model is allowed to
memorize eight frames, they will never appear in a 12k world-model run, and no
loss weight will change that — so run this after touching the decoder instead
of spending an hour of real training to find out.

Runs the current sub-pixel (`PixelShuffle`) decoder against a `nearest`
upsample control with identical seed/steps/data. Nearest replicates each latent
cell into a 2x2 block of identical values, so the following 3x3 conv cannot
recover position within a cell; at a 4x4 latent that cell is 16x16 px, which is
larger than any Crafter sprite. The control is here to keep that claim honest.

Usage:
    conda activate worldmodel
    python scripts/smoke_decoder_capacity.py
    python scripts/smoke_decoder_capacity.py --frames 8 --steps 600
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch import nn

from models.crafter_layout import avatar_slice, hud_slice
from models.decoder import Decoder
from models.encoder import Encoder
from models.preprocess import nchw_float_to_nhwc_uint8, nhwc_uint8_to_nchw_float
from training.device import autocast_context, configure_runtime, get_device, parse_amp
from training.replay_buffer import ReplayBuffer


def nearest_up_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """The old `Upsample(nearest)` + 2 convs stage, for the control run."""
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.SiLU(),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.SiLU(),
    )


def make_nearest_control(dec: Decoder, out_channels: int = 3) -> Decoder:
    """Swap `dec.up` for the pre-fix nearest-upsample stack, in place."""
    channels = dec.channels
    stages: list[nn.Module] = []
    prev = channels[0]
    for ch in channels[1:]:
        stages.append(nearest_up_block(prev, ch))
        prev = ch
    stages.append(
        nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(prev, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )
    )
    dec.up = nn.Sequential(*stages)
    return dec


def overfit(
    *,
    frames_u8: torch.Tensor,
    cfg: dict,
    device: torch.device,
    steps: int,
    variant: str,
    seed: int,
) -> tuple[float, dict[str, float], torch.Tensor]:
    """Overfit encode→decode on `frames_u8`; return (l1, region l1s, recon)."""
    enc_cfg = cfg["encoder"]
    dec_cfg = cfg.get("decoder", {})
    channels = tuple(int(c) for c in enc_cfg["channels"])
    embed_dim = int(enc_cfg["embed_dim"])

    torch.manual_seed(seed)
    encoder = Encoder(
        embed_dim=embed_dim, channels=channels, blocks=int(enc_cfg.get("blocks", 2))
    ).to(device)
    decoder = Decoder(
        embed_dim=embed_dim,
        channels=tuple(int(c) for c in dec_cfg.get("channels", [512, 256, 128, 64])),
        start_res=4,
        blocks=0,
    )
    if variant == "nearest":
        make_nearest_control(decoder)
    decoder = decoder.to(device)

    params = list(encoder.parameters()) + list(decoder.parameters())
    optim = torch.optim.Adam(params, lr=3.0e-4)
    amp_dtype = parse_amp(cfg["train"].get("amp", "bf16"), device)
    target = nhwc_uint8_to_nchw_float(frames_u8.to(device))

    encoder.train()
    decoder.train()
    for step in range(steps):
        optim.zero_grad(set_to_none=True)
        with autocast_context(device, amp_dtype):
            recon = decoder(encoder(target))
            loss = torch.nn.functional.l1_loss(recon, target)
        loss.backward()
        optim.step()
        if (step + 1) % max(1, steps // 4) == 0:
            print(f"    {variant:8s} step {step + 1:4d}/{steps}  l1={float(loss.detach()):.4f}")

    encoder.eval()
    decoder.eval()
    with torch.no_grad(), autocast_context(device, amp_dtype):
        recon = decoder(encoder(target)).float()
    regions = {
        "hud_l1": float(torch.nn.functional.l1_loss(hud_slice(recon), hud_slice(target))),
        "avatar_l1": float(
            torch.nn.functional.l1_loss(avatar_slice(recon), avatar_slice(target))
        ),
    }
    return float(torch.nn.functional.l1_loss(recon, target)), regions, recon.cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/m3_world_model.yaml"))
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out", type=Path, default=Path("results/m3/decoder_capacity.png")
    )
    args = parser.parse_args()

    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    configure_runtime(device)
    print(f"device: {device}")

    replay_path = Path(cfg["collect"]["out_path"])
    buffer = ReplayBuffer()
    buffer.load_state_dict(torch.load(replay_path, weights_only=False))
    print(f"replay: episodes={len(buffer)} steps={buffer.num_steps}")

    torch.manual_seed(args.seed)
    # One frame from each of `frames` sampled windows, so the batch is not 8
    # near-identical consecutive frames.
    batch = buffer.sample(args.frames, 8)
    frames_u8 = batch["obs"][:, -1]
    print(f"overfitting {tuple(frames_u8.shape)} for {args.steps} steps\n")

    results: dict[str, tuple[float, dict[str, float], torch.Tensor]] = {}
    for variant in ("subpixel", "nearest"):
        print(f"  variant={variant}")
        results[variant] = overfit(
            frames_u8=frames_u8,
            cfg=cfg,
            device=device,
            steps=args.steps,
            variant=variant,
            seed=args.seed,
        )
        print()

    print(f"{'variant':10s} {'full_l1':>9s} {'hud_l1':>9s} {'avatar_l1':>10s}")
    for variant, (l1, regions, _) in results.items():
        print(
            f"{variant:10s} {l1:9.4f} {regions['hud_l1']:9.4f} "
            f"{regions['avatar_l1']:10.4f}"
        )
    sub_l1 = results["subpixel"][0]
    near_l1 = results["nearest"][0]
    print(
        f"\nsub-pixel is {(1.0 - sub_l1 / max(near_l1, 1e-8)) * 100.0:+.1f}% "
        "full-frame L1 vs the nearest control"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        np.concatenate(
            [
                frames_u8[i].numpy(),
                nchw_float_to_nhwc_uint8(results["subpixel"][2][i : i + 1])[0].numpy(),
                nchw_float_to_nhwc_uint8(results["nearest"][2][i : i + 1])[0].numpy(),
            ],
            axis=1,
        )
        for i in range(frames_u8.shape[0])
    ]
    Image.fromarray(np.concatenate(rows, axis=0), mode="RGB").save(args.out)
    print(f"wrote {args.out}  (columns: real | sub-pixel | nearest)")


if __name__ == "__main__":
    main()
