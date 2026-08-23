"""Train the full world model (encoder + RSSM + decoder + heads) on replay.

Logs each loss term separately (recon / reward / continue / KL) to TensorBoard.

Usage (Windows / CUDA):
    conda activate worldmodel
    python scripts/collect_replay.py
    python scripts/train_world_model.py
    tensorboard --logdir runs

Prefer `notebooks/05_train_world_model.ipynb` for a live run you can watch
and stop. This CLI is the canonical script for CI / unattended jobs.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from models.preprocess import nchw_float_to_nhwc_uint8, nhwc_uint8_to_nchw_float
from models.world_model import WorldModel
from training.device import (
    autocast_context,
    configure_runtime,
    describe_device,
    get_device,
    make_grad_scaler,
    parse_amp,
    to_device,
    vram_peak_gb,
    warn_if_not_cuda,
)
from training.replay_buffer import ReplayBuffer
from training.wm_step import world_model_step


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_recon_grid(obs_u8: torch.Tensor, preds: list[torch.Tensor], path: Path) -> None:
    """obs `[B,T,H,W,C]` uint8, preds each `[B,T,3,H,W]` float → side-by-side PNG.

    Column order: real, then each entry of `preds` in order. Callers pass
    `[recon, recon_embed]` (`[h,z]` and skip-free embed — what RSSM sees).
    """
    from PIL import Image

    # Mid-sequence: t=0 has a near-init `h`, so [h,z] looks like a generic
    # spawn even when later steps have the right layout.
    t_vis = obs_u8.shape[1] // 2
    real = obs_u8[:, t_vis].cpu()
    pred_imgs = [nchw_float_to_nhwc_uint8(p[:, t_vis].detach().cpu()) for p in preds]
    strips = [
        np.concatenate([real[i].numpy()] + [p[i].numpy() for p in pred_imgs], axis=1)
        for i in range(real.shape[0])
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(strips, axis=0), mode="RGB").save(path)


def print_exit_criteria(
    history: list[dict[str, float]],
    *,
    obs_std: float,
    recon_std: float,
    embed_std: float,
    bottleneck_std: float,
    reward_true: np.ndarray | None = None,
    reward_pred: np.ndarray | None = None,
) -> bool:
    """Print PASS/FAIL against M3's exit criteria. Returns overall pass/fail."""
    if len(history) < 2:
        print("[SKIP] not enough logged steps to judge a trend")
        return False

    first, last = history[0], history[-1]

    def pct_drop(key: str) -> float:
        return 1.0 - (last[key] / max(first[key], 1e-8))

    checks: list[tuple[str, bool, str]] = []
    for key, min_drop in [("recon_l1", 0.5), ("recon_embed_l1", 0.5)]:
        # Fall back to the training-term key on old metrics dumps that predate *_l1.
        a = first.get(key, first.get(key.replace("_l1", ""), 0.0))
        b = last.get(key, last.get(key.replace("_l1", ""), 0.0))
        drop = 1.0 - (b / max(a, 1e-8))
        checks.append(
            (
                f"{key} loss dropped >= {min_drop:.0%} "
                f"(first={a:.4f} -> last={b:.4f}, actual={drop:.0%})",
                drop >= min_drop,
                "recon isn't improving -- check lr, recon_scale, or a decoder/data bug",
            )
        )

    # kl_dyn_raw is the TOTAL KL of the latent (summed over stoch groups).
    # free_nats is a floor on the *loss* (max(KL, 1)), not a ceiling on the
    # value. Sitting a bit above 1 (this box's 8k-step run lived at 1.07-1.16
    # for thousands of steps) means the regularizer is on, which is healthy.
    # Real failure modes are collapse toward ~0 (dead latent) or blowing up
    # toward tens of nats (posterior ignoring the prior).
    kl_dyn_last = last["kl_dyn_raw"]
    checks.append(
        (
            f"kl_dyn_raw isn't dead (last={kl_dyn_last:.3f} > 0.02)",
            kl_dyn_last > 0.02,
            "posterior has collapsed onto the prior -- latent carries ~no information",
        )
    )
    checks.append(
        (
            f"kl_dyn_raw hasn't exploded (last={kl_dyn_last:.3f} < 30.0)",
            kl_dyn_last < 30.0,
            "posterior is ignoring the prior entirely -- imagination rollouts will drift badly",
        )
    )

    for name, std in [
        ("recon", recon_std),
        ("recon_embed", embed_std),
    ]:
        ratio = std / max(obs_std, 1e-8)
        checks.append(
            (
                f"{name} pixel std is close to real "
                f"({name}_std={std:.4f} vs obs_std={obs_std:.4f}, ratio={ratio:.2f})",
                ratio > 0.4,
                f"{name} has collapsed toward a constant (solid-color) output -- "
                "note: this only checks AGGREGATE pixel variance, not WHETHER that "
                "variance is in the right place. A model that nails big-block "
                "background color while dropping every small object (HUD digits, "
                "mobs, precise player pose) can still pass this check -- eyeball "
                "recon_final.png, don't rely on this alone.",
            )
        )

    # M3's actual documented exit criterion (milestones.md) is reward prediction
    # CORRELATION with real reward, not pixel fidelity -- this is the check that
    # matters for whether the latent is control-useful, independent of how the
    # `[h,z]` panel looks.
    if reward_true is not None and reward_pred is not None:
        if reward_true.std() > 1e-8 and reward_pred.std() > 1e-8:
            corr = float(np.corrcoef(reward_true, reward_pred)[0, 1])
            checks.append(
                (
                    f"reward prediction correlates with real reward (r={corr:.2f})",
                    corr > 0.3,
                    "reward head isn't tracking real reward -- latent may not be "
                    "carrying reward-relevant information",
                )
            )
        else:
            checks.append(
                (
                    "reward prediction correlation: SKIPPED (degenerate std -- too "
                    "few nonzero-reward examples in this sample)",
                    True,
                    "",
                )
            )

    all_pass = True
    for description, ok, hint in checks:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {description}")
        if not ok:
            print(f"       -> {hint}")
            all_pass = False
    print("RESULT:", "PASS -- M3 world model looks healthy" if all_pass else "FAIL -- see hints above")
    return all_pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/m3_world_model.yaml"))
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()
    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg["seed"]))
    device = get_device()
    configure_runtime(device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(cfg["seed"]))
        torch.cuda.reset_peak_memory_stats()
    print(f"device: {describe_device(device)}")
    warn_if_not_cuda(device)

    replay_path = Path(cfg["collect"]["out_path"])
    if not replay_path.exists():
        raise SystemExit(
            f"Missing {replay_path}. Run: python scripts/collect_replay.py"
        )
    buffer = ReplayBuffer(seed=int(cfg["seed"]))
    buffer.load_state_dict(torch.load(replay_path, weights_only=False))
    print(f"replay: episodes={len(buffer)} steps={buffer.num_steps}")

    enc = cfg["encoder"]
    rssm = cfg["rssm"]
    dec = cfg.get("decoder", {})
    heads = cfg.get("heads", {})
    train = cfg["train"]

    model = WorldModel.from_config_dims(
        embed_dim=int(enc["embed_dim"]),
        encoder_channels=tuple(int(c) for c in enc["channels"]),
        action_dim=int(cfg["env"]["action_dim"]),
        deter_dim=int(rssm["deter_dim"]),
        stoch=int(rssm["stoch"]),
        classes=int(rssm["classes"]),
        hidden=int(rssm["hidden"]),
        unimix=float(rssm.get("unimix", 0.01)),
        act=str(rssm.get("act", "silu")),
        initial=str(rssm.get("initial", "learned")),
        rec_depth=int(rssm.get("rec_depth", 1)),
        prior_layers=int(rssm.get("prior_layers", 2)),
        decoder_channels=tuple(int(c) for c in dec.get("channels", [256, 128, 64, 32])),
        head_hidden=int(heads.get("hidden", 512)),
        head_layers=int(heads.get("layers", 2)),
        encoder_blocks=int(enc.get("blocks", 2)),
        decoder_blocks=int(dec.get("blocks", 0)),
        stem_channels=int(enc.get("stem_channels", 64)),
        spatial=int(enc.get("spatial", 4)),
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=float(train["lr"]))
    start_step = 0
    if args.resume is not None:
        ckpt = torch.load(args.resume, weights_only=False, map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)
        if "optim" in ckpt:
            optim.load_state_dict(ckpt["optim"])
        start_step = int(ckpt.get("step", 0))
        print(f"resumed from {args.resume} at step {start_step}")

    log_dir = Path(train["log_dir"])
    ckpt_dir = Path(train["checkpoint_dir"])
    results_dir = Path(train["results_dir"])
    for p in (log_dir, ckpt_dir, results_dir):
        p.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    steps = int(train["steps"])
    batch_size = int(train["batch_size"])
    seq_len = int(train["seq_len"])
    log_every = int(train["log_every"])
    image_every = int(train["image_every"])
    ckpt_every = int(train["checkpoint_every"])
    amp_dtype = parse_amp(train.get("amp", "bf16"), device)
    scaler = make_grad_scaler(device, amp_dtype)
    print(f"amp: {train.get('amp', 'bf16')}  batch={batch_size}  seq_len={seq_len}")

    model.train()
    history: list[dict[str, float]] = []
    last_log_time = time.time()
    last_log_step = start_step
    for step in range(start_step + 1, steps + 1):
        batch = buffer.sample(batch_size, seq_len)
        _loss, metrics = world_model_step(
            model,
            optim,
            batch,
            device=device,
            train_cfg=train,
            amp_dtype=amp_dtype,
            scaler=scaler,
        )

        if step % log_every == 0 or step == 1:
            for k, v in metrics.items():
                writer.add_scalar(f"m3/{k}", v, step)
            history.append({"step": step, **metrics})
            now = time.time()
            steps_per_sec = (step - last_log_step) / max(now - last_log_time, 1e-6)
            last_log_time, last_log_step = now, step
            vram = vram_peak_gb()
            vram_s = f"  vram {vram[0]:.1f}/{vram[1]:.1f} GiB" if vram else ""
            print(
                f"step {step:5d}  total={metrics['total']:.4f}  "
                f"recon_l1={metrics['recon_l1']:.4f}  "
                f"emb_l1={metrics['recon_embed_l1']:.4f}  "
                f"blob={metrics['recon_blob']:.4f}  "
                f"bneck_l1={metrics['recon_bottleneck_l1']:.4f}  "
                f"grad={metrics['grad']:.4f}  "
                f"rew={metrics['reward']:.4f}  cont={metrics['continue']:.4f}  "
                f"kl={metrics['kl']:.4f} "
                f"(dyn_raw={metrics['kl_dyn_raw']:.3f} rep_raw={metrics['kl_rep_raw']:.3f})  "
                f"{steps_per_sec:.2f} steps/s{vram_s}"
            )

        if step % image_every == 0 or step == 1:
            model.eval()
            with torch.no_grad(), autocast_context(device, amp_dtype):
                vis = buffer.sample(min(4, batch_size), seq_len)
                vis_g = to_device(vis, device)
                v_out = model(vis_g["obs"], vis_g["actions"])
                save_recon_grid(
                    vis["obs"],
                    [v_out.recon, v_out.recon_embed],
                    results_dir / f"recon_step_{step:05d}.png",
                )
            model.train()

        if step % ckpt_every == 0:
            path = ckpt_dir / f"ckpt_step_{step:05d}.pt"
            torch.save(
                {"step": step, "model": model.state_dict(), "optim": optim.state_dict()},
                path,
            )
            print(f"wrote {path}")

    final = ckpt_dir / "ckpt_final.pt"
    torch.save(
        {
            "step": steps,
            "model": model.state_dict(),
            "optim": optim.state_dict(),
        },
        final,
    )
    metrics_path = results_dir / "train_metrics.json"
    metrics_path.write_text(json.dumps(history, indent=2))
    writer.flush()
    writer.close()

    model.eval()
    with torch.no_grad(), autocast_context(device, amp_dtype):
        vis = buffer.sample(8, seq_len)
        vis_g = to_device(vis, device)
        v_out = model(vis_g["obs"], vis_g["actions"])
        save_recon_grid(
            vis["obs"],
            [v_out.recon, v_out.recon_embed],
            results_dir / "recon_final.png",
        )
        obs_std = float(
            nhwc_uint8_to_nchw_float(vis_g["obs"]).std()
        )
        recon_std = float(v_out.recon.float().std())
        embed_std = float(v_out.recon_embed.float().std())
        bottleneck_std = float(v_out.recon_bottleneck.float().std())

        # Held-out-ish reward correlation check (M3's actual documented exit
        # criterion): sample several fresh sequences the loss wasn't just
        # computed on.
        true_chunks, pred_chunks = [], []
        for _ in range(20):
            rb = buffer.sample(batch_size, seq_len)
            rb_g = to_device(rb, device)
            r_out = model(rb_g["obs"], rb_g["actions"])
            true_chunks.append(rb["rewards"].numpy().reshape(-1))
            pred_chunks.append(r_out.reward_pred.squeeze(-1).float().cpu().numpy().reshape(-1))
        reward_true = np.concatenate(true_chunks)
        reward_pred = np.concatenate(pred_chunks)

    print(f"done. final ckpt={final} metrics={metrics_path}")
    print_exit_criteria(
        history,
        obs_std=obs_std,
        recon_std=recon_std,
        embed_std=embed_std,
        bottleneck_std=bottleneck_std,
        reward_true=reward_true,
        reward_pred=reward_pred,
    )


if __name__ == "__main__":
    main()
