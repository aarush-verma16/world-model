"""Train the full world model (encoder + RSSM + decoder + heads) on replay.

Logs each loss term separately (recon / reward / continue / KL) to TensorBoard.

Usage:
    conda activate worldmodel
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    python scripts/collect_replay.py
    python scripts/train_world_model.py
    tensorboard --logdir runs
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
from training.device import get_device
from training.losses import world_model_loss
from training.replay_buffer import ReplayBuffer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_recon_grid(obs_u8: torch.Tensor, recon: torch.Tensor, path: Path) -> None:
    """obs `[B,T,H,W,C]` uint8, recon `[B,T,3,H,W]` float → side-by-side PNG."""
    from PIL import Image

    # Use first timestep of each batch item.
    real = obs_u8[:, 0].cpu()
    pred = nchw_float_to_nhwc_uint8(recon[:, 0].detach().cpu())
    strips = [
        np.concatenate([real[i].numpy(), pred[i].numpy()], axis=1)
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
    for key, min_drop in [("recon", 0.5), ("recon_embed", 0.5)]:
        drop = pct_drop(key)
        checks.append(
            (
                f"{key} loss dropped >= {min_drop:.0%} "
                f"(first={first[key]:.4f} -> last={last[key]:.4f}, actual={drop:.0%})",
                drop >= min_drop,
                "recon isn't improving -- check lr, recon_scale, or a decoder/data bug",
            )
        )

    # kl_dyn_raw is the TOTAL KL of the latent (summed over stoch groups). It's
    # expected to sit at/below free_nats for most (or all) of training -- that's
    # what the free-nats floor is for, not a target to exceed. Real failure
    # modes are collapse toward ~0 (dead latent) or blowing up (posterior
    # ignoring the prior, bad for imagination rollouts).
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

    for name, std in [("recon", recon_std), ("recon_embed", embed_std)]:
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
    print(f"device: {device}")
    if device.type != "mps":
        print(
            "WARNING: not running on MPS -- training will be ~10-20x slower on CPU. "
            "Check torch.backends.mps.is_available()."
        )

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
        decoder_channels=tuple(int(c) for c in dec.get("channels", [256, 128, 64, 32])),
        head_hidden=int(heads.get("hidden", 512)),
        head_layers=int(heads.get("layers", 2)),
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

    model.train()
    history: list[dict[str, float]] = []
    last_log_time = time.time()
    last_log_step = start_step
    for step in range(start_step + 1, start_step + steps + 1):
        batch = buffer.sample(batch_size, seq_len)
        obs = batch["obs"].to(device)
        actions = batch["actions"].to(device)
        rewards = batch["rewards"].to(device)
        cont = batch["cont"].to(device)

        out = model(obs, actions)
        b, t = obs.shape[:2]
        obs_f = nhwc_uint8_to_nchw_float(obs.reshape(b * t, *obs.shape[2:])).view(
            b, t, 3, 64, 64
        )
        loss = world_model_loss(
            obs=obs_f,
            recon=out.recon,
            recon_embed=out.recon_embed,
            reward=rewards,
            reward_pred=out.reward_pred,
            cont=cont,
            cont_logit=out.cont_logit,
            post_logits=out.rssm.posterior_logits,
            prior_logits=out.rssm.prior_logits,
            unimix=model.rssm.unimix,
            dyn_scale=float(train["dyn_scale"]),
            rep_scale=float(train["rep_scale"]),
            free_nats=float(train["free_nats"]),
            recon_scale=float(train["recon_scale"]),
            recon_embed_scale=float(train.get("recon_embed_scale", 1.0)),
            reward_scale=float(train["reward_scale"]),
            continue_scale=float(train["continue_scale"]),
            kl_scale=float(train["kl_scale"]),
            grad_scale=float(train.get("grad_scale", 0.0)),
            recon_loss_type=str(train.get("recon_loss", "l1")),
        )

        optim.zero_grad(set_to_none=True)
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        optim.step()

        metrics = {
            "total": float(loss.total.detach()),
            "recon": float(loss.recon.detach()),
            "recon_embed": float(loss.recon_embed.detach()),
            "grad": float(loss.grad.detach()),
            "reward": float(loss.reward.detach()),
            "continue": float(loss.continue_loss.detach()),
            "kl": float(loss.kl.detach()),
            "kl_dyn": float(loss.kl_dyn.detach()),
            "kl_rep": float(loss.kl_rep.detach()),
            "kl_dyn_raw": float(loss.kl_dyn_raw.detach()),
            "kl_rep_raw": float(loss.kl_rep_raw.detach()),
        }

        if step % log_every == 0 or step == 1:
            for k, v in metrics.items():
                writer.add_scalar(f"m3/{k}", v, step)
            history.append({"step": step, **metrics})
            now = time.time()
            steps_per_sec = (step - last_log_step) / max(now - last_log_time, 1e-6)
            last_log_time, last_log_step = now, step
            print(
                f"step {step:5d}  total={metrics['total']:.4f}  "
                f"recon={metrics['recon']:.4f}  emb={metrics['recon_embed']:.4f}  "
                f"grad={metrics['grad']:.4f}  "
                f"rew={metrics['reward']:.4f}  cont={metrics['continue']:.4f}  "
                f"kl={metrics['kl']:.4f} "
                f"(dyn_raw={metrics['kl_dyn_raw']:.3f} rep_raw={metrics['kl_rep_raw']:.3f})  "
                f"{steps_per_sec:.2f} steps/s"
            )

        if step % image_every == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                vis = buffer.sample(min(4, batch_size), seq_len)
                v_out = model(vis["obs"].to(device), vis["actions"].to(device))
                save_recon_grid(
                    vis["obs"], v_out.recon, results_dir / f"recon_step_{step:05d}.png"
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
            "step": start_step + steps,
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
    with torch.no_grad():
        vis = buffer.sample(8, seq_len)
        v_out = model(vis["obs"].to(device), vis["actions"].to(device))
        save_recon_grid(vis["obs"], v_out.recon, results_dir / "recon_final.png")
        obs_std = float(
            nhwc_uint8_to_nchw_float(vis["obs"].to(device)).std()
        )
        recon_std = float(v_out.recon.std())
        embed_std = float(v_out.recon_embed.std())

        # Held-out-ish reward correlation check (M3's actual documented exit
        # criterion): sample several fresh sequences the loss wasn't just
        # computed on.
        true_chunks, pred_chunks = [], []
        for _ in range(20):
            rb = buffer.sample(batch_size, seq_len)
            r_out = model(rb["obs"].to(device), rb["actions"].to(device))
            true_chunks.append(rb["rewards"].numpy().reshape(-1))
            pred_chunks.append(r_out.reward_pred.squeeze(-1).cpu().numpy().reshape(-1))
        reward_true = np.concatenate(true_chunks)
        reward_pred = np.concatenate(pred_chunks)

    print(f"done. final ckpt={final} metrics={metrics_path}")
    print_exit_criteria(
        history,
        obs_std=obs_std,
        recon_std=recon_std,
        embed_std=embed_std,
        reward_true=reward_true,
        reward_pred=reward_pred,
    )


if __name__ == "__main__":
    main()
