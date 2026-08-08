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
            reward_scale=float(train["reward_scale"]),
            continue_scale=float(train["continue_scale"]),
            kl_scale=float(train["kl_scale"]),
        )

        optim.zero_grad(set_to_none=True)
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        optim.step()

        metrics = {
            "total": float(loss.total.detach()),
            "recon": float(loss.recon.detach()),
            "reward": float(loss.reward.detach()),
            "continue": float(loss.continue_loss.detach()),
            "kl": float(loss.kl.detach()),
            "kl_dyn": float(loss.kl_dyn.detach()),
            "kl_rep": float(loss.kl_rep.detach()),
        }

        if step % log_every == 0 or step == 1:
            for k, v in metrics.items():
                writer.add_scalar(f"m3/{k}", v, step)
            history.append({"step": step, **metrics})
            print(
                f"step {step:5d}  total={metrics['total']:.4f}  "
                f"recon={metrics['recon']:.4f}  rew={metrics['reward']:.4f}  "
                f"cont={metrics['continue']:.4f}  kl={metrics['kl']:.4f} "
                f"(dyn={metrics['kl_dyn']:.3f} rep={metrics['kl_rep']:.3f})"
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
    print(f"done. final ckpt={final} metrics={metrics_path}")


if __name__ == "__main__":
    main()
