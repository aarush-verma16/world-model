"""Train the full Dreamer outer loop (M5): collect → WM → actor-critic.

Seeds the 700k world model and 20k actor-critic. Real-env eval return is the
skill curve (imagined λ-return is not). No geometric-mean Crafter score here.

Prefer `notebooks/08_train_outer_loop.ipynb` for a live run you can stop.
This CLI is the canonical loop the notebook imports.

    conda activate worldmodel
    python scripts/train_agent.py --config configs/m5_outer_loop.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from agents.actor_critic import Actor, Critic
from envs.crafter_env import register_crafter_envs
from models.world_model import WorldModel
from training.ckpt import resolve_resume
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
from training.evaluate import evaluate_policy, save_eval_gif
from training.imagine import decode_imagination
from training.outer_loop import (
    joint_payload,
    load_checkpoint,
    outer_cycle,
    save_checkpoint,
    save_replay,
)
from training.replay_buffer import ReplayBuffer
from training.returns import PercentileReturnNorm

from train_actor_critic import save_imagination_gif, save_imagination_strip
from train_world_model import build_model, set_seed


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_actor_critic(cfg: dict[str, Any], world_model: WorldModel, device: torch.device) -> tuple[Actor, Critic]:
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
    return actor, critic


def load_seed_world_model(m5_cfg: dict[str, Any], device: torch.device) -> tuple[WorldModel, dict[str, Any]]:
    """Build from the M3 yaml and load `world_model_ckpt` (trainable, not frozen)."""
    wm_cfg = load_yaml(Path(m5_cfg["world_model_config"]))
    model = build_model(wm_cfg).to(device)
    ckpt_path = Path(m5_cfg["world_model_ckpt"])
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"world-model checkpoint not found: {ckpt_path}. Train M3 first."
        )
    payload = torch.load(ckpt_path, weights_only=False, map_location=device)
    model.load_state_dict(payload["model"], strict=True)
    print(f"world model from {ckpt_path} (wm step {payload.get('step', '?')})")
    return model, wm_cfg


def load_seed_actor_critic(
    m5_cfg: dict[str, Any],
    actor: Actor,
    critic: Critic,
    ac_optim: torch.optim.Optimizer,
    retnorm: PercentileReturnNorm,
    device: torch.device,
) -> None:
    ckpt_path = Path(m5_cfg["actor_critic_ckpt"])
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"actor-critic checkpoint not found: {ckpt_path}. Train M4 first."
        )
    payload = torch.load(ckpt_path, weights_only=False, map_location=device)
    actor.load_state_dict(payload["actor"], strict=True)
    critic.load_state_dict(payload["critic"], strict=True)
    if "optim" in payload:
        try:
            ac_optim.load_state_dict(payload["optim"])
        except (ValueError, RuntimeError) as exc:
            print(f"skipping M4 optim state ({exc})")
    if "retnorm" in payload:
        retnorm.load_state_dict(payload["retnorm"])
    print(f"actor-critic from {ckpt_path} (ac step {payload.get('step', '?')})")


def load_replay(m5_cfg: dict[str, Any], train: dict[str, Any], *, resume: bool) -> ReplayBuffer:
    max_steps = train.get("replay_max_steps")
    buffer = ReplayBuffer(
        seed=int(m5_cfg.get("seed", 0)),
        max_steps=None if max_steps is None else int(max_steps),
    )
    replay_out = Path(train["replay_out"])
    seed_path = Path(m5_cfg["seed_replay"])
    if resume and replay_out.is_file():
        buffer.load_state_dict(torch.load(replay_out, weights_only=False))
        print(f"replay resumed {replay_out}: episodes={len(buffer)} steps={buffer.num_steps}")
        return buffer
    if not seed_path.is_file():
        raise FileNotFoundError(
            f"seed replay not found: {seed_path}. Collect with the M3 config first."
        )
    buffer.load_state_dict(torch.load(seed_path, weights_only=False))
    print(f"seed replay {seed_path}: episodes={len(buffer)} steps={buffer.num_steps}")
    return buffer


def make_envs(m5_cfg: dict[str, Any]) -> tuple[gym.Env, gym.Env]:
    register_crafter_envs()
    env_id = str(m5_cfg.get("env", {}).get("id", "CrafterReward-v1"))
    collect_env = gym.make(env_id)
    eval_env = gym.make(env_id)
    return collect_env, eval_env


def log_eval(
    result,
    env_steps: int,
    eval_history: list[dict[str, float]],
    writer: SummaryWriter | None,
) -> dict[str, float]:
    row = {"env_steps": float(env_steps), **result.as_metrics()}
    eval_history.append(row)
    if writer is not None:
        for k, v in result.as_metrics().items():
            writer.add_scalar(f"eval/{k}", v, env_steps)
    print(
        f"eval @ {env_steps}  return={result.mean_return:.3f}±{result.std_return:.3f}  "
        f"len={result.mean_length:.1f}  ach={result.mean_achievements:.2f}  "
        f"episodes={result.returns}",
        flush=True,
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/m5_outer_loop.yaml"))
    parser.add_argument(
        "--resume",
        default=None,
        help='Joint checkpoint, or "auto" for ckpt_latest in checkpoint_dir.',
    )
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    train = cfg["train"]
    set_seed(int(cfg["seed"]))
    device = get_device()
    configure_runtime(device)
    warn_if_not_cuda(device)
    print(f"device: {describe_device(device)}")

    world_model, wm_cfg = load_seed_world_model(cfg, device)
    wm_train_cfg = dict(wm_cfg["train"])
    actor, critic = make_actor_critic(cfg, world_model, device)
    wm_optim = torch.optim.Adam(world_model.parameters(), lr=float(train["wm_lr"]))
    ac_optim = torch.optim.Adam(
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
    replay_out = Path(train["replay_out"])
    for p in (ckpt_dir, results_dir, log_dir, replay_out.parent):
        p.mkdir(parents=True, exist_ok=True)

    resume_path = resolve_resume(args.resume, ckpt_dir)
    env_steps = 0
    wm_steps = 0
    ac_steps = 0
    collect_seed = int(cfg["seed"])
    if resume_path is not None:
        counters = load_checkpoint(
            resume_path, world_model, wm_optim, actor, critic, ac_optim, retnorm, device
        )
        env_steps = counters["env_steps"]
        wm_steps = counters["wm_steps"]
        ac_steps = counters["ac_steps"]
        collect_seed = counters["collect_seed"]
        print(f"resumed {resume_path} at env_steps={env_steps}")
    else:
        load_seed_actor_critic(cfg, actor, critic, ac_optim, retnorm, device)

    buffer = load_replay(cfg, train, resume=resume_path is not None)

    collect_env, eval_env = make_envs(cfg)
    collector = Collector(
        collect_env,
        world_model,
        actor,
        buffer,
        device=device,
        max_episode_steps=int(train["max_episode_steps"]),
        amp_dtype=amp_dtype,
        seed=collect_seed,
    )

    writer = SummaryWriter(log_dir=str(log_dir))
    history: list[dict] = []
    eval_history: list[dict[str, float]] = []
    metrics_path = results_dir / "train_metrics.json"
    eval_path = results_dir / "eval_metrics.json"
    if env_steps > 0 and metrics_path.is_file():
        prev = json.loads(metrics_path.read_text(encoding="utf-8"))
        history = [h for h in prev if int(h.get("env_steps", 0)) <= env_steps]
    if env_steps > 0 and eval_path.is_file():
        prev_e = json.loads(eval_path.read_text(encoding="utf-8"))
        eval_history = [h for h in prev_e if int(h.get("env_steps", 0)) <= env_steps]

    target = int(train["env_steps"])
    collect_every = int(train["collect_every"])
    eval_every = int(train["eval_every"])
    log_every = int(train["log_every"])
    image_every = int(train["image_every"])
    ckpt_every = int(train["checkpoint_every"])
    start_mode = str(train.get("start_mode", "all"))

    print(
        f"outer loop to {target} env steps  collect_every={collect_every}  "
        f"start_mode={start_mode}  (start={env_steps})",
        flush=True,
    )
    last_log_time = time.time()
    last_log_env = env_steps
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def run_eval(step: int) -> None:
        result = evaluate_policy(
            eval_env,
            world_model,
            actor,
            device=device,
            n_episodes=int(train["eval_episodes"]),
            max_steps=int(train.get("eval_max_steps", train["max_episode_steps"])),
            amp_dtype=amp_dtype,
            seed=int(train.get("eval_seed", 100_000)),
        )
        log_eval(result, step, eval_history, writer)
        if result.frames is not None:
            save_eval_gif(result.frames, results_dir / f"eval_step_{step:06d}.gif")
        eval_path.write_text(json.dumps(eval_history), encoding="utf-8")

    try:
        if env_steps == 0 or not eval_history:
            run_eval(env_steps)

        while env_steps < target:
            cycle = outer_cycle(
                collector,
                world_model,
                actor,
                critic,
                wm_optim,
                ac_optim,
                buffer,
                device=device,
                wm_train_cfg=wm_train_cfg,
                collect_every=collect_every,
                wm_updates=int(train["wm_updates"]),
                ac_updates=int(train["ac_updates"]),
                batch_size=int(train["batch_size"]),
                seq_len=int(train["seq_len"]),
                retnorm=retnorm,
                horizon=int(train["horizon"]),
                start_mode=start_mode,
                lam=float(train.get("lam", 0.95)),
                discount=float(train.get("discount", 0.997)),
                entropy_scale=float(train.get("entropy_scale", 3.0e-4)),
                amp_dtype=amp_dtype,
                scaler=scaler,
                wm_max_grad_norm=float(train.get("wm_max_grad_norm", 1000.0)),
                ac_max_grad_norm=float(train.get("ac_max_grad_norm", 100.0)),
            )
            env_steps += collect_every
            if cycle.wm_metrics is not None:
                wm_steps += int(train["wm_updates"])
            if cycle.ac_metrics is not None:
                ac_steps += int(train["ac_updates"])

            row: dict[str, Any] = {"env_steps": env_steps, "wm_steps": wm_steps, "ac_steps": ac_steps}
            if cycle.wm_metrics:
                row.update({f"wm_{k}": v for k, v in cycle.wm_metrics.items()})
            if cycle.ac_metrics:
                row.update({f"ac_{k}": v for k, v in cycle.ac_metrics.items()})
            row["collect_reward"] = float(cycle.collect["reward_mean"])
            row["collect_entropy"] = float(cycle.collect["entropy"])
            finished = cycle.collect.get("episodes") or []
            if finished:
                row["collect_ep_return"] = float(np.mean([e["return"] for e in finished]))
                row["collect_ep_len"] = float(np.mean([e["length"] for e in finished]))

            if env_steps % log_every == 0 or env_steps <= collect_every:
                now = time.time()
                dt = max(now - last_log_time, 1e-6)
                sps = (env_steps - last_log_env) / dt
                last_log_time = now
                last_log_env = env_steps
                row["env_steps_per_sec"] = sps
                history.append(row)
                if writer is not None:
                    for k, v in row.items():
                        if k == "env_steps":
                            continue
                        if isinstance(v, (int, float)) and np.isfinite(v):
                            writer.add_scalar(f"loop/{k}", float(v), env_steps)
                vram = vram_peak_gb()
                vram_s = f"  vram {vram[0]:.1f}/{vram[1]:.1f} GiB" if vram else ""
                ac_h = cycle.ac_metrics["entropy"] if cycle.ac_metrics else float("nan")
                wm_l1 = cycle.wm_metrics["recon_l1"] if cycle.wm_metrics else float("nan")
                print(
                    f"env {env_steps}/{target}  wm_l1={wm_l1:.4f}  ac_H={ac_h:.3f}  "
                    f"collect_r={row['collect_reward']:.4f}  "
                    f"({sps:.2f} env/s){vram_s}",
                    flush=True,
                )

            if env_steps % eval_every == 0:
                run_eval(env_steps)

            if cycle.rollout is not None and (
                env_steps % image_every == 0 or env_steps <= collect_every
            ):
                vis = decode_imagination(world_model, cycle.rollout.feat, max_starts=1)
                save_imagination_strip(vis, results_dir / f"imagine_step_{env_steps:06d}.png")
                save_imagination_gif(vis, results_dir / f"imagine_step_{env_steps:06d}.gif")

            if env_steps % ckpt_every == 0:
                payload = joint_payload(
                    env_steps=env_steps,
                    wm_steps=wm_steps,
                    ac_steps=ac_steps,
                    world_model=world_model,
                    wm_optim=wm_optim,
                    actor=actor,
                    critic=critic,
                    ac_optim=ac_optim,
                    retnorm=retnorm,
                    collect_seed=collector.next_seed,
                )
                save_checkpoint(ckpt_dir / f"ckpt_step_{env_steps}.pt", payload)
                save_checkpoint(ckpt_dir / "ckpt_latest.pt", payload)
                save_replay(buffer, replay_out)
                metrics_path.write_text(json.dumps(history), encoding="utf-8")
                eval_path.write_text(json.dumps(eval_history), encoding="utf-8")
                print(f"wrote {ckpt_dir / f'ckpt_step_{env_steps}.pt'}", flush=True)

        payload = joint_payload(
            env_steps=env_steps,
            wm_steps=wm_steps,
            ac_steps=ac_steps,
            world_model=world_model,
            wm_optim=wm_optim,
            actor=actor,
            critic=critic,
            ac_optim=ac_optim,
            retnorm=retnorm,
            collect_seed=collector.next_seed,
        )
        save_checkpoint(ckpt_dir / "ckpt_final.pt", payload)
        save_checkpoint(ckpt_dir / "ckpt_latest.pt", payload)
        save_replay(buffer, replay_out)
        metrics_path.write_text(json.dumps(history), encoding="utf-8")
        eval_path.write_text(json.dumps(eval_history), encoding="utf-8")
        print("done", ckpt_dir / "ckpt_final.pt")
    finally:
        collect_env.close()
        eval_env.close()
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
