"""Train the Dreamer outer loop: collect → WM → actor-critic.

M5 (`configs/m5_outer_loop.yaml`): 100k env steps, eval return.
M6 (`configs/m6_baseline.yaml`): continue M5 to 1M with official Crafter gmean.
M7 (`configs/m7_paper_online.yaml`): paper-style online 1M, fresh actor, size
from `world_model_config` (S keeps M6 WM; XL is ~200M and cannot load M6).

Prefer the matching notebook (`08` / `09` / `10`) for a live run you can stop.
This CLI is the canonical loop the notebooks import.

    conda activate worldmodel
    python scripts/train_agent.py --config configs/m7_paper_online.yaml --resume auto
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from agents.actor_critic import Actor, Critic, SlowCritic
from envs.crafter_env import register_crafter_envs
from models.world_model import WorldModel
from training.ckpt import resolve_outer_resume
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
from training.crafter_score import (
    append_jsonl,
    episode_jsonl_row,
    load_jsonl,
    score_from_episodes,
)
from training.evaluate import evaluate_policy, save_eval_gif
from training.ac_step import actor_critic_step
from training.imagine import decode_imagination, unfreeze_world_model
from training.outer_loop import (
    absorb_finished,
    crossed_interval,
    flush_episode_window,
    joint_payload,
    load_checkpoint,
    loop_updates,
    outer_cycle,
    save_checkpoint,
    save_replay,
)
from training.replay_buffer import ReplayBuffer, prefill_random_steps
from training.returns import PercentileReturnNorm
from training.wm_step import world_model_step

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


def make_slow_critic(
    cfg: dict[str, Any], critic: Critic, device: torch.device
) -> SlowCritic:
    """DreamerV3 slow critic. `critic.slow_target: false` disables the anchor."""
    critic_cfg = cfg.get("critic", {})
    return SlowCritic(
        critic,
        fraction=float(critic_cfg.get("slow_target_fraction", 0.02)),
        update_every=int(critic_cfg.get("slow_target_update", 1)),
    ).to(device)


def load_seed_world_model(m5_cfg: dict[str, Any], device: torch.device) -> tuple[WorldModel, dict[str, Any]]:
    """Build from `world_model_config` and optionally load a checkpoint.

    `world_model_ckpt` may be an M3 payload (`model`) or a joint outer-loop
    payload (`world_model`). Omit the ckpt (or set it null) to start from
    random weights — required when changing size S→XL; you cannot copy M6.
    """
    wm_cfg = load_yaml(Path(m5_cfg["world_model_config"]))
    model = build_model(wm_cfg).to(device)
    raw = m5_cfg.get("world_model_ckpt")
    if not raw:
        n_m = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"world model random init ({n_m:.1f}M params) from {m5_cfg['world_model_config']}")
        return model, wm_cfg
    ckpt_path = Path(raw)
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"world-model checkpoint not found: {ckpt_path}. Train M3 first."
        )
    payload = torch.load(ckpt_path, weights_only=False, map_location=device)
    if "world_model" in payload:
        model.load_state_dict(payload["world_model"], strict=True)
        print(f"world model from joint {ckpt_path} (env_steps {payload.get('env_steps', '?')})")
    elif "model" in payload:
        model.load_state_dict(payload["model"], strict=True)
        print(f"world model from {ckpt_path} (wm step {payload.get('step', '?')})")
    else:
        raise KeyError(f"{ckpt_path} has neither 'world_model' nor 'model'")
    return model, wm_cfg


def load_seed_actor_critic(
    m5_cfg: dict[str, Any],
    actor: Actor,
    critic: Critic,
    ac_optim: torch.optim.Optimizer,
    retnorm: PercentileReturnNorm,
    device: torch.device,
) -> None:
    raw = m5_cfg.get("actor_critic_ckpt")
    if not raw:
        raise FileNotFoundError("actor_critic_ckpt is empty")
    ckpt_path = Path(raw)
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
    if resume and replay_out.is_file():
        buffer.load_state_dict(torch.load(replay_out, weights_only=False))
        print(f"replay resumed {replay_out}: episodes={len(buffer)} steps={buffer.num_steps}")
        return buffer
    seed_raw = m5_cfg.get("seed_replay")
    if not seed_raw:
        print("replay empty (no seed_replay) — will prefill from collect")
        return buffer
    seed_path = Path(seed_raw)
    if not seed_path.is_file():
        raise FileNotFoundError(
            f"seed replay not found: {seed_path}. Collect with the M3 config first."
        )
    buffer.load_state_dict(torch.load(seed_path, weights_only=False))
    print(f"seed replay {seed_path}: episodes={len(buffer)} steps={buffer.num_steps}")
    return buffer


def overlay_wm_train(wm_train_cfg: dict[str, Any], train: dict[str, Any]) -> dict[str, Any]:
    """Copy KL / term scales from the outer-loop yaml when set (M7 paper 0.5/0.1)."""
    out = dict(wm_train_cfg)
    for key in (
        "dyn_scale",
        "rep_scale",
        "free_nats",
        "free_nats_dyn",
        "recon_scale",
        "reward_scale",
        "continue_scale",
        "kl_scale",
    ):
        if key in train and train[key] is not None:
            out[key] = train[key]
    return out


def prefill_replay(collector: Collector, buffer: ReplayBuffer, seq_len: int, steps: int) -> None:
    """Backward-compatible wrapper. Prefer `prefill_random_steps` on the collect env."""
    prefill_random_steps(
        collector.env,
        buffer,
        steps=int(steps),
        max_episode_steps=int(collector.max_episode_steps),
        seq_len=int(seq_len),
        seed=int(collector.next_seed),
    )


def pretrain_dreamer(
    world_model: WorldModel,
    wm_optim: torch.optim.Optimizer,
    actor: Actor,
    critic: Critic,
    ac_optim: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    *,
    device: torch.device,
    wm_train_cfg: dict[str, Any],
    train: dict[str, Any],
    retnorm: PercentileReturnNorm,
    slow_critic: SlowCritic | None,
    amp_dtype: torch.dtype | None,
    scaler: torch.amp.GradScaler,
) -> tuple[int, int]:
    """DreamerV3-torch `pretrain: 100`: WM-then-AC updates on the prefill, no collect.

    The JAX/torch reference runs `_train` (world model, then actor-critic) this
    many times on the random 2500-step buffer before the first env step of the
    outer loop. WM-only pretrain was a misread of that flag.
    """
    n = int(train.get("pretrain_steps", train.get("pretrain_wm_steps", 0)))
    if n <= 0:
        return 0, 0
    if buffer.num_steps < int(train["seq_len"]):
        print(
            f"skip pretrain: replay has {buffer.num_steps} steps, need seq_len="
            f"{train['seq_len']}",
            flush=True,
        )
        return 0, 0

    batch_size = int(train["batch_size"])
    seq_len = int(train["seq_len"])
    last_wm: dict[str, float] | None = None
    last_ac: dict[str, float] | None = None
    for _ in range(n):
        unfreeze_world_model(world_model)
        batch = buffer.sample(batch_size, seq_len)
        _loss, last_wm = world_model_step(
            world_model,
            wm_optim,
            batch,
            device=device,
            train_cfg=wm_train_cfg,
            amp_dtype=amp_dtype,
            scaler=scaler,
            max_grad_norm=float(train.get("wm_max_grad_norm", 1000.0)),
        )
        batch = buffer.sample(batch_size, seq_len)
        _loss, last_ac, _rollout = actor_critic_step(
            world_model,
            actor,
            critic,
            ac_optim,
            batch,
            device=device,
            retnorm=retnorm,
            horizon=int(train.get("horizon", 15)),
            start_mode=str(train.get("start_mode", "all")),
            lam=float(train.get("lam", 0.95)),
            discount=float(train.get("discount", 0.997)),
            entropy_scale=float(train.get("entropy_scale", 3.0e-4)),
            imag_gradient=str(train.get("imag_gradient", "reinforce")),
            imag_gradient_mix=float(train.get("imag_gradient_mix", 0.0)),
            slow_critic=slow_critic,
            amp_dtype=amp_dtype,
            scaler=scaler,
            max_grad_norm=float(train.get("ac_max_grad_norm", 100.0)),
        )
    recon = last_wm.get("recon_l1", float("nan")) if last_wm else float("nan")
    ent = last_ac.get("entropy", float("nan")) if last_ac else float("nan")
    print(
        f"pretrain {n} WM+AC (DreamerV3-torch pretrain)  "
        f"recon_l1={recon:.4f}  ac_H={ent:.3f}",
        flush=True,
    )
    return n, n


def pretrain_world_model(
    world_model: WorldModel,
    wm_optim: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    *,
    device: torch.device,
    wm_train_cfg: dict[str, Any],
    batch_size: int,
    seq_len: int,
    steps: int,
    amp_dtype: torch.dtype | None,
    scaler: torch.amp.GradScaler,
    max_grad_norm: float,
) -> None:
    """Deprecated alias: WM-only. Prefer `pretrain_dreamer` (joint WM+AC)."""
    n = int(steps)
    if n <= 0:
        return
    unfreeze_world_model(world_model)
    last: dict[str, float] | None = None
    for _ in range(n):
        batch = buffer.sample(int(batch_size), int(seq_len))
        _loss, last = world_model_step(
            world_model,
            wm_optim,
            batch,
            device=device,
            train_cfg=wm_train_cfg,
            amp_dtype=amp_dtype,
            scaler=scaler,
            max_grad_norm=float(max_grad_norm),
        )
    recon = last.get("recon_l1", float("nan")) if last else float("nan")
    print(f"WM pretrain {n} steps  recon_l1={recon:.4f}", flush=True)


def make_envs(m5_cfg: dict[str, Any]) -> tuple[gym.Env, gym.Env]:
    register_crafter_envs()
    env_id = str(m5_cfg.get("env", {}).get("id", "CrafterReward-v1"))
    collect_env = gym.make(env_id)
    eval_env = gym.make(env_id)
    return collect_env, eval_env


def record_finished_episodes(
    finished: list[dict[str, Any]],
    env_steps: int,
    episodes_path: Path,
    collect_log: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Append finished collect episodes to jsonl and the in-memory log."""
    for ep in finished:
        counts = ep.get("achievement_counts") or {}
        jl = episode_jsonl_row(
            env_steps=env_steps,
            length=int(ep["length"]),
            ep_return=float(ep["return"]),
            counts=counts,
        )
        append_jsonl(episodes_path, jl)
        collect_log.append(jl)
    return collect_log


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
        f"score={result.crafter_score:.3f}  episodes={result.returns}",
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
    wm_train_cfg = overlay_wm_train(dict(wm_cfg["train"]), train)
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
    amp_dtype = parse_amp(train.get("amp", "bf16"), device)
    scaler = make_grad_scaler(device, amp_dtype)

    ckpt_dir = Path(train["checkpoint_dir"])
    results_dir = Path(train["results_dir"])
    log_dir = Path(train["log_dir"])
    replay_out = Path(train["replay_out"])
    for p in (ckpt_dir, results_dir, log_dir, replay_out.parent):
        p.mkdir(parents=True, exist_ok=True)

    resume_path = resolve_outer_resume(
        args.resume, ckpt_dir, seed_joint=cfg.get("seed_joint_ckpt")
    )
    env_steps = 0
    wm_steps = 0
    ac_steps = 0
    collect_seed = int(cfg["seed"])
    if resume_path is not None:
        counters = load_checkpoint(
            resume_path,
            world_model,
            wm_optim,
            actor,
            critic,
            ac_optim,
            retnorm,
            device,
            slow_critic=slow_critic,
        )
        env_steps = counters["env_steps"]
        wm_steps = counters["wm_steps"]
        ac_steps = counters["ac_steps"]
        collect_seed = counters["collect_seed"]
        print(f"resumed {resume_path} at env_steps={env_steps}")
    elif not bool(cfg.get("reset_actor", False)) and cfg.get("actor_critic_ckpt"):
        load_seed_actor_critic(cfg, actor, critic, ac_optim, retnorm, device)
    else:
        print("actor-critic random init (reset_actor or no actor_critic_ckpt)")

    if cfg.get("seed_joint_ckpt") and env_steps < 100_000:
        raise RuntimeError(
            f"expected M5 100k joint ckpt, got env_steps={env_steps} from {resume_path}. "
            "Do not start M6 from M3/M4 alone."
        )

    buffer = load_replay(cfg, train, resume=resume_path is not None)

    collect_env, eval_env = make_envs(cfg)
    if resume_path is None:
        prefill_random_steps(
            collect_env,
            buffer,
            steps=int(train.get("prefill_steps", 0)),
            max_episode_steps=int(train["max_episode_steps"]),
            seq_len=int(train["seq_len"]),
            seed=int(cfg["seed"]),
        )
        pre_wm, pre_ac = pretrain_dreamer(
            world_model,
            wm_optim,
            actor,
            critic,
            ac_optim,
            buffer,
            device=device,
            wm_train_cfg=wm_train_cfg,
            train=train,
            retnorm=retnorm,
            slow_critic=slow_critic,
            amp_dtype=amp_dtype,
            scaler=scaler,
        )
        wm_steps += pre_wm
        ac_steps += pre_ac
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
    episodes_path = results_dir / "collect_episodes.jsonl"
    collect_log = load_jsonl(episodes_path)
    if env_steps > 0:
        collect_log = [r for r in collect_log if int(r.get("env_steps", 0)) <= env_steps]
    if env_steps > 0 and metrics_path.is_file():
        prev = json.loads(metrics_path.read_text(encoding="utf-8"))
        history = [h for h in prev if int(h.get("env_steps", 0)) <= env_steps]
    if env_steps > 0 and eval_path.is_file():
        prev_e = json.loads(eval_path.read_text(encoding="utf-8"))
        eval_history = [h for h in prev_e if int(h.get("env_steps", 0)) <= env_steps]

    target = int(train["env_steps"])
    collect_every = int(train["collect_every"])
    wm_updates, ac_updates = loop_updates(train)
    imag_gradient = str(train.get("imag_gradient", "both"))
    eval_every = int(train["eval_every"])
    log_every = int(train["log_every"])
    image_every = int(train["image_every"])
    ckpt_every = int(train["checkpoint_every"])
    replay_every = int(train.get("replay_every", ckpt_every))
    start_mode = str(train.get("start_mode", "all"))

    print(
        f"outer loop to {target} env steps  collect_every={collect_every}  "
        f"wm/ac_updates={wm_updates}/{ac_updates}  imag_gradient={imag_gradient}  "
        f"start_mode={start_mode}  (start={env_steps})",
        flush=True,
    )
    last_log_time = time.time()
    last_log_env = env_steps
    pending_lens: list[float] = []
    pending_rets: list[float] = []
    last_ep_len = float("nan")
    last_ep_ret = float("nan")
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
            t_eval = time.time()
            run_eval(env_steps)
            print(
                f"held-out eval done in {time.time() - t_eval:.0f}s "
                f"(not counted in env/s)",
                flush=True,
            )
        last_log_time = time.time()
        last_log_env = env_steps

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
                wm_updates=wm_updates,
                ac_updates=ac_updates,
                batch_size=int(train["batch_size"]),
                seq_len=int(train["seq_len"]),
                retnorm=retnorm,
                horizon=int(train["horizon"]),
                start_mode=start_mode,
                lam=float(train.get("lam", 0.95)),
                discount=float(train.get("discount", 0.997)),
                entropy_scale=float(train.get("entropy_scale", 3.0e-4)),
                imag_gradient=imag_gradient,
                imag_gradient_mix=float(train.get("imag_gradient_mix", 0.0)),
                slow_critic=slow_critic,
                amp_dtype=amp_dtype,
                scaler=scaler,
                wm_max_grad_norm=float(train.get("wm_max_grad_norm", 1000.0)),
                ac_max_grad_norm=float(train.get("ac_max_grad_norm", 100.0)),
            )
            prev_steps = env_steps
            env_steps += collect_every
            if cycle.wm_metrics is not None:
                wm_steps += wm_updates
            if cycle.ac_metrics is not None:
                ac_steps += ac_updates

            row: dict[str, Any] = {"env_steps": env_steps, "wm_steps": wm_steps, "ac_steps": ac_steps}
            if cycle.wm_metrics:
                row.update({f"wm_{k}": v for k, v in cycle.wm_metrics.items()})
            if cycle.ac_metrics:
                row.update({f"ac_{k}": v for k, v in cycle.ac_metrics.items()})
            row["collect_reward"] = float(cycle.collect["reward_mean"])
            row["collect_entropy"] = float(cycle.collect["entropy"])
            finished = cycle.collect.get("episodes") or []
            if finished:
                absorb_finished(finished, pending_lens, pending_rets)
                collect_log = record_finished_episodes(
                    finished, env_steps, episodes_path, collect_log
                )

            if (
                crossed_interval(prev_steps, env_steps, log_every)
                or env_steps >= target
            ):
                now = time.time()
                dt = max(now - last_log_time, 1e-6)
                sps = (env_steps - last_log_env) / dt
                last_log_time = now
                last_log_env = env_steps
                last_ep_len, last_ep_ret = flush_episode_window(
                    pending_lens, pending_rets, last_ep_len, last_ep_ret
                )
                if np.isfinite(last_ep_len):
                    row["collect_ep_len"] = last_ep_len
                    row["collect_ep_return"] = last_ep_ret
                row["env_steps_per_sec"] = sps
                if collect_log:
                    online_score, _ = score_from_episodes(
                        collect_log, budget=int(train["env_steps"])
                    )
                    row["online_crafter_score"] = online_score
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
                    f"ep_len={row.get('collect_ep_len', float('nan')):.0f}  "
                    f"collect_r={row['collect_reward']:.4f}  "
                    f"score={row.get('online_crafter_score', float('nan')):.3f}  "
                    f"({sps:.2f} env/s){vram_s}",
                    flush=True,
                )

            if crossed_interval(prev_steps, env_steps, eval_every):
                run_eval(env_steps)

            if cycle.rollout is not None and crossed_interval(
                prev_steps, env_steps, image_every
            ):
                vis = decode_imagination(world_model, cycle.rollout.feat, max_starts=1)
                save_imagination_strip(vis, results_dir / f"imagine_step_{env_steps:06d}.png")
                save_imagination_gif(vis, results_dir / f"imagine_step_{env_steps:06d}.gif")

            if crossed_interval(prev_steps, env_steps, ckpt_every):
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
                    slow_critic=slow_critic,
                )
                save_checkpoint(ckpt_dir / f"ckpt_step_{env_steps}.pt", payload)
                save_checkpoint(ckpt_dir / "ckpt_latest.pt", payload)
                metrics_path.write_text(json.dumps(history), encoding="utf-8")
                eval_path.write_text(json.dumps(eval_history), encoding="utf-8")
                print(f"wrote {ckpt_dir / f'ckpt_step_{env_steps}.pt'}", flush=True)
                gc.collect()

            if crossed_interval(prev_steps, env_steps, replay_every):
                save_replay(buffer, replay_out)
                print(f"wrote replay {replay_out}", flush=True)

    finally:
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
            slow_critic=slow_critic,
        )
        save_checkpoint(ckpt_dir / "ckpt_final.pt", payload)
        save_checkpoint(ckpt_dir / "ckpt_latest.pt", payload)
        save_replay(buffer, replay_out)
        metrics_path.write_text(json.dumps(history), encoding="utf-8")
        eval_path.write_text(json.dumps(eval_history), encoding="utf-8")
        collect_env.close()
        eval_env.close()
        writer.flush()
        writer.close()
        print("done", ckpt_dir / "ckpt_final.pt")


if __name__ == "__main__":
    main()
