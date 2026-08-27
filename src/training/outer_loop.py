"""One Dreamer outer-loop cycle: collect → world-model step → actor-critic step.

World-model parameters are frozen inside `actor_critic_step`. This module
unfreezes them before each WM update. Joint checkpoints store WM + actor +
critic + both optimizers + retnorm; the replay dump is a separate file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from agents.actor_critic import Actor, Critic
from models.world_model import WorldModel
from training.ac_step import actor_critic_step
from training.collect import Collector
from training.imagine import unfreeze_world_model
from training.replay_buffer import ReplayBuffer
from training.returns import PercentileReturnNorm
from training.wm_step import world_model_step


def loop_updates(train: dict[str, Any]) -> tuple[int, int]:
    """World-model / actor-critic updates per collect cycle.

    DreamerV3 `train_ratio` is replayed transitions trained per env step.
    Crafter uses 512 (`NM512/dreamerv3-torch`). With batch 16 × seq 32 that
    is 16 WM + 16 AC steps per 16 env steps. If `train_ratio` is omitted,
    `wm_updates` / `ac_updates` are used (M5/M6 16/1/1).
    """
    collect_every = int(train["collect_every"])
    batch = int(train["batch_size"])
    seq = int(train["seq_len"])
    raw_ratio = train.get("train_ratio")
    if raw_ratio is not None:
        n = max(1, int(round(float(raw_ratio) * collect_every / (batch * seq))))
        return n, n
    return int(train.get("wm_updates", 1)), int(train.get("ac_updates", 1))


def crossed_interval(prev: int, now: int, every: int) -> bool:
    """True when `now` crossed a multiple of `every` that `prev` had not.

    Env steps advance by `collect_every` (16), so `now % every == 0` never
    hits 5000. Compare integer buckets instead.
    """
    if every <= 0:
        return False
    return max(0, int(now)) // int(every) > max(0, int(prev)) // int(every)


@dataclass
class OuterCycleResult:
    """Metrics from one collect / WM / AC cycle."""

    collect: dict[str, Any]
    wm_metrics: dict[str, float] | None
    ac_metrics: dict[str, float] | None
    rollout: Any


def outer_cycle(
    collector: Collector,
    world_model: WorldModel,
    actor: Actor,
    critic: Critic,
    wm_optim: torch.optim.Optimizer,
    ac_optim: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    *,
    device: torch.device,
    wm_train_cfg: dict[str, Any],
    collect_every: int,
    wm_updates: int,
    ac_updates: int,
    batch_size: int,
    seq_len: int,
    retnorm: PercentileReturnNorm,
    horizon: int,
    start_mode: str,
    lam: float,
    discount: float,
    entropy_scale: float,
    imag_gradient: str = "both",
    amp_dtype: torch.dtype | None,
    scaler: torch.amp.GradScaler,
    wm_max_grad_norm: float = 1000.0,
    ac_max_grad_norm: float = 100.0,
) -> OuterCycleResult:
    """Collect `collect_every` env steps, then `wm_updates` WM and `ac_updates` AC.

    Returns the last WM / AC metric dicts (or None if that side ran 0 updates).
    """
    collect_stats = collector.collect(int(collect_every))

    wm_metrics: dict[str, float] | None = None
    for _ in range(int(wm_updates)):
        unfreeze_world_model(world_model)
        batch = buffer.sample(int(batch_size), int(seq_len))
        _loss, wm_metrics = world_model_step(
            world_model,
            wm_optim,
            batch,
            device=device,
            train_cfg=wm_train_cfg,
            amp_dtype=amp_dtype,
            scaler=scaler,
            max_grad_norm=float(wm_max_grad_norm),
        )

    ac_metrics: dict[str, float] | None = None
    rollout = None
    for _ in range(int(ac_updates)):
        batch = buffer.sample(int(batch_size), int(seq_len))
        _loss, ac_metrics, rollout = actor_critic_step(
            world_model,
            actor,
            critic,
            ac_optim,
            batch,
            device=device,
            retnorm=retnorm,
            horizon=int(horizon),
            start_mode=str(start_mode),
            lam=float(lam),
            discount=float(discount),
            entropy_scale=float(entropy_scale),
            imag_gradient=str(imag_gradient),
            amp_dtype=amp_dtype,
            scaler=scaler,
            max_grad_norm=float(ac_max_grad_norm),
        )

    return OuterCycleResult(
        collect=collect_stats,
        wm_metrics=wm_metrics,
        ac_metrics=ac_metrics,
        rollout=rollout,
    )


def joint_payload(
    *,
    env_steps: int,
    wm_steps: int,
    ac_steps: int,
    world_model: WorldModel,
    wm_optim: torch.optim.Optimizer,
    actor: Actor,
    critic: Critic,
    ac_optim: torch.optim.Optimizer,
    retnorm: PercentileReturnNorm,
    collect_seed: int,
) -> dict[str, Any]:
    """Checkpoint dict — replay is saved separately (`save_replay`)."""
    return {
        "env_steps": int(env_steps),
        "wm_steps": int(wm_steps),
        "ac_steps": int(ac_steps),
        "world_model": world_model.state_dict(),
        "wm_optim": wm_optim.state_dict(),
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "ac_optim": ac_optim.state_dict(),
        "retnorm": retnorm.state_dict(),
        "collect_seed": int(collect_seed),
    }


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    world_model: WorldModel,
    wm_optim: torch.optim.Optimizer,
    actor: Actor,
    critic: Critic,
    ac_optim: torch.optim.Optimizer,
    retnorm: PercentileReturnNorm,
    device: torch.device,
) -> dict[str, int]:
    """Restore weights / optim / retnorm. Returns step counters + collect_seed."""
    payload = torch.load(path, weights_only=False, map_location=device)
    world_model.load_state_dict(payload["world_model"], strict=True)
    actor.load_state_dict(payload["actor"], strict=True)
    critic.load_state_dict(payload["critic"], strict=True)
    if "wm_optim" in payload:
        wm_optim.load_state_dict(payload["wm_optim"])
    if "ac_optim" in payload:
        ac_optim.load_state_dict(payload["ac_optim"])
    if "retnorm" in payload:
        retnorm.load_state_dict(payload["retnorm"])
    return {
        "env_steps": int(payload.get("env_steps", 0)),
        "wm_steps": int(payload.get("wm_steps", 0)),
        "ac_steps": int(payload.get("ac_steps", 0)),
        "collect_seed": int(payload.get("collect_seed", 0)),
    }


def save_replay(buffer: ReplayBuffer, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(buffer.state_dict(), path)
