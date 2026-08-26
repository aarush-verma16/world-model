"""Real-environment policy evaluation for M5 (return, not geometric-mean score).

Uses the same stochastic STE actor as collect. Geometric-mean Crafter score
and 10k-step episodes are M6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from agents.actor_critic import Actor
from models.world_model import WorldModel
from training.collect import count_unlocked_achievements, rssm_policy_step


@dataclass
class EvalResult:
    """One eval call. `frames` is the last episode `[T, 64, 64, 3]` uint8."""

    returns: list[float] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    achievements: list[int] = field(default_factory=list)
    mean_return: float = 0.0
    std_return: float = 0.0
    mean_length: float = 0.0
    mean_achievements: float = 0.0
    frames: np.ndarray | None = None

    def as_metrics(self) -> dict[str, float]:
        return {
            "eval_return": self.mean_return,
            "eval_return_std": self.std_return,
            "eval_length": self.mean_length,
            "eval_achievements": self.mean_achievements,
        }


@torch.no_grad()
def evaluate_policy(
    env: Any,
    world_model: WorldModel,
    actor: Actor,
    *,
    device: torch.device,
    n_episodes: int = 4,
    max_steps: int = 400,
    amp_dtype: torch.dtype | None = None,
    seed: int = 100_000,
    keep_frames: bool = True,
) -> EvalResult:
    """Run `n_episodes` real rollouts. Does not write the replay buffer.

    Args:
        env: a second gym env — must not be the collect env.
        seed: base seed; episode `i` uses `seed + i`.
        keep_frames: store the last episode's observations for a GIF.
    """
    if n_episodes < 1:
        raise ValueError(f"n_episodes must be >= 1, got {n_episodes}")
    rssm = world_model.rssm
    returns: list[float] = []
    lengths: list[int] = []
    achievements: list[int] = []
    last_frames: list[np.ndarray] = []

    for ep in range(int(n_episodes)):
        obs, _info = env.reset(seed=int(seed) + ep)
        obs_np = np.asarray(obs, dtype=np.uint8)
        state = rssm.initial(1, device=device)
        prev_action = torch.zeros(1, rssm.action_dim, device=device)
        ep_ret = 0.0
        frames: list[np.ndarray] = [obs_np.copy()]
        n_ach = 0
        ep_len = 0
        for _ in range(int(max_steps)):
            state, action_oh, action_i, _entropy = rssm_policy_step(
                world_model,
                actor,
                obs_np,
                state,
                prev_action,
                device=device,
                amp_dtype=amp_dtype,
            )
            next_obs, reward, terminated, truncated, info = env.step(action_i)
            ep_ret += float(reward)
            ep_len += 1
            obs_np = np.asarray(next_obs, dtype=np.uint8)
            frames.append(obs_np.copy())
            prev_action = action_oh
            n_ach = count_unlocked_achievements(info if isinstance(info, dict) else None)
            if bool(terminated) or bool(truncated):
                break
        returns.append(ep_ret)
        lengths.append(ep_len)
        achievements.append(n_ach)
        last_frames = frames

    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns, ddof=0)) if len(returns) > 1 else 0.0
    frame_arr = None
    if keep_frames and last_frames:
        frame_arr = np.stack(last_frames, axis=0)
    return EvalResult(
        returns=returns,
        lengths=lengths,
        achievements=achievements,
        mean_return=mean_ret,
        std_return=std_ret,
        mean_length=float(np.mean(lengths)),
        mean_achievements=float(np.mean(achievements)),
        frames=frame_arr,
    )


def save_eval_gif(
    frames: np.ndarray,
    path: Path,
    *,
    duration_ms: int = 80,
    max_frames: int = 80,
) -> None:
    """`frames` `[T, H, W, 3]` uint8 → GIF. Thins long episodes for the dashboard."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    t = int(frames.shape[0])
    stride = max(1, t // int(max_frames)) if t > max_frames else 1
    seq = [
        Image.fromarray(np.asarray(frames[i], dtype=np.uint8), mode="RGB")
        for i in range(0, t, stride)
    ]
    if not seq:
        return
    seq[0].save(
        path,
        save_all=True,
        append_images=seq[1:],
        duration=duration_ms,
        loop=0,
    )
