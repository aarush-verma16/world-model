"""CUDA smoke for the M6 Crafter-score harness (not a 1M run).

Checks the official gmean formula and one short STE eval. Does not load
the M5 replay.

    conda activate worldmodel
    python scripts/smoke_crafter_score.py
"""

from __future__ import annotations

import math

import numpy as np
import torch

from pathlib import Path

from training.crafter_score import ACHIEVEMENT_NAMES, geometric_mean_score
from training.device import configure_runtime, describe_device, get_device, parse_amp, warn_if_not_cuda
from training.evaluate import evaluate_policy
from train_agent import load_yaml, make_actor_critic, make_envs
from train_world_model import build_model, set_seed


def main() -> None:
    percents = {n: 0.0 for n in ACHIEVEMENT_NAMES}
    assert geometric_mean_score(percents) == 0.0
    percents["collect_wood"] = 100.0
    expected = math.exp(math.log(101.0) / 22.0) - 1.0
    got = geometric_mean_score(percents)
    assert abs(got - expected) < 1e-12, (got, expected)
    print(f"gmean formula ok  (one 100% unlock -> {got:.4f})")

    set_seed(0)
    device = get_device()
    configure_runtime(device)
    warn_if_not_cuda(device)
    print(f"device: {describe_device(device)}")

    cfg = load_yaml(Path("configs/m6_baseline.yaml"))
    wm_cfg = load_yaml(Path(cfg["world_model_config"]))
    world_model = build_model(wm_cfg).to(device)
    actor, _critic = make_actor_critic(cfg, world_model, device)
    amp_dtype = parse_amp(cfg["train"].get("amp", "bf16"), device)
    collect_env, eval_env = make_envs(cfg)
    try:
        result = evaluate_policy(
            eval_env,
            world_model,
            actor,
            device=device,
            n_episodes=1,
            max_steps=8,
            amp_dtype=amp_dtype,
            seed=100_000,
        )
        print(
            f"eval return={result.mean_return:.3f}  len={result.mean_length:.1f}  "
            f"score={result.crafter_score:.4f}  n_pct={len(result.percents)}"
        )
        assert len(result.percents) == 22
        assert np.isfinite(result.crafter_score)
        if device.type == "cuda":
            torch.cuda.synchronize()
        print("SMOKE OK")
    finally:
        collect_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
