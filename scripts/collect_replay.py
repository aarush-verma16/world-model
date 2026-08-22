"""Collect random-policy Crafter episodes into a sequential replay dump.

Usage:
    conda activate worldmodel
    python scripts/collect_replay.py --config configs/m3_world_model.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from training.replay_buffer import collect_random_episodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/m3_world_model.yaml"))
    args = parser.parse_args()
    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    env = cfg["env"]
    collect = cfg["collect"]
    out_path = Path(collect["out_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"collecting {collect['num_episodes']} episodes "
        f"(max_steps={collect['max_episode_steps']}) from {env['id']}..."
    )
    buffer = collect_random_episodes(
        env_id=str(env["id"]),
        num_episodes=int(collect["num_episodes"]),
        max_episode_steps=int(collect["max_episode_steps"]),
        action_dim=int(env["action_dim"]),
        seed=int(cfg["seed"]),
    )
    torch.save(buffer.state_dict(), out_path)
    print(
        f"wrote {out_path}: episodes={len(buffer)} steps={buffer.num_steps}"
    )


if __name__ == "__main__":
    main()
