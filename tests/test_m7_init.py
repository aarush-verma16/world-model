"""M7 init: optional WM ckpt, joint payload key, reset_actor, empty replay."""

from __future__ import annotations

from pathlib import Path

import torch

from train_agent import load_replay, load_seed_world_model
from train_world_model import build_model


def test_load_seed_world_model_random_when_ckpt_null() -> None:
    cfg = {
        "world_model_config": "configs/sizes/dreamer_s.yaml",
        "world_model_ckpt": None,
    }
    device = torch.device("cpu")
    wm, wm_cfg = load_seed_world_model(cfg, device)
    assert wm_cfg["rssm"]["deter_dim"] == 512
    n = sum(p.numel() for p in wm.parameters())
    assert n > 1_000_000


def test_load_seed_world_model_from_joint_payload(tmp_path: Path) -> None:
    wm_cfg = {
        "world_model_config": "configs/sizes/dreamer_s.yaml",
        "world_model_ckpt": None,
    }
    device = torch.device("cpu")
    wm, _ = load_seed_world_model(wm_cfg, device)
    path = tmp_path / "joint.pt"
    torch.save({"world_model": wm.state_dict(), "env_steps": 99}, path)
    loaded, _ = load_seed_world_model(
        {"world_model_config": "configs/sizes/dreamer_s.yaml", "world_model_ckpt": str(path)},
        device,
    )
    a = next(wm.parameters()).detach()
    b = next(loaded.parameters()).detach()
    assert torch.equal(a, b)


def test_load_replay_allows_empty_seed() -> None:
    buf = load_replay(
        {"seed": 0, "seed_replay": None},
        {"replay_out": "data/_no_such_replay.pt", "replay_max_steps": 1000},
        resume=False,
    )
    assert len(buf) == 0


def test_xl_world_model_is_near_200m() -> None:
    import yaml

    cfg = yaml.safe_load(Path("configs/sizes/dreamer_xl.yaml").read_text(encoding="utf-8"))
    wm = build_model(cfg)
    n_m = sum(p.numel() for p in wm.parameters()) / 1e6
    assert 170.0 < n_m < 230.0, n_m
    assert wm.rssm.stoch == 32 and wm.rssm.classes == 32


def test_m7_paper_config_is_from_scratch() -> None:
    import yaml

    cfg = yaml.safe_load(Path("configs/m7_paper_online.yaml").read_text(encoding="utf-8"))
    assert cfg["world_model_config"] == "configs/sizes/dreamer_xl.yaml"
    assert cfg.get("world_model_ckpt") in (None, "")
    assert cfg.get("actor_critic_ckpt") in (None, "")
    assert cfg.get("seed_replay") in (None, "")
    assert cfg.get("reset_actor") is True
    assert "seed_joint_ckpt" not in cfg
    assert int(cfg["train"]["max_episode_steps"]) == 10_000
    assert int(cfg["actor"]["hidden"]) == 1024
    assert int(cfg["actor"]["layers"]) == 5
