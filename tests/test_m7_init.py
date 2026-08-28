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

    from training.outer_loop import loop_updates

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
    assert cfg["train"]["imag_gradient"] == "reinforce"
    assert float(cfg["train"]["train_ratio"]) == 512.0
    assert int(cfg["train"]["prefill_steps"]) == 2500
    assert int(cfg["train"]["pretrain_wm_steps"]) == 100
    assert "m7_paper_online" not in cfg["train"]["checkpoint_dir"]
    wm_u, ac_u = loop_updates(cfg["train"])
    assert wm_u == 16 and ac_u == 16


def test_prefill_random_reaches_step_target() -> None:
    import numpy as np

    from training.replay_buffer import ReplayBuffer, prefill_random_steps

    class Env:
        def __init__(self) -> None:
            self.t = 0
            self.action_space = type(
                "Space",
                (),
                {"n": 5, "sample": staticmethod(lambda: 1)},
            )()

        def reset(self, *, seed: int | None = None):
            self.t = 0
            return np.zeros((64, 64, 3), dtype=np.uint8), {}

        def step(self, action: int):
            self.t += 1
            done = self.t >= 20
            obs = np.zeros((64, 64, 3), dtype=np.uint8)
            return obs, 0.0, done, False, {}

    buf = ReplayBuffer(seed=0, max_steps=10_000)
    got = prefill_random_steps(
        Env(), buf, steps=80, max_episode_steps=20, seq_len=8, seed=0
    )
    assert got >= 80
    assert buf.num_steps >= 80
    assert len(buf) >= 4
    assert any(ep.obs.shape[0] >= 8 for ep in buf._episodes)


def test_m7_workstation_config_is_ratio_32() -> None:
    import yaml

    from training.outer_loop import loop_updates

    cfg = yaml.safe_load(Path("configs/m7_xl_workstation.yaml").read_text(encoding="utf-8"))
    assert cfg["world_model_config"] == "configs/sizes/dreamer_xl.yaml"
    assert cfg.get("world_model_ckpt") in (None, "")
    assert cfg.get("reset_actor") is True
    assert float(cfg["train"]["train_ratio"]) == 32.0
    assert cfg["train"]["imag_gradient"] == "reinforce"
    assert int(cfg["train"]["prefill_steps"]) == 2500
    assert "m7_xl_paper" not in cfg["train"]["checkpoint_dir"]
    assert "m7_paper_online" not in cfg["train"]["checkpoint_dir"]
    wm_u, ac_u = loop_updates(cfg["train"])
    assert wm_u == 1 and ac_u == 1


def test_loop_updates_train_ratio_and_explicit() -> None:
    from training.outer_loop import loop_updates

    n, m = loop_updates(
        {"collect_every": 16, "batch_size": 16, "seq_len": 32, "train_ratio": 32}
    )
    assert n == 1 and m == 1
    n, m = loop_updates(
        {"collect_every": 16, "batch_size": 16, "seq_len": 32, "train_ratio": 512}
    )
    assert n == 16 and m == 16
    n, m = loop_updates(
        {"collect_every": 16, "batch_size": 16, "seq_len": 32, "wm_updates": 1, "ac_updates": 1}
    )
    assert n == 1 and m == 1
