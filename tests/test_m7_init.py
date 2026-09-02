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
    assert buf.can_sample(8)


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


def test_m8_acfix_configs_enable_the_reference_recipe() -> None:
    """Both M8 configs must carry the finding-17 settings, not M7's defaults."""
    import yaml

    for path, wm_size in (
        ("configs/m8_s_acfix.yaml", "configs/sizes/dreamer_s.yaml"),
        ("configs/m8_xl_acfix.yaml", "configs/sizes/dreamer_xl.yaml"),
    ):
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        assert cfg["world_model_config"] == wm_size, path
        assert cfg.get("reset_actor") is True, path
        critic = cfg["critic"]
        assert critic["slow_target"] is True, path
        assert float(critic["slow_target_fraction"]) == 0.02, path
        assert int(critic["slow_target_update"]) == 1, path
        train = cfg["train"]
        assert train["imag_gradient"] == "reinforce", path
        assert int(train.get("pretrain_steps", train["pretrain_wm_steps"])) == 100, path
        assert float(train["entropy_scale"]) == 3.0e-4, path
        assert int(train["horizon"]) == 15, path
        # Never point an M8 run at an M7 checkpoint dir: those actors are dead.
        for key in ("checkpoint_dir", "log_dir", "results_dir"):
            assert "m7" not in train[key], (path, key)


def test_m8_s_acfix_seeds_from_the_m6_world_model() -> None:
    import yaml

    cfg = yaml.safe_load(Path("configs/m8_s_acfix.yaml").read_text(encoding="utf-8"))
    assert cfg["world_model_ckpt"] == "checkpoints/m6_baseline/ckpt_latest.pt"
    assert cfg.get("actor_critic_ckpt") in (None, "")
    assert cfg.get("seed_replay") in (None, "")


def test_overlay_wm_train_copies_paper_kl_scales() -> None:
    from train_agent import overlay_wm_train

    out = overlay_wm_train(
        {"dyn_scale": 1.0, "rep_scale": 0.5, "lr": 1e-4},
        {"dyn_scale": 0.5, "rep_scale": 0.1},
    )
    assert out["dyn_scale"] == 0.5
    assert out["rep_scale"] == 0.1
    assert out["lr"] == 1e-4
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


def test_m9_xl_is_a_new_dir_not_m8_resume() -> None:
    import yaml

    cfg = yaml.safe_load(Path("configs/m9_xl.yaml").read_text(encoding="utf-8"))
    train = cfg["train"]
    assert cfg["world_model_config"] == "configs/sizes/dreamer_xl.yaml"
    assert cfg.get("reset_actor") is True
    assert cfg.get("world_model_ckpt") in (None, "")
    assert train["imag_gradient"] == "reinforce"
    assert float(train["train_ratio"]) == 32.0
    for key in ("checkpoint_dir", "log_dir", "results_dir", "replay_out"):
        path = str(train[key]).replace("\\", "/")
        assert "m9_xl" in path, (key, path)
        assert "m8_xl_acfix" not in path, (key, path)
        assert "m7" not in path, (key, path)


def test_m10_continues_m9_at_ratio_128() -> None:
    import yaml

    from training.outer_loop import loop_updates

    cfg = yaml.safe_load(Path("configs/m10_xl_r128.yaml").read_text(encoding="utf-8"))
    train = cfg["train"]
    assert cfg.get("reset_actor") is False
    assert cfg["seed_joint_ckpt"] == "checkpoints/m9_xl/ckpt_step_400000.pt"
    assert cfg["seed_replay"] == "data/m9_xl_replay.pt"
    assert float(train["train_ratio"]) == 128.0
    wm_u, ac_u = loop_updates(train)
    assert wm_u == 4 and ac_u == 4
    for key in ("checkpoint_dir", "log_dir", "results_dir", "replay_out"):
        path = str(train[key]).replace("\\", "/")
        assert "m10_xl_r128" in path, (key, path)
        assert "m8_xl_acfix" not in path, (key, path)


def test_m11_from_scratch_ratio_128() -> None:
    import yaml

    from training.outer_loop import loop_updates

    cfg = yaml.safe_load(Path("configs/m11_xl_r128.yaml").read_text(encoding="utf-8"))
    train = cfg["train"]
    assert cfg.get("reset_actor") is True
    assert not cfg.get("seed_joint_ckpt")
    assert cfg.get("seed_replay") in (None, "")
    assert cfg.get("world_model_ckpt") in (None, "")
    assert cfg.get("actor_critic_ckpt") in (None, "")
    assert float(train["train_ratio"]) == 128.0
    wm_u, ac_u = loop_updates(train)
    assert wm_u == 4 and ac_u == 4
    for key in ("checkpoint_dir", "log_dir", "results_dir", "replay_out"):
        path = str(train[key]).replace("\\", "/")
        assert "m11_xl_r128" in path, (key, path)
        assert "m10_xl" not in path, (key, path)
        assert "m9_xl" not in path, (key, path)
        assert "m8_xl_acfix" not in path, (key, path)

def test_m12_from_scratch_ratio_512() -> None:
    import yaml

    from training.outer_loop import loop_updates

    cfg = yaml.safe_load(Path("configs/m12_xl_r512.yaml").read_text(encoding="utf-8"))
    train = cfg["train"]
    assert cfg.get("reset_actor") is True
    assert not cfg.get("seed_joint_ckpt")
    assert cfg.get("seed_replay") in (None, "")
    assert cfg.get("world_model_ckpt") in (None, "")
    assert cfg.get("actor_critic_ckpt") in (None, "")
    assert float(train["train_ratio"]) == 512.0
    wm_u, ac_u = loop_updates(train)
    assert wm_u == 16 and ac_u == 16
    assert int(train["eval_every"]) == 25000
    for key in ("checkpoint_dir", "log_dir", "results_dir", "replay_out"):
        path = str(train[key]).replace("\\", "/")
        assert "m12_xl_r512" in path, (key, path)
        assert "m7_" not in path, (key, path)
        assert "m9_xl" not in path, (key, path)
        assert "m10_xl" not in path, (key, path)
        assert "m11_xl" not in path, (key, path)


def test_m13_from_scratch_ratio_512_blocks2() -> None:
    import yaml

    from training.outer_loop import loop_updates

    cfg = yaml.safe_load(Path("configs/m13_xl_r512_b2.yaml").read_text(encoding="utf-8"))
    train = cfg["train"]
    size = yaml.safe_load(
        Path(cfg["world_model_config"]).read_text(encoding="utf-8")
    )
    assert int(size["encoder"]["blocks"]) == 2
    assert int(size["decoder"]["blocks"]) == 2
    assert int(size["rssm"]["deter_dim"]) == 2560
    assert cfg.get("reset_actor") is True
    assert not cfg.get("seed_joint_ckpt")
    assert cfg.get("seed_replay") in (None, "")
    assert cfg.get("world_model_ckpt") in (None, "")
    assert cfg.get("actor_critic_ckpt") in (None, "")
    assert float(train["train_ratio"]) == 512.0
    wm_u, ac_u = loop_updates(train)
    assert wm_u == 16 and ac_u == 16
    assert int(train["batch_size"]) == 16
    assert int(train["seq_len"]) == 32
    for key in ("checkpoint_dir", "log_dir", "results_dir", "replay_out"):
        path = str(train[key]).replace("\\", "/")
        assert "m13_xl_r512_b2" in path, (key, path)
        assert "m12_xl" not in path, (key, path)
        assert "m11_xl" not in path, (key, path)
        assert "m10_xl" not in path, (key, path)
        assert "m9_xl" not in path, (key, path)


def test_m14_from_scratch_ratio_32_blocks2() -> None:
    import yaml

    from training.outer_loop import loop_updates

    cfg = yaml.safe_load(Path("configs/m14_xl_r32_b2.yaml").read_text(encoding="utf-8"))
    train = cfg["train"]
    size = yaml.safe_load(
        Path(cfg["world_model_config"]).read_text(encoding="utf-8")
    )
    assert int(size["encoder"]["blocks"]) == 2
    assert int(size["decoder"]["blocks"]) == 2
    assert cfg.get("reset_actor") is True
    assert not cfg.get("seed_joint_ckpt")
    assert float(train["train_ratio"]) == 32.0
    wm_u, ac_u = loop_updates(train)
    assert wm_u == 1 and ac_u == 1
    for key in ("checkpoint_dir", "log_dir", "results_dir", "replay_out"):
        path = str(train[key]).replace("\\", "/")
        assert "m14_xl_r32_b2" in path, (key, path)
        assert "m13_xl" not in path, (key, path)
        assert "m12_xl" not in path, (key, path)
        assert "m9_xl" not in path, (key, path)


def test_m15_from_scratch_ratio_512_blocks2() -> None:
    import yaml

    from training.outer_loop import loop_updates

    cfg = yaml.safe_load(Path("configs/m15_xl_r512_b2.yaml").read_text(encoding="utf-8"))
    train = cfg["train"]
    size = yaml.safe_load(
        Path(cfg["world_model_config"]).read_text(encoding="utf-8")
    )
    assert int(size["encoder"]["blocks"]) == 2
    assert int(size["decoder"]["blocks"]) == 2
    assert cfg.get("reset_actor") is True
    assert not cfg.get("seed_joint_ckpt")
    assert float(train["train_ratio"]) == 512.0
    wm_u, ac_u = loop_updates(train)
    assert wm_u == 16 and ac_u == 16
    for key in ("checkpoint_dir", "log_dir", "results_dir", "replay_out"):
        path = str(train[key]).replace("\\", "/")
        assert "m15_xl_r512_b2" in path, (key, path)
        assert "m14_xl" not in path, (key, path)
        assert "m13_xl" not in path, (key, path)
        assert "m12_xl" not in path, (key, path)


def test_notebook_10_binds_m15_not_m14() -> None:
    """Pin notebook 10 to the paper 512 loop, not M14 ratio 32."""
    text = Path("notebooks/10_train_paper_online.ipynb").read_text(encoding="utf-8")
    assert "configs/m15_xl_r512_b2.yaml" in text
    assert 'CONFIG = Path(\\"configs/m14_xl_r32_b2.yaml\\")' not in text
    assert 'CONFIG = Path(\\"configs/m13_xl_r512_b2.yaml\\")' not in text
    assert "Need 512 and 16/16" in text

