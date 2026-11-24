"""The four room configs sample `use` the way the yaml says, and one step runs."""

from pathlib import Path

import numpy as np
import torch

from envs.room_env import USE, RoomEnv
from envs.room_teacher import RoomTeacher
from models.world_model import WorldModel
from training.ac_step import actor_critic_step
from training.returns import PercentileReturnNorm
from training.room_loop import _refuse_crafter_weights, load_run
from training.room_replay import RoomReplay
from training.wm_step import world_model_step

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [
    "configs/room_in_anchored.yaml",
    "configs/room_in_uniform.yaml",
    "configs/room_out_anchored.yaml",
    "configs/room_out_uniform.yaml",
]


def test_repeated_teacher_episodes_all_score() -> None:
    from training.room_loop import _teacher_episode

    env = RoomEnv(require_station_in_view=True, max_steps=80)
    teacher = RoomTeacher(station_in_view=True, hold_steps=4)
    for _ in range(4):
        episode = _teacher_episode(env, teacher)
        assert episode["return"] == 1.0
        assert episode["actions"][-1] == USE
        assert len(episode["actions"]) < 80


def test_four_room_configs_are_the_two_switches() -> None:
    runs = [load_run(ROOT / name) for name in CONFIGS]
    cells = {
        (run["name"], bool(run["station_in_view"]), float(run["train"]["craft_window_fraction"]))
        for run in runs
    }
    assert cells == {
        ("room_in_anchored", True, 0.5),
        ("room_in_uniform", True, 0.0),
        ("room_out_anchored", False, 0.5),
        ("room_out_uniform", False, 0.0),
    }
    for run in runs:
        assert int(run["_size"]["env"]["action_dim"]) == 5
        assert int(run["_size"]["local_map"]["cells"]) == 16
        assert int(run["train"]["seq_len"]) == 32
        assert int(run["train"]["hold_steps"]) >= int(run["train"]["seq_len"])


def test_a_crafter_checkpoint_is_refused() -> None:
    try:
        _refuse_crafter_weights({"world_model_ckpt": "checkpoints/m20_local_map/ckpt_final.pt"})
    except RuntimeError as exc:
        assert "Crafter" in str(exc)
    else:
        raise AssertionError("m20 weights were accepted")


def test_anchored_window_ends_on_use() -> None:
    env = RoomEnv(require_station_in_view=True)
    teacher = RoomTeacher(station_in_view=True, hold_steps=8)
    teacher.reset(env)
    obs, info = env.reset()
    teacher.reset(env)
    frames, actions, rewards, cont, mats, objs = [], [], [], [], [], []
    for _ in range(env.max_steps):
        action = teacher.act(env)
        frames.append(np.asarray(obs, dtype=np.uint8))
        mats.append(np.asarray(info["local_materials"], dtype=np.uint8))
        objs.append(np.asarray(info["local_objects"], dtype=np.uint8))
        obs, reward, terminated, truncated, info = env.step(action)
        actions.append(action)
        rewards.append(float(reward))
        cont.append(0.0 if terminated else 1.0)
        if terminated or truncated:
            break
    buffer = RoomReplay(cells=16, craft_actions=(USE,), seed=0)
    buffer.add_episode(frames, actions, rewards, cont, mats, objs, teacher=True)
    anchored = buffer.sample(4, 8, teacher_fraction=1.0, craft_window_fraction=1.0)
    assert int(anchored["actions"][0, -1]) == USE
    assert bool((anchored["actions"][:, -1] == USE).all())
    uniform = buffer.sample(8, 8, teacher_fraction=1.0, craft_window_fraction=0.0)
    assert uniform["local_mat"].shape[-1] == 16


def test_room_size_builds_five_actions_and_sixteen_cells() -> None:
    from training.room_loop import _build

    run = load_run(ROOT / "configs/room_in_anchored.yaml")
    model, actor, _critic, _wm, _ac = _build(run, torch.device("cpu"))
    assert int(model.rssm.action_dim) == 5
    assert model.local_map_head is not None
    assert int(model.local_map_head.cells) == 16
    assert int(actor.feat_dim) == int(model.feat_dim + model.actor_extra_dim)
    obs = torch.zeros(1, 64, 64, 3, dtype=torch.uint8)
    embed = model.encode(obs)
    assert embed.shape[0] == 1


def test_one_room_step_on_a_tiny_model() -> None:
    env = RoomEnv(require_station_in_view=True)
    teacher = RoomTeacher(station_in_view=True, hold_steps=8)
    obs, info = env.reset()
    teacher.reset(env)
    frames, actions, rewards, cont, mats, objs = [], [], [], [], [], []
    for _ in range(env.max_steps):
        action = teacher.act(env)
        frames.append(np.asarray(obs, dtype=np.uint8))
        mats.append(np.asarray(info["local_materials"], dtype=np.uint8))
        objs.append(np.asarray(info["local_objects"], dtype=np.uint8))
        obs, reward, terminated, truncated, info = env.step(action)
        actions.append(action)
        rewards.append(float(reward))
        cont.append(0.0 if terminated else 1.0)
        if terminated or truncated:
            break
    buffer = RoomReplay(cells=16, craft_actions=(USE,), seed=1)
    buffer.add_episode(frames, actions, rewards, cont, mats, objs, teacher=True)
    batch = buffer.sample(2, 8, teacher_fraction=1.0, craft_window_fraction=1.0)
    device = torch.device("cpu")
    model = WorldModel.from_config_dims(
        embed_dim=32,
        encoder_channels=(4, 8, 8, 8),
        action_dim=5,
        deter_dim=32,
        stoch=4,
        classes=4,
        hidden=32,
        decoder_channels=(8, 8, 8, 4),
        head_hidden=32,
        head_layers=1,
        encoder_blocks=1,
        decoder_blocks=1,
        reward_num_bins=15,
        local_map_cells=16,
        local_map_material_classes=4,
        local_map_object_classes=2,
    )
    model.actor_input_source = "truth"
    from agents.actor_critic import Actor, Critic

    feat = model.feat_dim + model.actor_extra_dim
    actor = Actor(feat, 5, hidden=32, layers=1)
    critic = Critic(feat, hidden=32, layers=1, num_bins=15)
    train = {
        "dyn_scale": 1.0,
        "rep_scale": 0.5,
        "free_nats": 1.0,
        "free_nats_dyn": 0.0,
        "recon_scale": 1.0,
        "reward_scale": 1.0,
        "continue_scale": 1.0,
        "kl_scale": 1.0,
        "local_scale": 1.0,
    }
    wm_optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    ac_optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-4)
    from training.device import make_grad_scaler

    scaler = make_grad_scaler(device, None)
    _loss, metrics = world_model_step(
        model, wm_optim, batch, device=device, train_cfg=train, amp_dtype=None, scaler=scaler
    )
    assert "local" in metrics
    _ac, ac_metrics, _roll = actor_critic_step(
        model,
        actor,
        critic,
        ac_optim,
        batch,
        device=device,
        retnorm=PercentileReturnNorm(),
        horizon=3,
        start_mode="all",
        amp_dtype=None,
        scaler=scaler,
        bc_scale=1.0,
        craft_ids=(USE,),
    )
    assert "entropy" in ac_metrics
    from training.collect import rssm_policy_step

    obs, info = env.reset()
    state = model.rssm.initial(1, device=device)
    prev = torch.zeros(1, 5)
    _state, _oh, action, _ent = rssm_policy_step(
        model, actor, obs, state, prev, device=device, amp_dtype=None, info=info
    )
    assert 0 <= action < 5
