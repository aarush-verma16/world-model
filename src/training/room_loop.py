"""Train the room. Four configs, one function. Not the Crafter loop.

A run is one switch: the station is in the crop or behind the agent, and
the cloning window is anchored on `use` or is an ordinary slice. The world
model is the small room size, never a Crafter checkpoint.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from agents.actor_critic import Actor, Critic
from envs.room_env import USE, RoomEnv
from envs.room_teacher import RoomTeacher
from models.heads import rssm_features
from training.ac_step import actor_critic_step
from training.collect import rssm_policy_step
from training.device import (
    configure_runtime,
    get_device,
    make_grad_scaler,
    parse_amp,
    require_free_vram,
)
from training.imagine import unfreeze_world_model
from training.outer_loop import loop_updates
from training.returns import PercentileReturnNorm
from training.room_replay import RoomReplay
from training.wm_step import world_model_step

ROOT = Path(__file__).resolve().parents[2]


def load_run(path: Path) -> dict[str, Any]:
    """Merge the run yaml over its size yaml. The run's `train` keys win."""
    run = yaml.safe_load(path.read_text(encoding="utf-8"))
    size_path = Path(run["world_model_config"])
    if not size_path.is_file():
        size_path = ROOT / size_path
    size = yaml.safe_load(size_path.read_text(encoding="utf-8"))
    if int(size["env"]["action_dim"]) != 5:
        raise RuntimeError(f"{size_path} action_dim must be 5 for the room")
    train = dict(size.get("train") or {})
    train.update(run.get("train") or {})
    run["train"] = train
    run["_size"] = size
    run["_path"] = str(path)
    _refuse_crafter_weights(run)
    return run


def _refuse_crafter_weights(run: dict[str, Any]) -> None:
    for key in ("world_model_ckpt", "actor_critic_ckpt"):
        raw = str(run.get(key) or "")
        if not raw:
            continue
        lowered = raw.replace("\\", "/").lower()
        if any(tag in lowered for tag in ("m20", "m21", "m22", "m23", "m24", "crafter")):
            raise RuntimeError(f"{key} points at a Crafter checkpoint: {raw}")


def _build(run: dict[str, Any], device: torch.device):
    import importlib.util

    script = ROOT / "scripts" / "train_world_model.py"
    spec = importlib.util.spec_from_file_location("train_world_model", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.build_model(run["_size"]).to(device)
    model.actor_input_source = str(run.get("actor_input_source", "truth"))
    feat = int(model.feat_dim + model.actor_extra_dim)
    actor_cfg = run.get("actor") or {}
    critic_cfg = run.get("critic") or {}
    actor = Actor(
        feat,
        int(model.rssm.action_dim),
        hidden=int(actor_cfg.get("hidden", 256)),
        layers=int(actor_cfg.get("layers", 2)),
        unimix=float(actor_cfg.get("unimix", 0.01)),
    ).to(device)
    critic = Critic(
        feat,
        hidden=int(critic_cfg.get("hidden", 256)),
        layers=int(critic_cfg.get("layers", 2)),
        num_bins=int(critic_cfg.get("num_bins", 255)),
        low=float(critic_cfg.get("low", -20.0)),
        high=float(critic_cfg.get("high", 20.0)),
    ).to(device)
    lr = float(run["train"]["lr"])
    wm_optim = torch.optim.Adam(model.parameters(), lr=lr)
    ac_optim = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=lr
    )
    return model, actor, critic, wm_optim, ac_optim


def _rollout(
    env: RoomEnv,
    policy,
    *,
    obs: np.ndarray | None = None,
    info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """`policy(obs, info) -> action int`. Stores the frame the action was taken on.

    Pass `obs` and `info` to continue an episode that was already reset.
    The teacher has to plan after that reset, so it cannot reset again here.
    """
    if obs is None or info is None:
        obs, info = env.reset()
    frames: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    cont: list[float] = []
    mats: list[np.ndarray] = []
    objs: list[np.ndarray] = []
    in_view: list[bool] = []
    while True:
        action = int(policy(obs, info))
        frames.append(np.asarray(obs, dtype=np.uint8))
        mats.append(np.asarray(info["local_materials"], dtype=np.uint8))
        objs.append(np.asarray(info["local_objects"], dtype=np.uint8))
        in_view.append(bool(info["station_in_view"]))
        obs, reward, terminated, truncated, info = env.step(action)
        actions.append(action)
        rewards.append(float(reward))
        cont.append(0.0 if terminated else 1.0)
        if terminated or truncated:
            return {
                "obs": frames,
                "actions": actions,
                "rewards": rewards,
                "cont": cont,
                "local_mat": mats,
                "local_obj": objs,
                "in_view": in_view,
                "return": float(sum(rewards)),
                "used": int(info.get("used", 0)),
            }


def _teacher_episode(env: RoomEnv, teacher: RoomTeacher) -> dict[str, Any]:
    obs, info = env.reset()
    teacher.reset(env)
    return _rollout(env, lambda _obs, _info: teacher.act(env), obs=obs, info=info)


class _Actor:
    """RSSM policy across steps of one episode. Rebuilt every episode."""

    def __init__(self, model, actor, device, amp) -> None:
        self.model = model
        self.actor = actor
        self.device = device
        self.amp = amp
        self.state = model.rssm.initial(1, device=device)
        self.prev = torch.zeros(1, model.rssm.action_dim, device=device)

    def __call__(self, obs, info) -> int:
        self.model.eval()
        self.actor.eval()
        self.state, _oh, action, _ent = rssm_policy_step(
            self.model,
            self.actor,
            obs,
            self.state,
            self.prev,
            device=self.device,
            amp_dtype=self.amp,
            info=info,
        )
        self.prev = torch.zeros(1, self.model.rssm.action_dim, device=self.device)
        self.prev[0, action] = 1.0
        return action


def _store(buffer: RoomReplay, episode: dict[str, Any], *, teacher: bool) -> None:
    buffer.add_episode(
        episode["obs"],
        episode["actions"],
        episode["rewards"],
        episode["cont"],
        episode["local_mat"],
        episode["local_obj"],
        teacher=teacher,
    )


@torch.no_grad()
def _station_recall(model, episode: dict[str, Any], device: torch.device) -> float:
    """Mode of `z_posterior`: fraction of station cells the field head names."""
    model.eval()
    obs = torch.from_numpy(np.stack(episode["obs"])).to(device).unsqueeze(0)
    actions = torch.from_numpy(np.asarray(episode["actions"], dtype=np.int64)).to(device).view(1, -1)
    out = model(obs, actions)
    z_mode_index = out.rssm.posterior_logits.argmax(dim=-1)
    z_mode = torch.nn.functional.one_hot(z_mode_index, out.rssm.posterior_logits.shape[-1])
    z_mode = z_mode.to(dtype=out.rssm.h.dtype)
    feat = rssm_features(out.rssm.h, z_mode)
    mat_logits, _obj = model.predict_local_map(feat)
    pred = mat_logits[0].argmax(dim=-1).cpu().numpy()
    true = np.stack(episode["local_mat"])
    station = true == 3
    if int(station.sum()) == 0:
        return float("nan")
    return float((pred[station] == 3).mean())


def _eval_actor(env, model, actor, device, amp, n: int) -> dict[str, float]:
    returns = []
    for _ in range(n):
        policy = _Actor(model, actor, device, amp)
        episode = _rollout(env, policy)
        returns.append(episode["return"])
    return {
        "eval_success": float(np.mean(np.asarray(returns) >= 1.0)),
        "eval_return": float(np.mean(returns)),
        "eval_episodes": float(n),
    }


def _save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _checkpoint(path: Path, model, actor, critic, wm_optim, ac_optim, retnorm, env_steps: int) -> None:
    _save(
        path,
        {
            "env_steps": int(env_steps),
            "world_model": model.state_dict(),
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "wm_optim": wm_optim.state_dict(),
            "ac_optim": ac_optim.state_dict(),
            "retnorm": retnorm.state_dict(),
        },
    )


def train_room(config_path: Path, *, log=print) -> dict[str, Any]:
    """Run one room config to `train.env_steps`. Saves on interrupt."""
    run = load_run(Path(config_path))
    train = run["train"]
    name = str(run["name"])
    dirs = run["dirs"]
    results = Path(dirs["results"])
    ckpt_dir = Path(dirs["checkpoints"])
    results.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    configure_runtime(device)
    if device.type == "cuda":
        log(f"vram free {require_free_vram(device, min_gib=4.0):.2f} GiB")
    torch.manual_seed(int(run.get("seed", 0)))
    np.random.seed(int(run.get("seed", 0)))
    model, actor, critic, wm_optim, ac_optim = _build(run, device)
    amp = parse_amp(str(train.get("amp", "bf16")), device)
    scaler = make_grad_scaler(device, amp)
    retnorm = PercentileReturnNorm()
    env_steps = 0
    latest = ckpt_dir / "ckpt_latest.pt"
    if str(run.get("resume", "auto")).lower() == "auto" and latest.is_file():
        payload = torch.load(latest, weights_only=False, map_location="cpu")
        model.load_state_dict(payload["world_model"])
        actor.load_state_dict(payload["actor"])
        critic.load_state_dict(payload["critic"])
        wm_optim.load_state_dict(payload["wm_optim"])
        ac_optim.load_state_dict(payload["ac_optim"])
        retnorm.load_state_dict(payload["retnorm"])
        env_steps = int(payload["env_steps"])
        del payload
        if device.type == "cuda":
            torch.cuda.empty_cache()
        log(f"resumed {name} at env {env_steps}")
    in_view = bool(run["station_in_view"])
    env = RoomEnv(
        max_steps=int(train.get("max_episode_steps", 160)),
        require_station_in_view=in_view,
    )
    teacher = RoomTeacher(
        station_in_view=in_view,
        hold_steps=int(train.get("hold_steps", 48)),
    )
    buffer = RoomReplay(cells=16, craft_actions=(USE,), seed=int(run.get("seed", 0)))
    replay_path = Path(dirs["results"]) / "replay.pt"
    if env_steps > 0 and replay_path.is_file():
        buffer.load(replay_path)
        log(f"replay {len(buffer)} episodes")
    target = int(train["env_steps"])
    seq_len = int(train["seq_len"])
    batch_size = int(train["batch_size"])
    metrics_path = results / "train_metrics.json"
    episodes_path = results / "episodes.jsonl"
    history: list[dict] = []
    if metrics_path.is_file() and env_steps > 0:
        history = [row for row in json.loads(metrics_path.read_text(encoding="utf-8")) if int(row["env_steps"]) <= env_steps]
    if env_steps == 0:
        for _ in range(int(train.get("teacher_episodes", 8))):
            episode = _teacher_episode(env, teacher)
            _store(buffer, episode, teacher=True)
            _append_episode(episodes_path, episode, env_steps, teacher=True)
        log(f"seeded {len(buffer)} teacher episodes, in_view={in_view}")
    log(
        f"{name} station_in_view={in_view} craft_window={train.get('craft_window_fraction')} "
        f"env {env_steps}/{target}"
    )
    t0 = time.perf_counter()
    try:
        while env_steps < target:
            unfreeze_world_model(model)
            model.train()
            actor.train()
            gained = 0
            need = int(train["collect_every"])
            while gained < need:
                policy = _Actor(model, actor, device, amp)
                episode = _rollout(env, policy)
                _store(buffer, episode, teacher=False)
                _append_episode(episodes_path, episode, env_steps + gained + len(episode["actions"]), teacher=False)
                gained += len(episode["actions"])
            env_steps += gained
            if not buffer.can_sample(seq_len):
                continue
            wm_n, ac_n = loop_updates(train, env_steps=env_steps)
            wm_metrics: dict[str, float] = {}
            ac_metrics: dict[str, float] = {}
            for i in range(wm_n):
                batch = buffer.sample(
                    batch_size,
                    seq_len,
                    teacher_fraction=float(train.get("teacher_fraction", 0.5)),
                    craft_window_fraction=float(train.get("craft_window_fraction", 0.5)),
                )
                unfreeze_world_model(model)
                _loss, wm_metrics = world_model_step(
                    model,
                    wm_optim,
                    batch,
                    device=device,
                    train_cfg=train,
                    amp_dtype=amp,
                    scaler=scaler,
                    metrics=(i + 1 == wm_n),
                )
            for i in range(ac_n):
                batch = buffer.sample(
                    batch_size,
                    seq_len,
                    teacher_fraction=float(train.get("teacher_fraction", 0.5)),
                    craft_window_fraction=float(train.get("craft_window_fraction", 0.5)),
                )
                _ac, ac_metrics, _roll = actor_critic_step(
                    model,
                    actor,
                    critic,
                    ac_optim,
                    batch,
                    device=device,
                    retnorm=retnorm,
                    horizon=int(train.get("horizon", 8)),
                    start_mode="all",
                    entropy_scale=float(train.get("entropy_scale", 3.0e-4)),
                    imag_gradient="reinforce",
                    amp_dtype=amp,
                    scaler=scaler,
                    bc_scale=float(train.get("bc_scale", 1.0)),
                    craft_ids=(USE,),
                    metrics=(i + 1 == ac_n),
                )
            row = {
                "env_steps": int(env_steps),
                "sps": gained / max(time.perf_counter() - t0, 1e-6),
            }
            row.update({f"wm_{k}": v for k, v in wm_metrics.items()})
            row.update({f"ac_{k}" if not k.startswith("ac_") else k: v for k, v in ac_metrics.items()})
            t0 = time.perf_counter()
            if int(train.get("eval_every", 0)):
                every = int(train["eval_every"])
                if env_steps // every != (env_steps - gained) // every:
                    teacher_ep = _teacher_episode(env, teacher)
                    _store(buffer, teacher_ep, teacher=True)
                    row["field_station_recall"] = _station_recall(model, teacher_ep, device)
                    row.update(
                        _eval_actor(
                            env, model, actor, device, amp, int(train.get("eval_episodes", 8))
                        )
                    )
                    log(
                        f"env {env_steps} success {row['eval_success']:.2f} "
                        f"station_recall {row['field_station_recall']:.2f}"
                    )
            history.append(row)
            log_every = int(train.get("log_every", 256))
            if env_steps // log_every != (env_steps - gained) // log_every:
                metrics_path.write_text(json.dumps(history), encoding="utf-8")
                log(
                    f"env {env_steps}/{target} wm_local {row.get('wm_local', float('nan')):.4f} "
                    f"ac_H {row.get('ac_entropy', float('nan')):.3f}"
                )
            if int(train.get("checkpoint_every", 0)):
                every = int(train["checkpoint_every"])
                if env_steps // every != (env_steps - gained) // every:
                    _checkpoint(
                        ckpt_dir / f"ckpt_step_{env_steps}.pt",
                        model, actor, critic, wm_optim, ac_optim, retnorm, env_steps,
                    )
                    buffer.save(replay_path)
    except KeyboardInterrupt:
        log("interrupt")
        raise
    finally:
        _checkpoint(latest, model, actor, critic, wm_optim, ac_optim, retnorm, env_steps)
        _checkpoint(ckpt_dir / "ckpt_final.pt", model, actor, critic, wm_optim, ac_optim, retnorm, env_steps)
        buffer.save(replay_path)
        metrics_path.write_text(json.dumps(history), encoding="utf-8")
        log(f"saved {name} at env {env_steps}")
    return {"name": name, "env_steps": env_steps}


def _append_episode(path: Path, episode: dict[str, Any], env_steps: int, *, teacher: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "env_steps": int(env_steps),
        "length": len(episode["actions"]),
        "return": episode["return"],
        "used": episode["used"],
        "teacher": int(teacher),
        "in_view_at_use": int(episode["in_view"][-1]) if episode["actions"] else 0,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
