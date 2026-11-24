"""Read the room field head on a correct teacher episode.

The logged eval recall was taken on whatever pose the env happened to be in.
This collects one in-view and one out-of-view demonstration after reset, then
scores the mode of z_posterior and a mode rollout of z_prior. Not a training run.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from envs.room_env import RoomEnv
from envs.room_teacher import RoomTeacher
from models.heads import rssm_features
from models.rssm import one_hot_action
from training.device import configure_runtime, get_device, require_free_vram

ROOT = Path(__file__).resolve().parents[1]
SIZE = ROOT / "configs" / "sizes" / "dreamer_s_room.yaml"
OUT = ROOT / "results" / "room_probe" / "summary.json"
NAMES = (
    "room_in_anchored",
    "room_in_uniform",
    "room_out_anchored",
    "room_out_uniform",
)
STATION = 3
PRIOR_AT = (1, 5, 15)


def build_model(device: torch.device):
    spec = importlib.util.spec_from_file_location(
        "train_world_model", ROOT / "scripts" / "train_world_model.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    cfg = yaml.safe_load(SIZE.read_text(encoding="utf-8"))
    model = module.build_model(cfg).to(device).eval()
    return model


def collect(station_in_view: bool) -> dict[str, np.ndarray]:
    env = RoomEnv(max_steps=160, require_station_in_view=station_in_view)
    teacher = RoomTeacher(station_in_view=station_in_view, hold_steps=48)
    obs, info = env.reset()
    teacher.reset(env)
    frames, actions, mats = [], [], []
    while True:
        action = teacher.act(env)
        frames.append(np.asarray(obs, dtype=np.uint8))
        mats.append(np.asarray(info["local_materials"], dtype=np.uint8))
        actions.append(action)
        obs, _reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    return {
        "obs": np.stack(frames),
        "act": np.asarray(actions, dtype=np.int64),
        "mat": np.stack(mats),
    }


def mode_stoch(logits: torch.Tensor) -> torch.Tensor:
    index = logits.argmax(dim=-1)
    return torch.nn.functional.one_hot(index, logits.shape[-1]).to(dtype=logits.dtype)


@torch.no_grad()
def posterior(model, episode: dict[str, np.ndarray], device: torch.device):
    obs = torch.from_numpy(episode["obs"]).to(device)
    actions = torch.from_numpy(episode["act"]).to(device)
    time = int(obs.shape[0])
    embed = model.encode(obs.unsqueeze(0))[0]
    act = one_hot_action(actions.view(1, time), model.rssm.action_dim)[0]
    state = model.rssm.initial(1, device=device)
    prev = torch.zeros(1, model.rssm.action_dim, device=device)
    pred = []
    prob_station = []
    h_steps = []
    z_steps = []
    true = torch.from_numpy(episode["mat"]).to(device)
    for t in range(time):
        first = torch.ones(1, device=device) if t == 0 else None
        state, _z_prior, _prior_logits, posterior_logits = model.rssm.obs_step(
            state, prev, embed[t : t + 1], first
        )
        z_mode = mode_stoch(posterior_logits)
        feat = rssm_features(state.h, z_mode)
        mat_logits, _obj = model.predict_local_map(feat)
        prob = torch.softmax(mat_logits[0].float(), dim=-1)
        pred.append(mat_logits[0].argmax(dim=-1).cpu().numpy())
        prob_station.append(prob[:, STATION].cpu().numpy())
        h_steps.append(state.h[0])
        z_steps.append(z_mode[0])
        prev = act[t : t + 1]
    return np.stack(pred), np.stack(prob_station), h_steps, z_steps, act


def counts(pred: np.ndarray, true: np.ndarray, prob_station: np.ndarray) -> dict:
    station = true == STATION
    other = ~station
    n_station = int(station.sum())
    return {
        "cells": int(true.size),
        "accuracy": float((pred == true).mean()),
        "station_cells": n_station,
        "station_recall": float((pred[station] == STATION).mean()) if n_station else None,
        "station_prob": float(prob_station[station].mean()) if n_station else None,
        "false_station": float((pred[other] == STATION).mean()) if int(other.sum()) else None,
    }


@torch.no_grad()
def prior_at_hold(model, episode, h_steps, z_steps, act, device) -> dict:
    """Roll the mode from the first hold frame. The pose does not change for 48 steps."""
    hold = int(np.argmax(episode["act"] == 0))
    # The walk is forwards, then no-ops. First no-op is the hold.
    if episode["act"][hold] != 0:
        return {}
    horizon = 15
    if hold + horizon >= len(episode["act"]):
        return {}
    h = h_steps[hold].unsqueeze(0)
    z = z_steps[hold].unsqueeze(0)
    future = episode["mat"][hold + 1 : hold + 1 + horizon]
    pred_rows = []
    for t in range(horizon):
        h, _sample, prior_logits = model.rssm.img_step(h, z, act[hold + t : hold + t + 1])
        z = mode_stoch(prior_logits)
        mat_logits, _obj = model.predict_local_map(rssm_features(h, z))
        pred_rows.append(mat_logits[0].argmax(dim=-1).cpu().numpy())
    pred = np.stack(pred_rows)
    out = {}
    for k in PRIOR_AT:
        true = future[k - 1]
        got = pred[k - 1]
        station = true == STATION
        other = ~station
        out[str(k)] = {
            "accuracy": float((got == true).mean()),
            "station_cells": int(station.sum()),
            "station_recall": float((got[station] == STATION).mean()) if int(station.sum()) else None,
            "false_station": float((got[other] == STATION).mean()) if int(other.sum()) else None,
        }
    return out


def main() -> None:
    device = get_device()
    configure_runtime(device)
    if device.type == "cuda":
        print(f"free {require_free_vram(device, min_gib=4.0):.2f} GiB", flush=True)
    episodes = {
        "in_view": collect(True),
        "out_of_view": collect(False),
    }
    for key, episode in episodes.items():
        station = int((episode["mat"] == STATION).sum())
        print(f"{key} length {len(episode['act'])} return-action {int(episode['act'][-1])} station_cells {station}", flush=True)
    summary = {}
    for name in NAMES:
        ckpt = ROOT / "checkpoints" / name / "ckpt_final.pt"
        model = build_model(device)
        payload = torch.load(ckpt, weights_only=False, map_location="cpu")
        model.load_state_dict(payload["world_model"], strict=True)
        del payload
        if device.type == "cuda":
            torch.cuda.empty_cache()
        summary[name] = {}
        for key, episode in episodes.items():
            pred, prob, h_steps, z_steps, act = posterior(model, episode, device)
            row = counts(pred, episode["mat"], prob)
            row["prior"] = prior_at_hold(model, episode, h_steps, z_steps, act, device)
            summary[name][key] = row
            print(
                f"{name} {key} acc {row['accuracy']:.3f} "
                f"station {row['station_recall']} prob {row['station_prob']} "
                f"false {row['false_station']}",
                flush=True,
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
