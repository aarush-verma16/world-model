"""Read the local field out of the saved world models.

Collects one fixed set of teacher lives, then asks each checkpoint whether
`z_posterior` names the 9x7 tiles and whether a mode rollout of `z_prior`
still names them. Not a training run.

Usage:
    conda run -n worldmodel python scripts/probe_field_head.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from envs.crafter_env import CrafterEnv  # noqa: E402
from models.heads import rssm_features  # noqa: E402
from models.rssm import one_hot_action  # noqa: E402
from training.crafter_rules import MATERIAL_NAMES  # noqa: E402
from training.crafter_teacher import CrafterTeacherV2  # noqa: E402
from training.device import configure_runtime, get_device, require_free_vram  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "train_world_model", ROOT / "scripts" / "train_world_model.py"
)
_train_world_model = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_train_world_model)
build_model = _train_world_model.build_model
save_video_pred_strip = _train_world_model.save_video_pred_strip

OUT = ROOT / "results" / "field_probe"
FIG = ROOT / "research paper" / "figures"
EPISODES = OUT / "episodes.npz"
SIZE_CFG = ROOT / "configs" / "sizes" / "dreamer_xl_paper_local.yaml"
CHECKPOINTS = (
    ("m22_field_craft", ROOT / "checkpoints" / "m22_field_craft" / "ckpt_step_100000.pt"),
    ("m23_uniform_window", ROOT / "checkpoints" / "m23_uniform_window" / "ckpt_step_100000.pt"),
    ("m24_walk_home", ROOT / "checkpoints" / "m24_walk_home" / "ckpt_step_100000.pt"),
)
HORIZON = 15
PRIOR_AT = (1, 5, 15)
CONTEXT = 8
STRIP_LEN = 24
N_EACH = 4
MAX_STEPS = 400


def material_code(name: str) -> int:
    return 1 + MATERIAL_NAMES.index(name)


def collect_episodes() -> dict[str, np.ndarray]:
    """Four in-crop station lives and four walk-home lives, one process."""
    if EPISODES.is_file():
        print(f"reusing {EPISODES}", flush=True)
        return dict(np.load(EPISODES))
    OUT.mkdir(parents=True, exist_ok=True)
    env = CrafterEnv(reward=True)
    obs_list: list[np.ndarray] = []
    act_list: list[np.ndarray] = []
    mat_list: list[np.ndarray] = []
    obj_list: list[np.ndarray] = []
    tag_list: list[np.ndarray] = []
    for i, field in enumerate((True, True, True, True, False, False, False, False)):
        teacher = CrafterTeacherV2(seed=0, craft_now=True, field_craft=field)
        teacher.reset()
        obs, info = env.reset(seed=2000 + i)
        frames = [np.asarray(obs, dtype=np.uint8)]
        mats = [np.asarray(info["local_materials"], dtype=np.uint8)]
        objs = [np.asarray(info["local_objects"], dtype=np.uint8)]
        actions: list[int] = []
        for _ in range(MAX_STEPS):
            action = int(teacher.act(env))
            obs, _reward, terminated, truncated, info = env.step(action)
            actions.append(action)
            frames.append(np.asarray(obs, dtype=np.uint8))
            mats.append(np.asarray(info["local_materials"], dtype=np.uint8))
            objs.append(np.asarray(info["local_objects"], dtype=np.uint8))
            if terminated or truncated:
                break
        # The action at t produces the frame at t+1, so drop the last frame
        # from the observation sequence and keep actions aligned with the
        # frames they were taken from.
        obs_list.append(np.stack(frames[:-1], axis=0))
        act_list.append(np.asarray(actions, dtype=np.int64))
        mat_list.append(np.stack(mats[:-1], axis=0))
        obj_list.append(np.stack(objs[:-1], axis=0))
        tag_list.append(np.asarray([1 if field else 0], dtype=np.int64))
        print(
            f"  collected {'field' if field else 'home'} ep {i} "
            f"len={len(actions)} table_cells={int((mat_list[-1] == material_code('table')).sum())} "
            f"furnace_cells={int((mat_list[-1] == material_code('furnace')).sum())}",
            flush=True,
        )
    env.close()
    packed = {
        "n": np.asarray([len(obs_list)], dtype=np.int64),
        **{f"obs_{i}": obs_list[i] for i in range(len(obs_list))},
        **{f"act_{i}": act_list[i] for i in range(len(act_list))},
        **{f"mat_{i}": mat_list[i] for i in range(len(mat_list))},
        **{f"obj_{i}": obj_list[i] for i in range(len(obj_list))},
        **{f"field_{i}": tag_list[i] for i in range(len(tag_list))},
    }
    np.savez_compressed(EPISODES, **packed)
    return packed


def episodes_from(packed: dict[str, np.ndarray]) -> list[dict[str, np.ndarray]]:
    n = int(packed["n"][0])
    return [
        {
            "obs": packed[f"obs_{i}"],
            "act": packed[f"act_{i}"],
            "mat": packed[f"mat_{i}"],
            "obj": packed[f"obj_{i}"],
            "field": packed[f"field_{i}"],
        }
        for i in range(n)
    ]


def mode_stoch(logits: torch.Tensor) -> torch.Tensor:
    """`logits` `[B, stoch, classes]` -> one-hot mode, same shape."""
    index = logits.argmax(dim=-1)
    return torch.nn.functional.one_hot(index, logits.shape[-1]).to(dtype=logits.dtype)


class Tally:
    def __init__(self, n_mat: int, n_obj: int) -> None:
        self.mat_correct = 0
        self.mat_total = 0
        self.obj_correct = 0
        self.obj_total = 0
        self.mat_hit = np.zeros(n_mat, dtype=np.int64)
        self.mat_count = np.zeros(n_mat, dtype=np.int64)
        self.mat_prob = np.zeros(n_mat, dtype=np.float64)

    def add_mat(
        self,
        pred_mat: np.ndarray,
        true_mat: np.ndarray,
        prob_true: np.ndarray | None = None,
    ) -> None:
        self.mat_correct += int((pred_mat == true_mat).sum())
        self.mat_total += int(true_mat.size)
        for code in range(self.mat_count.shape[0]):
            mask = true_mat == code
            self.mat_count[code] += int(mask.sum())
            self.mat_hit[code] += int((pred_mat[mask] == code).sum())
            if prob_true is not None:
                self.mat_prob[code] += float(prob_true[mask].sum())

    def add(
        self,
        pred_mat: np.ndarray,
        true_mat: np.ndarray,
        pred_obj: np.ndarray,
        true_obj: np.ndarray,
        prob_true: np.ndarray | None = None,
    ) -> None:
        self.add_mat(pred_mat, true_mat, prob_true)
        self.obj_correct += int((pred_obj == true_obj).sum())
        self.obj_total += int(true_obj.size)

    def as_dict(self) -> dict[str, float | int]:
        out: dict[str, float | int] = {
            "material_accuracy": self.mat_correct / max(self.mat_total, 1),
            "object_accuracy": (self.obj_correct / self.obj_total) if self.obj_total else float("nan"),
            "cells": int(self.mat_total),
        }
        for name in ("grass", "stone", "path", "coal", "iron", "table", "furnace"):
            code = material_code(name)
            count = int(self.mat_count[code])
            out[f"{name}_cells"] = count
            out[f"{name}_recall"] = (int(self.mat_hit[code]) / count) if count else None
            out[f"{name}_prob"] = (float(self.mat_prob[code]) / count) if count else None
        return out


@torch.no_grad()
def scan_episode(
    model: torch.nn.Module,
    episode: dict[str, np.ndarray],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
    """Return posterior material/object modes and the h / posterior-mode at each step.

    `h` and `z_posterior` are `[T, ...]`. `h` does not see the current frame.
    The posterior mode does.
    """
    obs = torch.from_numpy(episode["obs"]).to(device)
    actions = torch.from_numpy(episode["act"]).to(device)
    time = int(obs.shape[0])
    embeds = []
    for start in range(0, time, 32):
        chunk = obs[start : start + 32].unsqueeze(0)
        embeds.append(model.encode(chunk)[0])
    embed = torch.cat(embeds, dim=0)
    act = one_hot_action(actions.view(1, time), model.rssm.action_dim)[0]
    state = model.rssm.initial(1, device=device)
    prev = torch.zeros(1, model.rssm.action_dim, device=device)
    h_steps: list[torch.Tensor] = []
    z_steps: list[torch.Tensor] = []
    mat_pred: list[np.ndarray] = []
    obj_pred: list[np.ndarray] = []
    prob_rows: list[np.ndarray] = []
    true_mat = torch.from_numpy(episode["mat"]).to(device)
    for t in range(time):
        first = torch.ones(1, device=device) if t == 0 else None
        state, _z_prior, _prior_logits, posterior_logits = model.rssm.obs_step(
            state, prev, embed[t : t + 1], first
        )
        z_mode = mode_stoch(posterior_logits)
        feat = rssm_features(state.h, z_mode)
        mat_logits, obj_logits = model.predict_local_map(feat)
        mat_pred.append(mat_logits[0].argmax(dim=-1).cpu().numpy())
        obj_pred.append(obj_logits[0].argmax(dim=-1).cpu().numpy())
        prob = torch.softmax(mat_logits[0].float(), dim=-1)
        gathered = prob.gather(1, true_mat[t].long().unsqueeze(1)).squeeze(1)
        prob_rows.append(gathered.cpu().numpy())
        h_steps.append(state.h[0])
        z_steps.append(z_mode[0])
        prev = act[t : t + 1]
    return (
        np.stack(mat_pred, axis=0),
        np.stack(obj_pred, axis=0),
        np.stack(prob_rows, axis=0),
        torch.stack(h_steps, dim=0),
        torch.stack(z_steps, dim=0),
    )


@torch.no_grad()
def prior_field(
    model: torch.nn.Module,
    h0: torch.Tensor,
    z0: torch.Tensor,
    actions: torch.Tensor,
) -> np.ndarray:
    """Mode rollout. `actions` `[H, action_dim]`. Returns material modes `[H, 63]`."""
    h = h0.unsqueeze(0)
    z = z0.unsqueeze(0)
    mats: list[np.ndarray] = []
    for t in range(int(actions.shape[0])):
        h, _z_sample, prior_logits = model.rssm.img_step(h, z, actions[t : t + 1])
        z = mode_stoch(prior_logits)
        feat = rssm_features(h, z)
        mat_logits, _obj_logits = model.predict_local_map(feat)
        mats.append(mat_logits[0].argmax(dim=-1).cpu().numpy())
    return np.stack(mats, axis=0)


def score_model(name: str, ckpt: Path, episodes: list[dict[str, np.ndarray]], device: torch.device) -> dict:
    print(f"\n=== {name} ===", flush=True)
    with open(SIZE_CFG, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    model = build_model(cfg).to(device).eval()
    payload = torch.load(ckpt, weights_only=False, map_location="cpu")
    model.load_state_dict(payload["world_model"], strict=True)
    payload.clear()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    n_mat = int(model.local_map_head.n_material_classes)
    n_obj = int(model.local_map_head.n_object_classes)
    posterior = Tally(n_mat, n_obj)
    priors = {k: Tally(n_mat, n_obj) for k in PRIOR_AT}
    strip_choice: tuple[int, int] | None = None
    strip_score = -1

    for index, episode in enumerate(episodes):
        pred_mat, pred_obj, prob_true, h_steps, z_steps = scan_episode(model, episode, device)
        posterior.add(pred_mat, episode["mat"], pred_obj, episode["obj"], prob_true)
        time = int(episode["obs"].shape[0])
        actions = torch.from_numpy(episode["act"]).to(device)
        act = one_hot_action(actions.view(1, time), model.rssm.action_dim)[0]
        table = material_code("table")
        furnace = material_code("furnace")
        for t in range(CONTEXT, time - HORIZON, 32):
            future_act = act[t : t + HORIZON]
            pred_future = prior_field(model, h_steps[t], z_steps[t], future_act)
            true_future = episode["mat"][t + 1 : t + 1 + HORIZON]
            if true_future.shape[0] != HORIZON:
                continue
            for k in PRIOR_AT:
                priors[k].add_mat(pred_future[k - 1 : k], true_future[k - 1 : k])
            window_true = episode["mat"][t - CONTEXT : t + 1]
            station = int(((window_true == table) | (window_true == furnace)).sum())
            if station > strip_score and t + (STRIP_LEN - CONTEXT) < time:
                strip_score = station
                strip_choice = (index, t)
        del h_steps, z_steps
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  episode {index} len={time} posterior_cells={posterior.mat_total}", flush=True)

    if strip_choice is not None:
        index, t = strip_choice
        episode = episodes[index]
        start = t - CONTEXT + 1
        obs = torch.from_numpy(episode["obs"][start : start + STRIP_LEN]).unsqueeze(0).to(device)
        actions = torch.from_numpy(episode["act"][start : start + STRIP_LEN]).unsqueeze(0).to(device)
        pred = model.video_predict(obs, actions, context_len=CONTEXT)
        truth = obs[0, :CONTEXT].float() / 255.0
        ctx = pred.context_recon[0].permute(0, 2, 3, 1)
        imag = pred.imagined_recon[0]
        print(
            f"  context L1 {(ctx - truth).abs().mean().item():.4f} "
            f"imagined mean {imag.mean().item():.3f} std {imag.std().item():.3f}",
            flush=True,
        )
        FIG.mkdir(parents=True, exist_ok=True)
        path = FIG / f"field-head-open-loop-{name}.png"
        save_video_pred_strip(obs.detach().cpu(), pred.context_recon, pred.imagined_recon, CONTEXT, path)
        print(f"  strip {path.name} station_cells_in_context={strip_score}", flush=True)

    result = {
        "checkpoint": str(ckpt.relative_to(ROOT)).replace("\\", "/"),
        "posterior": posterior.as_dict(),
        "prior": {str(k): priors[k].as_dict() for k in PRIOR_AT},
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def majority_baseline(episodes: list[dict[str, np.ndarray]]) -> dict[str, float | int]:
    mats = np.concatenate([ep["mat"].reshape(-1) for ep in episodes], axis=0)
    mode = int(np.bincount(mats, minlength=13).argmax())
    tally = Tally(13, 8)
    tally.add_mat(np.full_like(mats, mode), mats)
    out = tally.as_dict()
    out["majority_code"] = mode
    out["majority_name"] = MATERIAL_NAMES[mode - 1] if mode > 0 else "outside"
    return out


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, float) and value != value:
        return None
    return value


def main() -> None:
    print("materials", list(enumerate(MATERIAL_NAMES)), flush=True)
    packed = collect_episodes()
    episodes = episodes_from(packed)
    device = get_device()
    configure_runtime(device)
    if device.type != "cuda":
        raise RuntimeError("field-head probe needs CUDA")
    print(f"vram free {require_free_vram(device):.2f} GiB", flush=True)
    baseline = majority_baseline(episodes)
    print("majority baseline", json.dumps(json_ready(baseline)), flush=True)
    objs = np.concatenate([ep["obj"].reshape(-1) for ep in episodes], axis=0)
    print("object counts", np.bincount(objs, minlength=8).tolist(), flush=True)
    results = {"baseline": baseline, "models": {}}
    for name, ckpt in CHECKPOINTS:
        results["models"][name] = score_model(name, ckpt, episodes, device)
    out = OUT / "summary.json"
    out.write_text(json.dumps(json_ready(results), indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
