"""Dump one probe frame and what a benchmark checkpoint's field head says about it.

Inference only (no training). Reuses the probe episodes cached by
`scripts/probe_field_head.py` and writes `paper/data/probes/field_example.npz`,
which `build_assets.py` turns into the qualitative field figure.

    c:/Users/Aarus/miniforge3/envs/worldmodel/python.exe paper/scripts/dump_field_example.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("probe_field_head", ROOT / "scripts" / "probe_field_head.py")
probe = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(probe)

OUT = ROOT / "paper" / "data" / "probes" / "field_example.npz"
RUN = "m22_field_craft"
HORIZON = 15


def pick_frame(episodes: list[dict[str, np.ndarray]]) -> tuple[int, int]:
    """Frame with the most station cells, preferring daylight; needs `HORIZON` steps of history."""
    table, furnace = probe.material_code("table"), probe.material_code("furnace")
    best, choice = -1.0, (0, HORIZON)
    for index, ep in enumerate(episodes):
        for t in range(HORIZON, int(ep["obs"].shape[0])):
            station = int(((ep["mat"][t] == table) | (ep["mat"][t] == furnace)).sum())
            bright = float(ep["obs"][t].mean()) / 255.0
            score = station + (0.5 if bright > 0.35 else 0.0) + 0.1 * bright
            if station >= 2 and score > best:
                best, choice = score, (index, t)
    return choice


@torch.no_grad()
def main() -> None:
    episodes = probe.episodes_from(probe.collect_episodes())
    index, t = pick_frame(episodes)
    ep = episodes[index]
    device = probe.get_device()
    probe.configure_runtime(device)

    with open(probe.SIZE_CFG, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    model = probe.build_model(cfg).to(device).eval()
    ckpt = dict(probe.CHECKPOINTS)[RUN]
    payload = torch.load(ckpt, weights_only=False, map_location="cpu")
    model.load_state_dict(payload["world_model"], strict=True)
    payload.clear()

    obs = torch.from_numpy(ep["obs"][: t + 1]).to(device)
    actions = torch.from_numpy(ep["act"][: t + 1]).to(device)
    embed = torch.cat([model.encode(obs[s : s + 32].unsqueeze(0))[0] for s in range(0, t + 1, 32)], dim=0)
    act = probe.one_hot_action(actions.view(1, -1), model.rssm.action_dim)[0]
    state = model.rssm.initial(1, device=device)
    prev = torch.zeros(1, model.rssm.action_dim, device=device)
    h_start = z_start = None
    post_probs = None
    for s in range(t + 1):
        first = torch.ones(1, device=device) if s == 0 else None
        state, _zp, _pl, post_logits = model.rssm.obs_step(state, prev, embed[s : s + 1], first)
        z_mode = probe.mode_stoch(post_logits)
        if s == t - HORIZON:
            h_start, z_start = state.h, z_mode
        if s == t:
            mat_logits, _ = model.predict_local_map(probe.rssm_features(state.h, z_mode))
            post_probs = torch.softmax(mat_logits[0].float(), dim=-1).cpu().numpy()
        prev = act[s : s + 1]

    h, z = h_start, z_start
    for s in range(t - HORIZON, t):
        h, _z_sample, prior_logits = model.rssm.img_step(h, z, act[s : s + 1])
        z = probe.mode_stoch(prior_logits)
    mat_logits, _ = model.predict_local_map(probe.rssm_features(h, z))
    prior_probs = torch.softmax(mat_logits[0].float(), dim=-1).cpu().numpy()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        frame=ep["obs"][t],
        true_mat=ep["mat"][t],
        post_probs=post_probs,
        prior_probs=prior_probs,
        episode=np.asarray([index]),
        step=np.asarray([t]),
        horizon=np.asarray([HORIZON]),
        run=np.asarray([RUN]),
    )
    table = probe.material_code("table")
    cells = ep["mat"][t] == table
    print(f"episode {index} step {t} table cells {int(cells.sum())} "
          f"post P(table) {post_probs[cells, table].round(3).tolist()} "
          f"prior P(table) {prior_probs[cells, table].round(3).tolist()}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
