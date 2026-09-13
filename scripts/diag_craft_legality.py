"""Read-only: did M17 craft/mine actions have the items Crafter requires?

Replay stores pixels + actions, not `info['inventory']`. Inventory is read
from the 2x9 HUD strip (finding 04) by matching Crafter's own ItemView
templates. No training, no GPU.

Crafter rules (package data.yaml):
  place_table        needs wood >= 2
  make_wood_pickaxe  needs wood >= 1 and a table in a 1-tile Moore neighbourhood
  collect_stone      needs wood_pickaxe >= 1 on the facing tile

    conda activate worldmodel
    python scripts/diag_craft_legality.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import crafter.constants as crafter_constants
from crafter.engine import ItemView, Textures

from models.crafter_layout import HUD_TOP, TILE, VIEW_W
from training.replay_buffer import ReplayBuffer

from train_agent import load_yaml

ITEM_NAMES: tuple[str, ...] = tuple(str(n) for n in crafter_constants.items)
ACTION_NAMES: tuple[str, ...] = tuple(str(n) for n in crafter_constants.actions)
ITEM_INDEX = {n: i for i, n in enumerate(ITEM_NAMES)}
ACTION_INDEX = {n: i for i, n in enumerate(ACTION_NAMES)}


def _slot_xy(item: str) -> tuple[int, int]:
    idx = ITEM_INDEX[item]
    return idx % VIEW_W, idx // VIEW_W


def _hud_slot(obs: np.ndarray, item: str) -> np.ndarray:
    """Crop one HUD cell. `obs` `[..., 64, 64, 3]` → `[..., 7, 7, 3]`."""
    sx, sy = _slot_xy(item)
    y0 = HUD_TOP + sy * TILE
    x0 = sx * TILE
    return obs[..., y0 : y0 + TILE, x0 : x0 + TILE, :]


def _templates() -> dict[str, np.ndarray]:
    """Per-item HUD crops for amounts 0..9. Shape `[10, 7, 7, 3]` uint8."""
    textures = Textures(crafter_constants.root / "assets")
    view = ItemView(textures, [VIEW_W, 2])
    unit = np.array([TILE, TILE])
    out: dict[str, np.ndarray] = {}
    for item in ITEM_NAMES:
        slots = np.zeros((10, TILE, TILE, 3), dtype=np.uint8)
        sx, sy = _slot_xy(item)
        for amount in range(1, 10):
            inv = {n: 0 for n in ITEM_NAMES}
            inv[item] = amount
            canvas = view(inv, unit)  # `[63, 14, 3]` in Crafter (x, y)
            hud = np.transpose(canvas, (1, 0, 2))
            slots[amount] = hud[sy * TILE : (sy + 1) * TILE, sx * TILE : (sx + 1) * TILE]
        out[item] = slots
    return out


def decode_item(obs: np.ndarray, item: str, templates: dict[str, np.ndarray]) -> np.ndarray:
    """`obs` `[N, 64, 64, 3]` → int amounts `[N]` in 0..9."""
    crop = _hud_slot(obs, item).astype(np.int16)
    tmpl = templates[item].astype(np.int16)
    err = np.abs(crop[:, None] - tmpl[None]).mean(axis=(2, 3, 4))
    return err.argmin(axis=1).astype(np.int16)


def _self_check(templates: dict[str, np.ndarray]) -> None:
    import crafter

    env = crafter.Env(seed=0)
    obs = np.asarray(env.reset(), dtype=np.uint8)
    decoded = {
        k: int(decode_item(obs[None], k, templates)[0])
        for k in ("health", "food", "drink", "energy", "wood", "sapling", "wood_pickaxe")
    }
    true = {k: int(env._player.inventory[k]) for k in decoded}
    if decoded != true:
        raise RuntimeError(f"HUD decoder mismatch: decoded={decoded} true={true}")
    print(f"HUD decoder ok on crafter.Env reset: {decoded}")


def _summarize(name: str, n: int, d: dict[str, int]) -> None:
    print(f"\n{name}  n={n}")
    if n == 0:
        print("  (none)")
        return
    for k, v in d.items():
        print(f"  {k:28s}  {v:7d}  {100.0 * v / n:6.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/m17_xl_paper.yaml"))
    parser.add_argument("--replay", type=Path, default=None)
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help="0 = all episodes in the dump.",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    replay_path = Path(args.replay) if args.replay is not None else Path(cfg["train"]["replay_out"])
    if not replay_path.is_file():
        raise FileNotFoundError(replay_path)

    print("building HUD templates from Crafter assets...")
    templates = _templates()
    _self_check(templates)

    print(f"loading {replay_path} (CPU, no training)...")
    buffer = ReplayBuffer(seed=0, max_steps=None)
    buffer.load_state_dict(torch.load(replay_path, weights_only=False, map_location="cpu"))
    n_eps = len(buffer)
    cap = n_eps if int(args.max_episodes) <= 0 else min(n_eps, int(args.max_episodes))
    print(f"episodes={n_eps} steps={buffer.num_steps}  scanning {cap}")

    i_table = ACTION_INDEX["place_table"]
    i_pick = ACTION_INDEX["make_wood_pickaxe"]
    i_do = ACTION_INDEX["do"]
    i_sleep = ACTION_INDEX["sleep"]
    i_plant = ACTION_INDEX["place_plant"]

    n_steps = 0
    n_table = n_pick = n_do = n_sleep = n_plant = 0
    table_no_wood = table_has_wood = table_ok = table_blocked = 0
    pick_no_wood = pick_has_wood = pick_ok = pick_blocked = 0
    pick_already = 0
    steps_wood2 = steps_wood1 = steps_has_pick = 0
    eps_table_ok = eps_pick_ok = 0
    eps_ever_wood2 = 0
    # Last-200 finished lives (same window as finding 37).
    last_table_ok = last_pick_ok = 0
    last_n = 0

    parts = buffer._parts()
    for ei, ep in enumerate(parts[:cap]):
        obs = np.asarray(ep.obs.numpy(), dtype=np.uint8)
        act = np.asarray(ep.actions.numpy(), dtype=np.int64)
        t = int(act.shape[0])
        if t < 2:
            continue
        wood = decode_item(obs, "wood", templates)
        pickaxe = decode_item(obs, "wood_pickaxe", templates)
        n_steps += t
        n_table += int((act == i_table).sum())
        n_pick += int((act == i_pick).sum())
        n_do += int((act == i_do).sum())
        n_sleep += int((act == i_sleep).sum())
        n_plant += int((act == i_plant).sum())
        steps_wood2 += int((wood >= 2).sum())
        steps_wood1 += int((wood >= 1).sum())
        steps_has_pick += int((pickaxe >= 1).sum())

        table_m = act[:-1] == i_table
        pick_m = act[:-1] == i_pick
        dw = wood[1:] - wood[:-1]
        dp = pickaxe[1:] - pickaxe[:-1]

        tw = table_m & (wood[:-1] < 2)
        th = table_m & (wood[:-1] >= 2)
        tok = th & (dw == -2)
        tbl = th & (dw != -2)
        table_no_wood += int(tw.sum())
        table_has_wood += int(th.sum())
        table_ok += int(tok.sum())
        table_blocked += int(tbl.sum())

        pw = pick_m & (wood[:-1] < 1)
        ph = pick_m & (wood[:-1] >= 1)
        pok = ph & (dw == -1) & (dp == 1)
        pbl = ph & ~((dw == -1) & (dp == 1))
        pick_no_wood += int(pw.sum())
        pick_has_wood += int(ph.sum())
        pick_ok += int(pok.sum())
        pick_blocked += int(pbl.sum())
        pick_already += int((pick_m & (pickaxe[:-1] >= 1)).sum())

        ep_table = bool(tok.any())
        ep_pick = bool(pok.any())
        if ep_table:
            eps_table_ok += 1
        if ep_pick:
            eps_pick_ok += 1
        if bool((wood >= 2).any()):
            eps_ever_wood2 += 1
        if ei >= cap - 200:
            last_n += 1
            last_table_ok += int(ep_table)
            last_pick_ok += int(ep_pick)

        if (ei + 1) % 400 == 0 or ei + 1 == cap:
            print(f"  scanned {ei + 1}/{cap} episodes, {n_steps} steps", flush=True)

    print(f"\nreplay {replay_path}")
    print(f"steps={n_steps}  episodes={cap}")
    print(
        f"action mix: sleep={100.0 * n_sleep / n_steps:.2f}%  "
        f"plant={100.0 * n_plant / n_steps:.2f}%  do={100.0 * n_do / n_steps:.2f}%  "
        f"table={100.0 * n_table / n_steps:.2f}%  pickaxe={100.0 * n_pick / n_steps:.2f}%"
    )
    print(
        f"inventory occupancy: wood>=2 {100.0 * steps_wood2 / n_steps:.2f}% of steps  "
        f"wood>=1 {100.0 * steps_wood1 / n_steps:.2f}%  "
        f"wood_pickaxe>=1 {100.0 * steps_has_pick / n_steps:.2f}%"
    )
    print(
        f"episodes that ever had wood>=2: {eps_ever_wood2}/{cap} "
        f"({100.0 * eps_ever_wood2 / cap:.1f}%)"
    )
    print(
        f"episodes with a *successful* place_table (wood -= 2): "
        f"{eps_table_ok}/{cap} ({100.0 * eps_table_ok / cap:.1f}%)"
    )
    print(
        f"episodes with a *successful* make_wood_pickaxe: "
        f"{eps_pick_ok}/{cap} ({100.0 * eps_pick_ok / cap:.1f}%)"
    )
    print(
        f"last-{last_n} lives: table success {last_table_ok} "
        f"({100.0 * last_table_ok / max(last_n, 1):.1f}%)  "
        f"pickaxe success {last_pick_ok} ({100.0 * last_pick_ok / max(last_n, 1):.1f}%)"
    )

    _summarize(
        "place_table attempts (t to t+1)",
        table_no_wood + table_has_wood,
        {
            "wood < 2 (illegal)": table_no_wood,
            "wood >= 2": table_has_wood,
            "success wood-=2": table_ok,
            "had wood, did not place": table_blocked,
        },
    )
    _summarize(
        "make_wood_pickaxe attempts (t to t+1)",
        pick_no_wood + pick_has_wood,
        {
            "wood < 1 (illegal)": pick_no_wood,
            "wood >= 1": pick_has_wood,
            "success wood-=1 pick+=1": pick_ok,
            "had wood, did not craft": pick_blocked,
            "already held a pickaxe": pick_already,
        },
    )
    print(
        "\nread: illegal = pressing craft without items. "
        "had-wood-but-failed = facing/terrain (table) or no table in range (pickaxe)."
    )


if __name__ == "__main__":
    main()
