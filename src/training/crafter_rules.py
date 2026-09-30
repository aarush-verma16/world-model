"""Ground-truth Crafter action legality (finding 39 fix, not a diagnostic).

Reuses `crafter.constants.place` / `.make` — the exact tables
`crafter.objects.Player._place` / `._make` check — so a masked-legal action
can never be illegal by construction. This is not a learned approximation
for the real env; it only needs the `info` dict `envs.crafter_env.CrafterEnv`
now returns (`inventory`, `facing_material`, `facing_object_present`,
`nearby_materials`).

Scope: every `place_*` and `make_*` action. `do`, movement, `sleep`, `noop`
stay always legal — Crafter's `do` is a general-purpose probabilistic action
(mine/attack/harvest depending on what's faced), not a hard uses/nearby
gate, and masking it is not what finding 39 measured (98.5% of illegal
presses were `place_table` with wood < 2, not a bad `do`).

`legal_action_mask` is the real-collect mask (exact). `legal_mask_from_counts`
is the imagination-side mask when the model has inventory but no spatial
head: it only has *predicted* inventory, so it can only enforce the `uses`
half of each recipe (finding 40). `legal_mask_from_facts_torch` is the M19
mask: predicted facing material, faced-object bit, and nearby materials
close that gap for `place_*` / `make_*`. `do` stays legal in every mask —
it also chops trees and attacks, and finding 39 was not a bad `do`.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor

import crafter.constants as crafter_constants

ACTION_NAMES: tuple[str, ...] = tuple(str(n) for n in crafter_constants.actions)
ACTION_INDEX: dict[str, int] = {n: i for i, n in enumerate(ACTION_NAMES)}
ACTION_DIM: int = len(ACTION_NAMES)

ITEM_NAMES: tuple[str, ...] = tuple(str(n) for n in crafter_constants.items)
ITEM_INDEX: dict[str, int] = {n: i for i, n in enumerate(ITEM_NAMES)}

MATERIAL_NAMES: tuple[str, ...] = tuple(str(n) for n in crafter_constants.materials)
MATERIAL_INDEX: dict[str, int] = {n: i for i, n in enumerate(MATERIAL_NAMES)}

# Crafter's rendered world view: 9 wide x 7 tall tiles (the bottom 2 rows of
# the 9x9 frame are the inventory HUD). Cell (x, y) is world
# `player.pos + (x, y) - (4, 3)`, the same offset `engine.LocalView` draws.
LOCAL_GRID: tuple[int, int] = (9, 7)
LOCAL_CELLS: int = LOCAL_GRID[0] * LOCAL_GRID[1]
# Material code 0 is "outside the world"; code i + 1 is MATERIAL_NAMES[i].
LOCAL_MATERIAL_CLASSES: int = 1 + len(MATERIAL_NAMES)
OBJECT_NAMES: tuple[str, ...] = (
    "none",
    "zombie",
    "skeleton",
    "cow",
    "plant",
    "plant_ripe",
    "arrow",
    "fence",
)
OBJECT_INDEX: dict[str, int] = {n: i for i, n in enumerate(OBJECT_NAMES)}
LOCAL_OBJECT_CLASSES: int = len(OBJECT_NAMES)

_PLACE: dict[str, dict[str, Any]] = dict(crafter_constants.place)
_MAKE: dict[str, dict[str, Any]] = dict(crafter_constants.make)

# Precompute which action indices are place_*/make_* recipes, in a fixed
# order, so the counts-only mask can be a pure array op (no python loop over
# dict items inside a training step).
_PLACE_ACTIONS: tuple[tuple[int, str, dict[str, Any]], ...] = tuple(
    (ACTION_INDEX[f"place_{name}"], name, spec)
    for name, spec in _PLACE.items()
    if f"place_{name}" in ACTION_INDEX
)
_MAKE_ACTIONS: tuple[tuple[int, str, dict[str, Any]], ...] = tuple(
    (ACTION_INDEX[f"make_{name}"], name, spec)
    for name, spec in _MAKE.items()
    if f"make_{name}" in ACTION_INDEX
)


def _uses_satisfied(inv: dict[str, int], uses: dict[str, int]) -> bool:
    return all(int(inv.get(k, 0)) >= v for k, v in uses.items())


def _place_legal(name: str, spec: dict[str, Any], info: dict[str, Any]) -> bool:
    if info.get("facing_object_present", False):
        return False
    if info.get("facing_material") not in spec["where"]:
        return False
    return _uses_satisfied(info.get("inventory") or {}, spec["uses"])


def _make_legal(name: str, spec: dict[str, Any], info: dict[str, Any]) -> bool:
    nearby = info.get("nearby_materials") or ()
    if not all(util in nearby for util in spec["nearby"]):
        return False
    return _uses_satisfied(info.get("inventory") or {}, spec["uses"])


def legal_action_mask(info: dict[str, Any] | None) -> np.ndarray:
    """Exact real-env legality. `info` is a `CrafterEnv.step`/`.reset` dict.

    Returns bool `[ACTION_DIM]`. A missing `info` (or a missing extra key)
    legalizes that action — never block on absent ground truth, only on a
    *known* violated recipe.
    """
    mask = np.ones(ACTION_DIM, dtype=bool)
    if not info:
        return mask
    for idx, name, spec in _PLACE_ACTIONS:
        mask[idx] = _place_legal(name, spec, info)
    for idx, name, spec in _MAKE_ACTIONS:
        mask[idx] = _make_legal(name, spec, info)
    return mask


def legal_mask_from_counts(
    counts: np.ndarray,
    item_names: Sequence[str] = ITEM_NAMES,
) -> np.ndarray:
    """Inventory-only legality for imagination (no facing/nearby ground truth).

    `counts` `[..., len(item_names)]` (raw counts, not normalized). Returns
    bool `[..., ACTION_DIM]`. Only gates the `uses` half of each recipe —
    document this gap wherever it is called (finding 40): a `place_table`
    can pass this mask with no legal facing tile, and a `make_wood_pickaxe`
    can pass with no table nearby. It still removes the dominant failure
    mode measured in finding 39 (wood < 2 on 98.5% of `place_table` presses).
    """
    name_to_i = {n: i for i, n in enumerate(item_names)}
    leading = counts.shape[:-1]
    mask = np.ones((*leading, ACTION_DIM), dtype=bool)

    def _uses_ok(spec_uses: dict[str, int]) -> np.ndarray:
        ok = np.ones(leading, dtype=bool)
        for item, amount in spec_uses.items():
            if item not in name_to_i:
                continue
            ok &= counts[..., name_to_i[item]] >= amount
        return ok

    for idx, _name, spec in _PLACE_ACTIONS:
        mask[..., idx] = _uses_ok(spec["uses"])
    for idx, _name, spec in _MAKE_ACTIONS:
        mask[..., idx] = _uses_ok(spec["uses"])
    return mask


def legal_mask_from_counts_torch(
    counts: Tensor,
    item_names: Sequence[str] = ITEM_NAMES,
) -> Tensor:
    """Torch, on-device version of `legal_mask_from_counts` (imagination).

    Kept separate (not just `torch.from_numpy`) so `training.imagine` never
    forces a GPU->CPU sync every imagined step. Same semantics/limits as the
    numpy version: inventory-`uses` only, no facing/nearby ground truth.
    """
    name_to_i = {n: i for i, n in enumerate(item_names)}
    leading = counts.shape[:-1]
    mask = torch.ones((*leading, ACTION_DIM), dtype=torch.bool, device=counts.device)

    def _uses_ok(spec_uses: dict[str, int]) -> Tensor:
        ok = torch.ones(leading, dtype=torch.bool, device=counts.device)
        for item, amount in spec_uses.items():
            if item not in name_to_i:
                continue
            ok &= counts[..., name_to_i[item]] >= amount
        return ok

    for idx, _name, spec in _PLACE_ACTIONS:
        mask[..., idx] = _uses_ok(spec["uses"])
    for idx, _name, spec in _MAKE_ACTIONS:
        mask[..., idx] = _uses_ok(spec["uses"])
    return mask


def inventory_vector(inventory: dict[str, Any] | None) -> np.ndarray:
    """`info['inventory']` -> fixed-order int array `[len(ITEM_NAMES)]`."""
    out = np.zeros(len(ITEM_NAMES), dtype=np.int64)
    if not inventory:
        return out
    for name, i in ITEM_INDEX.items():
        out[i] = int(inventory.get(name, 0))
    return out


def spatial_from_info(
    info: dict[str, Any] | None,
) -> tuple[int, float, np.ndarray] | None:
    """`info` -> `(facing_id, object_present, nearby_multihot)`.

    `facing_id` indexes `MATERIAL_NAMES`. `nearby_multihot` is float32
    `[len(MATERIAL_NAMES)]`. Returns `None` when `facing_material` is absent
    so replay can store `has_spatial=0` instead of a fake grass tile.
    """
    if not info or "facing_material" not in info:
        return None
    name = str(info["facing_material"])
    if name not in MATERIAL_INDEX:
        return None
    nearby = np.zeros(len(MATERIAL_NAMES), dtype=np.float32)
    for mat in info.get("nearby_materials") or ():
        idx = MATERIAL_INDEX.get(str(mat))
        if idx is not None:
            nearby[idx] = 1.0
    occupied = 1.0 if bool(info.get("facing_object_present")) else 0.0
    return MATERIAL_INDEX[name], occupied, nearby


def local_map_from_info(info: dict[str, Any] | None) -> tuple[np.ndarray, np.ndarray] | None:
    """`info` -> `(materials [LOCAL_CELLS] uint8, objects [LOCAL_CELLS] uint8)`.

    Flattened x-major (`x * 7 + y`). `None` when the env did not report a
    local map, so replay stores `has_local=0` instead of an all-outside grid.
    """
    if not info or "local_materials" not in info or "local_objects" not in info:
        return None
    mats = np.asarray(info["local_materials"], dtype=np.uint8).reshape(-1)
    objs = np.asarray(info["local_objects"], dtype=np.uint8).reshape(-1)
    if mats.shape != (LOCAL_CELLS,) or objs.shape != (LOCAL_CELLS,):
        return None
    return mats, objs


def legal_mask_from_facts_torch(
    counts: Tensor,
    facing_id: Tensor,
    facing_object: Tensor,
    nearby: Tensor,
) -> Tensor:
    """Imagination legality once facing and nearby are predicted (finding 45).

    `counts` `[..., n_items]`, `facing_id` `[...]` long, `facing_object`
    `[...]` (true when the faced tile is occupied), `nearby` `[..., n_materials]`
    (true when that material is in the 1-tile neighborhood). Returns bool
    `[..., ACTION_DIM]`.

    Starts from the inventory-`uses` mask, then requires `place_*` to face a
    material in that recipe's `where` list with no object on the tile, and
    `make_*` to have every `nearby` utility. `do`, movement, `sleep`, and
    `noop` stay legal: `do` mines, chops, and attacks.
    """
    mask = legal_mask_from_counts_torch(counts)
    occupied = (
        facing_object
        if facing_object.dtype == torch.bool
        else facing_object > 0.5
    )
    near = nearby if nearby.dtype == torch.bool else nearby > 0.5
    for idx, _name, spec in _PLACE_ACTIONS:
        where_ok = torch.zeros(facing_id.shape, dtype=torch.bool, device=facing_id.device)
        for mid in (MATERIAL_INDEX[m] for m in spec["where"] if m in MATERIAL_INDEX):
            where_ok = where_ok | (facing_id == mid)
        mask[..., idx] = mask[..., idx] & where_ok & ~occupied
    for idx, _name, spec in _MAKE_ACTIONS:
        near_ok = torch.ones(facing_id.shape, dtype=torch.bool, device=facing_id.device)
        for util in spec.get("nearby", ()):
            if util not in MATERIAL_INDEX:
                continue
            near_ok = near_ok & near[..., MATERIAL_INDEX[util]]
        mask[..., idx] = mask[..., idx] & near_ok
    return mask
