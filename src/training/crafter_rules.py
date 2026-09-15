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
is the imagination-side mask: it only has *predicted* inventory (no
ground-truth facing/nearby), so it can only enforce the `uses` half of each
recipe. That gap is the documented approximation boundary — see finding 40.
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
