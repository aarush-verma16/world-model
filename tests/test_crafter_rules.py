"""Ground-truth legality (finding 39 fix). Pure logic, no live Crafter env —
synthetic `info` dicts mirroring `crafter.constants.place` / `.make` exactly.
"""

from __future__ import annotations

import numpy as np
import torch

from training.crafter_rules import (
    ACTION_DIM,
    ACTION_INDEX,
    ITEM_NAMES,
    MATERIAL_NAMES,
    inventory_vector,
    legal_action_mask,
    legal_mask_from_counts,
    legal_mask_from_facts_torch,
    spatial_from_info,
)


def _base_info(**overrides) -> dict:
    info = {
        "inventory": {n: 0 for n in ITEM_NAMES},
        "facing_material": "grass",
        "facing_object_present": False,
        "nearby_materials": (),
    }
    info.update(overrides)
    return info


def test_missing_info_legalizes_everything() -> None:
    mask = legal_action_mask(None)
    assert mask.shape == (ACTION_DIM,)
    assert bool(mask.all())


def test_place_table_needs_two_wood_and_a_clear_where_tile() -> None:
    idx = ACTION_INDEX["place_table"]
    info = _base_info(inventory={**{n: 0 for n in ITEM_NAMES}, "wood": 1})
    assert not legal_action_mask(info)[idx]  # only 1 wood, needs 2
    info = _base_info(inventory={**{n: 0 for n in ITEM_NAMES}, "wood": 2})
    assert legal_action_mask(info)[idx]
    info = _base_info(
        inventory={**{n: 0 for n in ITEM_NAMES}, "wood": 2},
        facing_material="stone",  # not in place.table.where
    )
    assert not legal_action_mask(info)[idx]
    info = _base_info(
        inventory={**{n: 0 for n in ITEM_NAMES}, "wood": 2},
        facing_object_present=True,  # tile occupied
    )
    assert not legal_action_mask(info)[idx]


def test_make_wood_pickaxe_needs_table_nearby_and_wood() -> None:
    idx = ACTION_INDEX["make_wood_pickaxe"]
    info = _base_info(
        inventory={**{n: 0 for n in ITEM_NAMES}, "wood": 1}, nearby_materials=()
    )
    assert not legal_action_mask(info)[idx]  # no table nearby
    info = _base_info(
        inventory={**{n: 0 for n in ITEM_NAMES}, "wood": 1},
        nearby_materials=("table",),
    )
    assert legal_action_mask(info)[idx]
    info = _base_info(
        inventory={**{n: 0 for n in ITEM_NAMES}, "wood": 0},
        nearby_materials=("table",),
    )
    assert not legal_action_mask(info)[idx]  # table but no wood


def test_do_move_sleep_noop_always_legal() -> None:
    info = _base_info(inventory={n: 0 for n in ITEM_NAMES})
    mask = legal_action_mask(info)
    for name in ("noop", "do", "sleep", "move_left", "move_right", "move_up", "move_down"):
        assert bool(mask[ACTION_INDEX[name]])


def test_inventory_vector_fixed_order() -> None:
    inv = {"wood": 3, "stone": 1}
    vec = inventory_vector(inv)
    assert vec.shape == (len(ITEM_NAMES),)
    assert int(vec[ITEM_NAMES.index("wood")]) == 3
    assert int(vec[ITEM_NAMES.index("stone")]) == 1
    assert int(vec[ITEM_NAMES.index("coal")]) == 0
    assert inventory_vector(None).sum() == 0


def test_legal_mask_from_counts_batches_and_only_gates_uses() -> None:
    counts = np.zeros((2, len(ITEM_NAMES)), dtype=np.int64)
    counts[0, ITEM_NAMES.index("wood")] = 2
    counts[1, ITEM_NAMES.index("wood")] = 1
    mask = legal_mask_from_counts(counts)
    assert mask.shape == (2, ACTION_DIM)
    idx = ACTION_INDEX["place_table"]
    assert bool(mask[0, idx])  # 2 wood: uses-satisfied
    assert not bool(mask[1, idx])  # 1 wood: uses-unsatisfied
    # This mask has no facing/nearby ground truth -- it cannot see that
    # make_wood_pickaxe also needs a table nearby, only that wood >= 1.
    pick_idx = ACTION_INDEX["make_wood_pickaxe"]
    assert bool(mask[0, pick_idx])


def test_spatial_from_info_and_facts_mask_see_facing_and_nearby() -> None:
    info = _base_info(
        inventory={**{n: 0 for n in ITEM_NAMES}, "wood": 2},
        facing_material="stone",
        nearby_materials=("grass",),
    )
    facing_id, occupied, nearby = spatial_from_info(info)
    assert MATERIAL_NAMES[facing_id] == "stone"
    assert occupied == 0.0
    assert nearby[MATERIAL_NAMES.index("grass")] == 1.0
    assert spatial_from_info({}) is None

    counts = torch.zeros(1, len(ITEM_NAMES))
    counts[0, ITEM_NAMES.index("wood")] = 2
    face = torch.tensor([MATERIAL_NAMES.index("stone")])
    obj = torch.zeros(1)
    near = torch.zeros(1, len(MATERIAL_NAMES))
    mask = legal_mask_from_facts_torch(counts, face, obj, near)
    assert not bool(mask[0, ACTION_INDEX["place_table"]])
    assert bool(mask[0, ACTION_INDEX["do"]])

    face = torch.tensor([MATERIAL_NAMES.index("grass")])
    mask = legal_mask_from_facts_torch(counts, face, obj, near)
    assert bool(mask[0, ACTION_INDEX["place_table"]])
    assert not bool(mask[0, ACTION_INDEX["make_wood_pickaxe"]])

    near[0, MATERIAL_NAMES.index("table")] = 1.0
    counts[0, ITEM_NAMES.index("wood")] = 1
    mask = legal_mask_from_facts_torch(counts, face, obj, near)
    assert bool(mask[0, ACTION_INDEX["make_wood_pickaxe"]])
    assert bool(mask[0, ACTION_INDEX["do"]])
