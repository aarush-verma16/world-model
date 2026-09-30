"""Actor/critic input columns beyond raw RSSM `feat`.

M6/M17: `feat` alone. M18: append inventory counts / 9. M19: also append a
facing-material one-hot, a faced-object bit, and a nearby-material multi-hot.
M20: also append the 9x7 local map as per-cell material + object
distributions (finding 47).

Real collect uses ground truth from `CrafterEnv` info when
`world_model.actor_input_source == "truth"` (m18/m19). With `"predicted"`
(m20) real collect uses the model's own heads, exactly like imagination, so
the acting policy only ever sees pixels -> `[h, z]` -> heads; privileged
info is a training target, never an input. Imagination always decodes the
heads, because a rollout has no env. `do` is never masked here; the columns
exist so the actor can *see* "pickaxe and stone" and choose `do` itself.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from models.world_model import WorldModel
from training.crafter_rules import (
    ITEM_NAMES,
    MATERIAL_NAMES,
    inventory_vector,
    legal_mask_from_counts_torch,
    legal_mask_from_facts_torch,
    local_map_from_info,
    spatial_from_info,
)

ACTOR_INPUT_SOURCES = ("truth", "predicted")


def _spatial_columns(
    facing_id: Tensor,
    facing_object: Tensor,
    nearby: Tensor,
    *,
    n_materials: int,
    dtype: torch.dtype,
) -> Tensor:
    """`[..., n_materials + 1 + n_materials]`."""
    one_hot = torch.nn.functional.one_hot(facing_id.long(), n_materials).to(dtype=dtype)
    occupied = facing_object.to(dtype=dtype).reshape(*facing_id.shape, 1)
    return torch.cat([one_hot, occupied, nearby.to(dtype=dtype)], dim=-1)


def _local_columns_from_codes(
    local_mat: Tensor,
    local_obj: Tensor,
    *,
    n_material_classes: int,
    n_object_classes: int,
    dtype: torch.dtype,
) -> Tensor:
    """Codes `[..., C]` x2 -> one-hot `[..., C * (M + O)]` (materials first)."""
    mat = torch.nn.functional.one_hot(local_mat.long(), n_material_classes).to(dtype=dtype)
    obj = torch.nn.functional.one_hot(local_obj.long(), n_object_classes).to(dtype=dtype)
    return torch.cat([mat.flatten(-2), obj.flatten(-2)], dim=-1)


def _local_columns_predicted(world_model: WorldModel, feat_detached: Tensor) -> Tensor:
    """Head softmax `[..., C * (M + O)]`, same layout as `_local_columns_from_codes`.

    Probabilities rather than argmax: a 50/50 "stone or path" cell is more
    honest to the actor than a hard guess, and one-hot truth is the limit a
    confident head converges to.
    """
    mat_logits, obj_logits = world_model.predict_local_map(feat_detached)
    mat = torch.softmax(mat_logits.float(), dim=-1).to(dtype=feat_detached.dtype)
    obj = torch.softmax(obj_logits.float(), dim=-1).to(dtype=feat_detached.dtype)
    return torch.cat([mat.flatten(-2), obj.flatten(-2)], dim=-1)


def _check_source(world_model: WorldModel) -> str:
    source = str(getattr(world_model, "actor_input_source", "truth"))
    if source not in ACTOR_INPUT_SOURCES:
        raise ValueError(f"actor_input_source={source!r} not in {ACTOR_INPUT_SOURCES}")
    return source


def features_real(
    world_model: WorldModel,
    feat: Tensor,
    info: dict[str, Any] | None,
) -> Tensor:
    """`feat` `[1, feat_dim]` -> `[1, feat_dim + actor_extra_dim]` for a real env step.

    `"truth"` source reads `info`; `"predicted"` ignores it and decodes the
    heads (same columns imagination sees).
    """
    if _check_source(world_model) == "predicted":
        return features_imagined(world_model, feat.detach())[0]
    parts: list[Tensor] = [feat]
    if world_model.inventory_head is not None:
        inv = inventory_vector(info.get("inventory") if info else None)
        inv_t = torch.from_numpy(inv).to(device=feat.device, dtype=feat.dtype)
        parts.append(inv_t.unsqueeze(0) / 9.0)
    if world_model.spatial_head is not None:
        n = int(world_model.spatial_head.n_materials)
        parsed = spatial_from_info(info)
        if parsed is None:
            parts.append(torch.zeros(1, n + 1 + n, device=feat.device, dtype=feat.dtype))
        else:
            facing_id, occupied, nearby = parsed
            face_t = torch.tensor([facing_id], device=feat.device, dtype=torch.long)
            obj_t = torch.tensor([occupied], device=feat.device, dtype=feat.dtype)
            near_t = torch.from_numpy(np.asarray(nearby, dtype=np.float32)).to(
                device=feat.device, dtype=feat.dtype
            )
            parts.append(_spatial_columns(face_t, obj_t, near_t.unsqueeze(0), n_materials=n, dtype=feat.dtype))
    if world_model.local_map_head is not None:
        head = world_model.local_map_head
        parsed_local = local_map_from_info(info)
        if parsed_local is None:
            width = head.cells * (head.n_material_classes + head.n_object_classes)
            parts.append(torch.zeros(1, width, device=feat.device, dtype=feat.dtype))
        else:
            mats, objs = parsed_local
            parts.append(
                _local_columns_from_codes(
                    torch.from_numpy(mats.astype(np.int64)).to(feat.device).unsqueeze(0),
                    torch.from_numpy(objs.astype(np.int64)).to(feat.device).unsqueeze(0),
                    n_material_classes=head.n_material_classes,
                    n_object_classes=head.n_object_classes,
                    dtype=feat.dtype,
                )
            )
    if len(parts) == 1:
        return feat
    return torch.cat(parts, dim=-1)


def features_from_batch(
    world_model: WorldModel,
    feat_detached: Tensor,
    batch: dict[str, Tensor],
) -> Tensor:
    """Actor input for replay posteriors (behavior cloning, finding 48).

    Args:
        feat_detached: `[B, T, feat_dim]` posterior features, no grad.
        batch: replay batch from `ReplayBuffer.sample` on `feat_detached.device`.

    Returns:
        `[B, T, feat_dim + actor_extra_dim]`, built the same way real collect
        builds it: truth columns from the batch fields for `"truth"`, head
        decodes for `"predicted"`.
    """
    if _check_source(world_model) == "predicted":
        return features_imagined(world_model, feat_detached)[0]
    dtype = feat_detached.dtype
    parts: list[Tensor] = [feat_detached]
    if world_model.inventory_head is not None:
        parts.append(batch["inventory"].to(dtype=dtype) / 9.0)
    if world_model.spatial_head is not None:
        n = int(world_model.spatial_head.n_materials)
        parts.append(
            _spatial_columns(
                batch["facing"],
                batch["facing_object"],
                batch["nearby"],
                n_materials=n,
                dtype=dtype,
            )
        )
    if world_model.local_map_head is not None:
        head = world_model.local_map_head
        parts.append(
            _local_columns_from_codes(
                batch["local_mat"],
                batch["local_obj"],
                n_material_classes=head.n_material_classes,
                n_object_classes=head.n_object_classes,
                dtype=dtype,
            )
        )
    if len(parts) == 1:
        return feat_detached
    return torch.cat(parts, dim=-1)


def features_imagined(
    world_model: WorldModel,
    feat_detached: Tensor,
) -> tuple[Tensor, Tensor | None]:
    """`feat_detached` `[..., feat_dim]` -> `(feat_actor [..., feat_dim + extra], legal_mask)`.

    `legal_mask` is `None` when neither head exists (m6/m17). Inventory-only
    models get the uses-only mask. Spatial models get facing + nearby too.
    The local map only adds columns; it does not change the mask.
    """
    parts: list[Tensor] = [feat_detached]
    counts: Tensor | None = None
    if world_model.inventory_head is not None:
        inv_logits = world_model.predict_inventory(feat_detached)
        counts = inv_logits.argmax(dim=-1).float()
        parts.append(counts / 9.0)
    mask: Tensor | None = None
    if world_model.spatial_head is None:
        mask = None if counts is None else legal_mask_from_counts_torch(counts)
    else:
        mat_logits, obj_logit, near_logits = world_model.predict_spatial(feat_detached)
        facing_id = mat_logits.argmax(dim=-1)
        occupied = (obj_logit.squeeze(-1) > 0).to(dtype=feat_detached.dtype)
        nearby = (near_logits > 0).to(dtype=feat_detached.dtype)
        n = int(world_model.spatial_head.n_materials)
        if n != len(MATERIAL_NAMES):
            raise RuntimeError(
                f"spatial head n_materials={n} != len(MATERIAL_NAMES)={len(MATERIAL_NAMES)}"
            )
        parts.append(
            _spatial_columns(
                facing_id,
                occupied,
                nearby,
                n_materials=n,
                dtype=feat_detached.dtype,
            )
        )
        if counts is None:
            counts = torch.zeros(
                *facing_id.shape,
                len(ITEM_NAMES),
                device=feat_detached.device,
                dtype=feat_detached.dtype,
            )
        mask = legal_mask_from_facts_torch(counts, facing_id, occupied, nearby)
    if world_model.local_map_head is not None:
        parts.append(_local_columns_predicted(world_model, feat_detached))
    feat_actor = feat_detached if len(parts) == 1 else torch.cat(parts, dim=-1)
    return feat_actor, mask
