"""Actor/critic input columns beyond raw RSSM `feat`.

M6/M17: `feat` alone. M18: append inventory counts / 9. M19: also append a
facing-material one-hot, a faced-object bit, and a nearby-material multi-hot.

Real collect uses ground truth from `CrafterEnv` info. Imagination argmax /
threshold-decodes the heads, because a rollout has no env. `do` is never
masked here; the columns exist so the actor can *see* "pickaxe and stone"
and choose `do` itself.
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
    spatial_from_info,
)


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


def features_real(
    world_model: WorldModel,
    feat: Tensor,
    info: dict[str, Any] | None,
) -> Tensor:
    """`feat` `[1, feat_dim]` plus ground-truth columns. Same leading shape."""
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
    if len(parts) == 1:
        return feat
    return torch.cat(parts, dim=-1)


def features_imagined(
    world_model: WorldModel,
    feat_detached: Tensor,
) -> tuple[Tensor, Tensor | None]:
    """`(feat_actor, legal_mask)`.

    `legal_mask` is `None` when neither head exists (m6/m17). Inventory-only
    models get the uses-only mask. Spatial models get facing + nearby too.
    """
    parts: list[Tensor] = [feat_detached]
    counts: Tensor | None = None
    if world_model.inventory_head is not None:
        inv_logits = world_model.predict_inventory(feat_detached)
        counts = inv_logits.argmax(dim=-1).float()
        parts.append(counts / 9.0)
    if world_model.spatial_head is None:
        mask = None if counts is None else legal_mask_from_counts_torch(counts)
        feat_actor = feat_detached if len(parts) == 1 else torch.cat(parts, dim=-1)
        return feat_actor, mask

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
    return torch.cat(parts, dim=-1), mask
