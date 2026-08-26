"""Crafter 64x64 layout slices (world / HUD / avatar)."""

from __future__ import annotations

import torch

from models.crafter_layout import (
    AVATAR_H,
    AVATAR_W,
    HUD_H,
    HUD_W,
    WORLD_H,
    WORLD_W,
    avatar_slice,
    composite_hud,
    hud_slice,
    world_slice,
)


def test_layout_slice_shapes() -> None:
    img = torch.randn(2, 3, 64, 64)
    world = world_slice(img)
    hud = hud_slice(img)
    avatar = avatar_slice(img)
    assert world.shape == (2, 3, WORLD_H, WORLD_W)
    assert hud.shape == (2, 3, HUD_H, HUD_W)
    assert avatar.shape == (2, 3, AVATAR_H, AVATAR_W)


def test_world_and_hud_partition_the_frame() -> None:
    img = torch.zeros(1, 3, 64, 64)
    img[..., :WORLD_H, :WORLD_W] = 1.0
    img[..., WORLD_H : WORLD_H + HUD_H, :HUD_W] = 2.0
    assert torch.allclose(world_slice(img), torch.ones(1, 3, WORLD_H, WORLD_W))
    assert torch.allclose(hud_slice(img), torch.full((1, 3, HUD_H, HUD_W), 2.0))


def test_composite_hud_rejects_wrong_strip() -> None:
    world = torch.zeros(1, 3, 64, 64)
    bad = torch.zeros(1, 3, 8, 8)
    try:
        composite_hud(world, bad)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for HUD shape")
