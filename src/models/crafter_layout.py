"""Crafter 64x64 observation layout (from `crafter.env.Env.render`).

`view=(9, 9)`, `size=(64, 64)` → `unit = 64 // 9 = 7` px per tile.
16 inventory items → `ceil(16/9)=2` HUD rows.
Local view is 9×7 tiles; HUD is concatenated on y, then the canvas is
transposed, so the gym image is:

- world: rows `[0, 49)`, cols `[0, 63)` (7×9 tiles)
- HUD: rows `[49, 63)`, cols `[0, 63)` (2×9 slots)
- row 63 and col 63 are unused pad

The player is the centre tile of the local grid (`offset = grid // 2`).
"""

from __future__ import annotations

from torch import Tensor

TILE = 7
VIEW_W = 9
LOCAL_H = 7
HUD_ROWS = 2
WORLD_H = LOCAL_H * TILE  # 49
WORLD_W = VIEW_W * TILE  # 63
HUD_TOP = WORLD_H  # 49
HUD_H = HUD_ROWS * TILE  # 14
HUD_W = WORLD_W  # 63
# Local grid is [9, 7] (x, y); offset = [4, 3]; obs[h, w] = canvas[x, y].
PLAYER_Y = 3 * TILE  # 21
PLAYER_X = 4 * TILE  # 28
# 3×3 tiles around the player (facing / held item live in this crop).
AVATAR_Y0 = PLAYER_Y - TILE  # 14
AVATAR_X0 = PLAYER_X - TILE  # 21
AVATAR_H = 3 * TILE  # 21
AVATAR_W = 3 * TILE  # 21


def world_slice(img: Tensor) -> Tensor:
    """Local-view pixels `[..., C, 49, 63]` from `[..., C, 64, 64]`."""
    return img[..., :WORLD_H, :WORLD_W]


def hud_slice(img: Tensor) -> Tensor:
    """Inventory bar `[..., C, 14, 63]` from `[..., C, 64, 64]`."""
    return img[..., HUD_TOP : HUD_TOP + HUD_H, :HUD_W]


def avatar_slice(img: Tensor) -> Tensor:
    """3×3 tiles around the player `[..., C, 21, 21]`."""
    y1 = AVATAR_Y0 + AVATAR_H
    x1 = AVATAR_X0 + AVATAR_W
    return img[..., AVATAR_Y0:y1, AVATAR_X0:x1]


def composite_hud(world: Tensor, hud: Tensor) -> Tensor:
    """Paste a `[..., 3, 14, 63]` HUD strip onto a 64×64 world recon."""
    if hud.shape[-2:] != (HUD_H, HUD_W):
        raise ValueError(f"expected HUD [..., {HUD_H}, {HUD_W}], got {tuple(hud.shape)}")
    out = world.clone()
    out[..., HUD_TOP : HUD_TOP + HUD_H, :HUD_W] = hud
    return out
