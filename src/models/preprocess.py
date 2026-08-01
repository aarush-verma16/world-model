"""Pixel normalization helpers for the perception stack.

Convention for M1 (and later world-model image IO):
- Env / dataset frames: uint8 `[..., H, W, C]` in `[0, 255]`
- Network tensors: float32 `[..., C, H, W]` in `[-1, 1]`

Keeping this conversion in one place avoids the classic washed-out / inverted
reconstruction failure from mismatched encoder input vs decoder target scales.
"""

from __future__ import annotations

import torch
from torch import Tensor


def nhwc_uint8_to_nchw_float(frames: Tensor) -> Tensor:
    """Convert uint8 NHWC `[..., H, W, C]` to float NCHW `[..., C, H, W]` in [-1, 1].

    Args:
        frames: uint8 or float tensor with channel-last layout.

    Returns:
        float32 tensor in `[-1, 1]` with channel-first layout.
    """
    x = frames.float() / 255.0
    x = x.mul(2.0).sub(1.0)
    return x.movedim(-1, -3).contiguous()


def nchw_float_to_nhwc_uint8(frames: Tensor) -> Tensor:
    """Convert float NCHW `[-1, 1]` back to uint8 NHWC for logging / GIFs.

    Args:
        frames: float tensor `[..., C, H, W]` roughly in `[-1, 1]`.

    Returns:
        uint8 tensor `[..., H, W, C]` in `[0, 255]`.
    """
    x = frames.detach().clamp(-1.0, 1.0)
    x = x.add(1.0).mul(0.5).mul(255.0).round().to(torch.uint8)
    return x.movedim(-3, -1).contiguous()
