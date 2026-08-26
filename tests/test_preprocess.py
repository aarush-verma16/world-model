"""Pixel-scale invertibility. A swapped [-1,1] vs [0,1] convention is the
classic washed-out / inverted reconstruction failure.
"""

from __future__ import annotations

import torch

from models.preprocess import (
    nchw_float_to_nhwc_uint8,
    nchw_unit_to_nhwc_uint8,
    nhwc_uint8_to_nchw_float,
    nhwc_uint8_to_nchw_unit,
)


def test_m1_float_roundtrip_uint8() -> None:
    frames = torch.randint(0, 256, (2, 5, 64, 64, 3), dtype=torch.uint8)
    recon = nchw_float_to_nhwc_uint8(nhwc_uint8_to_nchw_float(frames))
    assert recon.shape == frames.shape
    assert recon.dtype == torch.uint8
    assert torch.equal(recon, frames)


def test_m3_unit_roundtrip_uint8() -> None:
    frames = torch.randint(0, 256, (2, 5, 64, 64, 3), dtype=torch.uint8)
    recon = nchw_unit_to_nhwc_uint8(nhwc_uint8_to_nchw_unit(frames))
    assert recon.shape == frames.shape
    assert torch.equal(recon, frames)


def test_nhwc_to_nchw_moves_channel_dim() -> None:
    frames = torch.zeros(3, 8, 64, 64, 3, dtype=torch.uint8)
    frames[..., 0] = 255  # red
    m1 = nhwc_uint8_to_nchw_float(frames)
    m3 = nhwc_uint8_to_nchw_unit(frames)
    assert m1.shape == (3, 8, 3, 64, 64)
    assert m3.shape == (3, 8, 3, 64, 64)
    assert torch.allclose(m1[:, :, 0], torch.ones(3, 8, 64, 64))
    assert torch.allclose(m3[:, :, 0], torch.ones(3, 8, 64, 64))
    assert torch.allclose(m1[:, :, 1], -torch.ones(3, 8, 64, 64))
    assert torch.allclose(m3[:, :, 1], torch.zeros(3, 8, 64, 64))


def test_black_and_white_extrema() -> None:
    black = torch.zeros(1, 4, 4, 3, dtype=torch.uint8)
    white = torch.full((1, 4, 4, 3), 255, dtype=torch.uint8)
    assert torch.allclose(nhwc_uint8_to_nchw_float(black), torch.full((1, 3, 4, 4), -1.0))
    assert torch.allclose(nhwc_uint8_to_nchw_float(white), torch.ones(1, 3, 4, 4))
    assert torch.allclose(nhwc_uint8_to_nchw_unit(black), torch.zeros(1, 3, 4, 4))
    assert torch.allclose(nhwc_uint8_to_nchw_unit(white), torch.ones(1, 3, 4, 4))
