"""CNN decoder: reconstructs Crafter frames from an embedding / RSSM feature.

Used by the M1 skip-free encode→decode path and by the M3 world model, which
decodes live from `feat = concat(h, z_posterior)` — no auxiliary decoder, no
frozen copy, no encoder-embedding bypass (see `models.world_model`).

Upsampling is **sub-pixel convolution** (`Conv2d(in, out*4)` + `PixelShuffle`),
not `Upsample(mode="nearest")`. Nearest replicates each latent cell into a
2x2 block of *identical* values, so the following 3x3 conv sees a
piecewise-constant input and cannot recover where inside a cell it is. With a
4x4 latent every cell is 16x16 pixels, so that discarded sub-cell phase is
exactly what a 7px tree / zombie / sapling and the 1-2px inventory digits are
made of — hence flat 16x16 blocks and blurred HUD no matter the loss weights.
Sub-pixel conv learns one filter per output sub-position instead, and ICNR
init (`icnr_`) starts those four filters identical so the layer *begins* as a
nearest upsample and cannot checkerboard.

Residual 3x3 blocks are optional per upsample stage (`blocks`, mirroring the
encoder's `ResidualBlock`), off by default (`blocks=0`) — a live residual
decoder is heavier than a plain one, so only turn it on if VRAM allows.

`output_activation`: `"tanh"` (default) targets pixels in `[-1, 1]` — this is
the M1 perception-autoencoder convention (`models.preprocess`). `"linear"`
targets `[0, 1]` with `mean = conv_out + 0.5` (no squashing activation),
matching DreamerV3's `cnn_sigmoid=False` decoder — used by the M3 world model.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from models.encoder import ResidualBlock


def icnr_(weight: Tensor, upscale: int = 2) -> None:
    """Init a sub-pixel conv so all `upscale**2` sub-filters start identical.

    A default-initialized `PixelShuffle` conv gives every output sub-position
    an independent filter, so the first forward pass is checkerboarded and the
    decoder burns early steps unlearning that. ICNR (Aitken et al. 2017) seeds
    one filter and replicates it, making the layer exactly a nearest upsample
    at init; the sub-filters only diverge once they earn it.

    Args:
        weight: conv weight `[out_ch, in_ch, kh, kw]`, `out_ch` divisible by
            `upscale**2`. Modified in place.
        upscale: `PixelShuffle` factor.
    """
    if weight.ndim != 4:
        raise ValueError(f"expected conv weight [O, I, kh, kw], got {tuple(weight.shape)}")
    out_ch, in_ch, kernel_h, kernel_w = weight.shape
    sub = upscale * upscale
    if out_ch % sub != 0:
        raise ValueError(f"out_channels={out_ch} not divisible by upscale**2={sub}")
    seed = torch.empty(out_ch // sub, in_ch, kernel_h, kernel_w, dtype=weight.dtype)
    nn.init.kaiming_uniform_(seed, a=math.sqrt(5))
    with torch.no_grad():
        # PixelShuffle reads output channel c from input channels
        # [c*sub, (c+1)*sub), so repeat_interleave lines the copies up per c.
        weight.copy_(seed.repeat_interleave(sub, dim=0))


def _subpixel_conv(in_ch: int, out_ch: int, upscale: int = 2) -> nn.Conv2d:
    """3x3 conv sized for `PixelShuffle(upscale)`, ICNR-initialized."""
    conv = nn.Conv2d(in_ch, out_ch * upscale * upscale, kernel_size=3, padding=1)
    icnr_(conv.weight, upscale)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)
    return conv


def _upsample_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """×2 sub-pixel upsample then a 3×3 refine conv."""
    return nn.Sequential(
        _subpixel_conv(in_ch, out_ch),
        nn.PixelShuffle(2),
        nn.SiLU(),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.SiLU(),
    )


def _to_rgb_block(in_ch: int, out_channels: int, output_activation: str = "tanh") -> nn.Sequential:
    """Final ×2 to `out_channels` at 64×64.

    Both convs run at 32×32 and the upsample is the last op, so this is
    *cheaper* than a 64×64 3x3 conv while giving the RGB layer a hidden conv
    to build 1-2px pixel-art detail with.

    `output_activation="tanh"` squashes to `[-1, 1]` (M1 convention).
    `"linear"` leaves the conv output as-is; `Decoder.from_map` adds `+ 0.5`
    to center an untrained network near gray in `[0, 1]` (DreamerV3
    `cnn_sigmoid=False`), rather than squashing through `sigmoid`/`tanh`.
    """
    layers: list[nn.Module] = [
        nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
        nn.SiLU(),
        _subpixel_conv(in_ch, out_channels),
        nn.PixelShuffle(2),
    ]
    if output_activation == "tanh":
        layers.append(nn.Tanh())
    elif output_activation != "linear":
        raise ValueError(f"unknown output_activation {output_activation!r}")
    return nn.Sequential(*layers)


class Decoder(nn.Module):
    """Embedding → image: `[B, embed_dim]` → `[B, 3, 64, 64]`.

    Path: embed → `start_res`×`start_res` map, then sub-pixel ×2 stages up to
    64×64, with `blocks` residual 3x3 pairs at each resolution (mirrors the
    encoder; `0` skips them). Pixel range is `[-1, 1]` for
    `output_activation="tanh"` or `[0, 1]` for `"linear"` — see module
    docstring.
    """

    def __init__(
        self,
        embed_dim: int = 8192,
        out_channels: int = 3,
        channels: tuple[int, ...] = (512, 256, 128, 64),
        start_res: int = 4,
        blocks: int = 0,
        output_activation: str = "tanh",
    ) -> None:
        super().__init__()
        if start_res not in (4, 8):
            raise ValueError(f"start_res must be 4 or 8, got {start_res}")
        if blocks < 0:
            raise ValueError(f"blocks must be >= 0, got {blocks}")
        # 4→8→16→32→64 needs 4 upsamples; 8→16→32→64 needs 3.
        n_up = 4 if start_res == 4 else 3
        if len(channels) != n_up:
            raise ValueError(
                f"with start_res={start_res} expected {n_up} channel sizes, got {channels}"
            )

        flat_dim = channels[0] * start_res * start_res
        self.channels0 = channels[0]
        self.start_res = start_res
        self.channels = tuple(channels)
        self.blocks = blocks
        self.output_activation = output_activation
        self.fc: nn.Module = (
            nn.Identity() if embed_dim == flat_dim else nn.Linear(embed_dim, flat_dim)
        )

        stages: list[nn.Module] = []
        prev = channels[0]
        for ch in channels[1:]:
            stages.append(_upsample_block(prev, ch))
            if blocks > 0:
                stages.append(nn.Sequential(*[ResidualBlock(ch) for _ in range(blocks)]))
            prev = ch
        stages.append(_to_rgb_block(prev, out_channels, output_activation))
        self.up = nn.Sequential(*stages)
        self.embed_dim = embed_dim
        self.flat_dim = flat_dim

    def from_map(self, feat_map: Tensor) -> Tensor:
        """Decode a `start_res` feature map: `[B, C, S, S]` → `[B, 3, 64, 64]`."""
        expected = (self.channels0, self.start_res, self.start_res)
        if feat_map.ndim != 4 or feat_map.shape[1:] != expected:
            raise ValueError(
                f"expected feat map [B, {expected[0]}, {expected[1]}, {expected[2]}], "
                f"got {tuple(feat_map.shape)}"
            )
        out = self.up(feat_map)
        if self.output_activation == "linear":
            out = out + 0.5
        if out.shape[1:] != (3, 64, 64):
            raise RuntimeError(f"decoder produced unexpected shape {tuple(out.shape)}")
        return out

    def forward(self, embed: Tensor) -> Tensor:
        """Decode embeddings to images.

        Args:
            embed: `[B, embed_dim]`.

        Returns:
            float images `[B, 3, 64, 64]` (range per `output_activation`).
        """
        if embed.ndim != 2 or embed.shape[1] != self.embed_dim:
            raise ValueError(
                f"expected embed shape [B, {self.embed_dim}], got {tuple(embed.shape)}"
            )
        x = self.fc(embed)
        x = x.view(-1, self.channels0, self.start_res, self.start_res)
        return self.from_map(x)
