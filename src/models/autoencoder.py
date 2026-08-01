"""M1 perception autoencoder with full-resolution U-Net skips.

Includes a 64x64 stem skip so the decoder can recover sharp Crafter sprites/UI.
A stem→RGB base path plus zero-init residual head gives a stable near-copy route
without shortcutting raw observation pixels into the output.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.SiLU(),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.SiLU(),
    )


class PerceptionAutoencoder(nn.Module):
    """Encode 64x64 frames to an embedding and decode with U-Net skips.

    Forward:
        obs `[B, 3, 64, 64]` → recon `[B, 3, 64, 64]`, embed `[B, embed_dim]`
    """

    def __init__(
        self,
        embed_dim: int = 8192,
        channels: tuple[int, ...] = (64, 128, 256, 512),
        stem_channels: int = 64,
    ) -> None:
        super().__init__()
        if len(channels) != 4:
            raise ValueError("expected 4 downsampling channel sizes")
        self.channels = tuple(channels)
        self.stem_channels = stem_channels
        self.embed_dim = embed_dim

        # Full-res stem so the decoder can copy fine sprites/UI via skips.
        self.stem = _conv_block(3, stem_channels)

        # Down: 64→32→16→8→4
        self.down = nn.ModuleList()
        prev = stem_channels
        for ch in channels:
            self.down.append(
                nn.Sequential(
                    nn.Conv2d(prev, ch, kernel_size=4, stride=2, padding=1),
                    nn.SiLU(),
                    _conv_block(ch, ch),
                )
            )
            prev = ch

        flat = channels[-1] * 4 * 4
        self.to_embed = nn.Identity() if embed_dim == flat else nn.Linear(flat, embed_dim)
        self.from_embed = nn.Identity() if embed_dim == flat else nn.Linear(embed_dim, flat)

        # Up blocks fuse with skips at 8, 16, 32, then 64 (stem).
        up_out = [channels[2], channels[1], channels[0], stem_channels]
        up_in = [channels[3], channels[2], channels[1], channels[0]]
        skip_ch = [channels[2], channels[1], channels[0], stem_channels]

        self.up = nn.ModuleList()
        for i in range(4):
            self.up.append(
                nn.ModuleDict(
                    {
                        "up": nn.Sequential(
                            nn.Upsample(scale_factor=2, mode="nearest"),
                            nn.Conv2d(up_in[i], up_out[i], kernel_size=3, padding=1),
                            nn.SiLU(),
                        ),
                        "fuse": _conv_block(up_out[i] + skip_ch[i], up_out[i]),
                    }
                )
            )

        # Near-copy path: project stem features to RGB, plus a zero-init residual.
        self.stem_to_rgb = nn.Conv2d(stem_channels, 3, kernel_size=1)
        self.delta_head = nn.Sequential(
            nn.Conv2d(stem_channels, stem_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(stem_channels, 3, kernel_size=1),
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

    def encode(self, obs: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Returns (embed, skips) where skips are [stem@64, d0@32, d1@16, d2@8, d3@4]."""
        if obs.ndim != 4 or obs.shape[1:] != (3, 64, 64):
            raise ValueError(f"expected [B,3,64,64], got {tuple(obs.shape)}")
        stem = self.stem(obs)
        skips: list[Tensor] = [stem]
        x = stem
        for block in self.down:
            x = block(x)
            skips.append(x)
        embed = self.to_embed(torch.flatten(x, 1))
        return embed, skips

    def decode(self, embed: Tensor, skips: list[Tensor]) -> Tensor:
        """Decode with U-Net skips including full-resolution stem."""
        if embed.ndim != 2 or embed.shape[1] != self.embed_dim:
            raise ValueError(f"expected embed [B,{self.embed_dim}], got {tuple(embed.shape)}")
        x = self.from_embed(embed).view(-1, self.channels[-1], 4, 4)
        fuse_skips = [skips[3], skips[2], skips[1], skips[0]]
        for block, skip in zip(self.up, fuse_skips, strict=True):
            x = block["up"](x)
            x = block["fuse"](torch.cat([x, skip], dim=1))
        base = self.stem_to_rgb(skips[0])
        delta = self.delta_head(x)
        return (base + delta).clamp(-1.0, 1.0)

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (recon, embed)."""
        embed, skips = self.encode(obs)
        recon = self.decode(embed, skips)
        return recon, embed
