"""CNN encoder: maps Crafter frames to a compact embedding.

DreamerV3's ImageEncoderResnet (cnn_blocks=2) downsamples with a stride-2
conv, then runs residual 3x3 blocks *at that resolution* before the next
downsample. A 4-layer stride-2 stack with no residuals aliases 8-12px
Crafter sprites/HUD into grass at the first downsample; the residual
blocks are what let those objects become features before they shrink to
the 4x4 flatten the RSSM consumes.

When `embed_dim == channels[-1] * 4 * 4`, the flatten is the embedding
(no mixing linear) so spatial layout is preserved.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def group_count(channels: int) -> int:
    """Largest GroupNorm group count that divides `channels`."""
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
    """Pre-activation residual block (DreamerV3 Conv2D `preact=True`)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = group_count(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        skip = x
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(F.silu(self.norm2(x)))
        return skip + x


class Encoder(nn.Module):
    """ResNet CNN: `[B, 3, 64, 64]` → `[B, embed_dim]`.

    Spatial path: 64 → 32 → 16 → 8 → 4. After each stride-2 conv, `blocks`
    residual 3x3 pairs run at that resolution (DreamerV3 default `blocks=2`).
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 8192,
        channels: tuple[int, ...] = (64, 128, 256, 512),
        blocks: int = 2,
    ) -> None:
        super().__init__()
        if len(channels) != 4:
            raise ValueError(f"expected 4 conv channel sizes, got {channels}")
        if blocks < 0:
            raise ValueError(f"blocks must be >= 0, got {blocks}")

        stages: list[nn.Module] = []
        prev = in_channels
        first_down: nn.Sequential | None = None
        for ch in channels:
            down = nn.Sequential(
                nn.Conv2d(prev, ch, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(group_count(ch), ch),
                nn.SiLU(),
            )
            if first_down is None:
                first_down = down
            parts: list[nn.Module] = [down]
            if blocks > 0:
                parts.extend(ResidualBlock(ch) for _ in range(blocks))
                parts.append(nn.SiLU())
            stages.append(nn.Sequential(*parts))
            prev = ch
        self.stages = nn.ModuleList(stages)
        # Tests/canary: first downsample conv. `conv[0]` is Conv2d.
        assert first_down is not None
        self.conv = first_down

        flat_dim = channels[-1] * 4 * 4
        self._flat_dim = flat_dim
        self.fc: nn.Module
        if embed_dim == flat_dim:
            self.fc = nn.Identity()
        else:
            self.fc = nn.Linear(flat_dim, embed_dim)
        self.embed_dim = embed_dim
        self.channels = tuple(channels)
        self.blocks = blocks

    def forward(self, obs: Tensor) -> Tensor:
        """Encode images.

        Args:
            obs: float images `[B, 3, 64, 64]` in `[-1, 1]`.

        Returns:
            embeddings `[B, embed_dim]`.
        """
        if obs.ndim != 4 or obs.shape[1:] != (3, 64, 64):
            raise ValueError(f"expected obs shape [B, 3, 64, 64], got {tuple(obs.shape)}")
        x = obs
        for stage in self.stages:
            x = stage(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)
