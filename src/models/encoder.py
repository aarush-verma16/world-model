"""CNN encoder: maps Crafter frames to a compact embedding.

Role in the Dreamer loop (M1): this is the "eyes" — compress a real 64x64x3
observation into an embedding vector. Later (M2+) the RSSM will consume this
embedding to form `z_posterior`. In M1 we only train encode→decode reconstruction.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class Encoder(nn.Module):
    """4-layer stride-2 CNN: `[B, 3, 64, 64]` → `[B, embed_dim]`.

    Spatial path: 64 → 32 → 16 → 8 → 4, then flatten (+ optional linear).
    When `embed_dim == channels[-1] * 4 * 4`, the flatten is the embedding
    (no mixing linear) so spatial layout is preserved for sharp reconstructions.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 8192,
        channels: tuple[int, ...] = (64, 128, 256, 512),
    ) -> None:
        super().__init__()
        if len(channels) != 4:
            raise ValueError(f"expected 4 conv channel sizes, got {channels}")

        layers: list[nn.Module] = []
        prev = in_channels
        for ch in channels:
            layers.extend(
                [
                    nn.Conv2d(prev, ch, kernel_size=4, stride=2, padding=1),
                    nn.SiLU(),
                ]
            )
            prev = ch
        self.conv = nn.Sequential(*layers)
        flat_dim = channels[-1] * 4 * 4
        self._flat_dim = flat_dim
        self.fc: nn.Module
        if embed_dim == flat_dim:
            self.fc = nn.Identity()
        else:
            self.fc = nn.Linear(flat_dim, embed_dim)
        self.embed_dim = embed_dim
        self.channels = tuple(channels)

    def forward(self, obs: Tensor) -> Tensor:
        """Encode images.

        Args:
            obs: float images `[B, 3, 64, 64]` in `[-1, 1]`.

        Returns:
            embeddings `[B, embed_dim]`.
        """
        if obs.ndim != 4 or obs.shape[1:] != (3, 64, 64):
            raise ValueError(f"expected obs shape [B, 3, 64, 64], got {tuple(obs.shape)}")
        x = self.conv(obs)
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)
